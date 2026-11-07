from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import torch


RESUME_SCHEMA_VERSION = 1
WARMUP_RECOVERY_KIND = "warmup_recovery"
PBT_RECOVERY_KIND = "pbt_recovery"
JOINT_RECOVERY_KIND = "joint_recovery"


class RecoveryError(RuntimeError):
    pass


def _fsync_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_write_bytes(path: Path, writer: Callable[[Any], None]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            writer(handle)
            _fsync_file(handle)
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    _atomic_write_bytes(Path(path), lambda handle: torch.save(dict(payload), handle))


def atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(Path(path), lambda handle: handle.write(encoded))


def signature(fields: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = json.loads(json.dumps(dict(fields), sort_keys=True, default=str))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"fields": normalized, "sha256": hashlib.sha256(encoded).hexdigest()}


def validate_signature(
    saved: Mapping[str, Any], current: Mapping[str, Any], *, strict: bool = True
) -> None:
    saved_fields = dict(saved.get("fields", {}))
    current_fields = dict(current.get("fields", {}))
    hard_keys = {
        "num_ground_stations",
        "num_gs_per_rb",
        "dqn_scheduler_impl",
        "dqn_state_dim",
        "dqn_action_dim",
        "power_state_dim",
        "power_action_dim",
        "phase_state_dim",
        "phase_groups",
        "phase_policy_dist",
        "alternating_schedule",
    }
    keys = set(saved_fields) | set(current_fields) if strict else hard_keys
    mismatches = {
        key: {"checkpoint": saved_fields.get(key), "runtime": current_fields.get(key)}
        for key in sorted(keys)
        if saved_fields.get(key) != current_fields.get(key)
    }
    if mismatches:
        raise RecoveryError(
            "Recovery checkpoint is incompatible with the current training configuration: "
            + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
        )


def capture_rng_state(*, env: Any = None) -> Dict[str, Any]:
    env_rng = getattr(env, "_rng", None) if env is not None else None
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "environment": (
            dict(env_rng.bit_generator.state)
            if env_rng is not None and hasattr(env_rng, "bit_generator")
            else None
        ),
    }


def restore_rng_state(state: Optional[Mapping[str, Any]], *, env: Any = None) -> None:
    if not state:
        return
    python_state = state.get("python", state.get("py"))
    numpy_state = state.get("numpy_global", state.get("np"))
    torch_state = state.get("torch_cpu", state.get("torch"))
    if python_state is None or numpy_state is None or torch_state is None:
        raise RecoveryError("RNG checkpoint is missing Python, NumPy, or Torch CPU state")
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    cuda_state = state.get("torch_cuda", state.get("cuda"))
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    env_state = state.get("environment")
    env_rng = getattr(env, "_rng", None) if env is not None else None
    if env_state is not None:
        if env_rng is None or not hasattr(env_rng, "bit_generator"):
            raise RecoveryError(
                "Checkpoint contains environment RNG state but runtime environment has no _rng"
            )
        env_rng.bit_generator.state = dict(env_state)


def replay_buffer_state(buffer: Any) -> Dict[str, Any]:
    return {
        "capacity": int(buffer.capacity),
        "ptr": int(buffer.ptr),
        "size": int(buffer.size),
        "states": buffer.states.copy(),
        "actions": buffer.actions.copy(),
        "rewards": buffer.rewards.copy(),
        "next_states": buffer.next_states.copy(),
        "dones": buffer.dones.copy(),
    }


def load_replay_buffer_state(buffer: Any, state: Mapping[str, Any]) -> None:
    if int(state["capacity"]) != int(buffer.capacity):
        raise ValueError(
            f"ReplayBuffer capacity mismatch: checkpoint={state['capacity']}, runtime={buffer.capacity}"
        )
    for name in ("states", "actions", "rewards", "next_states", "dones"):
        incoming = np.asarray(state[name])
        current = getattr(buffer, name)
        if incoming.shape != current.shape or incoming.dtype != current.dtype:
            raise ValueError(
                f"ReplayBuffer {name} mismatch: checkpoint={incoming.shape}/{incoming.dtype}, "
                f"runtime={current.shape}/{current.dtype}"
            )
        current[...] = incoming
    ptr = int(state["ptr"])
    size = int(state["size"])
    if not (0 <= ptr < buffer.capacity) or not (0 <= size <= buffer.capacity):
        raise ValueError(f"Invalid ReplayBuffer cursor: ptr={ptr}, size={size}")
    buffer.ptr = ptr
    buffer.size = size


def trajectory_buffer_state(buffer: Any) -> Dict[str, Any]:
    return {
        "states": list(buffer.states),
        "actions": list(buffer.actions),
        "rewards": list(buffer.rewards),
        "dones": list(buffer.dones),
        "log_probs": list(buffer.log_probs),
        "values": list(buffer.values),
    }


def load_trajectory_buffer_state(buffer: Any, state: Mapping[str, Any]) -> None:
    names = ("states", "actions", "rewards", "dones", "log_probs", "values")
    lengths = {len(state[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError(f"TrajectoryBuffer checkpoint has inconsistent lengths: {sorted(lengths)}")
    for name in names:
        setattr(buffer, name, list(state[name]))


def _validate_recovery_header(payload: Any, expected_kind: str, source: Path) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RecoveryError(f"{source} is a model artifact, not a training recovery checkpoint.")
    if payload.get("checkpoint_kind") != expected_kind or "resume_schema_version" not in payload:
        raise RecoveryError(f"{source} is a model artifact, not a training recovery checkpoint.")
    version = payload.get("resume_schema_version")
    if version != RESUME_SCHEMA_VERSION:
        raise RecoveryError(
            f"Unsupported recovery schema version {version!r}; expected {RESUME_SCHEMA_VERSION}."
        )
    return payload


def save_warmup_recovery(payload: Mapping[str, Any], path: Path) -> None:
    checkpoint = dict(payload)
    checkpoint.update(
        {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "checkpoint_kind": WARMUP_RECOVERY_KIND,
            "created_at": time.time(),
        }
    )
    atomic_torch_save(checkpoint, Path(path))


def load_warmup_recovery(path: Path, *, map_location: Any = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RecoveryError(f"Warmup recovery checkpoint does not exist: {path}")
    try:
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
    except Exception as exc:
        raise RecoveryError(f"Cannot read warmup recovery checkpoint {path}: {exc}") from exc
    return _validate_recovery_header(payload, WARMUP_RECOVERY_KIND, path)


def save_joint_recovery(payload: Mapping[str, Any], path: Path) -> None:
    checkpoint = dict(payload)
    checkpoint.update(
        resume_schema_version=RESUME_SCHEMA_VERSION,
        checkpoint_kind=JOINT_RECOVERY_KIND,
        created_at=time.time(),
    )
    atomic_torch_save(checkpoint, Path(path))


def load_joint_recovery(path: Path, *, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RecoveryError(f"Cannot read Joint recovery checkpoint {path}: {exc}") from exc
    return _validate_recovery_header(payload, JOINT_RECOVERY_KIND, Path(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cleanup_pbt_recovery(recovery_dir: Path, keep_generation: str) -> None:
    for path in recovery_dir.iterdir():
        if path.name in {"latest.json", keep_generation}:
            continue
        if path.is_dir() and (
            path.name.startswith("generation_") or path.name.startswith(".staging_")
        ):
            shutil.rmtree(path, ignore_errors=True)


def save_pbt_recovery(
    recovery_dir: Path,
    *,
    generation_id: int,
    chunk_id: int,
    global_episode: int,
    phase: str,
    compatibility: Mapping[str, Any],
    controller_state: Mapping[str, Any],
    worker_ids: Iterable[int],
    worker_payload: Callable[[int], Mapping[str, Any]],
) -> Path:
    recovery_dir = Path(recovery_dir)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    generation_name = f"generation_{int(generation_id):012d}"
    final_dir = recovery_dir / generation_name
    staging_dir = recovery_dir / f".staging_{generation_name}_{uuid.uuid4().hex}"
    if final_dir.exists():
        pointed_generation = None
        pointer_path = recovery_dir / "latest.json"
        if pointer_path.is_file():
            try:
                pointed_generation = json.loads(pointer_path.read_text(encoding="utf-8")).get(
                    "generation"
                )
            except Exception:
                pointed_generation = None
        if pointed_generation == generation_name:
            raise RecoveryError(f"Refusing to overwrite committed PBT generation: {final_dir}")
        shutil.rmtree(final_dir)
    staging_dir.mkdir(parents=False, exist_ok=False)

    worker_entries = []
    try:
        controller_path = staging_dir / "controller.ckpt"
        atomic_torch_save(
            {
                "resume_schema_version": RESUME_SCHEMA_VERSION,
                "checkpoint_kind": PBT_RECOVERY_KIND,
                "generation_id": int(generation_id),
                "chunk_id": int(chunk_id),
                "global_episode": int(global_episode),
                "controller": dict(controller_state),
            },
            controller_path,
        )
        for wid in sorted(int(w) for w in worker_ids):
            worker_path = staging_dir / f"worker_{wid:02d}.ckpt"
            wrapped = {
                "resume_schema_version": RESUME_SCHEMA_VERSION,
                "checkpoint_kind": PBT_RECOVERY_KIND,
                "generation_id": int(generation_id),
                "chunk_id": int(chunk_id),
                "global_episode": int(global_episode),
                "worker_id": int(wid),
                "worker": dict(worker_payload(wid)),
            }
            atomic_torch_save(wrapped, worker_path)
            worker_entries.append(
                {
                    "worker_id": wid,
                    "file": worker_path.name,
                    "sha256": _sha256_file(worker_path),
                }
            )

        manifest = {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "checkpoint_kind": PBT_RECOVERY_KIND,
            "created_at": time.time(),
            "generation_id": int(generation_id),
            "chunk_id": int(chunk_id),
            "global_episode": int(global_episode),
            "phase": str(phase),
            "compatibility": dict(compatibility),
            "controller": {
                "file": controller_path.name,
                "sha256": _sha256_file(controller_path),
            },
            "workers": worker_entries,
        }
        manifest_path = staging_dir / "manifest.json"
        atomic_json_save(manifest, manifest_path)
        _fsync_directory(staging_dir)
        os.replace(staging_dir, final_dir)
        _fsync_directory(recovery_dir)

        pointer = {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "checkpoint_kind": PBT_RECOVERY_KIND,
            "generation": generation_name,
            "manifest_sha256": _sha256_file(final_dir / "manifest.json"),
        }
        atomic_json_save(pointer, recovery_dir / "latest.json")
        _cleanup_pbt_recovery(recovery_dir, generation_name)
        return final_dir / "manifest.json"
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def load_pbt_recovery(
    run_dir: Path,
    *,
    expected_worker_ids: Optional[Iterable[int]] = None,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    recovery_dir = Path(run_dir) / "recovery"
    pointer_path = recovery_dir / "latest.json"
    if not pointer_path.is_file():
        raise RecoveryError(f"PBT committed recovery pointer does not exist: {pointer_path}")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RecoveryError(f"Cannot read PBT recovery pointer {pointer_path}: {exc}") from exc
    _validate_recovery_header(pointer, PBT_RECOVERY_KIND, pointer_path)

    generation_dir = recovery_dir / str(pointer.get("generation", ""))
    manifest_path = generation_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RecoveryError(f"Committed PBT recovery manifest is missing: {manifest_path}")
    if _sha256_file(manifest_path) != pointer.get("manifest_sha256"):
        raise RecoveryError(f"Committed PBT recovery manifest checksum mismatch: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RecoveryError(f"Cannot read PBT recovery manifest {manifest_path}: {exc}") from exc
    _validate_recovery_header(manifest, PBT_RECOVERY_KIND, manifest_path)

    entries = manifest.get("workers")
    if not isinstance(entries, list):
        raise RecoveryError("PBT recovery manifest has no worker list")
    actual_ids = {int(entry["worker_id"]) for entry in entries}
    if expected_worker_ids is not None:
        expected_ids = {int(w) for w in expected_worker_ids}
        if actual_ids != expected_ids:
            raise RecoveryError(
                f"PBT recovery worker set mismatch: checkpoint={sorted(actual_ids)}, expected={sorted(expected_ids)}"
            )

    workers: Dict[int, Dict[str, Any]] = {}
    generation_id = int(manifest["generation_id"])
    chunk_id = int(manifest["chunk_id"])
    global_episode = int(manifest["global_episode"])
    controller_entry = manifest.get("controller", {})
    controller_path = generation_dir / str(controller_entry.get("file", ""))
    if not controller_path.is_file():
        raise RecoveryError(f"Committed PBT recovery controller is missing: {controller_path}")
    if _sha256_file(controller_path) != controller_entry.get("sha256"):
        raise RecoveryError(
            f"Committed PBT recovery controller checksum mismatch: {controller_path}"
        )
    try:
        try:
            controller_wrapped = torch.load(
                controller_path, map_location=map_location, weights_only=False
            )
        except TypeError:
            controller_wrapped = torch.load(controller_path, map_location=map_location)
    except Exception as exc:
        raise RecoveryError(
            f"Cannot read PBT recovery controller {controller_path}: {exc}"
        ) from exc
    _validate_recovery_header(controller_wrapped, PBT_RECOVERY_KIND, controller_path)
    controller_identity = (
        int(controller_wrapped.get("generation_id", -1)),
        int(controller_wrapped.get("chunk_id", -1)),
        int(controller_wrapped.get("global_episode", -1)),
    )
    if controller_identity != (generation_id, chunk_id, global_episode):
        raise RecoveryError(
            f"PBT recovery controller generation mismatch: {controller_identity} != "
            f"{(generation_id, chunk_id, global_episode)}"
        )
    for entry in entries:
        wid = int(entry["worker_id"])
        worker_path = generation_dir / str(entry["file"])
        if not worker_path.is_file():
            raise RecoveryError(f"Committed PBT recovery worker is missing: {worker_path}")
        if _sha256_file(worker_path) != entry.get("sha256"):
            raise RecoveryError(f"Committed PBT recovery worker checksum mismatch: {worker_path}")
        try:
            try:
                wrapped = torch.load(worker_path, map_location=map_location, weights_only=False)
            except TypeError:
                wrapped = torch.load(worker_path, map_location=map_location)
        except Exception as exc:
            raise RecoveryError(f"Cannot read PBT recovery worker {worker_path}: {exc}") from exc
        _validate_recovery_header(wrapped, PBT_RECOVERY_KIND, worker_path)
        identity = (
            int(wrapped.get("generation_id", -1)),
            int(wrapped.get("chunk_id", -1)),
            int(wrapped.get("global_episode", -1)),
            int(wrapped.get("worker_id", -1)),
        )
        expected_identity = (generation_id, chunk_id, global_episode, wid)
        if identity != expected_identity:
            raise RecoveryError(
                f"PBT recovery generation mismatch for worker {wid}: {identity} != {expected_identity}"
            )
        workers[wid] = wrapped["worker"]

    return {
        "manifest": manifest,
        "controller": controller_wrapped["controller"],
        "workers": workers,
        "recovery_dir": recovery_dir,
    }
