import copy, hashlib, json, os, shutil
from pathlib import Path
import numpy as np
import torch
from atprl.training.sac_recovery import digest
from atprl.training.sac_v2 import (
    Store,
    LEGACY_SOURCES,
    validate_identity,
    rebuild,
    thin_policy,
    make_policy,
)
from atprl.training.sac_v2_diagnostics import Diagnostics
from atprl.training.sac_v2_io import (
    sha,
    atomic_bytes,
    atomic_json,
    descriptor,
    verify,
    read_json,
    lock,
)


def semantic_digest(value):
    def norm(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            return ["array", str(v.dtype), v.shape, hashlib.sha256(v.tobytes()).hexdigest()]
        if isinstance(v, np.generic):
            return norm(v.item())
        if isinstance(v, np.dtype):
            return ["dtype", str(v)]
        if isinstance(v, type):
            return ["type", v.__module__ + "." + v.__qualname__]
        if isinstance(v, np.random.Generator):
            return ["generator", norm(v.bit_generator.state)]
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in sorted(v.items(), key=lambda p: str(p[0]))}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, set):
            return sorted([norm(x) for x in v], key=str)
        if isinstance(v, float):
            return ["float", v.hex()]
        if isinstance(v, (str, int, bool, type(None))):
            return v
        if isinstance(v, bytes):
            return ["bytes", hashlib.sha256(v).hexdigest()]
        if hasattr(v, "__dict__"):
            return [type(v).__module__ + "." + type(v).__qualname__, norm(vars(v))]
        raise TypeError("Unsupported state identity type " + str(type(v)))

    return hashlib.sha256(json.dumps(norm(value), sort_keys=True).encode()).hexdigest()


def scientific_state_identity(s):
    keys = [
        "q",
        "target_q",
        "q_optimizer",
        "dqn_replay",
        "dqn_updates",
        "dqn_target_updates",
        "power",
        "phase",
        "environment",
        "rng",
        "kernel",
        "progress",
    ]
    return {k: semantic_digest(s[k]) for k in keys}


def read_legacy(path, job=None, allowed_sources=None):
    path = Path(path)
    meta = read_json(path.with_name("checkpoint_meta.json"))
    if sha(path) != meta["sha256"]:
        raise ValueError("Legacy checkpoint/meta checksum mismatch")
    s = torch.load(path, map_location="cpu", weights_only=False)
    if s.get("kind") != "sac_stp_recovery" or s.get("version") != 1:
        raise ValueError("Not legacy full SAC recovery")
    if s["source_signature"] not in (
        LEGACY_SOURCES if allowed_sources is None else allowed_sources
    ):
        raise ValueError("Unapproved migration source")
    validate_identity(s, job)
    if meta["episode"] != s["progress"]["episodes_done"]:
        raise ValueError("Legacy progress/meta mismatch")
    for name, key in [
        ("training.jsonl", "journal_bytes"),
        ("learner_updates.jsonl", "learner_bytes"),
    ]:
        f = path.with_name(name)
        n = s.get(key, 0)
        if n and (not f.exists() or f.stat().st_size < n):
            raise ValueError("Missing legacy journal prefix")
    selected = s["selection"].get("best_path")
    if selected and sha(path.parent / selected) != s["selection"]["best_sha256"]:
        raise ValueError("Missing committed legacy best")
    if s["stage"] == "stage2" and job is not None:
        kp = Path(s["config"]["kernel_checkpoint"])
        if kp != path.parent.parent / "stage1/kernel_best.ckpt":
            raise ValueError("Kernel belongs to another task")
        if sha(kp) != s["config"]["kernel_checkpoint_sha256"]:
            raise ValueError("Bound kernel file hash mismatch")
        if digest(torch.load(kp, map_location="cpu", weights_only=False)) != digest(s["kernel"]):
            raise ValueError("Bound kernel payload mismatch")
    return s


def import_diagnostics(s, path, store, retain_at_episode=None):
    d = Diagnostics()
    history = s["progress"]["history"]
    expected = s.get("learner_bytes", 0)
    f = Path(path).with_name("learner_updates.jsonl")
    seen = 0
    offset = 0
    stream = f.open("rb") if expected else None
    try:
        for row in history:
            d.begin(row["episode"])
            target = int(row["sac_updates"])
            while seen < target:
                if stream is None:
                    raise ValueError("Missing committed update diagnostics")
                line = stream.readline()
                offset += len(line)
                if not line or offset > expected:
                    raise ValueError("Diagnostic boundary mismatch")
                metric = json.loads(line)
                d.consume(metric)
                seen = int(metric["updates"])
            d.finish()
            if row["episode"] % 16 == 0 or row is history[-1]:
                state = d.preview(
                    store.root,
                    s["stage"],
                    f"import_{s['stage']}_{row['episode']:06d}",
                    s["progress"]["episodes_done"]
                    if retain_at_episode is None
                    else retain_at_episode,
                )
                d.published(state)
        if offset != expected:
            raise ValueError(f"Unused committed diagnostic bytes {offset} != {expected}")
    finally:
        if stream:
            stream.close()
    controller = s["phase"] if s["stage"] == "stage2" else s["power"]
    if d.state["consumed_update"] != controller["counters"]["updates"]:
        raise ValueError("Diagnostic/controller update mismatch")
    return d


def migrate(path, destination, job, *, allowed_sources=None, import_numeric=True):
    path = Path(path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with lock(destination / "worker.lock", blocking=False):
        store = Store(destination)
        if store.current_path.exists():
            state, m = store.load(job=job)
            if (destination / "MIGRATION.json").exists():
                record = read_json(destination / "MIGRATION.json")
                if path.exists() and record["legacy_sha256"] != sha(path):
                    raise ValueError("Existing migration has different source")
                if not path.exists() and not (destination / "LEGACY_RETIRED.json").exists():
                    raise ValueError("Missing unretired legacy source")
                return record
            original = read_legacy(path, job, allowed_sources)
            before = scientific_state_identity(original)
            if before != scientific_state_identity(state):
                raise ValueError("Incomplete migration receipt has diverged state")
            intent = read_json(destination / "MIGRATION_INTENT.json")
            record = {
                "passed": True,
                "job_id": job["key"],
                "seed": job["seed"],
                "lambda": job["lambda"],
                "legacy_checkpoint": str(path),
                "legacy_sha256": sha(path),
                "legacy_source_signature": original["source_signature"],
                "legacy_episode": original["progress"]["episodes_done"],
                "legacy_stage": original["stage"],
                "initial_cumulative_episode": m["cumulative_episode"],
                "converted_generation": m["generation"],
                "source_identity": before,
                "converted_identity": before,
                "numeric_history_imported": intent["numeric_history_imported"],
                "legacy_retirement_allowed": False,
                "receipt_recovered_after_publication": True,
            }
            atomic_json(destination / "MIGRATION.json", record)
            return record
        s = read_legacy(path, job, allowed_sources)
        before = scientific_state_identity(s)
        intent = {
            "job_id": job["key"],
            "legacy_sha256": sha(path),
            "legacy_source_signature": s["source_signature"],
            "numeric_history_imported": import_numeric,
        }
        if (destination / "MIGRATION_INTENT.json").exists() and read_json(
            destination / "MIGRATION_INTENT.json"
        ) != intent:
            raise ValueError("Migration staging identity conflict")
        atomic_json(destination / "MIGRATION_INTENT.json", intent)
        for name in ["spec.json", "FAILED.json", "SEED_BINDING.json", "RUNNING.json", "EXIT.json"]:
            original = path.parent.parent / name
            if original.exists():
                atomic_bytes(destination / "provenance" / ("legacy_" + name), original.read_bytes())
        stage_dir = "stage1" if s["stage"] == "warmup" else "stage2"
        expected = s.get("journal_bytes", 0)
        f = path.with_name("training.jsonl")
        data = f.read_bytes()[:expected] if expected else b""
        if len(data) != expected:
            raise ValueError("Short committed training journal")
        atomic_bytes(destination / stage_dir / "training.jsonl", data)
        for p in path.parent.glob("evaluation_ep*.json"):
            if int(p.stem.split("ep")[-1]) <= s["progress"]["episodes_done"]:
                atomic_bytes(destination / stage_dir / p.name, p.read_bytes())
        extra = {
            "legacy_rollback": {
                "checkpoint": str(path),
                "sha256": sha(path),
                "source_signature": s["source_signature"],
                "stage": s["stage"],
                "episode": s["progress"]["episodes_done"],
                "retired": False,
            },
            "persistent_objects": [],
            "legacy_selection": copy.deepcopy(s["selection"]),
        }
        if s["stage"] == "stage2":
            old_stage1 = path.parent.parent / "stage1"
            for p in old_stage1.glob("evaluation_ep*.json"):
                atomic_bytes(destination / "stage1" / p.name, p.read_bytes())
            jp = old_stage1 / "training.jsonl"
            if jp.exists():
                rows = jp.read_bytes().splitlines(keepends=True)[: s["config"]["stage1_episodes"]]
                atomic_bytes(destination / "stage1/training.jsonl", b"".join(rows))
            extra["stage1_complete"] = {
                "episode": s["config"]["stage1_episodes"],
                "source_job": job["key"],
                "kernel_origin_sha256": s["config"]["kernel_checkpoint_sha256"],
                "kernel_state_digest": digest(s["kernel"]),
            }
            if import_numeric:
                first = read_legacy(old_stage1 / "latest.ckpt", job, allowed_sources)
                if first["progress"]["episodes_done"] != s["config"]["stage1_episodes"]:
                    raise ValueError("Phase has incomplete Stage-I origin")
                prior_diagnostics = import_diagnostics(
                    first,
                    old_stage1 / "latest.ckpt",
                    store,
                    retain_at_episode=s["config"]["stage1_episodes"]
                    + s["progress"]["episodes_done"],
                )
                extra["initial_diagnostics_by_stage"] = {"warmup": prior_diagnostics.state}
                del first
            for p in list(old_stage1.glob("*endpoint*.ckpt")) + [old_stage1 / "kernel_best.ckpt"]:
                if p.exists():
                    extra["persistent_objects"].append(
                        store.put_object(
                            thin_policy(torch.load(p, map_location="cpu", weights_only=False)),
                            "policy",
                        )
                    )
        selected = s["selection"].get("best_path")
        if selected:
            b = torch.load(path.parent / selected, map_location="cpu", weights_only=False)
            pref = store.put_object(thin_policy(b), "policy")
            s["selection"] = dict(
                s["selection"], policy_ref=pref, best_path=pref["path"], best_sha256=pref["sha256"]
            )
            if s["stage"] == "warmup":
                s["selection"]["kernel_ref"] = store.put_object(b, "kernel")
        if (
            s["stage"] == "warmup"
            and s["progress"]["episodes_done"] == s["config"]["stage1_episodes"]
        ):
            if not s["selection"].get("kernel_ref"):
                raise ValueError("Completed Stage-I has no committed selected kernel")
            k = torch.load(
                verify(destination, s["selection"]["kernel_ref"]),
                map_location="cpu",
                weights_only=False,
            )
            extra["stage1_complete"] = {
                "episode": s["config"]["stage1_episodes"],
                "source_job": job["key"],
                "kernel_ref": s["selection"]["kernel_ref"],
                "kernel_state_digest": digest(k),
                "best": copy.deepcopy(s["selection"]),
            }
            endpoint_path = path.parent / f"kernel_endpoint_{s['config']['stage1_episodes']}.ckpt"
            if endpoint_path.exists():
                endpoint = thin_policy(
                    torch.load(endpoint_path, map_location="cpu", weights_only=False)
                )
            else:
                endpoint = make_policy(rebuild(s, job), "endpoint")
            ref = store.put_object(endpoint, "policy")
            extra["endpoints"] = {"warmup": ref}
            extra["persistent_objects"].extend([ref, s["selection"]["policy_ref"]])
        d = import_diagnostics(s, path, store) if import_numeric else Diagnostics()
        if not import_numeric:
            c = s["phase"] if s["stage"] == "stage2" else s["power"]
            d.state["consumed_update"] = c["counters"]["updates"]
            d.state["scope_start_update"] = c["counters"]["updates"] + 1
        tr = rebuild(s, job)
        m = store.publish(tr, d, extra=extra, legacy_state=s)
        decoded, _ = store.load(job=job)
        after = scientific_state_identity(decoded)
        if before != after:
            raise ValueError(
                "Migration changed complete scientific state: "
                + str([k for k in before if before[k] != after[k]])
            )
        record = {
            "passed": True,
            "job_id": job["key"],
            "seed": job["seed"],
            "lambda": job["lambda"],
            "legacy_checkpoint": str(path),
            "legacy_sha256": sha(path),
            "legacy_source_signature": s["source_signature"],
            "legacy_episode": s["progress"]["episodes_done"],
            "legacy_stage": s["stage"],
            "initial_cumulative_episode": m["cumulative_episode"],
            "converted_generation": m["generation"],
            "source_identity": before,
            "converted_identity": after,
            "numeric_history_imported": import_numeric,
            "legacy_retirement_allowed": False,
        }
        atomic_json(destination / "MIGRATION.json", record)
        return record
