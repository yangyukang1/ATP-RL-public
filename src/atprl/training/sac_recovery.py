import copy, hashlib, json, subprocess, time, os
from pathlib import Path
import numpy as np
import torch
from atprl.training.recovery import (
    atomic_torch_save,
    atomic_json_save,
    replay_buffer_state,
    load_replay_buffer_state,
    capture_rng_state,
    restore_rng_state,
)


def digest(value):
    def canonical(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            return dict(
                dtype=str(v.dtype), shape=v.shape, sha256=hashlib.sha256(v.tobytes()).hexdigest()
            )
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, dict):
            return {str(k): canonical(x) for k, x in sorted(v.items(), key=lambda p: str(p[0]))}
        if isinstance(v, (tuple, list)):
            return [canonical(x) for x in v]
        return v

    return hashlib.sha256(
        json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def physics(config):
    return {
        k: getattr(config, k)
        for k in [
            "num_ground_stations",
            "num_gs_per_rb",
            "topology_profile",
            "ris_num_elem_x",
            "ris_num_elem_y",
            "fairness_lambda",
            "csi_error_variance",
            "macro_steps_per_episode",
            "num_micro_steps",
            "dqn_scheduler_impl",
        ]
    }


def kernel_bundle(trainer, kind="sac_stage1_kernel"):
    rt = trainer.runtime
    return dict(
        version=1,
        kind=kind,
        algorithm="sac_stp",
        physics=physics(trainer.config),
        power_config=rt.power.state_dict()["config"],
        training_seed=trainer.config.seed,
        episode=trainer.episodes_done,
        total_steps=trainer.total_steps,
        q_network=copy.deepcopy(rt.q_network.state_dict()),
        target_q_network=copy.deepcopy(rt.target_q_network.state_dict()),
        q_optimizer=copy.deepcopy(rt.q_optimizer.state_dict()),
        dqn_replay=replay_buffer_state(rt.dqn_replay),
        dqn_updates=rt.dqn_updates,
        dqn_target_updates=rt.dqn_target_updates,
        power=rt.power.state_dict(),
    )


def load_kernel(rt, bundle):
    if (
        bundle.get("kind") != "sac_stage1_kernel"
        or bundle.get("version") != 1
        or bundle.get("algorithm") != "sac_stp"
    ):
        raise ValueError("Not a SAC Stage-I kernel")
    if bundle["physics"] != physics(rt.config) or bundle["training_seed"] != rt.config.seed:
        raise ValueError("Kernel physics/seed mismatch")
    rt.q_network.load_state_dict(bundle["q_network"])
    rt.target_q_network.load_state_dict(bundle["target_q_network"])
    rt.q_optimizer.load_state_dict(bundle["q_optimizer"])
    load_replay_buffer_state(rt.dqn_replay, bundle["dqn_replay"])
    rt.power.load_state_dict(bundle["power"])
    rt.dqn_updates = bundle["dqn_updates"]
    rt.dqn_target_updates = bundle["dqn_target_updates"]
    rt.freeze_kernel()


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_kernel(trainer, path):
    atomic_torch_save(kernel_bundle(trainer), Path(path))


def retry(operation, *args):
    for attempt in range(12):
        try:
            return operation(*args)
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.02 * 2**attempt, 0.5))


def write_json(path, value):
    retry(atomic_json_save, value, Path(path))


def write_torch(path, value):
    retry(atomic_torch_save, value, Path(path))


def copy_artifact(source, destination):
    from atprl.training.recovery import _atomic_write_bytes

    payload = Path(source).read_bytes()
    retry(_atomic_write_bytes, Path(destination), lambda f: f.write(payload))
    if file_sha(source) != file_sha(destination):
        raise ValueError("Artifact copy identity mismatch")


def source_signature():
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            "src/atprl/" + p.relative_to(root).as_posix(): file_sha(p)
            for p in sorted(root.rglob("*.py"))
        }
    )


def source_version():
    from atprl.provenance import git_state

    state = git_state()
    return dict(commit=state["commit"], dirty=state["dirty"])


def scientific_config(config):
    v = config.to_dict()
    for k in ["resume_checkpoint", "output_dir", "resume", "checkpoint", "kernel_checkpoint"]:
        v.pop(k, None)
    return v


def full_state(trainer, journal_bytes=0):
    rt = trainer.runtime
    return dict(
        version=1,
        kind="sac_stp_recovery",
        algorithm="sac_stp",
        config=trainer.config.to_dict(),
        config_identity=digest(scientific_config(trainer.config)),
        source_signature=source_signature(),
        source_version=source_version(),
        stage=rt.stage,
        kernel=copy.deepcopy(getattr(trainer, "kernel_source", None)),
        progress={
            k: copy.deepcopy(getattr(trainer, k))
            for k in [
                "episodes_done",
                "total_steps",
                "history",
                "last_mode",
                "current_epsilon",
                "dqn_revision",
            ]
            if hasattr(trainer, k)
        },
        selection=copy.deepcopy(
            getattr(trainer, "selection", dict(best_score=None, best_episode=None))
        ),
        q=copy.deepcopy(rt.q_network.state_dict()),
        target_q=copy.deepcopy(rt.target_q_network.state_dict()),
        q_optimizer=copy.deepcopy(rt.q_optimizer.state_dict()),
        dqn_replay=replay_buffer_state(rt.dqn_replay),
        dqn_updates=rt.dqn_updates,
        dqn_target_updates=rt.dqn_target_updates,
        power=rt.power.state_dict(),
        phase=None if rt.phase is None else rt.phase.state_dict(),
        environment=copy.deepcopy(rt.env),
        rng=capture_rng_state(env=rt.env),
        journal_bytes=journal_bytes,
        journal_rows=len(trainer.history),
        pid=os.getpid(),
    )


def save_recovery(trainer, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    journal = directory / "training.jsonl"
    payload = full_state(trainer, journal.stat().st_size if journal.exists() else 0)
    learner = directory / "learner_updates.jsonl"
    payload["learner_bytes"] = learner.stat().st_size if learner.exists() else 0
    write_torch(directory / "latest.ckpt", payload)
    write_json(
        directory / "checkpoint_meta.json",
        dict(
            stage=payload["stage"],
            episode=trainer.episodes_done,
            sha256=file_sha(directory / "latest.ckpt"),
            source_signature=payload["source_signature"],
            config_identity=payload["config_identity"],
        ),
    )


def restore_checkpoint(path, config=None, *, archive=True):
    from atprl.config.sac import SACRunConfig
    from atprl.training.sac_stp import SACWarmupTrainer, SACPhase2Trainer

    path = Path(path)
    s = load_torch(path)
    if (
        s.get("kind") != "sac_stp_recovery"
        or s.get("version") != 1
        or s.get("algorithm") != "sac_stp"
    ):
        raise ValueError("Not SAC recovery")
    config = config or SACRunConfig(**s["config"])
    if (
        digest(scientific_config(config)) != s["config_identity"]
        or source_signature() != s["source_signature"]
    ):
        raise ValueError("SAC recovery config/source mismatch")
    if archive:
        target = (
            path.parent
            / "recovery_sources"
            / f"ep{s['progress']['episodes_done']}_{file_sha(path)}.ckpt"
        )
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            with target.open("xb") as f:
                f.write(path.read_bytes())
                f.flush()
                os.fsync(f.fileno())
        if file_sha(target) != file_sha(path):
            raise ValueError("Immutable recovery source mismatch")
        summary = target.with_suffix(".json")
        if not summary.exists():
            live = path.parent / "progress.json"
            write_json(
                summary,
                dict(
                    source=str(path),
                    sha256=file_sha(path),
                    stage=s["stage"],
                    progress=s["progress"] | {"history": len(s["progress"]["history"])},
                    models_identity=digest(
                        dict(
                            q=s["q"],
                            power=s["power"]["models"],
                            phase=s["phase"]["models"] if s["phase"] else None,
                        )
                    ),
                    optimizer_identity=digest(
                        dict(
                            q=s["q_optimizer"],
                            power=s["power"]["optimizers"],
                            phase=s["phase"]["optimizers"] if s["phase"] else None,
                        )
                    ),
                    rng_identity=digest(s["rng"]),
                    power_replay_identity=digest(s["power"]["replay"]),
                    last_live_observation=json.loads(live.read_text(encoding="utf-8"))
                    if live.exists()
                    else None,
                    uncommitted_tail_may_require_recompute=True,
                ),
            )
    tr = (
        SACWarmupTrainer(config)
        if s["stage"] == "warmup"
        else SACPhase2Trainer(config, s["kernel"])
    )
    rt = tr.runtime
    rt.q_network.load_state_dict(s["q"])
    rt.target_q_network.load_state_dict(s["target_q"])
    rt.q_optimizer.load_state_dict(s["q_optimizer"])
    load_replay_buffer_state(rt.dqn_replay, s["dqn_replay"])
    rt.power.load_state_dict(s["power"])
    if s["phase"] is not None:
        rt.phase.load_state_dict(s["phase"])
    rt.dqn_updates = s["dqn_updates"]
    rt.dqn_target_updates = s["dqn_target_updates"]
    rt.env = s["environment"]
    for k, v in s["progress"].items():
        setattr(tr, k, v)
    tr.selection = s["selection"]
    journal = path.parent / "training.jsonl"
    if journal.exists():
        if journal.stat().st_size < s["journal_bytes"]:
            raise ValueError("Missing committed journal bytes")
        with journal.open("r+b") as f:
            f.truncate(s["journal_bytes"])
    elif s["journal_bytes"]:
        raise ValueError("Missing committed journal")
    learner = path.parent / "learner_updates.jsonl"
    expected = s.get("learner_bytes", 0)
    if learner.exists():
        if learner.stat().st_size < expected:
            raise ValueError("Missing committed learner journal")
        with learner.open("r+b") as f:
            f.truncate(expected)
    elif expected:
        raise ValueError("Missing learner journal")
    restore_rng_state(s["rng"], env=rt.env)
    return tr


def policy_bundle(trainer, identity):
    rt = trainer.runtime
    if rt.stage == "warmup":
        p = kernel_bundle(trainer)
    else:
        p = dict(
            version=1,
            kind="sac_stp_policy",
            algorithm="sac_stp",
            config=trainer.config.to_dict(),
            kernel=copy.deepcopy(trainer.kernel_source),
            phase=rt.phase.state_dict(),
            episode=trainer.episodes_done,
            cumulative_episode=trainer.config.stage1_episodes + trainer.episodes_done,
            training_seed=trainer.config.seed,
        )
    p.update(
        artifact_identity=identity,
        source_signature=source_signature(),
        source_version=source_version(),
        config=trainer.config.to_dict(),
    )
    return p
