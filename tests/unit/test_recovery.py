from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atprl.rl.buffers import ReplayBuffer, TrajectoryBuffer
from atprl.training.recovery import (
    RecoveryError,
    capture_rng_state,
    load_pbt_recovery,
    load_replay_buffer_state,
    load_trajectory_buffer_state,
    load_warmup_recovery,
    restore_rng_state,
    replay_buffer_state,
    save_pbt_recovery,
    save_warmup_recovery,
    signature,
    trajectory_buffer_state,
    validate_signature,
)
from atprl.training.pbt.controller import (
    LogPaths,
    NullSummaryWriter,
    UnifiedCkpt,
    WorkerState,
    _iter_crossed_boundary_indices,
    _pulse_tick_after_ready_boundary,
    exploit_copy,
    on_ready_boundary,
)


class _Env:
    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_runtime(seed: int = 7):
    _seed_all(seed)
    model = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return {
        "model": model,
        "target": target,
        "optimizer": optimizer,
        "env": _Env(seed + 1),
        "episodes_done": 0,
        "total_steps": 0,
        "epsilon": 1.0,
        "best": float("-inf"),
        "validation_cursor": 0,
    }


def _run_units(runtime, count: int) -> None:
    for _ in range(count):
        x = torch.randn(3)
        target_value = (
            random.random() + float(np.random.random()) + float(runtime["env"]._rng.random())
        )
        loss = (runtime["model"](x).sum() - target_value) ** 2
        runtime["optimizer"].zero_grad()
        loss.backward()
        runtime["optimizer"].step()
        runtime["episodes_done"] += 1
        runtime["total_steps"] += 3
        runtime["epsilon"] *= 0.97
        runtime["best"] = max(runtime["best"], -float(loss.detach()))
        if runtime["episodes_done"] % 4 == 0:
            runtime["target"].load_state_dict(runtime["model"].state_dict())
        if runtime["episodes_done"] % 8 == 0:
            runtime["validation_cursor"] = runtime["episodes_done"]


def _warmup_payload(runtime, run_dir: Path, *, n: int = 20, k: int = 5):
    return {
        "training_phase": "warmup",
        "run_dir": str(run_dir),
        "compatibility": signature(
            {
                "num_ground_stations": n,
                "num_gs_per_rb": k,
                "dqn_scheduler_impl": "enumerative",
                "dqn_state_dim": 3,
                "dqn_action_dim": 2,
                "power_state_dim": 4,
                "power_action_dim": 3,
                "phase_state_dim": 5,
                "phase_groups": 4,
                "phase_policy_dist": "gaussian_latent",
                "alternating_schedule": [["DQN", 6], ["POWER", 1]],
            }
        ),
        "models": {
            "q_network": runtime["model"].state_dict(),
            "target_q_network": runtime["target"].state_dict(),
        },
        "optimizers": {"q_optimizer": runtime["optimizer"].state_dict()},
        "progress": {
            "episodes_done": runtime["episodes_done"],
            "total_steps": runtime["total_steps"],
            "current_epsilon": runtime["epsilon"],
            "segment_idx": 0,
            "segment_ep_remaining": 6,
        },
        "bookkeeping": {"best_eval_return": runtime["best"]},
        "logging": {"last_validation_episode": runtime["validation_cursor"]},
        "rng": capture_rng_state(env=runtime["env"]),
    }


def _restore_runtime(runtime, payload) -> None:
    runtime["model"].load_state_dict(payload["models"]["q_network"])
    runtime["target"].load_state_dict(payload["models"]["target_q_network"])
    runtime["optimizer"].load_state_dict(payload["optimizers"]["q_optimizer"])
    runtime["episodes_done"] = int(payload["progress"]["episodes_done"])
    runtime["total_steps"] = int(payload["progress"]["total_steps"])
    runtime["epsilon"] = float(payload["progress"]["current_epsilon"])
    runtime["best"] = float(payload["bookkeeping"]["best_eval_return"])
    runtime["validation_cursor"] = int(payload["logging"]["last_validation_episode"])
    restore_rng_state(payload["rng"], env=runtime["env"])


def _assert_state_equal(left, right) -> None:
    assert left["episodes_done"] == right["episodes_done"]
    assert left["total_steps"] == right["total_steps"]
    assert left["epsilon"] == right["epsilon"]
    assert left["best"] == right["best"]
    assert left["validation_cursor"] == right["validation_cursor"]
    for name in ("model", "target"):
        for key, value in left[name].state_dict().items():
            torch.testing.assert_close(value, right[name].state_dict()[key], rtol=0, atol=0)
    assert left["optimizer"].state_dict()["state"]
    left_opt = left["optimizer"].state_dict()["state"]
    right_opt = right["optimizer"].state_dict()["state"]
    assert left_opt.keys() == right_opt.keys()
    for param_id in left_opt:
        for key, value in left_opt[param_id].items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, right_opt[param_id][key], rtol=0, atol=0)
            else:
                assert value == right_opt[param_id][key]


def test_warmup_uninterrupted_32_matches_16_resume_16(tmp_path: Path):
    uninterrupted = _make_runtime()
    _run_units(uninterrupted, 32)
    expected_next = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        float(uninterrupted["env"]._rng.random()),
    )

    first = _make_runtime()
    _run_units(first, 16)
    checkpoint = tmp_path / "run" / "recovery" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    save_warmup_recovery(_warmup_payload(first, tmp_path / "run"), checkpoint)

    _seed_all(999)
    resumed = _make_runtime(999)
    loaded = load_warmup_recovery(checkpoint)
    _restore_runtime(resumed, loaded)
    _run_units(resumed, 16)

    _assert_state_equal(uninterrupted, resumed)
    actual_next = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        float(resumed["env"]._rng.random()),
    )
    assert expected_next[0] == actual_next[0]
    assert expected_next[1] == actual_next[1]
    torch.testing.assert_close(expected_next[2], actual_next[2], rtol=0, atol=0)
    assert expected_next[3] == actual_next[3]


def test_rng_restore_replays_python_numpy_torch_and_environment():
    _seed_all(11)
    env = _Env(12)
    state = capture_rng_state(env=env)
    expected = (random.random(), np.random.random(), torch.rand(4), env._rng.random())
    restore_rng_state(state, env=env)
    actual = (random.random(), np.random.random(), torch.rand(4), env._rng.random())
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
    assert expected[3] == actual[3]


def test_warmup_compatibility_rejects_n_or_k_mismatch(tmp_path: Path):
    runtime = _make_runtime()
    saved = _warmup_payload(runtime, tmp_path, n=20, k=5)["compatibility"]
    validate_signature(saved, saved)
    for n, k in ((20, 2), (20, 4)):
        current = _warmup_payload(runtime, tmp_path, n=n, k=k)["compatibility"]
        with pytest.raises(RecoveryError, match="incompatible"):
            validate_signature(saved, current)


def test_legacy_model_artifact_is_not_recovery(tmp_path: Path):
    artifact = tmp_path / "q_network_phase1_best.pth"
    torch.save({"weight": torch.ones(1)}, artifact)
    with pytest.raises(RecoveryError, match="model artifact"):
        load_warmup_recovery(artifact)


def test_unknown_recovery_schema_is_rejected(tmp_path: Path):
    checkpoint = tmp_path / "latest.ckpt"
    torch.save(
        {"checkpoint_kind": "warmup_recovery", "resume_schema_version": 999},
        checkpoint,
    )
    with pytest.raises(RecoveryError, match="Unsupported recovery schema"):
        load_warmup_recovery(checkpoint)


def test_warmup_atomic_overwrite_preserves_old_on_replace_failure(tmp_path: Path, monkeypatch):
    path = tmp_path / "recovery" / "latest.ckpt"
    runtime = _make_runtime()
    save_warmup_recovery(_warmup_payload(runtime, tmp_path), path)
    old_episode = load_warmup_recovery(path)["progress"]["episodes_done"]
    _run_units(runtime, 1)

    def fail_replace(_src, _dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("atprl.training.recovery.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_warmup_recovery(_warmup_payload(runtime, tmp_path), path)
    assert load_warmup_recovery(path)["progress"]["episodes_done"] == old_episode


def test_buffer_checkpoint_round_trip():
    replay = ReplayBuffer(capacity=4, state_dim=2, action_dim=1)
    replay.store(np.array([1, 2], np.float32), 1, 3.0, np.array([4, 5], np.float32), False)
    restored_replay = ReplayBuffer(capacity=4, state_dim=2, action_dim=1)
    load_replay_buffer_state(restored_replay, replay_buffer_state(replay))
    assert restored_replay.ptr == replay.ptr and restored_replay.size == replay.size
    np.testing.assert_array_equal(restored_replay.states, replay.states)

    trajectory = TrajectoryBuffer()
    trajectory.store(
        torch.ones(2), torch.zeros(1), 1.0, False, torch.tensor(0.2), torch.tensor(0.3)
    )
    restored_trajectory = TrajectoryBuffer()
    load_trajectory_buffer_state(restored_trajectory, trajectory_buffer_state(trajectory))
    assert len(restored_trajectory) == 1
    torch.testing.assert_close(restored_trajectory.states[0], trajectory.states[0])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pbt_compatibility():
    return signature(
        {
            "num_ground_stations": 20,
            "num_gs_per_rb": 5,
            "dqn_scheduler_impl": "sequential",
            "dqn_state_dim": 10,
            "dqn_action_dim": 20,
            "power_state_dim": 30,
            "power_action_dim": 6,
            "phase_state_dim": 40,
            "phase_groups": 4,
            "phase_policy_dist": "von_mises",
            "alternating_schedule": {"chunk_len": 16},
        }
    )


def _save_pbt(run_dir: Path, generation: int, phase: str = "formal"):
    controller = {
        "phase": phase,
        "adapt_progress": 16 if phase != "adapt" else 8,
        "formal_progress": 16 if phase == "formal" else 0,
        "stop_state": {"best_ready_score_ever": 1.0, "best_smoothed": 0.9, "no_improve_ready": 2},
        "master_rng": capture_rng_state(),
    }
    save_pbt_recovery(
        run_dir / "recovery",
        generation_id=generation,
        chunk_id=generation,
        global_episode=generation * 16,
        phase=phase,
        compatibility=_pbt_compatibility(),
        controller_state=controller,
        worker_ids=range(4),
        worker_payload=lambda wid: {
            "wid": wid,
            "mode": "P1" if wid % 2 else "P2",
            "global_episode": generation * 16,
            "cooldown_remaining": wid,
            "pending_event_exploit": 0,
            "pending_event_flip": 0,
            "pending_p1_local_reheat": 0,
            "hp": {"lr": 0.001 + wid},
            "ckpt": {"weight": torch.tensor([generation, wid])},
        },
    )


@pytest.mark.parametrize("phase", ["adapt", "formal"])
def test_pbt_phase_progress_and_four_worker_generation_round_trip(tmp_path: Path, phase: str):
    run_dir = tmp_path / phase
    _save_pbt(run_dir, 1, phase)
    loaded = load_pbt_recovery(run_dir, expected_worker_ids=range(4))
    assert loaded["manifest"]["phase"] == phase
    assert set(loaded["workers"]) == {0, 1, 2, 3}
    assert {worker["global_episode"] for worker in loaded["workers"].values()} == {16}
    assert loaded["controller"]["phase"] == phase


def test_pbt_missing_worker_is_rejected(tmp_path: Path):
    run_dir = tmp_path / "run"
    _save_pbt(run_dir, 1)
    pointer = json.loads((run_dir / "recovery" / "latest.json").read_text(encoding="utf-8"))
    (run_dir / "recovery" / pointer["generation"] / "worker_02.ckpt").unlink()
    with pytest.raises(RecoveryError, match="missing"):
        load_pbt_recovery(run_dir, expected_worker_ids=range(4))


def test_pbt_mixed_generation_is_rejected(tmp_path: Path):
    run_dir = tmp_path / "run"
    _save_pbt(run_dir, 1)
    recovery = run_dir / "recovery"
    pointer_path = recovery / "latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    generation_dir = recovery / pointer["generation"]
    worker_path = generation_dir / "worker_01.ckpt"
    wrapped = torch.load(worker_path, weights_only=False)
    wrapped["generation_id"] = 0
    torch.save(wrapped, worker_path)
    manifest_path = generation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(entry for entry in manifest["workers"] if entry["worker_id"] == 1)["sha256"] = _sha(
        worker_path
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    pointer["manifest_sha256"] = _sha(manifest_path)
    pointer_path.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
    with pytest.raises(RecoveryError, match="generation mismatch"):
        load_pbt_recovery(run_dir, expected_worker_ids=range(4))


def test_pbt_storage_is_bounded_and_staging_is_ignored(tmp_path: Path):
    run_dir = tmp_path / "run"
    _save_pbt(run_dir, 1)
    stale = run_dir / "recovery" / ".staging_generation_broken"
    stale.mkdir()
    (stale / "worker_00.ckpt").write_bytes(b"partial")
    assert (
        load_pbt_recovery(run_dir, expected_worker_ids=range(4))["manifest"]["generation_id"] == 1
    )

    _save_pbt(run_dir, 2)
    entries = sorted(path.name for path in (run_dir / "recovery").iterdir())
    assert entries == ["generation_000000000002", "latest.json"]
    assert (
        load_pbt_recovery(run_dir, expected_worker_ids=range(4))["manifest"]["generation_id"] == 2
    )


def test_pbt_partial_worker_write_keeps_previous_commit(tmp_path: Path):
    run_dir = tmp_path / "run"
    _save_pbt(run_dir, 1)

    def fail_on_worker_two(wid: int):
        if wid == 2:
            raise OSError("simulated worker 2 write failure")
        return {"wid": wid, "global_episode": 32}

    with pytest.raises(OSError, match="worker 2"):
        save_pbt_recovery(
            run_dir / "recovery",
            generation_id=2,
            chunk_id=2,
            global_episode=32,
            phase="formal",
            compatibility=_pbt_compatibility(),
            controller_state={"phase": "formal", "master_rng": capture_rng_state()},
            worker_ids=range(4),
            worker_payload=fail_on_worker_two,
        )

    loaded = load_pbt_recovery(run_dir, expected_worker_ids=range(4))
    assert loaded["manifest"]["generation_id"] == 1
    assert not list((run_dir / "recovery").glob(".staging_*"))


def test_pbt_master_rng_and_committed_event_state_resume(tmp_path: Path):
    _seed_all(123)
    state = capture_rng_state()
    expected = [random.random() for _ in range(4)]
    restore_rng_state(state)
    run_dir = tmp_path / "run"
    _save_pbt(run_dir, 1)
    loaded = load_pbt_recovery(run_dir, expected_worker_ids=range(4))
    for worker in loaded["workers"].values():
        assert worker["pending_event_exploit"] == 0
        assert worker["pending_event_flip"] == 0
    restore_rng_state(loaded["controller"]["master_rng"])
    assert [random.random() for _ in range(4)] == expected


class _DummyTrainer:
    def __init__(self):
        self.args = SimpleNamespace(copy_optimizer=False, phase2_only=True, freeze_dqn_power=True)

    def set_hyperparams(self, **_kwargs):
        return None

    def reset_buffers(self, **_kwargs):
        return None


def _dummy_worker(tmp_path: Path, wid: int, mode: str) -> WorkerState:
    tensor_state = {"weight": torch.tensor([float(wid)])}
    ckpt = UnifiedCkpt(
        q_network=tensor_state,
        target_q_network=tensor_state,
        power_actor=tensor_state,
        power_critic=tensor_state,
        phase_actor=tensor_state,
        phase_critic=tensor_state,
        counters={"episodes_done": 16},
    )
    hp = {
        "p1_lr_dqn": 0.01,
        "p1_lr_power_ppo": 0.01,
        "p1_entropy_coef_power": 0.01,
        "p2_lr_dqn": 0.01,
        "p2_lr_power_ppo": 0.01,
        "p2_entropy_coef_power": 0.01,
        "p2_lr_phase_ppo": 0.01,
        "p2_entropy_coef_phase": 0.01,
    }
    trainer = _DummyTrainer()
    worker_dir = tmp_path / f"worker_{wid:02d}"
    (worker_dir / "checkpoints").mkdir(parents=True)
    (worker_dir / "ready_events.jsonl").write_text("", encoding="utf-8")
    return WorkerState(
        wid=wid,
        mode=mode,
        cooldown_remaining=0,
        global_episode=16,
        last_daily_score=float("nan"),
        last_ready_score=float("nan"),
        ckpt=ckpt,
        p1_trainer=trainer,
        p2_trainer=trainer,
        writer=NullSummaryWriter(worker_dir),
        log_dir=worker_dir,
        daily_curve_json_path=worker_dir / "curve_daily.json",
        hp=dict(hp),
        hp_base=dict(hp),
    )


def test_pbt_committed_exploit_flip_and_mutation_are_not_replayed(tmp_path: Path):
    workers = {wid: _dummy_worker(tmp_path, wid, "P2" if wid == 3 else "P1") for wid in range(4)}
    log_paths = LogPaths(
        run_root=tmp_path,
        global_tb_dir=tmp_path / "global",
        worker_dirs={wid: worker.log_dir for wid, worker in workers.items()},
        pbt_summary_csv=tmp_path / "pbt_summary.csv",
        early_stop_log=tmp_path / "early_stop.log",
    )
    log_paths.pbt_summary_csv.write_text(
        "ready_idx,best_id,best_score,mean_score,std_score,num_exploit,num_flip,num_in_cooldown,eligible_ids\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        num_exploit=1,
        copy_optimizer=False,
        p12_flip_prob=1.0,
        p21_flip_prob=1.0,
        cooldown_episodes=32,
        keep_last_ready_ckpts=1,
        hp_lr_perturb_prob=1.0,
        hp_lr_factor_down=0.5,
        hp_lr_factor_up=0.5,
        hp_lr_bias_down=1.0,
        hp_lr_min=0.0,
        hp_lr_max_factor=2.0,
        hp_lr_perturb_dqn=False,
        hp_ent_perturb_prob=1.0,
        hp_ent_factor_down=0.5,
        hp_ent_factor_up=0.5,
        hp_ent_bias_down=1.0,
        hp_ent_min=0.0,
        hp_ent_max=1.0,
        hp_perturb_power_in_p2=False,
    )
    stats = on_ready_boundary(
        workers,
        args,
        ready_idx=1,
        log_paths=log_paths,
        writer_global=NullSummaryWriter(log_paths.global_tb_dir),
        precomputed_scores={0: 10.0, 1: 8.0, 2: 2.0, 3: 1.0},
    )
    assert stats.exploit_pairs == [(0, 3)]
    assert stats.num_flip == 1
    assert workers[3].mode == "P1"
    assert workers[3].hp["p1_lr_power_ppo"] == pytest.approx(0.005)
    assert workers[3].hp["p1_entropy_coef_power"] == pytest.approx(0.005)

    _pulse_tick_after_ready_boundary(workers)
    assert workers[3].pending_event_exploit == 0
    assert workers[3].pending_event_flip == 0

    save_pbt_recovery(
        tmp_path / "recovery",
        generation_id=1,
        chunk_id=1,
        global_episode=16,
        phase="formal",
        compatibility=_pbt_compatibility(),
        controller_state={"phase": "formal", "ready_index": 1},
        worker_ids=range(4),
        worker_payload=lambda wid: {
            "wid": wid,
            "mode": workers[wid].mode,
            "cooldown_remaining": workers[wid].cooldown_remaining,
            "pending_event_exploit": workers[wid].pending_event_exploit,
            "pending_event_flip": workers[wid].pending_event_flip,
            "hp": workers[wid].hp,
        },
    )
    resumed = load_pbt_recovery(tmp_path, expected_worker_ids=range(4))["workers"][3]
    assert resumed["mode"] == "P1"
    assert resumed["hp"]["p1_lr_power_ppo"] == pytest.approx(0.005)
    assert resumed["pending_event_exploit"] == 0
    assert resumed["pending_event_flip"] == 0
    assert list(_iter_crossed_boundary_indices(16, 32, 16)) == [2]


def test_pbt_copy_optimizer_flag_only_controls_exploit_inheritance(tmp_path: Path):
    src = _dummy_worker(tmp_path / "src", 0, "P1")
    dst = _dummy_worker(tmp_path / "dst", 1, "P1")
    src.ckpt.q_optimizer = {"step": torch.tensor(10)}
    dst.ckpt.q_optimizer = {"step": torch.tensor(3)}

    exploit_copy(dst, src, copy_optimizer=False)
    torch.testing.assert_close(dst.ckpt.q_network["weight"], src.ckpt.q_network["weight"])
    assert int(dst.ckpt.q_optimizer["step"]) == 3

    dst_with_copy = _dummy_worker(tmp_path / "dst_copy", 2, "P1")
    dst_with_copy.ckpt.q_optimizer = {"step": torch.tensor(4)}
    exploit_copy(dst_with_copy, src, copy_optimizer=True)
    assert int(dst_with_copy.ckpt.q_optimizer["step"]) == 10
