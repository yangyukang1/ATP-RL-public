import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atprl.config.loader import cli_args, load_config
from atprl.training.pbt import controller as pbt
from atprl.training.recovery import capture_rng_state, load_pbt_recovery, save_pbt_recovery

ROOT = Path(__file__).resolve().parents[2]


def worker_state(wid, episode, mode="P2"):
    state = {"weight": torch.tensor([float(wid), float(episode)])}
    ckpt = pbt.UnifiedCkpt(
        **{
            key: copy.deepcopy(state)
            for key in (
                "q_network",
                "target_q_network",
                "power_actor",
                "power_critic",
                "phase_actor",
                "phase_critic",
            )
        },
        counters={"segment_idx": 0, "segment_ep_remaining": 1},
    )
    return dict(
        wid=wid,
        mode=mode,
        global_episode=episode,
        cooldown_remaining=wid,
        pending_event_exploit=0,
        pending_event_flip=0,
        pending_p1_local_reheat=0,
        hp={"lr": 0.001 + wid},
        ckpt=pbt.ckpt_to_dict(ckpt),
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/paper_main.yaml")
    args = pbt.parse_args_and_validate([*cli_args(config), "--resume_run_dir", str(tmp_path)])
    args.resume_run_dir = None
    args.runs_root = str(tmp_path)
    args.exp_name = "schedule"
    args.adapt_episodes = 32
    args.max_pbt_episodes = 64
    events = {"adapt": [], "formal": []}

    def init_worker(args, wid, **kwargs):
        worker = SimpleNamespace(**worker_state(wid, 0))
        worker.ckpt = pbt.dict_to_ckpt(worker.ckpt)
        return worker

    def restore(worker, snapshot):
        vars(worker).update(copy.deepcopy(snapshot))
        worker.ckpt = pbt.dict_to_ckpt(worker.ckpt)

    def snapshot(worker, train_info=None):
        state = copy.deepcopy(vars(worker))
        state["ckpt"] = pbt.ckpt_to_dict(worker.ckpt)
        return state

    def adapt(args, log_paths, **kwargs):
        events["adapt"].append((kwargs["adapt_mode"], kwargs["window_episodes"]))
        assert kwargs["adapt_mode"] == "P2"
        snapshots = copy.deepcopy(kwargs["input_snapshots"])
        if snapshots is None:
            snapshots = {wid: worker_state(wid, 0) for wid in range(args.n_workers)}
        for snapshot in snapshots.values():
            snapshot["mode"] = "P2"
            snapshot["global_episode"] += kwargs["window_episodes"]
        episode = min(snapshot["global_episode"] for snapshot in snapshots.values())
        kwargs["on_committed_chunk"](snapshots, kwargs["window_episodes"], episode)
        return snapshots

    def formal(args, workers, log_paths, *, chunk_len):
        events["formal"].append(
            dict(
                modes=[workers[wid].mode for wid in range(args.n_workers)],
                episodes=[workers[wid].global_episode for wid in range(args.n_workers)],
                chunk_len=chunk_len,
                segment_remaining=[
                    workers[wid].ckpt.counters["segment_ep_remaining"]
                    for wid in range(args.n_workers)
                ],
            )
        )
        for worker in workers.values():
            worker.global_episode += chunk_len
            worker.ckpt.q_network["weight"] += 1

    monkeypatch.setattr(pbt, "parse_args_and_validate", lambda argv: args)
    monkeypatch.setattr(pbt, "init_worker", init_worker)
    monkeypatch.setattr(pbt, "_apply_adapt_snapshot_to_worker", restore)
    monkeypatch.setattr(pbt, "_build_adapt_snapshot", snapshot)
    monkeypatch.setattr(pbt, "_run_adapt_window_parallel_spawn", adapt)
    monkeypatch.setattr(pbt, "_run_formal_chunk_parallel_spawn", formal)
    return args, events, tmp_path / "schedule"


def seed_recovery(args, output, phase, episode):
    snapshots = {
        wid: worker_state(wid, episode, "P2" if wid % 2 else "P1") for wid in range(args.n_workers)
    }
    save_pbt_recovery(
        output / "recovery",
        generation_id=1,
        chunk_id=1,
        global_episode=episode,
        phase=phase,
        compatibility=pbt._pbt_resume_signature(args),
        controller_state=dict(
            phase=phase,
            adapt_progress=min(episode, args.adapt_episodes),
            formal_progress=max(0, episode - args.adapt_episodes),
            stop_state={},
            master_rng=capture_rng_state(),
        ),
        worker_ids=range(args.n_workers),
        worker_payload=lambda wid: snapshots[wid],
    )
    args.resume_run_dir = str(output)
    return snapshots


def test_adaptation_enters_formal_p1_without_another_training_window(harness):
    args, events, output = harness
    pbt.run_pbt_master([])
    assert events["adapt"] == [("P2", 32)]
    assert [event["episodes"] for event in events["formal"]] == [[32] * 4, [48] * 4]
    assert events["formal"][0]["modes"] == ["P1"] * 4
    assert events["formal"][0]["segment_remaining"] == [6] * 4
    saved = load_pbt_recovery(output, expected_worker_ids=range(4))
    assert saved["controller"]["phase"] == "formal"
    assert saved["controller"]["adapt_progress"] == 32
    assert saved["controller"]["formal_progress"] == 32
    assert saved["manifest"]["global_episode"] == 64


def test_adaptation_resume_runs_only_remaining_adaptation(harness):
    args, events, output = harness
    seed_recovery(args, output, "adapt", 16)
    pbt.run_pbt_master([])
    assert events["adapt"] == [("P2", 16)]
    assert events["formal"][0]["episodes"] == [32] * 4
    assert events["formal"][0]["modes"] == ["P1"] * 4
    assert events["formal"][0]["segment_remaining"] == [6] * 4


def test_budget_tail_uses_a_partial_final_chunk(harness):
    args, events, output = harness
    args.max_pbt_episodes = 56
    pbt.run_pbt_master([])
    assert [event["chunk_len"] for event in events["formal"]] == [16, 8]
    saved = load_pbt_recovery(output, expected_worker_ids=range(4))
    assert saved["manifest"]["global_episode"] == 56
    assert saved["controller"]["formal_progress"] == 24


def test_formal_resume_preserves_worker_modes_and_model_state(harness):
    args, events, output = harness
    snapshots = seed_recovery(args, output, "formal", 48)
    pbt.run_pbt_master([])
    assert events["adapt"] == []
    assert len(events["formal"]) == 1
    assert events["formal"][0]["modes"] == ["P1", "P2", "P1", "P2"]
    saved = load_pbt_recovery(output, expected_worker_ids=range(4))
    for wid, snapshot in snapshots.items():
        torch.testing.assert_close(
            saved["workers"][wid]["ckpt"]["q_network"]["weight"],
            snapshot["ckpt"]["q_network"]["weight"] + 1,
        )
        assert saved["workers"][wid]["cooldown_remaining"] == wid


def test_unknown_recovery_phase_is_rejected_before_training(harness):
    args, events, output = harness
    seed_recovery(args, output, "unsupported", 16)
    with pytest.raises(pbt.RecoveryError, match="Unsupported PBT recovery phase"):
        pbt.run_pbt_master([])
    assert events == {"adapt": [], "formal": []}


def test_adaptation_parser_accepts_only_phase_training():
    parser = argparse.ArgumentParser()
    pbt._add_schedule_group(parser)
    assert parser.parse_args([]).adapt_mode == "P2"
    with pytest.raises(SystemExit):
        parser.parse_args(["--adapt_mode", "P1"])


@pytest.mark.parametrize("chunk", [8, 16])
def test_real_phase_adaptation_handoff_runs_six_dqn_then_ten_power_episodes(
    tmp_path, monkeypatch, chunk
):
    from atprl.envs.leo_sat import LeoSatEnv
    from atprl.training import stage1, stage2

    config = load_config(ROOT / "configs/paper_main.yaml")
    args = pbt.parse_args_and_validate([*cli_args(config), "--resume_run_dir", str(tmp_path)])
    args.n_workers = 1
    args.ris_num_elem_x = args.ris_num_elem_y = 2
    args.p1_dqn_batch_size = 4
    args.p1_ppo_power_buffer_size = args.p2_ppo_phase_buffer_size = 4
    args.p1_ppo_power_epochs = args.p2_ppo_phase_epochs = 1
    args.enable_tb = False

    def environment(config):
        return LeoSatEnv(dict(config, max_steps_per_episode=4, num_micro_steps=2))

    monkeypatch.setattr(stage1, "LeoSatEnv", environment)
    monkeypatch.setattr(stage2, "LeoSatEnv", environment)
    writer = pbt.NullSummaryWriter(tmp_path)
    worker = pbt.init_worker(args, 0, log_dir=tmp_path, writer=writer, skip_warmup_bootstrap=True)
    pbt.train_chunk(worker, 16)
    snapshot = pbt._build_adapt_snapshot(worker, train_info=None)
    workers = pbt._initialize_formal_workers(
        args,
        SimpleNamespace(worker_dirs={0: tmp_path}),
        {0: writer},
        {0: snapshot},
        resume_formal=False,
    )
    worker = workers[0]
    phase_before = {
        key: value.clone() for key, value in worker.p1_trainer.phase_actor.state_dict().items()
    }
    modes = []
    current_mode = [None]
    agent = next(iter(worker.p1_trainer.agents.values()))
    select = agent.select_action
    step = worker.p1_trainer.env.step

    def trace(*args, **kwargs):
        current_mode[0] = "POWER" if kwargs.get("deterministic_dqn") else "DQN"
        return select(*args, **kwargs)

    def trace_step(actions):
        modes.append(current_mode[0])
        return step(actions)

    monkeypatch.setattr(agent, "select_action", trace)
    monkeypatch.setattr(worker.p1_trainer.env, "step", trace_step)
    pbt.train_chunk(worker, chunk)
    assert modes == ["DQN"] * 24 + ["POWER"] * ((chunk - 6) * 4)
    assert worker.global_episode == 16 + chunk
    assert worker.p1_trainer.episodes_done == 16 + chunk
    for key, value in phase_before.items():
        torch.testing.assert_close(
            worker.p1_trainer.phase_actor.state_dict()[key], value, rtol=0, atol=0
        )
    if chunk == 8:
        assert worker.ckpt.counters["power_episode_offset"] == 2
        paused = pbt._build_adapt_snapshot(worker, train_info=None)
        resumed = pbt._initialize_formal_workers(
            args,
            SimpleNamespace(worker_dirs={0: tmp_path}),
            {0: writer},
            {0: paused},
            resume_formal=True,
        )[0]
        assert resumed.p1_trainer.power_episode_offset == 2
        pbt.train_chunk(resumed, 8)
        assert resumed.global_episode == resumed.p1_trainer.episodes_done == 32
        assert resumed.p1_trainer.power_episode_offset == 0
        continuous = pbt._initialize_formal_workers(
            args,
            SimpleNamespace(worker_dirs={0: tmp_path}),
            {0: writer},
            {0: snapshot},
            resume_formal=False,
        )[0]
        pbt.train_chunk(continuous, 16)
        for model in ("q_network", "power_actor", "power_critic", "phase_actor"):
            expected = getattr(continuous.p1_trainer, model).state_dict()
            actual = getattr(resumed.p1_trainer, model).state_dict()
            for key, value in expected.items():
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
