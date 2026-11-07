import json
import sys
from pathlib import Path

import pytest

from atprl.config.loader import cli_args, load_config
from atprl.envs.leo_sat import LeoSatEnv
from atprl.evaluation import utils
from atprl.runtime import configure_cpu_threads
from atprl.training import stage1, stage2, warmup
from atprl.training.pbt import controller as pbt
from atprl.training.recovery import load_pbt_recovery

ROOT = Path(__file__).resolve().parents[2]


def small_environment(config):
    return LeoSatEnv(dict(config, max_steps_per_episode=4, num_micro_steps=2))


def configure_child():
    configure_cpu_threads()
    for module in (stage1, stage2, utils):
        module.LeoSatEnv = small_environment


def adaptation_worker(*args, **kwargs):
    configure_child()
    pbt._adapt_worker_process_main(*args, **kwargs)


def formal_worker(*args, **kwargs):
    configure_child()
    pbt._formal_chunk_worker_process_main(*args, **kwargs)


def evaluation_worker(*args, **kwargs):
    configure_child()
    pbt._ready_eval_worker_process_main(*args, **kwargs)


@pytest.mark.parametrize("endpoint", [40, 48])
def test_four_worker_training_exploitation_and_process_resume(tmp_path, monkeypatch, endpoint):
    for module in (warmup, stage1, stage2, utils):
        monkeypatch.setattr(module, "LeoSatEnv", small_environment)
    monkeypatch.chdir(tmp_path)
    config = load_config(ROOT / "configs/base/warmup.yaml")
    config["args"].update(
        device="cpu",
        max_train_episodes=16,
        eval_every_episodes=16,
        curve_k=1,
        checkpoint_every_episodes=16,
        dqn_batch_size=4,
        ppo_power_buffer_size=4,
        ppo_power_epochs=1,
        ris_num_elem_x=2,
        ris_num_elem_y=2,
    )
    monkeypatch.setattr(sys, "argv", ["warmup", *cli_args(config)])
    warmup.main()
    kernel = next(tmp_path.rglob("q_network_phase1_best.pth")).parent

    config = load_config(ROOT / "configs/paper_main.yaml")
    config["args"].update(
        exp_name="pbt_process_smoke",
        runs_root=str(tmp_path),
        warmup_model_dir=str(kernel),
        n_workers=4,
        adapt_episodes=16,
        max_pbt_episodes=32,
        ready_every_episodes=16,
        cooldown_episodes=32,
        eval_every_episodes=16,
        k_daily=1,
        k_ready=1,
        p12_flip_prob=1.0,
        p21_flip_prob=0.0,
        p1_dqn_batch_size=4,
        p1_ppo_power_buffer_size=4,
        p2_ppo_phase_buffer_size=4,
        p1_ppo_power_epochs=1,
        p2_ppo_phase_epochs=1,
        ris_num_elem_x=2,
        ris_num_elem_y=2,
    )
    monkeypatch.setattr(pbt, "_adapt_worker_process_main", adaptation_worker)
    monkeypatch.setattr(pbt, "_formal_chunk_worker_process_main", formal_worker)
    monkeypatch.setattr(pbt, "_ready_eval_worker_process_main", evaluation_worker)
    pbt.run_pbt_master(cli_args(config))
    output = tmp_path / "pbt_process_smoke"
    first = load_pbt_recovery(output, expected_worker_ids=range(4))
    assert first["controller"]["phase"] == "formal"
    assert first["controller"]["adapt_progress"] == 16
    assert first["controller"]["formal_progress"] == 16
    assert first["manifest"]["global_episode"] == 32
    assert sum(worker["mode"] == "P2" for worker in first["workers"].values()) == 1

    config["args"].update(max_pbt_episodes=endpoint, resume_run_dir=str(output))
    pbt.run_pbt_master(cli_args(config))
    resumed = load_pbt_recovery(output, expected_worker_ids=range(4))
    assert resumed["manifest"]["generation_id"] > first["manifest"]["generation_id"]
    assert resumed["manifest"]["global_episode"] == endpoint
    assert resumed["controller"]["adapt_progress"] == 16
    assert resumed["controller"]["formal_progress"] == endpoint - 16
    for wid in range(4):
        assert resumed["workers"][wid]["global_episode"] == endpoint
        assert resumed["workers"][wid]["ckpt"]["counters"]["episodes_done"] == endpoint
        events = [
            json.loads(line)
            for line in (output / f"worker_{wid:02d}/ready_events.jsonl").read_text().splitlines()
        ]
        assert len(events) == (2 if endpoint == 48 else 1)
    from tests.integration.test_evaluation_configs import evaluate_trained_model

    evaluate_trained_model("atp_rl", output, tmp_path / "final_evaluation", monkeypatch)
