import contextlib
import importlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from atprl.cli import METHODS, main
from atprl.config.loader import cli_args, load_config, validate_config
from atprl.envs.leo_sat import LeoSatEnv
from atprl.training.joint_runtime import JointRuntime, make_args

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("method", METHODS)
def test_complete_method_command_materializes(method):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), pytest.raises(SystemExit) as result:
        main(["train", "--method", method, "--seed", "3", "--dry-run"])
    assert result.value.code == 0
    commands = [json.loads(line)["command"] for line in output.getvalue().splitlines()]
    assert len(commands) == (2 if method in ("atp_rl", "stp_rl", "rnd_stp", "wmmse_stp") else 1)
    if method in ("rnd_stp", "wmmse_stp"):
        flag, value = (
            ("--scheduler_mode", "random") if method == "rnd_stp" else ("--power_mode", "wmmse")
        )
        for command in commands:
            assert command[command.index(flag) + 1] == value


def test_all_configs_resolve_and_target_real_flags():
    import ast
    from atprl.cli import MODULES
    from atprl.training.joint_runtime import DEFAULTS as JOINT_DEFAULTS
    from atprl.evaluation.conventional_config import DEFAULTS as CONVENTIONAL_DEFAULTS

    for path in (ROOT / "configs").rglob("*.yaml"):
        config = load_config(path)
        validate_config(config, for_run=False)
        if config["entrypoint"] == "sac_stp":
            continue
        module = importlib.import_module(MODULES[config["entrypoint"]])
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        flags = {
            a.value[2:]
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument"
            for a in n.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")
        }
        if config["entrypoint"] == "joint":
            flags.update(JOINT_DEFAULTS)
        if config["entrypoint"] == "conventional":
            flags.update(CONVENTIONAL_DEFAULTS)
        assert {key for key, value in config["args"].items() if value is not None} <= flags, path


def test_joint_uses_all_users_without_action_conditioning():
    config = load_config(ROOT / "configs/methods/joint_rl.yaml")
    from atprl.training.joint_runtime import DEFAULTS

    args = make_args(**{k: v for k, v in config["args"].items() if k in DEFAULTS})
    runtime = JointRuntime(args)
    assert runtime.phase_state_dim == 114
    agent = next(iter(runtime.agents.values()))
    obs = runtime.env.reset()[runtime.env.agents[0]]
    a = agent._build_phase_state(obs, (0, 1), np.array([0.2, 0.4, 0.4]))
    b = agent._build_phase_state(obs, (2, 3), np.array([0.8, 0.1, 0.1]))
    np.testing.assert_array_equal(a, b)


def test_wrapped_sac_pipeline_starts_without_external_kernel(tmp_path):
    from tests.integration.test_sac_stp import smoke_config
    from atprl.training.sac_stp import run_seed
    from atprl.training.sac_recovery import load_torch

    config = replace(
        smoke_config(),
        stage="pipeline",
        phase_distribution="wrapped_normal_v1",
        power_sac=dict(capacity=128, batch_size=8, learning_starts=8, kappa=1.5),
        evaluation_seeds=[10001],
        eval_every_episodes=16,
        auto_final_evaluation=False,
    )
    assert run_seed(config, tmp_path)
    policy = load_torch(tmp_path / "stage2/policy_best.ckpt")
    assert policy["phase"]["phase_distribution"] == "wrapped_normal_v1"
    assert policy["kernel"]["power"]["config"]["kappa"] == 1.5


@pytest.mark.parametrize("method", ["stp_rl", "rnd_stp", "wmmse_stp", "no_ris"])
def test_warmup_checkpoint_and_method_specific_actions(method, tmp_path, monkeypatch):
    from atprl.training import warmup
    from atprl.training.recovery import load_warmup_recovery

    path = ROOT / (
        f"configs/methods/{method}.yaml" if method == "no_ris" else f"configs/warmup/{method}.yaml"
    )
    config = load_config(path)
    config["args"].update(
        device="cpu",
        max_train_episodes=2,
        eval_every_episodes=2,
        curve_k=1,
        checkpoint_every_episodes=2,
        cycle_dqn_before_power=1,
        ppo_power_opt_episodes=1,
        ppo_power_epochs=1,
        ppo_power_buffer_size=4,
        ris_num_elem_x=2,
        ris_num_elem_y=2,
    )
    original_env = warmup.LeoSatEnv

    def small_env(config):
        config = dict(
            config, max_steps_per_episode=4, num_micro_steps=2, ris_num_elem_x=2, ris_num_elem_y=2
        )
        return original_env(config)

    monkeypatch.setattr(warmup, "LeoSatEnv", small_env)
    monkeypatch.chdir(tmp_path)
    with patch.object(sys, "argv", ["warmup", *cli_args(config)]):
        warmup.main()
    checkpoint = next(tmp_path.rglob("recovery/latest.ckpt"))
    payload = load_warmup_recovery(checkpoint)
    assert payload["progress"]["episodes_done"] == 2
    if method == "no_ris":
        from atprl.evaluation.no_ris import evaluate

        assert payload["no_ris_best_policy"] is not None
        pack = evaluate(checkpoint, tmp_path / "evaluation", seeds=[10001, 10038])
        assert pack["environment"]["disable_ris_path"]
        assert pack["seed_list"] == [10001, 10038]
        assert evaluate(checkpoint, tmp_path / "evaluation", seeds=[10001, 10038]) == pack
    else:
        from atprl.training import stage2
        import torch

        model_dir = next(tmp_path.rglob("q_network_phase1_best.pth")).parent
        frozen_q = torch.load(model_dir / "q_network_phase1_best.pth", weights_only=True)
        config = load_config(ROOT / f"configs/methods/{method}.yaml")
        config["args"].update(
            device="cpu",
            phase1_model_dir=str(model_dir),
            max_train_episodes=2,
            eval_every_episodes=2,
            curve_k=1,
            ppo_phase_opt_episodes=2,
            ppo_phase_epochs=1,
            ppo_phase_buffer_size=4,
        )
        monkeypatch.setattr(stage2, "LeoSatEnv", small_env)
        stage_dir = tmp_path / "stage2"
        stage_dir.mkdir()
        monkeypatch.chdir(stage_dir)
        with patch.object(sys, "argv", ["stage2", *cli_args(config)]):
            stage2.main()
        q_file = next(stage_dir.rglob("q_network_final.pth"))
        actual = torch.load(q_file, weights_only=True)
        for key, value in frozen_q.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        from tests.integration.test_evaluation_configs import evaluate_trained_model

        evaluate_trained_model(method, q_file.parent, tmp_path / "final_evaluation", monkeypatch)


def test_joint_cli_evaluates_endpoint(tmp_path):
    from atprl.training import joint

    args = [
        "joint",
        "--device",
        "cpu",
        "--seeds",
        "3",
        "--output_root",
        str(tmp_path),
        "--max_train_episodes",
        "2",
        "--max_steps_per_episode",
        "8",
        "--rollout_length",
        "8",
        "--learning_starts",
        "4",
        "--dqn_batch_size",
        "4",
        "--replay_capacity",
        "64",
        "--eval_every_episodes",
        "1",
        "--selection_every_episodes",
        "1",
        "--checkpoint_every_episodes",
        "1",
        "--k_daily",
        "1",
        "--k_ready",
        "1",
        "--num_eval_seeds",
        "2",
        "--ris_num_elem_x",
        "2",
        "--ris_num_elem_y",
        "2",
        "--phase_state_mode",
        "all_users_no_action",
    ]
    with patch.object(sys, "argv", args):
        joint.main()
    final = json.loads((tmp_path / "seed_3/final_evaluation.json").read_text())
    assert final["artifact_role"] == "fixed_training_endpoint"
    assert final["selected_episode"] == 2
    assert (tmp_path / "seed_3/models/endpoint.ckpt").is_file()


def test_periodic_trains_only_power_with_balanced_service(tmp_path, monkeypatch):
    from atprl.training import periodic
    from atprl.rl.algorithms import DQN

    config = load_config(ROOT / "configs/methods/periodic.yaml")
    config["args"].update(
        device="cpu",
        max_train_episodes=2,
        eval_every_episodes=2,
        curve_k=1,
        ppo_power_opt_episodes=2,
        ppo_power_epochs=1,
        ppo_power_buffer_size=4,
        ris_num_elem_x=2,
        ris_num_elem_y=2,
    )
    original = periodic.LeoSatEnv
    monkeypatch.setattr(
        periodic,
        "LeoSatEnv",
        lambda config: original(dict(config, max_steps_per_episode=4, num_micro_steps=2)),
    )
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(DQN, "update", side_effect=AssertionError("Periodic must not train DQN")),
        patch.object(sys, "argv", ["periodic", *cli_args(config)]),
    ):
        periodic.main()
    assert list(tmp_path.rglob("power_actor_phase1_best.pth"))
    pairs, targets = periodic._validate_periodic_cycles_for_topology(4, 2)
    counts = np.bincount(np.asarray(pairs).reshape(-1), minlength=4)
    assert len(set(counts)) == 1
    assert len(set(np.bincount(targets, minlength=4))) == 1
    from tests.integration.test_evaluation_configs import evaluate_trained_model

    model_dir = next(tmp_path.rglob("power_actor_phase1_best.pth")).parent
    evaluate_trained_model("periodic", model_dir, tmp_path / "final_evaluation", monkeypatch)


def test_no_ris_saves_the_budget_endpoint_between_checkpoint_intervals(tmp_path, monkeypatch):
    from atprl.training import warmup
    from atprl.training.recovery import load_warmup_recovery
    from atprl.evaluation.no_ris import evaluate

    config = load_config(ROOT / "configs/methods/no_ris.yaml")
    config["args"].update(
        max_train_episodes=24,
        eval_every_episodes=16,
        curve_k=1,
        checkpoint_every_episodes=16,
        dqn_batch_size=4,
        ppo_power_buffer_size=4,
        ppo_power_epochs=1,
        ris_num_elem_x=2,
        ris_num_elem_y=2,
    )

    def environment(config):
        config.update(max_steps_per_episode=4, num_micro_steps=2)
        return LeoSatEnv(config)

    monkeypatch.setattr(warmup, "LeoSatEnv", environment)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["warmup", *cli_args(config)])
    warmup.main()
    checkpoint = next(tmp_path.rglob("recovery/latest.ckpt"))
    payload = load_warmup_recovery(checkpoint)
    assert payload["progress"]["episodes_done"] == 24
    assert payload["no_ris_runtime"]["args"]["max_train_episodes"] == 24
    result = evaluate(checkpoint, tmp_path / "final_evaluation", seeds=[10001])
    assert result["seed_list"] == [10001]
