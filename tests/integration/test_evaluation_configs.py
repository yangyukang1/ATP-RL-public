import sys
import json
from pathlib import Path

import numpy as np
import pytest

from atprl.config.loader import cli_args, load_config
from atprl.evaluation import runner

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "method", ["atp_rl", "stp_rl", "rnd_stp", "wmmse_stp", "periodic", "paper_all"]
)
@pytest.mark.parametrize("enabled", [True, False])
def test_evaluation_boolean_values_reach_the_real_parser(method, enabled, tmp_path, monkeypatch):
    config = load_config(ROOT / f"configs/evaluation/{method}.yaml")
    paths = [
        "pbt_run_dir",
        "phase2_model_dir",
        "periodic_phase1_model_dir",
        "atp_pbt_run_dir",
        "stp_phase2_model_dir",
        "rnd_stp_phase2_model_dir",
        "wmmse_phase2_model_dir",
        "periodic_phase1_model_dir_batch",
    ]
    config["args"].update({name: str(tmp_path) for name in paths})
    config["args"].update(show_progress=enabled, stream_write=enabled)
    monkeypatch.setattr(sys, "argv", ["evaluate", *cli_args(config)])
    args = runner.parse_args()
    assert args.show_progress is enabled
    assert args.stream_write is enabled
    assert args.run_batch is (method == "paper_all")
    assert args.include_mrt_features


def evaluate_trained_model(method, model_path, output, monkeypatch):
    from atprl.envs.leo_sat import LeoSatEnv
    from atprl.evaluation import utils
    from atprl.training import stage2

    def environment(config):
        config.update(
            max_steps_per_episode=4, num_micro_steps=2, ris_num_elem_x=2, ris_num_elem_y=2
        )
        return LeoSatEnv(config)

    monkeypatch.setattr(stage2, "LeoSatEnv", environment)
    monkeypatch.setattr(utils, "LeoSatEnv", environment)
    monkeypatch.setattr(runner, "LeoSatEnv", environment)
    source_key = {"atp_rl": "pbt_run_dir", "periodic": "periodic_phase1_model_dir"}.get(
        method, "phase2_model_dir"
    )
    config = load_config(ROOT / f"configs/evaluation/{method}.yaml")
    config["args"].update({source_key: str(model_path)})
    config["args"].update(
        output_dir=str(output),
        num_eval_seeds=2,
        show_progress=False,
        stream_write=True,
        phase2_phase_ckpt_strict=True,
        phase1_ckpt_strict=True,
    )
    monkeypatch.setattr(sys, "argv", ["evaluate", *cli_args(config)])
    runner.main()
    result = json.loads((output / "eval_metrics_pack.json").read_text())
    assert result["seed_list"] == [10001, 10038]
    assert len(result["metrics"]) == 4
    assert all(np.isfinite(row["mean"]) for row in result["metrics"].values())
    return result
