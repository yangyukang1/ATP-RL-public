from pathlib import Path
from types import SimpleNamespace

from atprl.config.budgets import TOTAL_TRAINING_EPISODES, WARMUP_EPISODES, POST_WARMUP_EPISODES
from atprl.config.loader import load_config
from atprl.config.sac import SACRunConfig
from atprl.training.pbt.controller import init_stop_state, soft_stop_update_on_ready

ROOT = Path(__file__).resolve().parents[2]


def test_all_learning_methods_share_the_cumulative_episode_budget():
    warmup = load_config(ROOT / "configs/base/warmup.yaml")["args"]["max_train_episodes"]
    totals = {}
    for method in ("stp_rl", "rnd_stp", "wmmse_stp"):
        config = load_config(ROOT / f"configs/methods/{method}.yaml")
        totals[method] = warmup + config["args"]["max_train_episodes"]
        assert config["args"]["early_stop_patience_eval"] == 0
    atp = load_config(ROOT / "configs/methods/atp_rl.yaml")["args"]
    totals["atp_rl"] = warmup + atp["max_pbt_episodes"]
    assert atp["patience_ready"] == 0
    assert atp["n_workers"] == 4
    for method in ("joint_rl", "no_ris", "periodic"):
        totals[method] = load_config(ROOT / f"configs/methods/{method}.yaml")["args"][
            "max_train_episodes"
        ]
    sac = SACRunConfig(**load_config(ROOT / "configs/methods/stp_sac.yaml")["args"])
    totals["stp_sac"] = sac.stage1_episodes + sac.stage2_episodes
    assert warmup == WARMUP_EPISODES == 1440
    assert POST_WARMUP_EPISODES == 15552
    assert set(totals.values()) == {TOTAL_TRAINING_EPISODES} == {16992}
    conventional = load_config(ROOT / "configs/methods/conventional.yaml")
    assert "max_train_episodes" not in conventional["args"]


def test_disabled_pbt_early_stopping_cannot_shorten_the_budget():
    args = SimpleNamespace(patience_ready=0, min_delta=15.0, smooth_alpha=0.2)
    workers = {0: SimpleNamespace(last_ready_score=1.0)}
    state = init_stop_state()
    for index in range(100):
        assert not soft_stop_update_on_ready(workers, args, state, ready_idx=index, log_paths=None)
