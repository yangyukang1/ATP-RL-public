import ast
import io
import os
import tokenize
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import j0
from threadpoolctl import threadpool_info

from atprl.config.loader import cli_args, load_config
from atprl.envs.leo_sat import LeoSatEnv
from atprl.nn.models import (
    ActorNetwork,
    CriticNetwork,
    PhaseActorGaussianLatent,
    PhaseCritic,
    QNetwork,
)
from atprl.runtime import THREAD_ENV, configure_cpu_threads
from atprl.training.joint_runtime import build_env_config, make_args
from atprl.training.pbt.controller import build_worker_args, parse_args_and_validate

ROOT = Path(__file__).resolve().parents[2]


def test_physical_parameters_and_time_resolution():
    config = build_env_config(make_args())
    expected = {
        "num_ground_stations": 4,
        "num_gs_per_rb": 2,
        "num_resource_blocks": 1,
        "satellite_altitude_m": 600000,
        "carrier_frequency_GHz": 4.0,
        "satellite_max_power_dBm": 50.0,
        "terrestrial_cell_power_dBm": 35.0,
        "noise_power_dBm": -114.0,
        "active_ris_noise_power_dBm": -100.0,
        "ris_reflection_power_limit_dBm": -5.0,
        "ris_elem_max_amp_gain_dB": 40.0,
        "satellite_antenna_gain_dBi": 24.3,
        "ground_station_antenna_gain_dBi": 24.3,
        "uav_rx_antenna_gain_dBi": 15.0,
        "k_factor_sat_to_ris_linear": 30.0,
        "k_factor_ris_to_gs_linear": 10.0,
        "rician_k_factor": 10.0,
        "sic_detection_threshold_dBm": -94.0,
        "csi_error_variance": 0.01,
        "max_steps_per_episode": 1024,
        "fairness_lambda": 3.0,
        "fairness_beta": 0.05,
    }
    for key, value in expected.items():
        assert config[key] == value, key
    env = LeoSatEnv(config)
    assert env.ris_num_groups == 4
    assert config["ris_num_elem_x"] * config["ris_num_elem_y"] == 256
    assert env.micro_step_interval_s == pytest.approx(0.002)
    assert env._compute_jakes_rho(50.0, 0.002) == pytest.approx(j0(2 * np.pi * 50 * 0.002))
    assert config["uav_initial_position_xyz"][2] == 80.0
    assert np.linalg.norm(config["uav_velocity_per_step_xyz"]) / 0.04 == pytest.approx(10.0)


def test_pbt_worker_profiles_and_training_chunks(tmp_path):
    config = load_config(ROOT / "configs/paper_main.yaml")
    args = parse_args_and_validate([*cli_args(config), "--resume_run_dir", str(tmp_path)])
    assert (args.n_workers, args.chunk_len, args.num_exploit) == (4, 16, 1)
    assert (args.adapt_episodes, args.ready_every_episodes, args.cooldown_episodes) == (
        2464,
        224,
        448,
    )
    assert (args.p12_flip_prob, args.p21_flip_prob) == (0.33, 0.25)
    for i, (lr, entropy) in enumerate(zip([1.0, 0.7, 1.3, 0.3], [1.0, 0.8, 1.3, 1.0])):
        p1, p2 = build_worker_args(args, i)
        assert p1.device == p2.device == "cpu"
        assert (p1.cycle_dqn_before_power, p1.ppo_power_opt_episodes) == (6, 10)
        assert p2.ppo_phase_opt_episodes == 16
        assert (p1.dqn_batch_size, p1.dqn_update_freq, p1.target_update_freq) == (128, 4, 8)
        assert p1.lr_dqn == pytest.approx(5e-4 * lr)
        assert p1.lr_power_ppo == pytest.approx(3e-4 * lr)
        assert p2.lr_phase_ppo == pytest.approx(6e-5 * lr)
        assert p1.entropy_coef_power == pytest.approx(0.012 * entropy)
        assert p2.entropy_coef_phase == pytest.approx(0.006 * entropy)
        assert p1.ppo_power_buffer_size == p2.ppo_phase_buffer_size == 1024
        assert p1.gamma == p2.gamma == 0.9
        assert p1.gae_lambda == p2.gae_lambda == 0.95


def test_network_architectures():
    models = [
        (QNetwork(10, 6), [64, 64, 6]),
        (ActorNetwork(10, 3, 100.0), [64, 64, 3, 3]),
        (CriticNetwork(10), [64, 64, 1]),
        (PhaseCritic(10), [128, 128, 1]),
    ]
    for model, expected in models:
        assert [
            layer.out_features for layer in model.modules() if isinstance(layer, torch.nn.Linear)
        ] == expected
    phase = PhaseActorGaussianLatent(10, 4, learn_amp=False)
    assert [layer.out_features for layer in phase.body if isinstance(layer, torch.nn.Linear)] == [
        128,
        128,
    ]
    assert isinstance(phase.body[1], torch.nn.ReLU)
    assert isinstance(phase.body[2], torch.nn.LayerNorm)


def test_cpu_thread_limit_is_idempotent_and_covers_blas():
    configure_cpu_threads()
    configure_cpu_threads()
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert all(os.environ[name] == "1" for name in THREAD_ENV)
    assert all(pool["num_threads"] == 1 for pool in threadpool_info())


def test_comments_and_docstrings_remain_concise_english():
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        comments = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        ]
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    assert len(doc) <= 180, path
                    comments.append(doc)
        assert all(comment.isascii() for comment in comments), path
