from __future__ import annotations

from atprl.config.budgets import POST_WARMUP_EPISODES

import os
import argparse
import copy
import json
import random
import time
import traceback
import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shutil
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from atprl.training.recovery import (
    RecoveryError,
    capture_rng_state,
    load_pbt_recovery,
    load_replay_buffer_state,
    load_trajectory_buffer_state,
    restore_rng_state,
    save_pbt_recovery,
    signature,
    validate_signature,
)

import math
from atprl.envs.topologies import resolve_topology
from atprl.config.kernel_signature import (
    KERNEL_DIMENSION_FIELDS,
    read_kernel_signature,
    validate_checkpoint_metadata,
)


MODE_P1 = "P1"
MODE_P2 = "P2"
MODE_TO_INT = {MODE_P1: 0, MODE_P2: 1}

TB_EVENT_EXPLOIT = "event_exploit"
TB_EVENT_FLIP = "event_flip"

try:
    from atprl.training.stage1 import Phase1TrainerPBT
except Exception as e:
    raise ImportError(
        "Failed to import Phase1TrainerPBT from atprl.training.stage1. "
        "Please ensure this file is on PYTHONPATH and the class name is correct."
    ) from e

try:
    from atprl.training.stage2 import (
        Phase2Trainer,
    )
except Exception as e:
    raise ImportError(
        "Failed to import Phase2Trainer from atprl.training.stage2. "
        "Please ensure this file is on PYTHONPATH and the class name is correct."
    ) from e

try:
    from atprl.evaluation.utils import (
        FIXED_VALIDATION_SEEDS,
        get_eval_config,
        run_multi_seed_validation_metrics,
        build_eval_curve_record,
    )
except Exception as e:
    raise ImportError(
        "Failed to import unified eval helpers from atprl.evaluation.utils. "
        "Please ensure this file is on PYTHONPATH and the names are correct."
    ) from e


def _add_warmup_group(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Warmup (Phase-I kernel) input (STRICT)")

    g.add_argument(
        "--tb_flush_secs",
        type=int,
        default=600,
        help="TensorBoard flush seconds (bigger => fewer disk writes).",
    )
    g.add_argument(
        "--tb_max_queue",
        type=int,
        default=500,
        help="TensorBoard max_queue (bigger => fewer disk writes, more RAM).",
    )
    g.add_argument(
        "--tb_p2_diag_every",
        type=int,
        default=32,
        help="Downsample P2 diagnostics tags every N global steps.",
    )
    g.add_argument(
        "--keep_last_ready_ckpts",
        type=int,
        default=8,
        help="Keep only last K ready checkpoints per worker (older will be deleted).",
    )

    g.add_argument(
        "--warmup_model_dir",
        type=str,
        default=None,
        help=(
            "阶段一 warmup 模型目录（唯一输入）。\n"
            "必须至少包含 old warmup 三件套：\n"
            "  - q_network_phase1(.pth/_best.pth)\n"
            "  - power_actor_phase1(.pth/_best.pth)\n"
            "  - power_critic_phase1(.pth/_best.pth)\n"
            "若同时还包含：\n"
            "  - phase_actor_phase1(.pth/_best.pth)\n"
            "  - phase_critic_phase1(.pth/_best.pth)\n"
            "则自动识别为 new warmup，并在所有 worker 初始化时统一继承 DQN + Power + Phase。"
        ),
    )
    g.add_argument(
        "--warmup_ckpt",
        type=str,
        default="best",
        choices=["best", "final"],
        help="选择 warmup checkpoint：best 或 final。best 将优先 *_best.pth，若缺失则回退到 .pth。",
    )
    g.add_argument(
        "--warmup_tag",
        type=str,
        default=None,
        choices=["l0", "l1", "l2", "l3", "l4", "l5", "l6"],
        help="仅用于日志/文件名标记（不参与逻辑）。总控脚本只允许一个 warmup_tag。",
    )
    g.add_argument(
        "--warmup_ckpt_strict",
        action="store_true",
        help="若设置且 --warmup_ckpt=best，则 *_best.pth 缺失时直接报错（不回退到 .pth）。",
    )


def _add_schedule_group(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("PBT schedule (core)")
    g.add_argument("--n_workers", type=int, default=4)
    g.add_argument("--chunk_len", type=int, default=16, help="每次 train_for 的 episode 数。")

    g.add_argument(
        "--eval_every_episodes",
        "--daily_every_episodes",
        dest="eval_every_episodes",
        type=int,
        default=32,
        help="Trigger unified eval at episode multiples. Alias: --daily_every_episodes",
    )

    g.add_argument("--ready_every_episodes", type=int, default=224, help="Ready eval 频率（R）。")
    g.add_argument("--cooldown_episodes", type=int, default=448, help="Cooldown（C）。")

    g.add_argument(
        "--adapt_episodes", type=int, default=2464, help="适配窗口 W_adapt 的 episode 数。"
    )
    g.add_argument(
        "--adapt_mode",
        type=str,
        default="P2",
        choices=["P2"],
        help="Use phase-only training during adaptation.",
    )


def _add_eval_seed_group(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Eval seeds")
    g.add_argument("--k_daily", type=int, default=3, help="Daily eval 使用的 seeds 数。")
    g.add_argument(
        "--k_ready",
        type=int,
        default=15,
        help="Ready eval 使用的 seeds 数（排名 single score 的均值来源）。",
    )
    g.add_argument("--eval_seed_base", type=int, default=10001, help="评估 seed 基准。")
    g.add_argument(
        "--eval_seed_stride", type=int, default=37, help="用于生成 seed 的步长（可选）。"
    )


def _add_exploit_explore_group(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Exploit / Explore (PBT events)")
    g.add_argument(
        "--exploit_rule",
        type=str,
        default="truncation",
        choices=["truncation"],
        help="当前固定为 truncation：bottom_k 复制 top_k。",
    )
    g.add_argument(
        "--num_exploit",
        type=int,
        default=1,
        help="每次 ready 执行的复制对数（如 1: bottom1<-top1）。",
    )
    g.add_argument(
        "--p12_flip_prob", type=float, default=0.33, help="被复制者若当前 P1，翻到 P2 的概率。"
    )
    g.add_argument(
        "--p21_flip_prob", type=float, default=0.25, help="被复制者若当前 P2，翻到 P1 的概率。"
    )
    g.add_argument(
        "--copy_optimizer",
        action="store_true",
        help="若设置：exploit 时复制 optimizer 状态；否则只复制网络权重（更稳但收敛慢一些）。",
    )


def _add_hp_evolve_group(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("HP Evolve (small-population PBT)")

    g.add_argument(
        "--use_worker_init_profiles",
        action="store_true",
        help="Enable small-population init diversity.",
    )
    g.add_argument(
        "--worker_lr_mults",
        type=str,
        default="1.0,0.7,1.3,0.3",
        help="Comma-separated LR multipliers for wid=0.. (applied to P1/P2 lrs).",
    )
    g.add_argument(
        "--worker_ent_mults",
        type=str,
        default="1.0,0.8,1.3,1.0",
        help="Comma-separated entropy multipliers for wid=0.. (applied to P1/P2 entropies).",
    )

    g.add_argument(
        "--hp_lr_perturb_prob",
        type=float,
        default=0.25,
        help="Prob to perturb LR after exploit (dst only).",
    )
    g.add_argument("--hp_lr_factor_down", type=float, default=0.8)
    g.add_argument("--hp_lr_factor_up", type=float, default=1.2)
    g.add_argument(
        "--hp_lr_bias_down",
        type=float,
        default=0.67,
        help="P(select down) when LR perturb triggers.",
    )
    g.add_argument("--hp_lr_min", type=float, default=1e-6)
    g.add_argument(
        "--hp_lr_max_factor", type=float, default=1.05, help="Max = lr_base * this factor."
    )
    g.add_argument(
        "--hp_lr_perturb_dqn", action="store_true", help="Also perturb DQN LR in the next mode."
    )

    g.add_argument(
        "--hp_ent_perturb_prob",
        type=float,
        default=0.15,
        help="Prob to perturb entropy after exploit.",
    )
    g.add_argument("--hp_ent_factor_down", type=float, default=0.9)
    g.add_argument("--hp_ent_factor_up", type=float, default=1.1)
    g.add_argument(
        "--hp_ent_bias_down",
        type=float,
        default=0.67,
        help="P(select down) when entropy perturb triggers.",
    )
    g.add_argument("--hp_ent_min", type=float, default=0.004)
    g.add_argument("--hp_ent_max", type=float, default=0.02)

    g.add_argument(
        "--hp_perturb_power_in_p2",
        action="store_true",
        help="If set, also perturb Power-PPO LR/entropy in P2 (only when P2 actually trains power).",
    )


def _add_stop_group(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Stop conditions (hard + soft)")
    g.add_argument(
        "--max_pbt_episodes",
        type=int,
        default=POST_WARMUP_EPISODES,
        help="硬停止：达到该 global_episode 即停止。",
    )
    g.add_argument(
        "--patience_ready",
        type=int,
        default=0,
        help="Stop after this many stagnant ready evaluations; 0 disables early stopping.",
    )
    g.add_argument("--min_delta", type=float, default=15.0, help="软停止：提升阈值。")
    g.add_argument(
        "--smooth_alpha",
        type=float,
        default=0.2,
        help="软停止：EWMA alpha（只用于早停，不用于排名）。",
    )


def _add_logging_group(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Logging / IO")
    g.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="实验名（用于 runs/pbt_exp/<exp_name>）。缺省将自动生成。",
    )
    g.add_argument("--runs_root", type=str, default="runs/pbt_exp", help="TensorBoard 根目录。")

    g.add_argument(
        "--enable_tb",
        action="store_true",
        default=False,
        help="Enable TensorBoard event logging. Default: disabled to avoid generating event files.",
    )
    g.add_argument(
        "--seed", type=int, default=42, help="总控随机种子（worker seed 将基于此偏移）。"
    )
    g.add_argument(
        "--worker_seed_stride",
        type=int,
        default=1000,
        help="不同 worker 的 seed 偏移步长：seed + wid*stride。",
    )
    g.add_argument(
        "--resume_run_dir",
        type=str,
        default=None,
        help="Resume this PBT experiment from its latest committed recovery snapshot.",
    )
    g.add_argument(
        "--resume_strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the complete recovery compatibility signature to match.",
    )


def _add_passthrough_groups(parser: argparse.ArgumentParser) -> None:
    g1 = parser.add_argument_group("Passthrough to Phase-I trainer (prefix: p1_)")
    g1.add_argument("--p1_device", type=str, default="cpu")
    g1.add_argument("--p1_max_train_episodes", type=int, default=1600)

    g1.add_argument(
        "--p1_num_ground_stations",
        type=int,
        default=None,
        help="Topology N for Phase-I trainer. If None, follow p2_num_ground_stations.",
    )
    g1.add_argument(
        "--p1_num_gs_per_rb",
        type=int,
        default=None,
        help="Topology K for Phase-I trainer. If None, follow p2_num_gs_per_rb.",
    )
    g1.add_argument(
        "--p1_topology_profile",
        type=str,
        default=None,
        help="Fixed topology profile for Phase-I. If None, follow Phase-II.",
    )

    g1.add_argument("--p1_cycle_dqn_before_power", type=int, default=6)
    g1.add_argument("--p1_cycle_power", type=int, default=1)
    g1.add_argument("--p1_dqn_batch_size", type=int, default=128)
    g1.add_argument("--p1_dqn_update_freq", type=int, default=4)
    g1.add_argument("--p1_target_update_freq", type=int, default=8)
    g1.add_argument("--p1_epsilon_decay_episodes", type=int, default=0)

    g1.add_argument("--p1_entropy_coef_power", type=float, default=0.012)

    g1.add_argument("--p1_epsilon_start", type=float, default=0.05)
    g1.add_argument("--p1_epsilon_end", type=float, default=0.05)

    g1.add_argument(
        "--p1_local_epsilon_reheat_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable local DQN epsilon reheating in Phase-I only when a real P2->P1 switch happens.",
    )
    g1.add_argument("--p1_local_epsilon_reheat_start", type=float, default=0.50)
    g1.add_argument("--p1_local_epsilon_reheat_end", type=float, default=0.12)
    g1.add_argument("--p1_local_epsilon_reheat_decay_dqn_episodes", type=int, default=64)
    g1.add_argument(
        "--p1_local_epsilon_reheat_log_tb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether Phase-I trainer writes local epsilon reheating diagnostics to TensorBoard.",
    )

    g1.add_argument(
        "--p1_scheduler_mode",
        type=str,
        default="dqn",
        choices=["dqn", "random"],
        help="Discrete scheduler source for Phase-I trainer. "
        "'dqn' keeps original greedy/frozen-DQN scheduling; "
        "'random' enables Random Scheduling baseline in Phase-I.",
    )
    g1.add_argument(
        "--p1_dqn_scheduler_impl",
        choices=["enumerative"],
        default="enumerative",
        help="Phase-I DQN architecture; must match Phase-II because PBT copies scheduler weights.",
    )

    g1.add_argument(
        "--p1_power_mode",
        type=str,
        default="ppo",
        choices=["ppo", "wmmse"],
        help="Power backend for Phase-I trainer. "
        "'ppo' keeps original Power-PPO behavior; "
        "'wmmse' uses WMMSE runtime power generation.",
    )

    g1.add_argument("--p1_wmmse_rb_index", type=int, default=0)
    g1.add_argument("--p1_wmmse_max_iters", type=int, default=50)
    g1.add_argument("--p1_wmmse_tol", type=float, default=1e-6)
    g1.add_argument("--p1_wmmse_scipy_method", type=str, default="SLSQP")
    g1.add_argument("--p1_wmmse_scipy_options", type=str, default=None)
    g1.add_argument("--p1_wmmse_sic_repair_max_iters", type=int, default=30)
    g1.add_argument("--p1_wmmse_sic_repair_step_ratio", type=float, default=0.05)

    g1.add_argument("--p1_ppo_power_opt_episodes", type=int, default=10)
    g1.add_argument("--p1_ppo_power_buffer_size", type=int, default=1024)
    g1.add_argument("--p1_ppo_power_epochs", type=int, default=12)

    g1.add_argument("--p1_lr_dqn", type=float, default=5e-4)
    g1.add_argument("--p1_lr_power_ppo", type=float, default=3e-4)

    g1.add_argument("--p1_gamma", type=float, default=0.9)
    g1.add_argument("--p1_gae_lambda", type=float, default=0.95)

    g1.add_argument("--p1_min_rate_req", type=float, default=0.10)
    g1.add_argument("--p1_csi_error_variance", type=float, default=0.01)
    g1.add_argument(
        "--p1_rho",
        type=float,
        default=None,
        help="Override rho for Phase-I. If None, follows global --rho or default.",
    )
    g1.add_argument("--p1_active_ris_noise_power_dBm", type=float, default=-100.0)

    g1.add_argument(
        "--p1_disable_ris_path",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="No-RIS baseline gate for Phase-I trainer. "
        "If None, it can follow Phase-II setting in build_worker_args; "
        "otherwise explicitly controls Phase-I physical No-RIS.",
    )

    g1.add_argument(
        "--p1_uav_rx_antenna_gain_dBi",
        type=float,
        default=None,
        help="UAV Rx antenna gain for Sat->UAV(RIS) link (dBi). If None, follow p2.",
    )
    g1.add_argument(
        "--p1_ris_elem_max_amp_gain_dB",
        type=float,
        default=None,
        help="Per-element max amplification gain upper bound (dB, POWER gain). If None, follow p2.",
    )
    g1.add_argument(
        "--p1_ris_reflection_power_limit_dBm",
        type=float,
        default=None,
        help="Active-RIS reflection power budget for Phase-I. If None, follow Phase-II.",
    )

    g1.add_argument("--p1_fairness_alpha", type=float, default=0.0)
    g1.add_argument("--p1_fairness_beta", type=float, default=0.01)
    g1.add_argument("--p1_fairness_epsilon", type=float, default=0.01)
    g1.add_argument("--p1_fairness_lambda", type=float, default=3.0)
    g1.add_argument("--p1_sic_detection_threshold_dBm", type=float, default=-94.0)
    g1.add_argument("--p1_sic_penalty", type=float, default=0.0)

    g1.add_argument(
        "--p1_include_sr_incident_pow_groups",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(PBT) Whether to append per-RIS-group incident power into Phase-I base observation. "
        "Default: None (unspecified). If None, it may follow Phase-II setting; finally defaults to False.",
    )

    g1.add_argument("--p1_train_reseed_each_episode", action="store_true")
    g1.add_argument("--p1_sat_user_gain_db_offsets", type=str, default=None)
    g1.add_argument("--p1_force_bs_gain_db", type=float, default=None)

    g1.add_argument("--p1_eval_every_episodes", type=int, default=0)
    g1.add_argument("--p1_early_stop_patience_eval", type=int, default=0)
    g1.add_argument("--p1_min_improvement", type=float, default=0.0)

    g1.add_argument(
        "--p1_phase_model_include_mrt_features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(PBT) Whether Phase-I should feed MRT features to the Phase-II (phase) model when calling it. "
        "This does NOT change DQN/Power observation. Default: None (follow Phase-II).",
    )

    g1.add_argument(
        "--p1_phase_model_state_mode",
        type=str,
        default=None,
        help="(PBT) State mode used when Phase-I calls the frozen phase model. "
        "Use 'legacy_full' or 'compact_selected'. "
        "Default: None (follow Phase-II phase_state_mode).",
    )

    g1.add_argument(
        "--p1_phase_model_compact_include_mrt_features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(PBT) Whether compact Phase-I frozen-phase state should append selected-user MRT features. "
        "Default: None (follow Phase-II compact MRT setting).",
    )

    g1.add_argument(
        "--p1_phase_model_policy_dist",
        type=str,
        default=None,
        choices=["gaussian_latent", "von_mises"],
        help="(PBT) Frozen phase-model policy distribution used by Phase-I when calling the phase model. "
        "Default: None (follow Phase-II phase_policy_dist).",
    )

    g1.add_argument(
        "--p1_mrt_casc_abs_tau", type=float, default=None, help="Default: None (follow p2)."
    )

    g2 = parser.add_argument_group("Passthrough to Phase-II trainer (prefix: p2_)")
    g2.add_argument("--p2_device", type=str, default="cpu")

    g2.add_argument(
        "--p2_num_ground_stations", type=int, default=4, help="Topology N for Phase-II trainer."
    )
    g2.add_argument(
        "--p2_num_gs_per_rb", type=int, default=2, help="Topology K for Phase-II trainer."
    )
    g2.add_argument("--p2_topology_profile", type=str, default="paper_n4")

    g2.add_argument(
        "--p2_scheduler_mode",
        type=str,
        default="dqn",
        choices=["dqn", "random"],
        help="Discrete scheduler source for Phase-II trainer. "
        "'dqn' keeps original frozen greedy-DQN scheduling; "
        "'random' enables Random Scheduling baseline in Phase-II.",
    )
    g2.add_argument(
        "--p2_dqn_scheduler_impl",
        choices=["enumerative"],
        default="enumerative",
        help="Phase-II DQN architecture; must match Phase-I because PBT copies scheduler weights.",
    )

    g2.add_argument(
        "--p2_power_mode",
        type=str,
        default="ppo",
        choices=["ppo", "wmmse"],
        help="Frozen power backend for Phase-II trainer. "
        "'ppo' keeps original frozen Power-PPO kernel; "
        "'wmmse' uses WMMSE runtime frozen power generation.",
    )

    g2.add_argument("--p2_wmmse_rb_index", type=int, default=0)
    g2.add_argument("--p2_wmmse_max_iters", type=int, default=50)
    g2.add_argument("--p2_wmmse_tol", type=float, default=1e-6)
    g2.add_argument("--p2_wmmse_scipy_method", type=str, default="SLSQP")
    g2.add_argument("--p2_wmmse_scipy_options", type=str, default=None)
    g2.add_argument("--p2_wmmse_sic_repair_max_iters", type=int, default=30)
    g2.add_argument("--p2_wmmse_sic_repair_step_ratio", type=float, default=0.05)

    g2.add_argument(
        "--p2_phase2_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run Phase-II only mode (recommended in your two-stage framework).",
    )
    g2.add_argument(
        "--p2_freeze_dqn_power",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze DQN/Power during Phase-II (recommended together with --p2_phase2_only).",
    )

    g2.add_argument("--p2_phase1_model_dir", type=str, default=None)
    g2.add_argument("--p2_phase1_ckpt", type=str, choices=["best", "final"], default=None)
    g2.add_argument(
        "--p2_phase1_ckpt_strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether strict-best loading is required for Phase-II kernel source. "
        "If None, inherit from master --warmup_ckpt_strict.",
    )

    g2.add_argument("--p2_max_train_episodes", type=int, default=1600000)
    g2.add_argument("--p2_eval_every_episodes", type=int, default=0)
    g2.add_argument("--p2_early_stop_patience_eval", type=int, default=0)
    g2.add_argument("--p2_min_improvement", type=float, default=0.0)

    g2.add_argument("--p2_cycle_dqn_before_power", type=int, default=2)
    g2.add_argument("--p2_cycle_power", type=int, default=1)
    g2.add_argument("--p2_cycle_dqn_before_phase", type=int, default=2)
    g2.add_argument("--p2_cycle_phase", type=int, default=1)
    g2.add_argument("--p2_ppo_power_opt_episodes", type=int, default=8)

    g2.add_argument("--p2_phase_delay_cycles", type=int, default=0)
    g2.add_argument("--p2_phase_delay_episodes", type=int, default=0)

    g2.add_argument("--p2_epsilon_start", type=float, default=0.05)
    g2.add_argument("--p2_epsilon_end", type=float, default=0.05)
    g2.add_argument("--p2_epsilon_decay_episodes", type=int, default=0)

    g2.add_argument("--p2_eps_floor_start", type=float, default=0.0)
    g2.add_argument("--p2_eps_floor_end", type=float, default=0.0)
    g2.add_argument("--p2_eps_floor_decay_steps", type=int, default=0)

    g2.add_argument("--p2_ppo_phase_opt_episodes", type=int, default=16)
    g2.add_argument("--p2_ppo_phase_buffer_size", type=int, default=1024)
    g2.add_argument("--p2_ppo_phase_epochs", type=int, default=6)
    g2.add_argument("--p2_lr_phase_ppo", type=float, default=6e-5)
    g2.add_argument("--p2_entropy_coef_phase", type=float, default=0.006)
    g2.add_argument("--p2_gamma", type=float, default=0.9)
    g2.add_argument("--p2_gae_lambda", type=float, default=0.95)

    g2.add_argument("--p2_lr_dqn", type=float, default=5e-4)
    g2.add_argument("--p2_lr_power_ppo", type=float, default=3e-4)
    g2.add_argument("--p2_entropy_coef_power", type=float, default=0.012)
    g2.add_argument("--p2_ppo_power_epochs", type=int, default=12)
    g2.add_argument("--p2_ppo_power_buffer_size", type=int, default=1024)

    g2.add_argument("--p2_min_rate_req", type=float, default=0.10)
    g2.add_argument("--p2_csi_error_variance", type=float, default=0.01)
    g2.add_argument(
        "--p2_rho",
        type=float,
        default=None,
        help="Override rho for Phase-II. If None, follows global --rho or default.",
    )
    g2.add_argument("--p2_active_ris_noise_power_dBm", type=float, default=-100.0)

    g2.add_argument(
        "--p2_disable_ris_path",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="No-RIS baseline gate for Phase-II trainer. "
        "If enabled, Phase-II runs under No-RIS PHY.",
    )

    g2.add_argument(
        "--p2_uav_rx_antenna_gain_dBi",
        type=float,
        default=15.0,
        help="UAV Rx antenna gain for Sat->UAV(RIS) link (dBi).",
    )
    g2.add_argument(
        "--p2_ris_elem_max_amp_gain_dB",
        type=float,
        default=40.0,
        help="Per-element max amplification gain upper bound (dB, POWER gain).",
    )
    g2.add_argument(
        "--p2_ris_reflection_power_limit_dBm",
        type=float,
        default=-5.0,
        help="Active-RIS reflection power budget for Phase-II.",
    )

    g2.add_argument("--p2_sic_detection_threshold_dBm", type=float, default=-94.0)

    g2.add_argument("--p2_fairness_alpha", type=float, default=0.0)
    g2.add_argument("--p2_fairness_beta", type=float, default=0.01)
    g2.add_argument("--p2_fairness_epsilon", type=float, default=0.01)
    g2.add_argument("--p2_fairness_lambda", type=float, default=3.0)
    g2.add_argument("--p2_lambda_boost", type=float, default=0.0)
    g2.add_argument("--p2_sic_penalty", type=float, default=0.0)

    g2.add_argument(
        "--p2_phase2_power_deterministic_mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mean for power head during Phase-II PHASE segment.",
    )

    g2.add_argument("--p2_phase_wrap_mode", type=str, default="pi", choices=["pi", "2pi"])

    g2.add_argument(
        "--p2_include_sr_incident_pow_groups",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(PBT) Whether to append per-RIS-group incident power into Phase-II base observation. "
        "Default: None (unspecified). Finally defaults to False.",
    )

    g2.add_argument(
        "--p2_include_mrt_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append MRT features in Phase-II (recommended in your recent runs).",
    )
    g2.add_argument("--p2_mrt_casc_abs_tau", type=float, default=1e-12)

    g2.add_argument(
        "--p2_phase_state_mode",
        type=str,
        default="compact_selected",
        choices=["legacy_full", "compact_selected"],
        help="Phase-II phase-state mode. "
        "'legacy_full' keeps old logic; "
        "'compact_selected' keeps only selected-user-related blocks plus power_shares.",
    )

    g2.add_argument(
        "--p2_phase_compact_include_mrt_features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether compact Phase-II phase-state should append selected-user MRT features. "
        "Requires --p2_phase_state_mode compact_selected and --p2_include_mrt_features.",
    )

    g2.add_argument(
        "--p2_phase_policy_dist",
        type=str,
        default="gaussian_latent",
        choices=["gaussian_latent", "von_mises"],
        help="Phase-II phase policy distribution. "
        "'gaussian_latent' keeps the original 2G/3G latent PPO logic; "
        "'von_mises' enables theta-only circular phase policy with 1G action dim.",
    )

    g2.add_argument(
        "--p2_learn_amp_weights", type=str, default="false", choices=["auto", "true", "false"]
    )
    g2.add_argument("--p2_force_phase2_uniform_amp", action="store_true")

    g2.add_argument("--p2_phase2_alt_enable", action="store_true")
    g2.add_argument("--p2_phase2_alt_theta_episodes", type=int, default=12)
    g2.add_argument("--p2_phase2_alt_amp_episodes", type=int, default=4)
    g2.add_argument("--p2_phase2_alt_warmup_theta_only_episodes", type=int, default=0)

    g2.add_argument("--p2_train_reseed_each_episode", action="store_true")
    g2.add_argument("--p2_sat_user_gain_db_offsets", type=str, default=None)
    g2.add_argument("--p2_force_bs_gain_db", type=float, default=None)
    g2.add_argument(
        "--p2_save_model",
        action="store_true",
        help="Passthrough: Whether Phase-II trainer should save models explicitly (usually handled by PBT).",
    )
    g2.add_argument("--p2_eval_env_seed", type=int, default=24681357)
    g2.add_argument("--p2_eval_every_steps", type=int, default=0)
    g2.add_argument("--p2_eval_tag_steps", type=str, default="StepEval")


def _resolve_learn_amp_like(raw_value, include_sr_incident_pow_groups: bool) -> bool:
    if isinstance(raw_value, bool):
        return bool(raw_value)

    v = str(raw_value).strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "auto":
        return bool(include_sr_incident_pow_groups)

    raise ValueError(f"Invalid learn_amp_weights-like value: {raw_value}")


def parse_args_and_validate(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("PBT Master Controller (Blocks 0-3)")

    parser.add_argument(
        "--ris_num_elem_x", type=int, default=16, help="RIS elements along x (columns), default 16"
    )
    parser.add_argument(
        "--ris_num_elem_y", type=int, default=16, help="RIS elements along y (rows), default 16"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.95,
        help="Global temporal correlation coefficient (rho) for both phases if not overridden.",
    )

    _add_logging_group(parser)
    _add_warmup_group(parser)
    _add_schedule_group(parser)
    _add_eval_seed_group(parser)
    _add_exploit_explore_group(parser)
    _add_stop_group(parser)
    _add_passthrough_groups(parser)
    _add_hp_evolve_group(parser)

    args = parser.parse_args(argv)

    if args.resume_run_dir:
        resume_run_dir = Path(args.resume_run_dir).resolve()
        if not resume_run_dir.is_dir():
            raise FileNotFoundError(f"resume_run_dir does not exist: {resume_run_dir}")
        args.resume_run_dir = str(resume_run_dir)
        args.runs_root = str(resume_run_dir.parent)
        args.exp_name = resume_run_dir.name

    if args.n_workers <= 0:
        raise ValueError("--n_workers must be positive.")
    if args.chunk_len <= 0:
        raise ValueError("--chunk_len must be positive.")
    if args.ready_every_episodes <= 0 or args.eval_every_episodes <= 0:
        raise ValueError("--ready_every_episodes and --eval_every_episodes must be positive.")
    if args.k_ready <= 0 or args.k_daily <= 0:
        raise ValueError("--k_ready and --k_daily must be positive.")

    args.adapt_mode = str(getattr(args, "adapt_mode", "P2")).upper().strip()
    if args.adapt_mode not in (MODE_P1, MODE_P2):
        raise ValueError(f"Invalid --adapt_mode: {args.adapt_mode}")

    if args.adapt_mode == MODE_P2:
        if int(args.adapt_episodes) <= 0:
            print(
                "[PBT Master][WARN] adapt_mode=P2 but adapt_episodes<=0. "
                "This is allowed, but then there is no dedicated P2 adapt window."
            )

        if float(args.p21_flip_prob) >= float(args.p12_flip_prob):
            print(
                "[PBT Master][WARN] adapt_mode=P2 but p21_flip_prob >= p12_flip_prob. "
                "This is allowed, but it is not P2-dominant and may weaken early phase-first training."
            )

    p1_chunk = int(args.p1_cycle_dqn_before_power) + int(args.p1_cycle_power) * int(
        args.p1_ppo_power_opt_episodes
    )
    if p1_chunk != int(args.chunk_len):
        raise ValueError(
            f"[ChunkMismatch] Phase-1 chunk={p1_chunk} != chunk_len={args.chunk_len}. "
            f"Expected: dqn({args.p1_cycle_dqn_before_power}) + power_blocks({args.p1_cycle_power})*"
            f"ppo_power_opt_episodes({args.p1_ppo_power_opt_episodes})."
        )

    if int(args.p2_ppo_phase_opt_episodes) != int(args.chunk_len):
        raise ValueError(
            f"[ChunkMismatch] Phase-2 ppo_phase_opt_episodes={args.p2_ppo_phase_opt_episodes} "
            f"must equal chunk_len={args.chunk_len} for strict 1-chunk=16 episodes."
        )

    p1_sched = str(getattr(args, "p1_scheduler_mode", "dqn")).strip().lower()
    p2_sched = str(getattr(args, "p2_scheduler_mode", "dqn")).strip().lower()
    p1_impl = str(getattr(args, "p1_dqn_scheduler_impl", "enumerative")).strip().lower()
    p2_impl = str(getattr(args, "p2_dqn_scheduler_impl", "enumerative")).strip().lower()
    if p1_impl != p2_impl:
        raise ValueError(
            "P1/P2 dqn_scheduler_impl must match because one worker checkpoint shares scheduler weights: "
            f"P1={p1_impl}, P2={p2_impl}"
        )

    p1_power_mode = str(getattr(args, "p1_power_mode", "ppo")).strip().lower()
    p2_power_mode = str(getattr(args, "p2_power_mode", "ppo")).strip().lower()

    if p1_power_mode not in ("ppo", "wmmse"):
        raise ValueError(f"Invalid --p1_power_mode: {p1_power_mode}")
    if p2_power_mode not in ("ppo", "wmmse"):
        raise ValueError(f"Invalid --p2_power_mode: {p2_power_mode}")

    if p1_power_mode != p2_power_mode:
        print(
            "[PBT Master][WARN] P1/P2 power_mode are inconsistent: "
            f"P1={p1_power_mode}, P2={p2_power_mode}. "
            "This is allowed, but it usually breaks the semantic purity of a "
            "single power-backend baseline."
        )

    if p1_sched not in ("dqn", "random"):
        raise ValueError(f"Invalid --p1_scheduler_mode: {p1_sched}")
    if p2_sched not in ("dqn", "random"):
        raise ValueError(f"Invalid --p2_scheduler_mode: {p2_sched}")

    if (p1_sched == "random") != (p2_sched == "random"):
        print(
            "[PBT Master][WARN] P1/P2 scheduler_mode are inconsistent: "
            f"P1={p1_sched}, P2={p2_sched}. "
            "This is allowed, but it usually breaks the semantic purity of the "
            "Random Scheduling baseline."
        )

    p2_phase_state_mode = str(getattr(args, "p2_phase_state_mode", "legacy_full")).strip().lower()
    if p2_phase_state_mode not in ("legacy_full", "compact_selected"):
        raise ValueError(f"Invalid --p2_phase_state_mode: {p2_phase_state_mode}")

    p1_phase_state_mode_raw = getattr(args, "p1_phase_model_state_mode", None)
    if p1_phase_state_mode_raw not in (None, ""):
        p1_phase_state_mode = str(p1_phase_state_mode_raw).strip().lower()
        if p1_phase_state_mode not in ("legacy_full", "compact_selected"):
            raise ValueError(f"Invalid --p1_phase_model_state_mode: {p1_phase_state_mode}")

    if bool(getattr(args, "p2_phase_compact_include_mrt_features", False)):
        if p2_phase_state_mode != "compact_selected":
            raise ValueError(
                "--p2_phase_compact_include_mrt_features requires --p2_phase_state_mode compact_selected."
            )
        if not bool(getattr(args, "p2_include_mrt_features", False)):
            raise ValueError(
                "--p2_phase_compact_include_mrt_features requires --p2_include_mrt_features."
            )

    p1_compact_mrt_raw = getattr(args, "p1_phase_model_compact_include_mrt_features", None)
    if p1_compact_mrt_raw is True and p1_phase_state_mode_raw not in (None, ""):
        if str(p1_phase_state_mode_raw).strip().lower() != "compact_selected":
            raise ValueError(
                "--p1_phase_model_compact_include_mrt_features requires "
                "--p1_phase_model_state_mode compact_selected."
            )

    p2_phase_policy_dist = (
        str(getattr(args, "p2_phase_policy_dist", "gaussian_latent")).strip().lower()
    )
    if p2_phase_policy_dist not in ("gaussian_latent", "von_mises"):
        raise ValueError(f"Invalid --p2_phase_policy_dist: {p2_phase_policy_dist}")

    p1_phase_policy_dist_raw = getattr(args, "p1_phase_model_policy_dist", None)
    if p1_phase_policy_dist_raw not in (None, ""):
        p1_phase_policy_dist = str(p1_phase_policy_dist_raw).strip().lower()
        if p1_phase_policy_dist not in ("gaussian_latent", "von_mises"):
            raise ValueError(f"Invalid --p1_phase_model_policy_dist: {p1_phase_policy_dist}")

    p2_learn_amp = _resolve_learn_amp_like(
        getattr(args, "p2_learn_amp_weights", "false"),
        bool(getattr(args, "p2_include_sr_incident_pow_groups", False)),
    )

    if p2_phase_policy_dist == "von_mises":
        if bool(p2_learn_amp):
            raise ValueError(
                "--p2_phase_policy_dist von_mises requires --p2_learn_amp_weights false."
            )
        if bool(getattr(args, "p2_phase2_alt_enable", False)):
            raise ValueError(
                "--p2_phase_policy_dist von_mises is not compatible with --p2_phase2_alt_enable."
            )

    if p1_phase_policy_dist_raw not in (None, ""):
        p1_learn_amp = _resolve_learn_amp_like(
            getattr(args, "p1_learn_amp_weights", getattr(args, "p2_learn_amp_weights", "false")),
            bool(getattr(args, "p1_include_sr_incident_pow_groups", False)),
        )
        if p1_phase_policy_dist == "von_mises" and bool(p1_learn_amp):
            raise ValueError(
                "--p1_phase_model_policy_dist von_mises requires Phase-I learn_amp_weights to resolve to false."
            )

    if args.resume_run_dir:
        if args.warmup_model_dir in (None, ""):
            args.warmup_model_dir = str(Path(args.resume_run_dir))
        args.warmup_kind = "recovery"
    else:
        if args.warmup_model_dir in (None, ""):
            raise ValueError("--warmup_model_dir is required for a new PBT run")
        if args.warmup_tag in (None, ""):
            raise ValueError("--warmup_tag is required for a new PBT run")
        warmup_dir = Path(args.warmup_model_dir)
        if not warmup_dir.exists():
            raise FileNotFoundError(f"warmup_model_dir does not exist: {warmup_dir}")

        candidate_dirs = [warmup_dir, warmup_dir / "models"]
        resolved_dir = None
        resolved_info = None

        for d in candidate_dirs:
            if not d.is_dir():
                continue
            try:
                info = _inspect_warmup_assets(
                    d,
                    prefer=args.warmup_ckpt,
                    strict_best=args.warmup_ckpt_strict,
                )
                resolved_dir = d
                resolved_info = info
                break
            except FileNotFoundError:
                continue

        if resolved_dir is None:
            raise FileNotFoundError(
                f"warmup_model_dir does not contain a valid old/new warmup. Tried: {candidate_dirs}\n"
                f"Required old-warmup files:\n"
                f"  - q_network_phase1(.pth/_best.pth)\n"
                f"  - power_actor_phase1(.pth/_best.pth)\n"
                f"  - power_critic_phase1(.pth/_best.pth)\n"
                f"Optional new-warmup phase pair (must appear together):\n"
                f"  - phase_actor_phase1(.pth/_best.pth)\n"
                f"  - phase_critic_phase1(.pth/_best.pth)\n"
                f"(according to warmup_ckpt={args.warmup_ckpt}, strict_best={bool(args.warmup_ckpt_strict)})."
            )

        args.warmup_model_dir = str(resolved_dir)
        args.warmup_kind = str(resolved_info["kind"])

        print(
            f"[PBT Master] resolved warmup_model_dir={args.warmup_model_dir} "
            f"(detected {args.warmup_kind} warmup)"
        )

    if args.adapt_episodes % args.ready_every_episodes != 0:
        print(
            f"[PBT Master][WARN] adapt_episodes ({args.adapt_episodes}) is not a multiple of "
            f"ready_every_episodes ({args.ready_every_episodes}). This is allowed but not recommended."
        )
    if args.cooldown_episodes < args.ready_every_episodes:
        print(
            f"[PBT Master][WARN] cooldown_episodes ({args.cooldown_episodes}) < ready_every_episodes "
            f"({args.ready_every_episodes}). This is allowed but may make ranking unstable."
        )

    if not (0.0 <= args.p12_flip_prob <= 1.0) or not (0.0 <= args.p21_flip_prob <= 1.0):
        raise ValueError("Flip probabilities must be in [0, 1].")
    if args.num_exploit < 0:
        raise ValueError("--num_exploit must be >= 0.")
    if args.num_exploit >= args.n_workers:
        raise ValueError("--num_exploit must be < n_workers.")

    if not (0.0 < args.smooth_alpha <= 1.0):
        raise ValueError("--smooth_alpha must be in (0, 1].")

    if args.exp_name is None or str(args.exp_name).strip() == "":
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"pbt_{args.warmup_tag}_{ts}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args


def _pick_ckpt_path(model_dir: Path, base_name: str, prefer: str, strict_best: bool) -> Path:
    final_path = model_dir / f"{base_name}.pth"
    best_path = model_dir / f"{base_name}_best.pth"
    if prefer == "best":
        if best_path.is_file():
            return best_path
        if strict_best:
            raise FileNotFoundError(f"Requested best ckpt but not found: {best_path}")
        return final_path
    return final_path


def _pick_optional_ckpt_path(model_dir: Path, base_name: str, prefer: str) -> Path:
    final_path = model_dir / f"{base_name}.pth"
    best_path = model_dir / f"{base_name}_best.pth"
    if prefer == "best" and best_path.is_file():
        return best_path
    return final_path


def _inspect_warmup_assets(model_dir: Path, prefer: str, strict_best: bool) -> Dict[str, Any]:
    model_dir = Path(model_dir)

    q_path = _pick_ckpt_path(model_dir, "q_network_phase1", prefer, strict_best)
    pa_path = _pick_ckpt_path(model_dir, "power_actor_phase1", prefer, strict_best)
    pc_path = _pick_ckpt_path(model_dir, "power_critic_phase1", prefer, strict_best)

    if not q_path.is_file():
        raise FileNotFoundError(f"Warmup missing DQN ckpt: {q_path}")
    if not pa_path.is_file():
        raise FileNotFoundError(f"Warmup missing power_actor ckpt: {pa_path}")
    if not pc_path.is_file():
        raise FileNotFoundError(f"Warmup missing power_critic ckpt: {pc_path}")

    pha_path = _pick_optional_ckpt_path(model_dir, "phase_actor_phase1", prefer)
    phc_path = _pick_optional_ckpt_path(model_dir, "phase_critic_phase1", prefer)

    has_phase_actor = pha_path.is_file()
    has_phase_critic = phc_path.is_file()

    if has_phase_actor != has_phase_critic:
        raise FileNotFoundError(
            "[Warmup] Incomplete new warmup detected: "
            f"phase_actor exists={has_phase_actor}, phase_critic exists={has_phase_critic}. "
            "For new warmup, phase_actor_phase1 and phase_critic_phase1 must either both exist or both be absent.\n"
            f"  - phase_actor candidate: {pha_path}\n"
            f"  - phase_critic candidate: {phc_path}"
        )

    has_phase = bool(has_phase_actor and has_phase_critic)
    kind = "new" if has_phase else "old"

    return {
        "kind": kind,
        "has_phase": has_phase,
        "paths": {
            "q_network": q_path,
            "power_actor": pa_path,
            "power_critic": pc_path,
            "phase_actor": pha_path if has_phase else None,
            "phase_critic": phc_path if has_phase else None,
        },
    }


def _warmup_has_required_ckpts(model_dir: Path, prefer: str, strict_best: bool) -> bool:
    try:
        _inspect_warmup_assets(model_dir, prefer=prefer, strict_best=strict_best)
        return True
    except FileNotFoundError:
        return False


def _load_warmup_assets(model_dir: Path, prefer: str, strict_best: bool) -> Dict[str, Any]:
    info = _inspect_warmup_assets(model_dir, prefer=prefer, strict_best=strict_best)
    p = info["paths"]

    q_state = torch.load(p["q_network"], map_location="cpu", weights_only=True)
    pa_state = torch.load(p["power_actor"], map_location="cpu", weights_only=True)
    pc_state = torch.load(p["power_critic"], map_location="cpu", weights_only=True)

    signature = read_kernel_signature(model_dir)
    out = {
        "kind": info["kind"],
        "has_phase": bool(info["has_phase"]),
        "meta": None if signature is None else {"kernel_signature": signature},
        "q_network": q_state,
        "target_q_network": q_state,
        "power_actor": pa_state,
        "power_critic": pc_state,
        "phase_actor": None,
        "phase_critic": None,
    }

    if info["has_phase"]:
        out["phase_actor"] = torch.load(p["phase_actor"], map_location="cpu", weights_only=True)
        out["phase_critic"] = torch.load(p["phase_critic"], map_location="cpu", weights_only=True)

    return out


@dataclass
class UnifiedCkpt:
    q_network: Dict[str, Any]
    target_q_network: Dict[str, Any]
    power_actor: Dict[str, Any]
    power_critic: Dict[str, Any]
    phase_actor: Dict[str, Any]
    phase_critic: Dict[str, Any]

    q_optimizer: Optional[Dict[str, Any]] = None
    power_actor_optim: Optional[Dict[str, Any]] = None
    power_critic_optim: Optional[Dict[str, Any]] = None
    phase_actor_optim: Optional[Dict[str, Any]] = None
    phase_critic_optim: Optional[Dict[str, Any]] = None

    counters: Optional[Dict[str, Any]] = None
    rng: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, Any]] = None


@dataclass
class WorkerState:
    wid: int
    mode: str
    cooldown_remaining: int
    global_episode: int

    last_daily_score: float
    last_ready_score: float

    ckpt: UnifiedCkpt

    p1_trainer: Any
    p2_trainer: Any

    writer: SummaryWriter
    log_dir: Path

    daily_curve_json_path: Path

    hp: Dict[str, float]
    hp_base: Dict[str, float]

    pending_event_exploit: int = 0
    pending_event_flip: int = 0

    pending_p1_local_reheat: int = 0


def _ws_get_ep_done(self) -> int:
    return int(self.global_episode)


def _ws_set_ep_done(self, v: int) -> None:
    self.global_episode = int(v)


WorkerState.episodes_done = property(_ws_get_ep_done, _ws_set_ep_done)


@dataclass
class ReadyRoundStats:
    ready_idx: int
    eligible_ids: List[int]
    scores_all: Dict[int, float]
    scores_eligible: Dict[int, float]

    best_id: int
    best_score: float
    mean_score: float
    std_score: float

    num_exploit: int
    num_flip: int
    num_in_cooldown: int

    exploit_pairs: List[Tuple[int, int]]


def ckpt_from_p1(p1_trainer: Any, *, include_optims: bool = True) -> UnifiedCkpt:
    if not hasattr(p1_trainer, "checkpoint_dict"):
        raise TypeError("p1_trainer must have checkpoint_dict().")
    d = p1_trainer.checkpoint_dict()

    return UnifiedCkpt(
        q_network=d["q_network"],
        target_q_network=d["target_q_network"],
        power_actor=d["power_actor"],
        power_critic=d["power_critic"],
        phase_actor=d["phase_actor"],
        phase_critic=d["phase_critic"],
        q_optimizer=d.get("q_optimizer") if include_optims else None,
        power_actor_optim=d.get("power_actor_optim") if include_optims else None,
        power_critic_optim=d.get("power_critic_optim") if include_optims else None,
        phase_actor_optim=None,
        phase_critic_optim=None,
        counters=d.get("counters", None),
        rng=d.get("rng", None),
        meta={"source": "p1_checkpoint_dict", **dict(d.get("meta", {}))},
        runtime={"source": "p1", "buffers": d.get("buffers", {})},
    )


def ckpt_from_p2(p2_trainer: Any, *, include_optims: bool = True) -> UnifiedCkpt:
    if not hasattr(p2_trainer, "state_dict"):
        raise TypeError("p2_trainer must have state_dict().")
    s = p2_trainer.state_dict()

    m = s.get("models", {})
    o = s.get("optims", {})
    c = s.get("counters", None)
    r = s.get("rng", None)
    meta = s.get("meta", None)

    return UnifiedCkpt(
        q_network=m["q_network"],
        target_q_network=m["target_q_network"],
        power_actor=m["power_actor"],
        power_critic=m["power_critic"],
        phase_actor=m["phase_actor"],
        phase_critic=m["phase_critic"],
        q_optimizer=o.get("q_optimizer") if include_optims else None,
        power_actor_optim=o.get("power_actor_optim") if include_optims else None,
        power_critic_optim=o.get("power_critic_optim") if include_optims else None,
        phase_actor_optim=o.get("phase_actor_optim") if include_optims else None,
        phase_critic_optim=o.get("phase_critic_optim") if include_optims else None,
        counters=c,
        rng=r,
        meta=meta,
        runtime={"source": "p2", "buffers": s.get("buffers", {})},
    )


def ckpt_to_dict(ckpt: UnifiedCkpt) -> Dict[str, Any]:
    return asdict(ckpt)


def dict_to_ckpt(d: Dict[str, Any]) -> UnifiedCkpt:
    return UnifiedCkpt(**d)


class SafeSummaryWriter:
    def __init__(
        self,
        log_dir: str,
        *,
        allow_fn=None,
        flush_secs: int = 300,
        max_queue: int = 1000,
        tag_step_every: dict = None,
    ):
        self._enabled = True
        self._allow_fn = allow_fn
        self._tag_step_every = tag_step_every or {}
        self._writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=int(flush_secs),
            max_queue=int(max_queue),
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _should_log(self, tag: str, global_step):
        if not self._enabled:
            return False
        if self._allow_fn is not None and (not self._allow_fn(tag)):
            return False
        if global_step is None:
            return True
        try:
            step_i = int(global_step)
        except Exception:
            return True
        for prefix, every in self._tag_step_every.items():
            if tag.startswith(prefix) and int(every) > 1:
                return (step_i % int(every)) == 0
        return True

    def add_scalar(self, tag, scalar_value, global_step=None):
        if not self._should_log(tag, global_step):
            return
        try:
            self._writer.add_scalar(tag, scalar_value, global_step)
        except OSError as e:
            print(f"[WARN] TB write failed (OSError): {e}. Disabling writer.")
            self._enabled = False
        except Exception as e:
            print(f"[WARN] TB write failed: {e}. Disabling writer.")
            self._enabled = False

    def add_histogram(self, *args, **kwargs):
        return

    def add_image(self, *args, **kwargs):
        return

    def add_text(self, *args, **kwargs):
        return

    def add_figure(self, *args, **kwargs):
        return

    def add_graph(self, *args, **kwargs):
        return

    def add_embedding(self, *args, **kwargs):
        return

    def flush(self):
        if self._enabled:
            try:
                self._writer.flush()
            except Exception:
                pass

    def close(self):
        if self._enabled:
            try:
                self._writer.close()
            except Exception:
                pass

    def __getattr__(self, item):
        return getattr(self._writer, item)


class NullSummaryWriter:
    def __init__(self, log_dir=None):
        self._enabled = False

        self.log_dir = None if log_dir is None else str(log_dir)

    @property
    def enabled(self) -> bool:
        return False

    def get_logdir(self):
        return self.log_dir

    def add_scalar(self, *args, **kwargs):
        return

    def add_histogram(self, *args, **kwargs):
        return

    def add_image(self, *args, **kwargs):
        return

    def add_text(self, *args, **kwargs):
        return

    def add_figure(self, *args, **kwargs):
        return

    def add_graph(self, *args, **kwargs):
        return

    def add_embedding(self, *args, **kwargs):
        return

    def flush(self):
        return

    def close(self):
        return

    def __getattr__(self, item):
        def _noop(*args, **kwargs):
            return

        return _noop


def _writer_is_enabled(writer) -> bool:
    if writer is None:
        return False
    try:
        return bool(getattr(writer, "enabled", True))
    except Exception:
        return True


def _make_worker_allow_fn():
    allowed_worker_prefixes = (
        "worker/mode",
        "worker/cooldown_remaining",
        "worker/episodes_done",
        "worker/EvalDailyReturn",
        "worker/EvalReadyReturn",
        "worker/event_exploit",
        "worker/event_flip",
    )

    def allow_worker(tag: str) -> bool:
        return tag.startswith(allowed_worker_prefixes)

    return allow_worker


def _make_worker_tag_step_every(tb_p2_diag_every: int) -> Dict[str, int]:
    return {}


def _build_safe_worker_writer(
    log_dir,
    *,
    tb_flush_secs: int = 300,
    tb_max_queue: int = 1000,
    tb_p2_diag_every: int = 10,
    enabled: bool = True,
):
    if not bool(enabled):
        return NullSummaryWriter(log_dir=log_dir)

    return SafeSummaryWriter(
        str(log_dir),
        allow_fn=_make_worker_allow_fn(),
        flush_secs=int(tb_flush_secs),
        max_queue=int(tb_max_queue),
        tag_step_every=_make_worker_tag_step_every(tb_p2_diag_every),
    )


def _torch_load_any(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ipc_dir_for_worker(log_dir: Path) -> Path:
    d = Path(log_dir) / "_ipc"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_ipc_snapshot_path(log_dir: Path, *, wid: int, tag: str) -> Path:
    ts = int(time.time_ns())
    pid = int(os.getpid())
    return _ipc_dir_for_worker(log_dir) / f"{tag}_w{int(wid):02d}_pid{pid}_{ts}.pt"


def _save_ipc_snapshot(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _safe_unlink(path) -> None:
    if path is None:
        return
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except Exception:
        pass


@dataclass
class LogPaths:
    run_root: Path
    global_tb_dir: Path
    worker_dirs: Dict[int, Path]

    pbt_summary_csv: Path
    early_stop_log: Path


def prepare_log_dirs(args: argparse.Namespace) -> LogPaths:
    run_root = Path(args.runs_root) / str(args.exp_name)
    run_root.mkdir(parents=True, exist_ok=True)

    global_tb_dir = run_root / "global"
    global_tb_dir.mkdir(parents=True, exist_ok=True)

    worker_dirs: Dict[int, Path] = {}
    for wid in range(int(args.n_workers)):
        wdir = run_root / f"worker_{wid:02d}"
        (wdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        worker_dirs[wid] = wdir

        _ensure_mode_timeline_csv(wdir / "mode_timeline.csv")
        _ensure_jsonl_exists(wdir / "ready_events.jsonl")

    pbt_summary_csv = run_root / "pbt_summary.csv"
    _ensure_pbt_summary_csv(pbt_summary_csv)

    early_stop_log = run_root / "early_stop.log"
    if not early_stop_log.exists():
        early_stop_log.write_text("", encoding="utf-8")

    return LogPaths(
        run_root=run_root,
        global_tb_dir=global_tb_dir,
        worker_dirs=worker_dirs,
        pbt_summary_csv=pbt_summary_csv,
        early_stop_log=early_stop_log,
    )


def make_writers(
    log_paths: LogPaths,
    n_workers: int,
    *,
    tb_flush_secs: int = 300,
    tb_max_queue: int = 1000,
    tb_p2_diag_every: int = 10,
    enable_tb: bool = False,
):
    if not bool(enable_tb):
        writer_global = NullSummaryWriter(log_dir=log_paths.global_tb_dir)
        writers_worker = {
            wid: NullSummaryWriter(log_dir=log_paths.worker_dirs[wid])
            for wid in range(int(n_workers))
        }
        return writers_worker, writer_global

    allowed_global = {
        "global/ready_round_idx",
        "global/best_eval_ready_return",
        "global/mean_eval_ready_return",
        "global/std_eval_ready_return",
        "global/num_exploit_this_ready",
        "global/num_flip_this_ready",
        "global/num_in_cooldown",
        "global/best_worker_id",
    }

    def allow_global(tag: str) -> bool:
        return tag in allowed_global

    writer_global = SafeSummaryWriter(
        str(log_paths.global_tb_dir),
        allow_fn=allow_global,
        flush_secs=int(tb_flush_secs),
        max_queue=int(tb_max_queue),
    )

    writers_worker = {}
    for wid in range(n_workers):
        writers_worker[wid] = _build_safe_worker_writer(
            log_paths.worker_dirs[wid],
            tb_flush_secs=int(tb_flush_secs),
            tb_max_queue=int(tb_max_queue),
            tb_p2_diag_every=int(tb_p2_diag_every),
        )

    return writers_worker, writer_global


def tb_write_worker_chunk(
    writer,
    global_step: int,
    *,
    mode: int,
    cooldown_remaining: int,
    episodes_done: int,
    time_per_episode: Optional[float] = None,
    event_exploit: int = 0,
    event_flip: int = 0,
):
    if not _writer_is_enabled(writer):
        return

    writer.add_scalar("worker/mode", mode, global_step)
    writer.add_scalar("worker/cooldown_remaining", cooldown_remaining, global_step)
    writer.add_scalar("worker/episodes_done", episodes_done, global_step)
    writer.add_scalar(f"worker/{TB_EVENT_EXPLOIT}", event_exploit, global_step)
    writer.add_scalar(f"worker/{TB_EVENT_FLIP}", event_flip, global_step)


def tb_write_worker_daily_eval(
    writer: SummaryWriter, *, global_episode: int, daily_return: float
) -> None:
    writer.add_scalar("worker/EvalDailyReturn", float(daily_return), global_episode)


def tb_write_worker_ready_eval(
    writer: SummaryWriter, *, global_episode: int, ready_return: float
) -> None:
    writer.add_scalar("worker/EvalReadyReturn", float(ready_return), global_episode)


def tb_write_global_ready_summary(
    writer_global,
    *,
    ready_round_idx: int,
    best_eval_ready_return: float,
    mean_eval_ready_return: float,
    std_eval_ready_return: float,
    num_exploit_this_ready: int,
    num_flip_this_ready: int,
    num_in_cooldown: int,
    best_worker_id: int,
):
    if not _writer_is_enabled(writer_global):
        return

    s = ready_round_idx
    writer_global.add_scalar("global/best_eval_ready_return", best_eval_ready_return, s)
    writer_global.add_scalar("global/mean_eval_ready_return", mean_eval_ready_return, s)
    writer_global.add_scalar("global/std_eval_ready_return", std_eval_ready_return, s)
    writer_global.add_scalar("global/num_exploit_this_ready", num_exploit_this_ready, s)
    writer_global.add_scalar("global/num_flip_this_ready", num_flip_this_ready, s)
    writer_global.add_scalar("global/num_in_cooldown", num_in_cooldown, s)
    writer_global.add_scalar("global/best_worker_id", best_worker_id, s)
    writer_global.add_scalar("global/ready_round_idx", ready_round_idx, s)


def _prune_ready_checkpoints(checkpoints_dir: Path, keep_last_k: int):
    if keep_last_k is None:
        return
    keep_last_k = int(keep_last_k)
    if keep_last_k <= 0:
        return
    ckpts = sorted(checkpoints_dir.glob("ready_*.ckpt"))
    if len(ckpts) <= keep_last_k:
        return
    to_delete = ckpts[:-keep_last_k]
    for p in to_delete:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        hp_p = p.with_suffix(".hp.json")
        try:
            hp_p.unlink(missing_ok=True)
        except Exception:
            pass


def save_ready_ckpt(
    worker_dir: Path,
    ready_idx: int,
    ckpt_dict: dict,
    hp: dict,
    *,
    keep_last_k: int = None,
    ready_score: float = None,
):
    checkpoints_dir = worker_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / f"ready_{ready_idx:04d}.ckpt"
    hp_path = checkpoints_dir / f"ready_{ready_idx:04d}.hp.json"

    torch.save(ckpt_dict, ckpt_path)
    with open(hp_path, "w", encoding="utf-8") as f:
        json.dump(hp, f, ensure_ascii=False, indent=2)

    if (ready_score is not None) and (ready_score == ready_score):
        best_meta_path = checkpoints_dir / "best_meta.json"
        prev_best = float("-inf")
        prev_best_idx = None

        if best_meta_path.exists():
            try:
                with open(best_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                prev_best = float(meta.get("best_ready_score", float("-inf")))
                prev_best_idx = meta.get("best_ready_idx", None)
            except Exception:
                prev_best = float("-inf")
                prev_best_idx = None

        if float(ready_score) > float(prev_best):
            best_ckpt_path = checkpoints_dir / "best.ckpt"
            best_hp_path = checkpoints_dir / "best.hp.json"

            shutil.copy2(ckpt_path, best_ckpt_path)
            shutil.copy2(hp_path, best_hp_path)

            new_meta = {
                "best_ready_idx": int(ready_idx),
                "best_ready_score": float(ready_score),
                "source_ready_ckpt": ckpt_path.name,
                "prev_best_ready_idx": prev_best_idx,
                "prev_best_ready_score": float(prev_best) if prev_best == prev_best else None,
                "time": float(time.time()),
            }

            tmp = best_meta_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(new_meta, f, ensure_ascii=False, indent=2)
            tmp.replace(best_meta_path)

    _prune_ready_checkpoints(checkpoints_dir, keep_last_k=keep_last_k)


def append_mode_timeline(
    csv_path: Path,
    *,
    global_episode: int,
    mode: str,
    cooldown_remaining: int,
    daily_score: float,
    ready_score: float,
) -> None:
    line = (
        f"{global_episode},{mode},{int(cooldown_remaining)},{daily_score:.6f},{ready_score:.6f}\n"
    )
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(line)


def append_eval_curve_json(json_path: Path, record: dict, *, meta: dict = None) -> None:
    json_path = Path(json_path)
    payload = {"meta": dict(meta or {}), "records": []}

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict):
                payload["meta"] = dict(old.get("meta", payload["meta"]) or payload["meta"])
                payload["records"] = list(old.get("records", []) or [])
        except Exception:
            payload = {"meta": dict(meta or {}), "records": []}

    payload["records"].append(record)

    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(json_path)


def append_ready_event_jsonl(jsonl_path: Path, event: Dict[str, Any]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def append_pbt_summary_csv(
    csv_path: Path,
    stats: ReadyRoundStats,
) -> None:
    eligible_str = ";".join(str(x) for x in stats.eligible_ids)
    line = (
        f"{stats.ready_idx},{stats.best_id},{stats.best_score:.6f},"
        f"{stats.mean_score:.6f},{stats.std_score:.6f},"
        f'{stats.num_exploit},{stats.num_flip},{stats.num_in_cooldown},"{eligible_str}"\n'
    )
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(line)


def _ensure_mode_timeline_csv(path: Path) -> None:
    if not path.exists():
        path.write_text(
            "global_episode,mode,cooldown_remaining,last_daily_score,last_ready_score\n",
            encoding="utf-8",
        )


def _ensure_pbt_summary_csv(path: Path) -> None:
    if not path.exists():
        path.write_text(
            "ready_idx,best_id,best_score,mean_score,std_score,num_exploit,num_flip,num_in_cooldown,eligible_ids\n",
            encoding="utf-8",
        )


def _ensure_jsonl_exists(path: Path) -> None:
    if not path.exists():
        path.write_text("", encoding="utf-8")


def build_worker_args(
    master_args: argparse.Namespace, wid: int
) -> Tuple[argparse.Namespace, argparse.Namespace]:
    args_p1 = copy.deepcopy(master_args)
    args_p2 = copy.deepcopy(master_args)

    seed_base = int(getattr(master_args, "seed", 42))
    stride = int(getattr(master_args, "worker_seed_stride", 1000))
    worker_seed = seed_base + wid * stride
    args_p1.seed = worker_seed
    args_p2.seed = worker_seed

    master_exp = str(getattr(master_args, "exp_name", "exp")).strip()
    wtag = f"w{wid:02d}"
    exp_worker = f"pbt_{master_exp}_{wtag}"
    args_p1.exp_name = exp_worker
    args_p2.exp_name = exp_worker

    run_root = Path(getattr(master_args, "runs_root", "runs/pbt_exp"))
    args_p1.log_dir = str(run_root / master_exp / f"worker_{wid:02d}")
    args_p2.log_dir = str(run_root / master_exp / f"worker_{wid:02d}")

    p2_num_ground_stations = int(getattr(master_args, "p2_num_ground_stations", 3))
    p2_num_gs_per_rb = int(getattr(master_args, "p2_num_gs_per_rb", 2))

    p1_num_ground_stations_raw = getattr(master_args, "p1_num_ground_stations", None)
    p1_num_gs_per_rb_raw = getattr(master_args, "p1_num_gs_per_rb", None)

    p1_num_ground_stations = (
        p2_num_ground_stations
        if p1_num_ground_stations_raw in (None, "")
        else int(p1_num_ground_stations_raw)
    )
    p1_num_gs_per_rb = (
        p2_num_gs_per_rb if p1_num_gs_per_rb_raw in (None, "") else int(p1_num_gs_per_rb_raw)
    )

    def _validate_topology(tag: str, n_users: int, k_users: int, profile: str):
        if n_users <= 0:
            raise ValueError(f"[PBT] {tag}: num_ground_stations must be > 0, got {n_users}")
        if k_users <= 0:
            raise ValueError(f"[PBT] {tag}: num_gs_per_rb must be > 0, got {k_users}")
        if k_users > n_users:
            raise ValueError(
                f"[PBT] {tag}: num_gs_per_rb={k_users} cannot exceed num_ground_stations={n_users}"
            )
        resolve_topology(profile, n_users)

    p2_profile = str(getattr(master_args, "p2_topology_profile", "paper_n4"))
    p1_profile = str(getattr(master_args, "p1_topology_profile", p2_profile) or p2_profile)
    _validate_topology("P1", p1_num_ground_stations, p1_num_gs_per_rb, p1_profile)
    _validate_topology("P2", p2_num_ground_stations, p2_num_gs_per_rb, p2_profile)

    args_p1.num_ground_stations = int(p1_num_ground_stations)
    args_p1.num_gs_per_rb = int(p1_num_gs_per_rb)

    args_p2.num_ground_stations = int(p2_num_ground_stations)
    args_p2.num_gs_per_rb = int(p2_num_gs_per_rb)

    print(
        f"[PBT Master][worker {wid:02d}] "
        f"P1 topology: N={args_p1.num_ground_stations}, K={args_p1.num_gs_per_rb}; "
        f"P2 topology: N={args_p2.num_ground_stations}, K={args_p2.num_gs_per_rb}"
    )

    if (
        args_p1.num_ground_stations != args_p2.num_ground_stations
        or args_p1.num_gs_per_rb != args_p2.num_gs_per_rb
        or p1_profile != p2_profile
    ):
        raise ValueError(
            f"[PBT] P1/P2 topology mismatch for worker {wid:02d}: "
            f"P1=(N={args_p1.num_ground_stations},K={args_p1.num_gs_per_rb},profile={p1_profile}), "
            f"P2=(N={args_p2.num_ground_stations},K={args_p2.num_gs_per_rb},profile={p2_profile}). "
            "PBT shares scheduler/power checkpoints, so P1 and P2 must use the same topology."
        )

    skip_p1_keys = {"p1_num_ground_stations", "p1_num_gs_per_rb"}
    for k, v in vars(master_args).items():
        if k.startswith("p1_") and k not in skip_p1_keys:
            setattr(args_p1, k[len("p1_") :], v)

    skip_p2_keys = {"p2_num_ground_stations", "p2_num_gs_per_rb"}
    for k, v in vars(master_args).items():
        if k.startswith("p2_") and k not in skip_p2_keys:
            setattr(args_p2, k[len("p2_") :], v)

    p2_disable_ris = getattr(args_p2, "disable_ris_path", None)
    if p2_disable_ris is None:
        args_p2.disable_ris_path = False
    else:
        args_p2.disable_ris_path = bool(p2_disable_ris)

    p1_disable_ris = getattr(args_p1, "disable_ris_path", None)
    if p1_disable_ris is None:
        args_p1.disable_ris_path = bool(args_p2.disable_ris_path)
    else:
        args_p1.disable_ris_path = bool(p1_disable_ris)

    args_p1.scheduler_mode = str(getattr(args_p1, "scheduler_mode", "dqn")).strip().lower()
    if args_p1.scheduler_mode not in ("dqn", "random"):
        args_p1.scheduler_mode = "dqn"

    args_p2.scheduler_mode = str(getattr(args_p2, "scheduler_mode", "dqn")).strip().lower()
    if args_p2.scheduler_mode not in ("dqn", "random"):
        args_p2.scheduler_mode = "dqn"

    global_rho = float(getattr(master_args, "rho", 0.95))

    p1_rho_override = getattr(args_p1, "rho", None)
    if p1_rho_override is None:
        args_p1.rho = global_rho
    else:
        args_p1.rho = float(p1_rho_override)

    p2_rho_override = getattr(args_p2, "rho", None)
    if p2_rho_override is None:
        args_p2.rho = global_rho
    else:
        args_p2.rho = float(p2_rho_override)

    def _parse_mults(s: str) -> List[float]:
        arr = []
        for x in str(s).split(","):
            x = x.strip()
            if not x:
                continue
            arr.append(float(x))
        return arr

    if bool(getattr(master_args, "use_worker_init_profiles", False)):
        lr_mults = _parse_mults(getattr(master_args, "worker_lr_mults", ""))
        ent_mults = _parse_mults(getattr(master_args, "worker_ent_mults", ""))

        lr_m = lr_mults[wid] if wid < len(lr_mults) else 1.0
        ent_m = ent_mults[wid] if wid < len(ent_mults) else 1.0

        for f in ["lr_dqn", "lr_power_ppo"]:
            if hasattr(args_p1, f):
                setattr(args_p1, f, float(getattr(args_p1, f)) * lr_m)
        for f in ["lr_dqn", "lr_power_ppo", "lr_phase_ppo"]:
            if hasattr(args_p2, f):
                setattr(args_p2, f, float(getattr(args_p2, f)) * lr_m)

        if hasattr(args_p1, "entropy_coef_power"):
            args_p1.entropy_coef_power = float(args_p1.entropy_coef_power) * ent_m
        if hasattr(args_p2, "entropy_coef_phase"):
            args_p2.entropy_coef_phase = float(args_p2.entropy_coef_phase) * ent_m
        if hasattr(args_p2, "entropy_coef_power"):
            args_p2.entropy_coef_power = float(args_p2.entropy_coef_power) * ent_m

    shared_phase_fields = [
        "include_sr_incident_pow_groups",
        "learn_amp_weights",
        "phase_wrap_mode",
        "uav_rx_antenna_gain_dBi",
        "ris_elem_max_amp_gain_dB",
        "ris_reflection_power_limit_dBm",
        "topology_profile",
    ]

    for f in shared_phase_fields:
        if (not hasattr(args_p1, f)) or (getattr(args_p1, f) is None):
            if hasattr(args_p2, f):
                setattr(args_p1, f, getattr(args_p2, f))

    if hasattr(args_p1, "power_mode"):
        args_p1.power_mode = str(getattr(args_p1, "power_mode", "ppo")).strip().lower()
        if args_p1.power_mode not in ("ppo", "wmmse"):
            args_p1.power_mode = "ppo"

    if hasattr(args_p2, "power_mode"):
        args_p2.power_mode = str(getattr(args_p2, "power_mode", "ppo")).strip().lower()
        if args_p2.power_mode not in ("ppo", "wmmse"):
            args_p2.power_mode = "ppo"

    shared_wmmse_fields = [
        "wmmse_rb_index",
        "wmmse_max_iters",
        "wmmse_tol",
        "wmmse_scipy_method",
        "wmmse_scipy_options",
        "wmmse_sic_repair_max_iters",
        "wmmse_sic_repair_step_ratio",
    ]

    for f in shared_wmmse_fields:
        if (not hasattr(args_p1, f)) or (getattr(args_p1, f) is None):
            if hasattr(args_p2, f):
                setattr(args_p1, f, getattr(args_p2, f))

    if getattr(args_p2, "include_sr_incident_pow_groups", None) is None:
        args_p2.include_sr_incident_pow_groups = False
    if getattr(args_p1, "include_sr_incident_pow_groups", None) is None:
        args_p1.include_sr_incident_pow_groups = False

    if bool(args_p1.include_sr_incident_pow_groups) != bool(args_p2.include_sr_incident_pow_groups):
        raise ValueError(
            "[PBT] include_sr_incident_pow_groups must be consistent between Phase-I and Phase-II.\n"
            f"  - Phase-I include_sr_incident_pow_groups = {args_p1.include_sr_incident_pow_groups}\n"
            f"  - Phase-II include_sr_incident_pow_groups = {args_p2.include_sr_incident_pow_groups}\n"
            "Fix: set only Phase-II flag (recommended) and keep Phase-I as None so it follows, e.g.\n"
            "  --p2_include_sr_incident_pow_groups   (or --no-p2_include_sr_incident_pow_groups)\n"
            "Or explicitly set both."
        )

    args_p1.include_mrt_features = False

    args_p2.include_mrt_features = bool(getattr(args_p2, "include_mrt_features", False))

    p1_phase_mrt = getattr(args_p1, "phase_model_include_mrt_features", None)
    if p1_phase_mrt is None:
        args_p1.phase_model_include_mrt_features = bool(args_p2.include_mrt_features)
    else:
        args_p1.phase_model_include_mrt_features = bool(p1_phase_mrt)

    if getattr(args_p1, "mrt_casc_abs_tau", None) is None:
        args_p1.mrt_casc_abs_tau = float(getattr(args_p2, "mrt_casc_abs_tau", 1e-12))
    else:
        args_p1.mrt_casc_abs_tau = float(args_p1.mrt_casc_abs_tau)

    args_p2.mrt_casc_abs_tau = float(getattr(args_p2, "mrt_casc_abs_tau", 1e-12))

    args_p2.phase_state_mode = (
        str(getattr(args_p2, "phase_state_mode", "legacy_full")).strip().lower()
    )
    if args_p2.phase_state_mode not in ("legacy_full", "compact_selected"):
        raise ValueError(f"[PBT] Invalid Phase-II phase_state_mode: {args_p2.phase_state_mode}")

    args_p2.phase_compact_include_mrt_features = bool(
        getattr(args_p2, "phase_compact_include_mrt_features", False)
    )

    if args_p2.phase_compact_include_mrt_features:
        if args_p2.phase_state_mode != "compact_selected":
            raise ValueError(
                "[PBT] Phase-II compact MRT requires phase_state_mode='compact_selected'."
            )
        if not bool(args_p2.include_mrt_features):
            raise ValueError("[PBT] Phase-II compact MRT requires include_mrt_features=True.")

    p1_phase_state_mode_raw = getattr(args_p1, "phase_model_state_mode", None)
    if p1_phase_state_mode_raw in (None, ""):
        args_p1.phase_model_state_mode = str(args_p2.phase_state_mode)
    else:
        args_p1.phase_model_state_mode = str(p1_phase_state_mode_raw).strip().lower()

    if args_p1.phase_model_state_mode not in ("legacy_full", "compact_selected"):
        raise ValueError(
            f"[PBT] Invalid Phase-I phase_model_state_mode: {args_p1.phase_model_state_mode}"
        )

    p1_compact_mrt_raw = getattr(args_p1, "phase_model_compact_include_mrt_features", None)
    if p1_compact_mrt_raw is None:
        args_p1.phase_model_compact_include_mrt_features = bool(
            getattr(args_p2, "phase_compact_include_mrt_features", False)
        )
    else:
        args_p1.phase_model_compact_include_mrt_features = bool(p1_compact_mrt_raw)

    if args_p1.phase_model_compact_include_mrt_features:
        if args_p1.phase_model_state_mode != "compact_selected":
            raise ValueError(
                "[PBT] Phase-I compact MRT requires phase_model_state_mode='compact_selected'."
            )
        if not bool(args_p1.phase_model_include_mrt_features):
            raise ValueError(
                "[PBT] Phase-I compact MRT requires phase_model_include_mrt_features=True."
            )

    args_p2.phase_policy_dist = (
        str(getattr(args_p2, "phase_policy_dist", "gaussian_latent")).strip().lower()
    )
    if args_p2.phase_policy_dist not in ("gaussian_latent", "von_mises"):
        raise ValueError(f"[PBT] Invalid Phase-II phase_policy_dist: {args_p2.phase_policy_dist}")

    p1_phase_policy_raw = getattr(args_p1, "phase_model_policy_dist", None)
    if p1_phase_policy_raw in (None, ""):
        args_p1.phase_model_policy_dist = str(args_p2.phase_policy_dist)
    else:
        args_p1.phase_model_policy_dist = str(p1_phase_policy_raw).strip().lower()

    if args_p1.phase_model_policy_dist not in ("gaussian_latent", "von_mises"):
        raise ValueError(
            f"[PBT] Invalid Phase-I phase_model_policy_dist: {args_p1.phase_model_policy_dist}"
        )

    p2_learn_amp = _resolve_learn_amp_like(
        getattr(args_p2, "learn_amp_weights", "false"),
        bool(getattr(args_p2, "include_sr_incident_pow_groups", False)),
    )
    p1_learn_amp = _resolve_learn_amp_like(
        getattr(args_p1, "learn_amp_weights", getattr(args_p2, "learn_amp_weights", "false")),
        bool(getattr(args_p1, "include_sr_incident_pow_groups", False)),
    )

    if args_p2.phase_policy_dist == "von_mises":
        if bool(p2_learn_amp):
            raise ValueError(
                "[PBT] Phase-II phase_policy_dist='von_mises' requires learn_amp_weights=False."
            )
        if bool(getattr(args_p2, "phase2_alt_enable", False)):
            raise ValueError(
                "[PBT] Phase-II phase_policy_dist='von_mises' is not compatible with phase2_alt_enable=True."
            )

    if args_p1.phase_model_policy_dist == "von_mises":
        if bool(p1_learn_amp):
            raise ValueError(
                "[PBT] Phase-I frozen phase_model_policy_dist='von_mises' requires learn_amp_weights=False."
            )

    if bool(getattr(args_p2, "phase2_only", False)):
        if getattr(args_p2, "phase1_model_dir", None) in (None, ""):
            args_p2.phase1_model_dir = str(Path(master_args.warmup_model_dir))
        else:
            args_p2.phase1_model_dir = str(Path(args_p2.phase1_model_dir))

        if getattr(args_p2, "phase1_ckpt", None) in (None, ""):
            args_p2.phase1_ckpt = str(getattr(master_args, "warmup_ckpt", "best"))
        else:
            args_p2.phase1_ckpt = str(args_p2.phase1_ckpt)

        if getattr(args_p2, "phase1_ckpt_strict", None) is None:
            args_p2.phase1_ckpt_strict = bool(getattr(master_args, "warmup_ckpt_strict", False))
        else:
            args_p2.phase1_ckpt_strict = bool(args_p2.phase1_ckpt_strict)

    if hasattr(args_p1, "phase1_phase_source"):
        args_p1.phase1_phase_source = "phase_mean"
    if hasattr(args_p1, "phase1_phase_pass_every_step"):
        args_p1.phase1_phase_pass_every_step = True
    if hasattr(args_p1, "phase1_fix_phase_per_episode"):
        args_p1.phase1_fix_phase_per_episode = False
    if hasattr(args_p1, "phase1_randomize_episode_phase"):
        args_p1.phase1_randomize_episode_phase = False

    if hasattr(args_p2, "force_phase2_uniform_amp"):
        setattr(
            args_p1, "force_phase2_uniform_amp", bool(getattr(args_p2, "force_phase2_uniform_amp"))
        )
    else:
        setattr(args_p1, "force_phase2_uniform_amp", False)

    args_p1.include_mrt_features = False

    p1_phase_mrt = getattr(args_p1, "phase_model_include_mrt_features", None)
    if p1_phase_mrt is None:
        args_p1.phase_model_include_mrt_features = bool(
            getattr(args_p2, "include_mrt_features", False)
        )
    else:
        args_p1.phase_model_include_mrt_features = bool(p1_phase_mrt)

    p2_tau = getattr(args_p2, "mrt_casc_abs_tau", None)
    if p2_tau is not None:
        args_p1.mrt_casc_abs_tau = float(p2_tau)

    args_p1.eval_every_episodes = 0
    args_p1.early_stop_patience_eval = 0

    args_p2.eval_every_episodes = 0
    args_p2.early_stop_patience_eval = 0

    return args_p1, args_p2


def _resolve_kernel_source(
    master_args: argparse.Namespace,
    args_p2: argparse.Namespace,
) -> Tuple[Path, str, bool]:
    kernel_dir = getattr(args_p2, "phase1_model_dir", None)
    if kernel_dir in (None, ""):
        kernel_dir = getattr(master_args, "warmup_model_dir")
    kernel_dir = Path(kernel_dir)

    kernel_ckpt = getattr(args_p2, "phase1_ckpt", None)
    if kernel_ckpt in (None, ""):
        kernel_ckpt = str(getattr(master_args, "warmup_ckpt", "best"))
    else:
        kernel_ckpt = str(kernel_ckpt)

    kernel_strict_best = getattr(args_p2, "phase1_ckpt_strict", None)
    if kernel_strict_best is None:
        kernel_strict_best = bool(getattr(master_args, "warmup_ckpt_strict", False))
    else:
        kernel_strict_best = bool(kernel_strict_best)

    return kernel_dir, kernel_ckpt, kernel_strict_best


def _load_warmup_kernel_only(
    model_dir: Path, prefer: str, strict_best: bool
) -> Dict[str, Dict[str, Any]]:
    model_dir = Path(model_dir)

    def _pick(base: str) -> Path:
        final_p = model_dir / f"{base}.pth"
        best_p = model_dir / f"{base}_best.pth"
        if prefer == "best":
            if best_p.is_file():
                return best_p
            if strict_best:
                raise FileNotFoundError(f"Requested best ckpt but not found: {best_p}")
            return final_p
        return final_p

    q_path = _pick("q_network_phase1")
    pa_path = _pick("power_actor_phase1")
    pc_path = _pick("power_critic_phase1")

    if not q_path.is_file():
        raise FileNotFoundError(f"Warmup missing DQN ckpt: {q_path}")
    if not pa_path.is_file():
        raise FileNotFoundError(f"Warmup missing power_actor ckpt: {pa_path}")
    if not pc_path.is_file():
        raise FileNotFoundError(f"Warmup missing power_critic ckpt: {pc_path}")

    q_state = torch.load(q_path, map_location="cpu", weights_only=True)
    pa_state = torch.load(pa_path, map_location="cpu", weights_only=True)
    pc_state = torch.load(pc_path, map_location="cpu", weights_only=True)

    return {
        "q_network": q_state,
        "target_q_network": q_state,
        "power_actor": pa_state,
        "power_critic": pc_state,
    }


def init_worker(
    master_args: argparse.Namespace,
    wid: int,
    *,
    log_dir: Path,
    writer: SummaryWriter,
    skip_warmup_bootstrap: bool = False,
    quiet_bootstrap: bool = False,
) -> WorkerState:
    args_p1, args_p2 = build_worker_args(master_args, wid)

    setattr(args_p2, "skip_phase1_kernel_load", bool(skip_warmup_bootstrap))

    if not quiet_bootstrap:
        print(
            f"[PBT Master][worker {wid:02d}] "
            f"P1 scheduler_mode={getattr(args_p1, 'scheduler_mode', 'dqn')}, "
            f"P1 power_mode={getattr(args_p1, 'power_mode', 'ppo')}, "
            f"P2 scheduler_mode={getattr(args_p2, 'scheduler_mode', 'dqn')}, "
            f"P2 power_mode={getattr(args_p2, 'power_mode', 'ppo')}"
        )

    p1_trainer = Phase1TrainerPBT(args_p1, log_writer=writer)
    p2_trainer = Phase2Trainer(args_p2, log_dir=log_dir, writer=writer)

    if not bool(skip_warmup_bootstrap):
        kernel_dir, kernel_ckpt, kernel_strict_best = _resolve_kernel_source(master_args, args_p2)

        warmup_sd = _load_warmup_assets(
            kernel_dir,
            prefer=kernel_ckpt,
            strict_best=kernel_strict_best,
        )

        if not quiet_bootstrap:
            print(
                f"[PBT Master][worker {wid:02d}] "
                f"loading {warmup_sd['kind']} warmup from {kernel_dir}"
            )

        validate_checkpoint_metadata(
            warmup_sd.get("meta"),
            p1_trainer.kernel_signature,
            context=f"Warmup model directory {kernel_dir}",
            dimension_fields=KERNEL_DIMENSION_FIELDS,
            warn_legacy=False,
        )
        if warmup_sd.get("meta") is None:
            warmup_sd["meta"] = {"_legacy_warning_emitted": True}

        p1_trainer.q_network.load_state_dict(warmup_sd["q_network"], strict=True)
        p1_trainer.target_q_network.load_state_dict(warmup_sd["target_q_network"], strict=True)
        p1_trainer.power_actor.load_state_dict(warmup_sd["power_actor"], strict=True)
        p1_trainer.power_critic.load_state_dict(warmup_sd["power_critic"], strict=True)

        p2_trainer.load_kernel_state(
            {
                "models": {
                    "q_network": warmup_sd["q_network"],
                    "target_q_network": warmup_sd["target_q_network"],
                    "power_actor": warmup_sd["power_actor"],
                    "power_critic": warmup_sd["power_critic"],
                },
                "meta": warmup_sd.get("meta"),
            },
            strict=True,
            load_optim=False,
        )

        if bool(warmup_sd["has_phase"]):
            if hasattr(p1_trainer, "attach_phase_models"):
                p1_trainer.attach_phase_models(
                    warmup_sd["phase_actor"],
                    warmup_sd["phase_critic"],
                    strict=True,
                )
            else:
                p1_trainer.phase_actor.load_state_dict(warmup_sd["phase_actor"], strict=True)
                p1_trainer.phase_critic.load_state_dict(warmup_sd["phase_critic"], strict=True)
                p1_trainer.phase_actor.eval()
                p1_trainer.phase_critic.eval()

            p2_trainer.phase_actor.load_state_dict(warmup_sd["phase_actor"], strict=True)
            p2_trainer.phase_critic.load_state_dict(warmup_sd["phase_critic"], strict=True)

    p1_trainer.reset_buffers(clear_replay=True, clear_power=True, reset_scheduler=True)
    p2_trainer.reset_buffers(clear_replay=True, clear_power=True, clear_phase=True)

    ckpt = ckpt_from_p2(
        p2_trainer,
        include_optims=True,
    )

    hp = {
        "p1_lr_dqn": float(getattr(p1_trainer.args, "lr_dqn", 0.0)),
        "p1_lr_power_ppo": float(getattr(p1_trainer.args, "lr_power_ppo", 0.0)),
        "p1_entropy_coef_power": float(getattr(p1_trainer.args, "entropy_coef_power", 0.0)),
        "p2_lr_dqn": float(getattr(p2_trainer.args, "lr_dqn", 0.0)),
        "p2_lr_power_ppo": float(getattr(p2_trainer.args, "lr_power_ppo", 0.0)),
        "p2_entropy_coef_power": float(getattr(p2_trainer.args, "entropy_coef_power", 0.0)),
        "p2_lr_phase_ppo": float(getattr(p2_trainer.args, "lr_phase_ppo", 0.0)),
        "p2_entropy_coef_phase": float(getattr(p2_trainer.args, "entropy_coef_phase", 0.0)),
    }
    hp_base = dict(hp)

    init_mode = str(getattr(master_args, "adapt_mode", "P2")).upper().strip()
    if init_mode not in (MODE_P1, MODE_P2):
        init_mode = MODE_P2

    worker = WorkerState(
        wid=wid,
        mode=init_mode,
        cooldown_remaining=0,
        global_episode=0,
        last_daily_score=float("nan"),
        last_ready_score=float("nan"),
        ckpt=ckpt,
        p1_trainer=p1_trainer,
        p2_trainer=p2_trainer,
        writer=writer,
        log_dir=Path(log_dir),
        daily_curve_json_path=Path(log_dir) / "curve_daily.json",
        pending_event_exploit=0,
        pending_event_flip=0,
        hp=hp,
        hp_base=hp_base,
    )
    return worker


def _unified_to_p1_checkpoint_dict(ckpt: UnifiedCkpt) -> Dict[str, Any]:
    return {
        "q_network": ckpt.q_network,
        "target_q_network": ckpt.target_q_network,
        "power_actor": ckpt.power_actor,
        "power_critic": ckpt.power_critic,
        "phase_actor": ckpt.phase_actor,
        "phase_critic": ckpt.phase_critic,
        "q_optimizer": ckpt.q_optimizer,
        "power_actor_optim": ckpt.power_actor_optim,
        "power_critic_optim": ckpt.power_critic_optim,
        "counters": ckpt.counters,
        "rng": ckpt.rng,
        "buffers": (
            dict(ckpt.runtime or {}).get("buffers", {})
            if dict(ckpt.runtime or {}).get("source") == "p1"
            else {}
        ),
        "meta": dict(ckpt.meta or {}),
    }


def _unified_to_p2_kernel_state(
    ckpt: UnifiedCkpt, *, include_optims: bool = True
) -> Dict[str, Any]:
    s = {
        "meta": dict(ckpt.meta or {}),
        "models": {
            "q_network": ckpt.q_network,
            "target_q_network": ckpt.target_q_network,
            "power_actor": ckpt.power_actor,
            "power_critic": ckpt.power_critic,
        },
    }
    if include_optims and (ckpt.q_optimizer is not None):
        s["optims"] = {
            "q_optimizer": ckpt.q_optimizer,
            "power_actor_optim": ckpt.power_actor_optim,
            "power_critic_optim": ckpt.power_critic_optim,
        }
    return s


def _filter_state_dict_by_shape(module, incoming_sd: dict):
    if incoming_sd is None:
        return None, []
    cur_sd = module.state_dict()
    filtered = {}
    skipped = []
    for k, v in incoming_sd.items():
        if (
            (k in cur_sd)
            and hasattr(v, "shape")
            and hasattr(cur_sd[k], "shape")
            and (tuple(v.shape) == tuple(cur_sd[k].shape))
        ):
            filtered[k] = v
        else:
            skipped.append(k)
    return filtered, skipped


from typing import Optional


def apply_ckpt_to_p1(
    worker: WorkerState, *, load_optim: Optional[bool] = None, strict: bool = True
) -> None:
    p1 = worker.p1_trainer
    ckpt = worker.ckpt

    if load_optim is None:
        load_optim_final = bool(getattr(p1.args, "copy_optimizer", True))
    else:
        load_optim_final = bool(load_optim)

    d = _unified_to_p1_checkpoint_dict(ckpt)

    can_load_optims = (
        load_optim_final
        and (ckpt.q_optimizer is not None)
        and (ckpt.power_actor_optim is not None)
        and (ckpt.power_critic_optim is not None)
    )

    if hasattr(p1, "load_checkpoint_dict") and can_load_optims:
        p1.load_checkpoint_dict(d, strict=strict)
    else:
        validate_checkpoint_metadata(
            ckpt.meta,
            p1.kernel_signature,
            context="PBT unified checkpoint for Phase-I",
        )
        p1.q_network.load_state_dict(ckpt.q_network, strict=strict)
        p1.target_q_network.load_state_dict(ckpt.target_q_network, strict=strict)
        p1.power_actor.load_state_dict(ckpt.power_actor, strict=strict)
        p1.power_critic.load_state_dict(ckpt.power_critic, strict=strict)

        c = ckpt.counters or {}
        if hasattr(p1, "total_steps") and ("total_steps" in c):
            p1.total_steps = int(c["total_steps"])
        if hasattr(p1, "episodes_done") and ("episodes_done" in c):
            p1.episodes_done = int(c["episodes_done"])
        if hasattr(p1, "current_epsilon") and ("current_epsilon" in c):
            p1.current_epsilon = float(c["current_epsilon"])
        if hasattr(p1, "segment_idx") and ("segment_idx" in c):
            p1.segment_idx = int(c["segment_idx"])
        if hasattr(p1, "segment_ep_remaining") and ("segment_ep_remaining" in c):
            p1.segment_ep_remaining = int(c["segment_ep_remaining"])
        if hasattr(p1, "power_episode_offset"):
            p1.power_episode_offset = int(c.get("power_episode_offset", 0))
        if hasattr(p1, "local_epsilon_reheat_active") and ("local_reheat_active" in c):
            p1.local_epsilon_reheat_active = bool(c["local_reheat_active"])
        if hasattr(p1, "local_epsilon_reheat_progress_dqn") and ("local_reheat_progress_dqn" in c):
            p1.local_epsilon_reheat_progress_dqn = int(c["local_reheat_progress_dqn"])
        if hasattr(p1, "local_epsilon_reheat_last_trigger") and ("local_reheat_last_trigger" in c):
            p1.local_epsilon_reheat_last_trigger = c["local_reheat_last_trigger"]

    if hasattr(p1, "set_phase_source"):
        p1.set_phase_source("phase_mean")

    if hasattr(p1, "attach_phase_models"):
        pa_sd, _ = _filter_state_dict_by_shape(p1.phase_actor, ckpt.phase_actor)
        pc_sd, _ = _filter_state_dict_by_shape(p1.phase_critic, ckpt.phase_critic)
        p1.attach_phase_models(pa_sd, pc_sd, strict=False)


from typing import Optional


def apply_ckpt_to_p2(
    worker: WorkerState, *, load_optim: Optional[bool] = None, strict: bool = True
):
    assert worker.p2_trainer is not None
    assert worker.ckpt is not None

    if load_optim is None:
        load_optim_final = bool(getattr(worker.p2_trainer.args, "copy_optimizer", True))
    else:
        load_optim_final = bool(load_optim)

    kernel = _unified_to_p2_kernel_state(worker.ckpt, include_optims=load_optim_final)

    if load_optim_final:
        optims = kernel.get("optims", None)
        if not optims:
            load_optim_final = False

    worker.p2_trainer.load_kernel_state(kernel, strict=strict, load_optim=load_optim_final)

    worker.p2_trainer.phase_actor.load_state_dict(worker.ckpt.phase_actor, strict=strict)
    worker.p2_trainer.phase_critic.load_state_dict(worker.ckpt.phase_critic, strict=strict)
    c = worker.ckpt.counters or {}
    p2 = worker.p2_trainer

    def _get_int(*keys, default=None):
        for k in keys:
            if k in c and c[k] is not None:
                return int(c[k])
        return default

    def _get_float(*keys, default=None):
        for k in keys:
            if k in c and c[k] is not None:
                return float(c[k])
        return default

    if hasattr(p2, "total_steps"):
        v = _get_int("total_steps", default=getattr(p2, "total_steps", 0))
        if v is not None:
            p2.total_steps = v

    if hasattr(p2, "episodes_done"):
        v = _get_int("episodes_done", "episode_count", default=getattr(p2, "episodes_done", 0))
        if v is not None:
            p2.episodes_done = v
    if hasattr(p2, "episode_count"):
        v = _get_int("episode_count", "episodes_done", default=getattr(p2, "episode_count", 0))
        if v is not None:
            p2.episode_count = v

    if hasattr(p2, "current_epsilon"):
        v = _get_float("current_epsilon", default=getattr(p2, "current_epsilon", 0.0))
        if v is not None:
            p2.current_epsilon = v

    if hasattr(p2, "set_mode"):
        p2.set_mode("P2", clear_buffers_on_switch=False)
    else:
        p2.kernel_frozen = True
        p2.segment_order = [("PHASE", int(getattr(p2.args, "cycle_phase", 1)))]
        p2.segment_idx = 0
        p2.segment_ep_remaining = int(p2.segment_order[0][1])

    if (
        load_optim_final
        and (worker.ckpt.phase_actor_optim is not None)
        and (worker.ckpt.phase_critic_optim is not None)
    ):
        worker.p2_trainer.phase_actor_optim.load_state_dict(worker.ckpt.phase_actor_optim)
        worker.p2_trainer.phase_critic_optim.load_state_dict(worker.ckpt.phase_critic_optim)

    worker.p2_trainer.args.freeze_dqn_power = True
    worker.p2_trainer.args.phase2_only = True
    runtime = dict(worker.ckpt.runtime or {})
    if runtime.get("source") == "p2":
        buffers = dict(runtime.get("buffers", {}))
        replay_state = buffers.get("replay")
        if replay_state is None:
            p2.replay_buffer.clear()
        else:
            load_replay_buffer_state(p2.replay_buffer, replay_state)
        for name, runtime_buffers in (("power", p2.power_buffers), ("phase", p2.phase_buffers)):
            for buffer in runtime_buffers.values():
                buffer.clear()
            for aid, buffer_state in buffers.get(name, {}).items():
                if aid not in runtime_buffers:
                    raise ValueError(
                        f"Unknown {name} trajectory buffer in unified checkpoint: {aid}"
                    )
                load_trajectory_buffer_state(runtime_buffers[aid], buffer_state)
    p2._last_rng_state = worker.ckpt.rng
    restore_rng_state(worker.ckpt.rng, env=p2.env)


def writeback_unified_ckpt(worker: WorkerState) -> None:
    if worker.mode == MODE_P1:
        worker.ckpt = ckpt_from_p1(worker.p1_trainer, include_optims=True)
    elif worker.mode == MODE_P2:
        worker.ckpt = ckpt_from_p2(worker.p2_trainer, include_optims=True)
    else:
        raise ValueError(f"Unknown worker.mode: {worker.mode}")


def reset_worker_buffers_all(
    worker: WorkerState,
    *,
    clear_replay: bool = True,
    clear_power: bool = True,
    clear_phase: bool = True,
    reset_scheduler: bool = True,
) -> None:
    if worker.p1_trainer is not None and hasattr(worker.p1_trainer, "reset_buffers"):
        try:
            worker.p1_trainer.reset_buffers(
                clear_replay=clear_replay,
                clear_power=clear_power,
                reset_scheduler=reset_scheduler,
            )
        except TypeError:
            worker.p1_trainer.reset_buffers()

    if worker.p2_trainer is not None and hasattr(worker.p2_trainer, "reset_buffers"):
        try:
            worker.p2_trainer.reset_buffers(
                clear_replay=clear_replay,
                clear_power=clear_power,
                clear_phase=clear_phase,
            )
        except TypeError:
            worker.p2_trainer.reset_buffers()


def clean_buffers_after_event(
    worker: WorkerState,
    *,
    reason: str,
    prev_mode: Optional[str] = None,
    new_mode: Optional[str] = None,
) -> None:
    _ = (reason, prev_mode, new_mode)
    reset_worker_buffers_all(
        worker,
        clear_replay=True,
        clear_power=True,
        clear_phase=True,
        reset_scheduler=True,
    )


def train_chunk(
    worker: WorkerState,
    chunk_len: int,
    *,
    allow_early_stop_p2: bool = False,
    strict_sync: bool = True,
) -> Dict[str, Any]:
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be positive, got {chunk_len}")

    t0 = time.time()

    if worker.mode == MODE_P1:
        apply_ckpt_to_p1(worker, load_optim=None, strict=strict_sync)

        if int(getattr(worker, "pending_p1_local_reheat", 0)) != 0:
            if hasattr(worker.p1_trainer, "on_mode_transition"):
                worker.p1_trainer.on_mode_transition(prev_mode=MODE_P2, new_mode=MODE_P1)
            worker.pending_p1_local_reheat = 0

        worker.p1_trainer.train_for(
            int(chunk_len),
            allow_internal_eval=False,
            allow_early_stop=False,
        )

        writeback_unified_ckpt(worker)

    elif worker.mode == MODE_P2:
        apply_ckpt_to_p2(worker, strict=strict_sync)

        worker.p2_trainer.train_for(
            int(chunk_len),
            allow_internal_eval=False,
            allow_early_stop=bool(allow_early_stop_p2),
        )

    else:
        raise ValueError(f"Unknown worker.mode: {worker.mode}")

    worker.global_episode += int(chunk_len)

    writeback_unified_ckpt(worker)

    elapsed = time.time() - t0
    return {
        "elapsed_sec": float(elapsed),
        "sec_per_episode": float(elapsed / max(1, int(chunk_len))),
    }


def _seed_offset_by_ready(args, ready_idx: int) -> int:
    stride = int(getattr(args, "eval_seed_stride", 0) or 0)
    if stride <= 0:
        return 0
    return int(ready_idx) * stride


def make_daily_seed_list(args, daily_idx: int) -> list:
    k = int(getattr(args, "k_daily", 0) or 0)
    if k <= 0:
        return []
    return list(FIXED_VALIDATION_SEEDS)[:k]


def make_ready_seed_list(args, ready_idx: int) -> list:
    k = int(getattr(args, "k_ready", 0) or 0)
    if k <= 0:
        return []

    start = int(getattr(args, "k_daily", 0) or 0)
    seeds = list(FIXED_VALIDATION_SEEDS)[start : start + k]

    if len(seeds) < k:
        seeds = list(FIXED_VALIDATION_SEEDS)[:k]

    return seeds


def daily_eval(worker: WorkerState, args, daily_idx: int, tag_ep: int) -> float:
    apply_ckpt_to_p2(worker, load_optim=False, strict=True)

    seed_list = make_daily_seed_list(args, daily_idx)

    profile_str = (
        "phase2"
        if bool(getattr(worker.p2_trainer.args, "include_mrt_features", False))
        else "phase2_no_mrt"
    )

    raw_train_cfg = dict(worker.p2_trainer.env.config)
    eval_cfg = get_eval_config(
        base_config=raw_train_cfg,
        mode="standard",
        profile=profile_str,
        force_fixed_positions=True,
    )

    pack = run_multi_seed_validation_metrics(
        base_env_config=eval_cfg,
        agents=worker.p2_trainer.agents,
        args=worker.p2_trainer.args,
        seed_list=seed_list,
        profile=profile_str,
        mode="standard",
        force_fixed_positions=True,
    )

    record = build_eval_curve_record(
        episodes_done=int(tag_ep),
        metrics_pack=pack,
    )

    append_eval_curve_json(
        worker.daily_curve_json_path,
        record,
        meta={
            "exp_name": str(getattr(args, "exp_name", "")),
            "wid": int(worker.wid),
            "script": "train_PBT_v64.py",
            "curve_kind": "daily_like_detailed_values",
        },
    )

    score = float(pack["metrics"]["undiscounted_return"]["mean"])
    tb_write_worker_daily_eval(
        worker.writer,
        global_episode=int(tag_ep),
        daily_return=score,
    )

    worker.last_daily_score = score
    return score


def ready_eval(worker: WorkerState, master_args, ready_idx: int) -> float:
    assert worker.p2_trainer is not None
    assert worker.writer is not None

    apply_ckpt_to_p2(worker, load_optim=False, strict=True)

    seed_list = make_ready_seed_list(master_args, ready_idx)

    profile_str = (
        "phase2"
        if bool(getattr(worker.p2_trainer.args, "include_mrt_features", False))
        else "phase2_no_mrt"
    )

    raw_train_cfg = dict(worker.p2_trainer.env.config)
    eval_cfg = get_eval_config(
        base_config=raw_train_cfg,
        mode="standard",
        profile=profile_str,
        force_fixed_positions=True,
    )

    pack = run_multi_seed_validation_metrics(
        base_env_config=eval_cfg,
        agents=worker.p2_trainer.agents,
        args=worker.p2_trainer.args,
        seed_list=seed_list,
        profile=profile_str,
        mode="standard",
        force_fixed_positions=True,
    )

    score = float(pack["metrics"]["undiscounted_return"]["mean"])

    worker.last_ready_score = score
    tb_write_worker_ready_eval(
        worker.writer,
        global_episode=int(worker.global_episode),
        ready_return=score,
    )
    return score


def _clone_ckpt(src: UnifiedCkpt) -> UnifiedCkpt:
    return dict_to_ckpt(ckpt_to_dict(src))


def _strip_optims(ckpt: UnifiedCkpt) -> UnifiedCkpt:
    c = _clone_ckpt(ckpt)
    c.q_optimizer = None
    c.power_actor_optim = None
    c.power_critic_optim = None
    c.phase_actor_optim = None
    c.phase_critic_optim = None
    return c


def exploit_copy(dst: WorkerState, src: WorkerState, *, copy_optimizer: bool) -> None:
    if copy_optimizer:
        dst.ckpt = _clone_ckpt(src.ckpt)
    else:
        dst_optims = {
            name: copy.deepcopy(getattr(dst.ckpt, name))
            for name in (
                "q_optimizer",
                "power_actor_optim",
                "power_critic_optim",
                "phase_actor_optim",
                "phase_critic_optim",
            )
        }
        dst.ckpt = _strip_optims(src.ckpt)
        for name, optimizer_state in dst_optims.items():
            setattr(dst.ckpt, name, optimizer_state)
    dst.ckpt.runtime = None

    dst.hp = dict(src.hp)
    dst.hp_base = dict(src.hp_base)

    dst.p1_trainer.set_hyperparams(
        lr_dqn=dst.hp["p1_lr_dqn"],
        lr_power_ppo=dst.hp["p1_lr_power_ppo"],
        entropy_power=dst.hp["p1_entropy_coef_power"],
    )
    dst.p2_trainer.set_hyperparams(
        lr_power_ppo=dst.hp["p2_lr_power_ppo"],
        entropy_power=dst.hp["p2_entropy_coef_power"],
        lr_phase_ppo=dst.hp["p2_lr_phase_ppo"],
        entropy_phase=dst.hp["p2_entropy_coef_phase"],
    )

    clean_buffers_after_event(dst, reason="exploit", prev_mode=dst.mode, new_mode=dst.mode)


def _choose_factor(down: float, up: float, bias_down: float) -> float:
    return float(down) if random.random() < float(bias_down) else float(up)


def _maybe_perturb_next_mode_hparams(
    w: WorkerState, *, prev_mode: str, new_mode: str, args: argparse.Namespace
) -> None:
    """Perturb the hyperparameters of the destination training stage after exploitation."""
    lr_keys: List[str] = []
    ent_keys: List[str] = []

    if new_mode == MODE_P1:
        lr_keys.append("p1_lr_power_ppo")
        ent_keys.append("p1_entropy_coef_power")

        if bool(getattr(args, "hp_lr_perturb_dqn", False)):
            lr_keys.append("p1_lr_dqn")

    elif new_mode == MODE_P2:
        lr_keys.append("p2_lr_phase_ppo")
        ent_keys.append("p2_entropy_coef_phase")

        p2a = w.p2_trainer.args
        p2_trains_power = not bool(getattr(p2a, "phase2_only", False)) and not bool(
            getattr(p2a, "freeze_dqn_power", False)
        )

        if bool(getattr(args, "hp_perturb_power_in_p2", False)) and p2_trains_power:
            lr_keys.append("p2_lr_power_ppo")
            ent_keys.append("p2_entropy_coef_power")

    if random.random() < float(getattr(args, "hp_lr_perturb_prob", 0.0)):
        factor = _choose_factor(args.hp_lr_factor_down, args.hp_lr_factor_up, args.hp_lr_bias_down)
        for k in lr_keys:
            old = float(w.hp.get(k, 0.0))
            base = float(w.hp_base.get(k, old if old > 0 else 1.0))
            new = old * factor
            new = max(float(args.hp_lr_min), min(new, base * float(args.hp_lr_max_factor)))
            w.hp[k] = float(new)

    if random.random() < float(getattr(args, "hp_ent_perturb_prob", 0.0)):
        factor = _choose_factor(
            args.hp_ent_factor_down, args.hp_ent_factor_up, args.hp_ent_bias_down
        )
        for k in ent_keys:
            old = float(w.hp.get(k, 0.0))
            new = old * factor
            new = max(float(args.hp_ent_min), min(new, float(args.hp_ent_max)))
            w.hp[k] = float(new)

    if hasattr(w.p1_trainer, "set_hyperparams"):
        w.p1_trainer.set_hyperparams(
            lr_dqn=w.hp["p1_lr_dqn"],
            lr_power_ppo=w.hp["p1_lr_power_ppo"],
            entropy_power=w.hp["p1_entropy_coef_power"],
        )

    if hasattr(w.p2_trainer, "set_hyperparams"):
        w.p2_trainer.set_hyperparams(
            lr_power_ppo=w.hp["p2_lr_power_ppo"],
            entropy_power=w.hp["p2_entropy_coef_power"],
            lr_phase_ppo=w.hp["p2_lr_phase_ppo"],
            entropy_phase=w.hp["p2_entropy_coef_phase"],
        )


def _maybe_flip_mode(w: WorkerState, args: argparse.Namespace) -> bool:
    r = random.random()
    if w.mode == MODE_P1:
        if r < float(args.p12_flip_prob):
            w.mode = MODE_P2
            return True
        return False
    elif w.mode == MODE_P2:
        if r < float(args.p21_flip_prob):
            w.mode = MODE_P1
            return True
        return False
    else:
        raise ValueError(f"Unknown mode: {w.mode}")


def on_ready_boundary(
    workers: Dict[int, WorkerState],
    args: argparse.Namespace,
    ready_idx: int,
    log_paths: LogPaths,
    writer_global: SummaryWriter,
    precomputed_scores: Optional[Dict[int, float]] = None,
) -> ReadyRoundStats:
    scores_all: Dict[int, float] = {}

    if precomputed_scores is None:
        for wid, w in workers.items():
            s = ready_eval(w, args, ready_idx)
            scores_all[wid] = float(s)
    else:
        missing = sorted(set(workers.keys()) - set(precomputed_scores.keys()))
        if missing:
            raise RuntimeError(f"[ReadyMaster] precomputed_scores missing workers: {missing}")

        for wid, w in workers.items():
            s = float(precomputed_scores[wid])
            scores_all[wid] = s
            w.last_ready_score = float(s)

    eligible_ids = [wid for wid, w in workers.items() if int(w.cooldown_remaining) <= 0]
    scores_eligible = {wid: scores_all[wid] for wid in eligible_ids}

    num_in_cooldown = sum(1 for w in workers.values() if int(w.cooldown_remaining) > 0)

    need_pairs = int(args.num_exploit)
    can_exploit = (need_pairs > 0) and (len(eligible_ids) >= max(2, 2 * need_pairs))

    if len(eligible_ids) > 0:
        eligible_sorted = sorted(eligible_ids, key=lambda wid: (-scores_all[wid], wid))
        best_id = eligible_sorted[0]
        eligible_scores_list = [scores_all[wid] for wid in eligible_sorted]
        mean_score = float(np.mean(eligible_scores_list))
        std_score = float(np.std(eligible_scores_list))
        best_score = float(scores_all[best_id])
    else:
        eligible_sorted = []
        best_id = -1
        best_score = float("nan")
        mean_score = float("nan")
        std_score = float("nan")

    exploit_pairs: List[Tuple[int, int]] = []
    num_flip = 0
    num_exploit_done = 0

    exploited_from_by_wid: Dict[int, Optional[int]] = {wid: None for wid in workers.keys()}
    did_flip_by_wid: Dict[int, bool] = {wid: False for wid in workers.keys()}
    mode_before_by_wid: Dict[int, str] = {wid: w.mode for wid, w in workers.items()}

    tb_write_global_ready_summary(
        writer_global,
        ready_round_idx=int(ready_idx),
        best_eval_ready_return=float(best_score),
        mean_eval_ready_return=float(mean_score),
        std_eval_ready_return=float(std_score),
        num_exploit_this_ready=0,
        num_flip_this_ready=0,
        num_in_cooldown=int(num_in_cooldown),
        best_worker_id=int(best_id),
    )

    if can_exploit:
        k = int(args.num_exploit)
        top_ids = eligible_sorted[:k]
        bottom_ids = list(reversed(eligible_sorted[-k:]))

        for src_id, dst_id in zip(top_ids, bottom_ids):
            src = workers[src_id]
            dst = workers[dst_id]

            exploit_copy(dst, src, copy_optimizer=bool(args.copy_optimizer))
            exploit_pairs.append((src_id, dst_id))
            exploited_from_by_wid[dst_id] = int(src_id)
            num_exploit_done += 1

            dst.pending_event_exploit = 1

            prev_mode = dst.mode
            did_flip = _maybe_flip_mode(dst, args)
            did_flip_by_wid[dst_id] = bool(did_flip)

            if did_flip:
                num_flip += 1
                dst.pending_event_flip = 1

                clean_buffers_after_event(
                    dst, reason="flip", prev_mode=prev_mode, new_mode=dst.mode
                )

            _maybe_perturb_next_mode_hparams(dst, prev_mode=prev_mode, new_mode=dst.mode, args=args)
            if (
                str(prev_mode).upper().strip() == MODE_P2
                and str(dst.mode).upper().strip() == MODE_P1
            ):
                dst.pending_p1_local_reheat = 1
            else:
                dst.pending_p1_local_reheat = 0
            dst.cooldown_remaining = int(args.cooldown_episodes)

    num_in_cooldown_after = sum(1 for w in workers.values() if int(w.cooldown_remaining) > 0)

    tb_write_global_ready_summary(
        writer_global,
        ready_round_idx=int(ready_idx),
        best_eval_ready_return=float(best_score),
        mean_eval_ready_return=float(mean_score),
        std_eval_ready_return=float(std_score),
        num_exploit_this_ready=int(num_exploit_done),
        num_flip_this_ready=int(num_flip),
        num_in_cooldown=int(num_in_cooldown_after),
        best_worker_id=int(best_id),
    )

    for wid, w in workers.items():
        wdir = log_paths.worker_dirs[wid]

        save_ready_ckpt(
            worker_dir=wdir,
            ready_idx=int(ready_idx),
            ckpt_dict=ckpt_to_dict(w.ckpt),
            hp=w.hp,
            keep_last_k=args.keep_last_ready_ckpts,
            ready_score=float(scores_all[wid]),
        )

        rank_in_eligible = None
        if wid in eligible_sorted:
            rank_in_eligible = int(eligible_sorted.index(wid))

        event = {
            "ready_idx": int(ready_idx),
            "wid": int(wid),
            "ready_score": float(scores_all[wid]),
            "eligible": bool(wid in eligible_ids),
            "rank_in_eligible": rank_in_eligible,
            "exploited_from": exploited_from_by_wid[wid],
            "did_flip": bool(did_flip_by_wid[wid]),
            "mode_before": str(mode_before_by_wid[wid]),
            "mode_after": str(w.mode),
            "cooldown_remaining_after": int(w.cooldown_remaining),
            "eligible_ids": list(eligible_ids),
            "exploit_pairs": list(exploit_pairs),
        }
        append_ready_event_jsonl(wdir / "ready_events.jsonl", event)

    stats = ReadyRoundStats(
        ready_idx=int(ready_idx),
        eligible_ids=list(eligible_ids),
        scores_all=dict(scores_all),
        scores_eligible=dict(scores_eligible),
        best_id=int(best_id),
        best_score=float(best_score),
        mean_score=float(mean_score),
        std_score=float(std_score),
        num_exploit=int(num_exploit_done),
        num_flip=int(num_flip),
        num_in_cooldown=int(num_in_cooldown_after),
        exploit_pairs=list(exploit_pairs),
    )

    append_pbt_summary_csv(log_paths.pbt_summary_csv, stats)
    return stats


def tick_after_chunk(
    worker: WorkerState,
    chunk_len: int,
    *,
    train_info: Optional[Dict[str, Any]] = None,
    write_mode_timeline: bool = True,
) -> None:
    if chunk_len < 0:
        raise ValueError(f"chunk_len must be non-negative, got {chunk_len}")

    if int(chunk_len) > 0:
        worker.cooldown_remaining = max(0, int(worker.cooldown_remaining) - int(chunk_len))

    time_per_episode = None
    if isinstance(train_info, dict):
        tpe = train_info.get("sec_per_episode", None)
        if tpe is not None:
            time_per_episode = float(tpe)

    tb_write_worker_chunk(
        worker.writer,
        global_step=int(worker.global_episode),
        mode=MODE_TO_INT.get(worker.mode, -1),
        cooldown_remaining=int(worker.cooldown_remaining),
        episodes_done=int(worker.global_episode),
        time_per_episode=time_per_episode,
        event_exploit=int(getattr(worker, "pending_event_exploit", 0)),
        event_flip=int(getattr(worker, "pending_event_flip", 0)),
    )

    if write_mode_timeline:
        try:
            csv_path = Path(worker.log_dir) / "mode_timeline.csv"
        except Exception:
            pass

    worker.pending_event_exploit = 0
    worker.pending_event_flip = 0


@dataclass
class StopState:
    best_ready_score_ever: float = float("-inf")
    best_smoothed: Optional[float] = None
    no_improve_ready: int = 0


def init_stop_state() -> StopState:
    return StopState()


def _get_global_episode_now(workers: Dict[int, WorkerState]) -> int:
    if not workers:
        return 0
    return int(max(int(w.global_episode) for w in workers.values()))


def hard_stop_reached(workers: Dict[int, WorkerState], args: argparse.Namespace) -> bool:
    ge = _get_global_episode_now(workers)
    return ge >= int(args.max_pbt_episodes)


def _write_early_stop_log(
    log_path: Path,
    *,
    reason: str,
    ready_idx: int,
    global_episode: int,
    best_this_ready: float,
    best_ready_score_ever: float,
    best_smoothed: float,
    no_improve_ready: int,
    patience_ready: int,
    min_delta: float,
    smooth_alpha: float,
) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"[{ts}] EARLY_STOP_TRIGGERED reason={reason}\n"
        f"  ready_idx={ready_idx}\n"
        f"  global_episode={global_episode}\n"
        f"  best_this_ready={best_this_ready:.6f}\n"
        f"  best_ready_score_ever={best_ready_score_ever:.6f}\n"
        f"  best_smoothed={best_smoothed:.6f}\n"
        f"  no_improve_ready={no_improve_ready} / patience_ready={patience_ready}\n"
        f"  min_delta={min_delta}\n"
        f"  smooth_alpha={smooth_alpha}\n"
        f"\n"
    )
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(msg)


def soft_stop_update_on_ready(
    workers: Dict[int, WorkerState],
    args: argparse.Namespace,
    stop_state: StopState,
    *,
    ready_idx: int,
    log_paths: LogPaths,
    writer_global: Optional[SummaryWriter] = None,
) -> bool:
    if not workers or int(args.patience_ready) <= 0:
        return False

    scores = []
    for w in workers.values():
        s = float(w.last_ready_score)
        if s == s:
            scores.append(s)

    if len(scores) == 0:
        return False

    best_this_ready = float(max(scores))

    if best_this_ready > float(stop_state.best_ready_score_ever):
        stop_state.best_ready_score_ever = float(best_this_ready)

    alpha = float(args.smooth_alpha)
    if stop_state.best_smoothed is None:
        prev_smoothed = float(best_this_ready)
        stop_state.best_smoothed = float(best_this_ready)
        improvement = float("inf")
    else:
        prev_smoothed = float(stop_state.best_smoothed)
        stop_state.best_smoothed = alpha * float(best_this_ready) + (1.0 - alpha) * prev_smoothed
        improvement = float(stop_state.best_smoothed - prev_smoothed)

    if improvement <= float(args.min_delta):
        stop_state.no_improve_ready += 1
    else:
        stop_state.no_improve_ready = 0

    if writer_global is not None:
        ridx = int(ready_idx)
        writer_global.add_scalar("early_stop/best_this_ready", float(best_this_ready), ridx)
        writer_global.add_scalar(
            "early_stop/best_ready_score_ever", float(stop_state.best_ready_score_ever), ridx
        )
        writer_global.add_scalar("early_stop/best_smoothed", float(stop_state.best_smoothed), ridx)
        writer_global.add_scalar(
            "early_stop/no_improve_ready", int(stop_state.no_improve_ready), ridx
        )
        writer_global.add_scalar("early_stop/improvement_smoothed", float(improvement), ridx)

    if int(stop_state.no_improve_ready) >= int(args.patience_ready):
        ge = _get_global_episode_now(workers)
        _write_early_stop_log(
            log_paths.early_stop_log,
            reason="soft_stop_ready_plateau",
            ready_idx=int(ready_idx),
            global_episode=int(ge),
            best_this_ready=float(best_this_ready),
            best_ready_score_ever=float(stop_state.best_ready_score_ever),
            best_smoothed=float(stop_state.best_smoothed),
            no_improve_ready=int(stop_state.no_improve_ready),
            patience_ready=int(args.patience_ready),
            min_delta=float(args.min_delta),
            smooth_alpha=float(args.smooth_alpha),
        )
        return True

    return False


def _get_min_global_episode_now(workers: Dict[int, WorkerState]) -> int:
    if not workers:
        return 0
    return int(min(int(w.global_episode) for w in workers.values()))


def _boundary_crossed(prev_ep: int, new_ep: int, every: int) -> bool:
    if every <= 0:
        raise ValueError(f"every must be positive, got {every}")
    prev_ep = int(prev_ep)
    new_ep = int(new_ep)
    if new_ep < prev_ep:
        return False
    return (prev_ep // every) != (new_ep // every)


def _iter_crossed_boundary_indices(prev_ep: int, new_ep: int, every: int) -> List[int]:
    if every <= 0:
        raise ValueError(f"every must be positive, got {every}")
    prev_bucket = int(prev_ep) // int(every)
    new_bucket = int(new_ep) // int(every)
    if new_bucket <= prev_bucket:
        return []
    return list(range(prev_bucket + 1, new_bucket + 1))


def _maybe_run_daily_eval(
    w: WorkerState,
    args: argparse.Namespace,
    prev_ep: int,
    new_ep: int,
) -> None:
    """Evaluate every crossed monitoring boundary on the worker-local episode axis."""
    every = int(getattr(args, "eval_every_episodes", 0) or 0)
    if every <= 0:
        return

    for idx in _iter_crossed_boundary_indices(prev_ep, new_ep, every=every):
        tag_ep = idx * every

        daily_idx = max(0, idx - 1)

        w.last_daily_score = daily_eval(w, args, daily_idx=daily_idx, tag_ep=tag_ep)


def _obj_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {k: _obj_to_cpu(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_obj_to_cpu(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_obj_to_cpu(v) for v in obj)

    return copy.deepcopy(obj)


def _build_adapt_snapshot(
    worker: WorkerState, train_info: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "wid": int(worker.wid),
        "mode": str(worker.mode),
        "cooldown_remaining": int(worker.cooldown_remaining),
        "global_episode": int(worker.global_episode),
        "last_daily_score": float(worker.last_daily_score),
        "last_ready_score": float(worker.last_ready_score),
        "pending_event_exploit": int(getattr(worker, "pending_event_exploit", 0)),
        "pending_event_flip": int(getattr(worker, "pending_event_flip", 0)),
        "pending_p1_local_reheat": int(getattr(worker, "pending_p1_local_reheat", 0)),
        "hp": copy.deepcopy(getattr(worker, "hp", {})),
        "hp_base": copy.deepcopy(getattr(worker, "hp_base", {})),
        "ckpt": _obj_to_cpu(ckpt_to_dict(worker.ckpt)),
        "train_info": _obj_to_cpu(train_info) if train_info is not None else None,
    }


def _apply_adapt_snapshot_to_worker(worker: WorkerState, snapshot: Dict[str, Any]) -> None:
    worker.mode = str(snapshot["mode"])
    worker.cooldown_remaining = int(snapshot["cooldown_remaining"])
    worker.global_episode = int(snapshot["global_episode"])

    worker.last_daily_score = float(snapshot["last_daily_score"])
    worker.last_ready_score = float(snapshot["last_ready_score"])

    worker.pending_event_exploit = int(snapshot.get("pending_event_exploit", 0))
    worker.pending_event_flip = int(snapshot.get("pending_event_flip", 0))

    worker.pending_p1_local_reheat = int(snapshot.get("pending_p1_local_reheat", 0))

    if "hp" in snapshot and snapshot["hp"] is not None:
        worker.hp = dict(snapshot["hp"])
    if "hp_base" in snapshot and snapshot["hp_base"] is not None:
        worker.hp_base = dict(snapshot["hp_base"])

    worker.ckpt = dict_to_ckpt(snapshot["ckpt"])

    load_optim_flag = any(
        getattr(worker.ckpt, name) is not None
        for name in (
            "q_optimizer",
            "power_actor_optim",
            "power_critic_optim",
            "phase_actor_optim",
            "phase_critic_optim",
        )
    )

    apply_ckpt_to_p1(worker, load_optim=load_optim_flag, strict=True)
    apply_ckpt_to_p2(worker, load_optim=load_optim_flag, strict=True)

    if hasattr(worker.p1_trainer, "set_hyperparams") and getattr(worker, "hp", None):
        worker.p1_trainer.set_hyperparams(
            lr_dqn=worker.hp["p1_lr_dqn"],
            lr_power_ppo=worker.hp["p1_lr_power_ppo"],
            entropy_power=worker.hp["p1_entropy_coef_power"],
        )

    if hasattr(worker.p2_trainer, "set_hyperparams") and getattr(worker, "hp", None):
        worker.p2_trainer.set_hyperparams(
            lr_power_ppo=worker.hp["p2_lr_power_ppo"],
            entropy_power=worker.hp["p2_entropy_coef_power"],
            lr_phase_ppo=worker.hp["p2_lr_phase_ppo"],
            entropy_phase=worker.hp["p2_entropy_coef_phase"],
        )


def _adapt_worker_process_main(
    master_args: argparse.Namespace,
    wid: int,
    log_dir_str: str,
    adapt_mode: str,
    conn,
    *,
    input_snapshot_path: Optional[str] = None,
) -> None:
    writer = None
    try:
        log_dir = Path(log_dir_str)

        writer = _build_safe_worker_writer(
            log_dir=log_dir,
            tb_flush_secs=int(getattr(master_args, "tb_flush_secs", 600)),
            tb_max_queue=int(getattr(master_args, "tb_max_queue", 500)),
            tb_p2_diag_every=int(getattr(master_args, "tb_p2_diag_every", 32)),
            enabled=False,
        )

        snapshot_path = None if input_snapshot_path in (None, "") else Path(input_snapshot_path)
        has_input_snapshot = bool(snapshot_path is not None and snapshot_path.exists())

        worker = init_worker(
            master_args,
            int(wid),
            log_dir=log_dir,
            writer=writer,
            skip_warmup_bootstrap=bool(has_input_snapshot),
            quiet_bootstrap=True,
        )

        if has_input_snapshot:
            snap_payload = _torch_load_any(snapshot_path, map_location="cpu")
            if not isinstance(snap_payload, dict):
                raise RuntimeError(
                    f"[AdaptChild] worker {wid} input snapshot must be a dict, got {type(snap_payload)}"
                )
            _apply_adapt_snapshot_to_worker(worker, snap_payload)

        worker.mode = str(adapt_mode).upper().strip()

        conn.send(
            {
                "ok": True,
                "op": "INIT_DONE",
                "wid": int(wid),
            }
        )

        while True:
            msg = conn.recv()
            op = str(msg.get("op", "")).upper().strip()

            if op == "TRAIN_ADAPT_CHUNK":
                chunk_len = int(msg["chunk_len"])

                prev_ep = int(worker.global_episode)
                train_info = train_chunk(worker, chunk_len=chunk_len, allow_early_stop_p2=False)
                _maybe_run_daily_eval(
                    worker,
                    master_args,
                    prev_ep=prev_ep,
                    new_ep=int(worker.global_episode),
                )
                tick_after_chunk(
                    worker,
                    int(chunk_len),
                    train_info=train_info,
                    write_mode_timeline=False,
                )

                conn.send(
                    {
                        "ok": True,
                        "op": "TRAIN_DONE",
                        "wid": int(wid),
                        "snapshot": _build_adapt_snapshot(worker, train_info=train_info),
                    }
                )
                continue

            if op == "CLOSE":
                conn.send(
                    {
                        "ok": True,
                        "op": "CLOSE_DONE",
                        "wid": int(wid),
                        "snapshot": _build_adapt_snapshot(worker, train_info=None),
                    }
                )
                break

            raise RuntimeError(f"[AdaptChild] worker {wid} got unknown op={op}")

    except EOFError:
        pass
    except Exception:
        try:
            conn.send(
                {
                    "ok": False,
                    "op": "ERROR",
                    "wid": int(wid),
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


def _recv_or_raise_adapt_msg(conn, wid: int) -> Dict[str, Any]:
    msg = conn.recv()
    if not isinstance(msg, dict):
        raise RuntimeError(f"[AdaptMaster] worker {wid} returned invalid message: {type(msg)}")

    if not bool(msg.get("ok", False)):
        err = str(msg.get("error", "unknown child error"))
        raise RuntimeError(f"[AdaptMaster] worker {wid} failed:\n{err}")

    return msg


def _run_adapt_window_parallel_spawn(
    args: argparse.Namespace,
    log_paths,
    *,
    window_episodes: Optional[int] = None,
    adapt_mode: Optional[str] = None,
    input_snapshots: Optional[Dict[int, Dict[str, Any]]] = None,
    window_tag: str = "ADAPT",
    on_committed_chunk=None,
) -> Dict[int, Dict[str, Any]]:
    run_mode = (
        str(adapt_mode if adapt_mode is not None else getattr(args, "adapt_mode", "P2"))
        .upper()
        .strip()
    )
    if run_mode != MODE_P2:
        raise ValueError(f"Invalid adaptation mode: {run_mode}")

    target_window_episodes = int(
        getattr(args, "adapt_episodes", 0) if window_episodes is None else window_episodes
    )
    if target_window_episodes <= 0:
        return copy.deepcopy(input_snapshots) if input_snapshots is not None else {}

    ctx = mp.get_context("spawn")

    parent_conns: Dict[int, Any] = {}
    procs: Dict[int, mp.Process] = {}
    latest_snapshots: Dict[int, Dict[str, Any]] = {}
    input_snapshot_paths: Dict[int, Path] = {}

    global_eps: Dict[int, int] = {}
    window_progress: Dict[int, int] = {}
    start_eps: Dict[int, int] = {}

    for wid in range(int(args.n_workers)):
        ge0 = 0
        if input_snapshots is not None and wid in input_snapshots:
            ge0 = int(input_snapshots[wid].get("global_episode", 0))
        start_eps[wid] = int(ge0)
        global_eps[wid] = int(ge0)
        window_progress[wid] = 0

    try:
        for wid in range(int(args.n_workers)):
            snapshot_path_str = None

            if input_snapshots is not None and wid in input_snapshots:
                init_path = _make_ipc_snapshot_path(
                    log_paths.worker_dirs[wid],
                    wid=int(wid),
                    tag=f"{str(window_tag).lower()}_init",
                )
                _save_ipc_snapshot(init_path, input_snapshots[wid])
                input_snapshot_paths[wid] = init_path
                snapshot_path_str = str(init_path)

            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_adapt_worker_process_main,
                args=(
                    args,
                    int(wid),
                    str(log_paths.worker_dirs[wid]),
                    run_mode,
                    child_conn,
                ),
                kwargs={
                    "input_snapshot_path": snapshot_path_str,
                },
                daemon=False,
            )
            p.start()
            child_conn.close()

            parent_conns[wid] = parent_conn
            procs[wid] = p

        for wid, conn in parent_conns.items():
            msg = _recv_or_raise_adapt_msg(conn, wid)
            if str(msg.get("op", "")) != "INIT_DONE":
                raise RuntimeError(
                    f"[AdaptMaster] worker {wid} expected INIT_DONE, got {msg.get('op')}"
                )

        print(
            f"[PBT Master] entering P2 adaptation window (parallel): "
            f"tag={window_tag}, mode={run_mode}, window_episodes={int(target_window_episodes)}"
        )

        while True:
            win_min = min(window_progress.values()) if len(window_progress) > 0 else 0
            ge_min = min(global_eps.values()) if len(global_eps) > 0 else 0

            if win_min >= int(target_window_episodes):
                break
            if ge_min >= int(args.max_pbt_episodes):
                break

            remaining_to_window = int(target_window_episodes) - int(win_min)
            remaining_to_hard_stop = int(args.max_pbt_episodes) - int(ge_min)
            if remaining_to_window <= 0 or remaining_to_hard_stop <= 0:
                break

            chunk = min(
                int(args.chunk_len),
                int(remaining_to_window),
                int(remaining_to_hard_stop),
            )
            chunk = int(max(1, chunk))

            for conn in parent_conns.values():
                conn.send(
                    {
                        "op": "TRAIN_ADAPT_CHUNK",
                        "chunk_len": int(chunk),
                    }
                )

            for wid, conn in parent_conns.items():
                msg = _recv_or_raise_adapt_msg(conn, wid)
                if str(msg.get("op", "")) != "TRAIN_DONE":
                    raise RuntimeError(
                        f"[AdaptMaster] worker {wid} expected TRAIN_DONE, got {msg.get('op')}"
                    )

                snap = msg.get("snapshot", None)
                if not isinstance(snap, dict):
                    raise RuntimeError(
                        f"[AdaptMaster] worker {wid} returned invalid snapshot: {type(snap)}"
                    )

                latest_snapshots[wid] = snap
                global_eps[wid] = int(snap.get("global_episode", 0))
                window_progress[wid] = int(global_eps[wid]) - int(start_eps[wid])

            if on_committed_chunk is not None:
                on_committed_chunk(
                    latest_snapshots,
                    int(min(window_progress.values())),
                    int(min(global_eps.values())),
                )

        for conn in parent_conns.values():
            try:
                conn.send({"op": "CLOSE"})
            except Exception:
                pass

        for wid, conn in parent_conns.items():
            try:
                msg = _recv_or_raise_adapt_msg(conn, wid)
                if str(msg.get("op", "")) == "CLOSE_DONE":
                    snap = msg.get("snapshot", None)
                    if isinstance(snap, dict):
                        latest_snapshots[wid] = snap
                        global_eps[wid] = int(snap.get("global_episode", 0))
                        window_progress[wid] = int(global_eps[wid]) - int(start_eps[wid])
            except Exception:
                pass

        return latest_snapshots

    finally:
        for wid, conn in parent_conns.items():
            try:
                conn.close()
            except Exception:
                pass

        for wid, p in procs.items():
            try:
                p.join(timeout=3.0)
            except Exception:
                pass
            try:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.0)
            except Exception:
                pass

        for wid, p in input_snapshot_paths.items():
            _safe_unlink(p)


def _recv_or_raise_formal_msg(conn, wid: int) -> Dict[str, Any]:
    msg = conn.recv()
    if not isinstance(msg, dict):
        raise RuntimeError(f"[FormalMaster] worker {wid} returned invalid message: {type(msg)}")

    if not bool(msg.get("ok", False)):
        err = str(msg.get("error", "unknown child error"))
        raise RuntimeError(f"[FormalMaster] worker {wid} failed:\n{err}")

    return msg


def _formal_chunk_worker_process_main(
    master_args: argparse.Namespace,
    wid: int,
    log_dir_str: str,
    input_snapshot_path: str,
    chunk_len: int,
    conn,
) -> None:
    writer = None
    output_snapshot_path = None

    try:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        log_dir = Path(log_dir_str)

        writer = _build_safe_worker_writer(
            log_dir=log_dir,
            tb_flush_secs=int(getattr(master_args, "tb_flush_secs", 600)),
            tb_max_queue=int(getattr(master_args, "tb_max_queue", 500)),
            tb_p2_diag_every=int(getattr(master_args, "tb_p2_diag_every", 32)),
            enabled=False,
        )

        worker = init_worker(
            master_args,
            int(wid),
            log_dir=log_dir,
            writer=writer,
            skip_warmup_bootstrap=True,
            quiet_bootstrap=True,
        )

        input_snapshot = _torch_load_any(input_snapshot_path, map_location="cpu")
        _apply_adapt_snapshot_to_worker(worker, input_snapshot)

        prev_ep = int(worker.global_episode)
        train_info = train_chunk(worker, chunk_len=int(chunk_len), allow_early_stop_p2=False)
        _maybe_run_daily_eval(
            worker, master_args, prev_ep=prev_ep, new_ep=int(worker.global_episode)
        )
        tick_after_chunk(worker, int(chunk_len), train_info=train_info, write_mode_timeline=False)

        output_snapshot = _build_adapt_snapshot(worker, train_info=train_info)
        output_snapshot_path = _make_ipc_snapshot_path(log_dir, wid=int(wid), tag="formal_out")
        _save_ipc_snapshot(output_snapshot_path, output_snapshot)

        conn.send(
            {
                "ok": True,
                "op": "TRAIN_DONE",
                "wid": int(wid),
                "snapshot_path": str(output_snapshot_path),
            }
        )

    except Exception:
        tb = traceback.format_exc()
        try:
            conn.send(
                {
                    "ok": False,
                    "wid": int(wid),
                    "error": tb,
                }
            )
        except Exception:
            pass
        raise

    finally:
        _safe_unlink(input_snapshot_path)

        try:
            conn.close()
        except Exception:
            pass

        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


def _run_formal_chunk_parallel_spawn(
    args: argparse.Namespace,
    workers: Dict[int, WorkerState],
    log_paths,
    *,
    chunk_len: int,
) -> Dict[int, Dict[str, Any]]:
    if int(chunk_len) <= 0:
        raise ValueError(f"chunk_len must be positive, got {chunk_len}")

    ctx = mp.get_context("spawn")

    parent_conns: Dict[int, Any] = {}
    procs: Dict[int, mp.Process] = {}
    latest_snapshots: Dict[int, Dict[str, Any]] = {}

    input_snapshot_paths: Dict[int, Path] = {}
    output_snapshot_paths: Dict[int, Path] = {}

    try:
        for wid, w in workers.items():
            parent_conn, child_conn = ctx.Pipe(duplex=False)

            input_snapshot = _build_adapt_snapshot(w, train_info=None)
            input_snapshot_path = _make_ipc_snapshot_path(
                log_paths.worker_dirs[int(wid)],
                wid=int(wid),
                tag="formal_in",
            )
            _save_ipc_snapshot(input_snapshot_path, input_snapshot)
            input_snapshot_paths[int(wid)] = input_snapshot_path

            p = ctx.Process(
                target=_formal_chunk_worker_process_main,
                args=(
                    args,
                    int(wid),
                    str(log_paths.worker_dirs[int(wid)]),
                    str(input_snapshot_path),
                    int(chunk_len),
                    child_conn,
                ),
                daemon=False,
            )
            p.start()
            child_conn.close()

            parent_conns[int(wid)] = parent_conn
            procs[int(wid)] = p

        for wid, conn in parent_conns.items():
            msg = _recv_or_raise_formal_msg(conn, wid)
            if str(msg.get("op", "")) != "TRAIN_DONE":
                raise RuntimeError(
                    f"[FormalMaster] worker {wid} expected TRAIN_DONE, got {msg.get('op')}"
                )

            snapshot_path = Path(msg["snapshot_path"])
            output_snapshot_paths[int(wid)] = snapshot_path

            snap = _torch_load_any(snapshot_path, map_location="cpu")
            latest_snapshots[int(wid)] = snap

    finally:
        for wid, conn in parent_conns.items():
            try:
                conn.close()
            except Exception:
                pass

        for wid, p in procs.items():
            try:
                p.join(timeout=30)
            except Exception:
                pass

            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    p.join(timeout=5)
                except Exception:
                    pass

        for p in input_snapshot_paths.values():
            _safe_unlink(p)

        for p in output_snapshot_paths.values():
            _safe_unlink(p)

    if len(latest_snapshots) != len(workers):
        missing = sorted(set(workers.keys()) - set(latest_snapshots.keys()))
        raise RuntimeError(f"[FormalMaster] missing snapshots from workers: {missing}")

    for wid, snap in latest_snapshots.items():
        _apply_adapt_snapshot_to_worker(workers[int(wid)], snap)

    return latest_snapshots


def _recv_or_raise_ready_msg(conn, wid: int) -> Dict[str, Any]:
    msg = conn.recv()
    if not isinstance(msg, dict):
        raise RuntimeError(f"[ReadyMaster] worker {wid} returned invalid message: {type(msg)}")

    if not bool(msg.get("ok", False)):
        err = str(msg.get("error", "unknown child error"))
        raise RuntimeError(f"[ReadyMaster] worker {wid} failed:\n{err}")

    return msg


def _ready_eval_worker_process_main(
    master_args: argparse.Namespace,
    wid: int,
    log_dir_str: str,
    input_snapshot: Dict[str, Any],
    ready_idx: int,
    conn,
) -> None:
    writer = None
    try:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        log_dir = Path(log_dir_str)

        writer = NullSummaryWriter(log_dir=log_dir)

        worker = init_worker(
            master_args,
            int(wid),
            log_dir=log_dir,
            writer=writer,
            skip_warmup_bootstrap=True,
            quiet_bootstrap=True,
        )

        _apply_adapt_snapshot_to_worker(worker, input_snapshot)

        score = ready_eval(worker, master_args, int(ready_idx))

        conn.send(
            {
                "ok": True,
                "op": "READY_DONE",
                "wid": int(wid),
                "ready_score": float(score),
                "global_episode": int(worker.global_episode),
            }
        )

    except Exception:
        tb = traceback.format_exc()
        try:
            conn.send(
                {
                    "ok": False,
                    "wid": int(wid),
                    "error": tb,
                }
            )
        except Exception:
            pass
        raise

    finally:
        try:
            conn.close()
        except Exception:
            pass

        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


def _run_ready_eval_parallel_spawn(
    args: argparse.Namespace,
    workers: Dict[int, WorkerState],
    log_paths,
    *,
    ready_idx: int,
) -> Dict[int, float]:
    ctx = mp.get_context("spawn")

    parent_conns: Dict[int, Any] = {}
    procs: Dict[int, mp.Process] = {}
    scores_all: Dict[int, float] = {}

    try:
        for wid, w in workers.items():
            parent_conn, child_conn = ctx.Pipe(duplex=False)

            input_snapshot = _build_adapt_snapshot(w, train_info=None)

            p = ctx.Process(
                target=_ready_eval_worker_process_main,
                args=(
                    args,
                    int(wid),
                    str(log_paths.worker_dirs[int(wid)]),
                    input_snapshot,
                    int(ready_idx),
                    child_conn,
                ),
                daemon=False,
            )
            p.start()
            child_conn.close()

            parent_conns[int(wid)] = parent_conn
            procs[int(wid)] = p

        for wid, conn in parent_conns.items():
            msg = _recv_or_raise_ready_msg(conn, wid)
            if str(msg.get("op", "")) != "READY_DONE":
                raise RuntimeError(
                    f"[ReadyMaster] worker {wid} expected READY_DONE, got {msg.get('op')}"
                )

            scores_all[int(wid)] = float(msg["ready_score"])

    finally:
        for wid, conn in parent_conns.items():
            try:
                conn.close()
            except Exception:
                pass

        for wid, p in procs.items():
            try:
                p.join(timeout=30)
            except Exception:
                pass

            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    p.join(timeout=5)
                except Exception:
                    pass

    if len(scores_all) != len(workers):
        missing = sorted(set(workers.keys()) - set(scores_all.keys()))
        raise RuntimeError(f"[ReadyMaster] missing ready scores from workers: {missing}")

    for wid, score in scores_all.items():
        workers[int(wid)].last_ready_score = float(score)

    return scores_all


def _sync_master_workers_after_ready_boundary(workers: Dict[int, WorkerState]) -> None:
    for w in workers.values():
        load_optim_flag = any(
            getattr(w.ckpt, name) is not None
            for name in (
                "q_optimizer",
                "power_actor_optim",
                "power_critic_optim",
                "phase_actor_optim",
                "phase_critic_optim",
            )
        )

        apply_ckpt_to_p1(w, load_optim=load_optim_flag, strict=True)
        apply_ckpt_to_p2(w, load_optim=load_optim_flag, strict=True)

        if hasattr(w.p1_trainer, "set_hyperparams"):
            w.p1_trainer.set_hyperparams(
                lr_dqn=w.hp["p1_lr_dqn"],
                lr_power_ppo=w.hp["p1_lr_power_ppo"],
                entropy_power=w.hp["p1_entropy_coef_power"],
            )

        if hasattr(w.p2_trainer, "set_hyperparams"):
            w.p2_trainer.set_hyperparams(
                lr_power_ppo=w.hp["p2_lr_power_ppo"],
                entropy_power=w.hp["p2_entropy_coef_power"],
                lr_phase_ppo=w.hp["p2_lr_phase_ppo"],
                entropy_phase=w.hp["p2_entropy_coef_phase"],
            )


def _pulse_tick_after_ready_boundary(workers: Dict[int, WorkerState]) -> None:
    for w in workers.values():
        tick_after_chunk(w, 0, train_info=None, write_mode_timeline=False)


def _pbt_resume_signature(args: argparse.Namespace) -> Dict[str, Any]:
    return signature(
        {
            "num_ground_stations": int(args.p2_num_ground_stations),
            "num_gs_per_rb": int(args.p2_num_gs_per_rb),
            "topology_profile_p1": str(args.p1_topology_profile or args.p2_topology_profile),
            "topology_profile_p2": str(args.p2_topology_profile),
            "dqn_scheduler_impl": str(args.p2_dqn_scheduler_impl),
            "dqn_state_dim": None,
            "dqn_action_dim": None,
            "power_state_dim": None,
            "power_action_dim": None,
            "phase_state_dim": None,
            "phase_groups": 4,
            "ris_array_elements": int(args.ris_num_elem_x) * int(args.ris_num_elem_y),
            "phase_policy_dist": str(args.p2_phase_policy_dist),
            "alternating_schedule": {
                "chunk_len": int(args.chunk_len),
                "p1_dqn": int(args.p1_cycle_dqn_before_power),
                "p1_power_blocks": int(args.p1_cycle_power),
                "p1_power_episodes": int(args.p1_ppo_power_opt_episodes),
                "p2_phase_episodes": int(args.p2_ppo_phase_opt_episodes),
            },
            "n_workers": int(args.n_workers),
            "adapt_mode": str(args.adapt_mode),
            "adapt_episodes": int(args.adapt_episodes),
            "training_schedule": "warmup_p2_adaptation_formal_pbt_v1",
            "ready_every_episodes": int(args.ready_every_episodes),
            "cooldown_episodes": int(args.cooldown_episodes),
            "num_exploit": int(args.num_exploit),
            "p12_flip_prob": float(args.p12_flip_prob),
            "p21_flip_prob": float(args.p21_flip_prob),
            "copy_optimizer": bool(args.copy_optimizer),
            "hp_lr_perturb_prob": float(args.hp_lr_perturb_prob),
            "hp_ent_perturb_prob": float(args.hp_ent_perturb_prob),
            "p1_power_mode": str(args.p1_power_mode),
            "p2_power_mode": str(args.p2_power_mode),
            "p1_scheduler_mode": str(args.p1_scheduler_mode),
            "p2_scheduler_mode": str(args.p2_scheduler_mode),
            "p2_phase_state_mode": str(args.p2_phase_state_mode),
            "p2_include_mrt_features": bool(args.p2_include_mrt_features),
            "p2_learn_amp_weights": str(args.p2_learn_amp_weights),
            "ris_num_elem_x": int(args.ris_num_elem_x),
            "ris_num_elem_y": int(args.ris_num_elem_y),
            "rho": float(args.rho),
            "csi_error_variance": float(args.p2_csi_error_variance),
            "fairness_lambda": float(args.p2_fairness_lambda),
            "ris_reflection_power_limit_dBm": float(args.p2_ris_reflection_power_limit_dBm),
        }
    )


def _initialize_formal_workers(
    args: argparse.Namespace,
    log_paths: LogPaths,
    writers: Dict[int, Any],
    snapshots: Dict[int, Dict[str, Any]],
    *,
    resume_formal: bool,
) -> Dict[int, WorkerState]:
    workers = {}
    for wid in range(int(args.n_workers)):
        has_snapshot = wid in snapshots
        worker = init_worker(
            args,
            wid,
            log_dir=log_paths.worker_dirs[wid],
            writer=writers[wid],
            skip_warmup_bootstrap=has_snapshot,
            quiet_bootstrap=has_snapshot,
        )
        if has_snapshot:
            _apply_adapt_snapshot_to_worker(worker, snapshots[wid])
        if not resume_formal:
            worker.mode = MODE_P1
            worker.ckpt.counters = dict(worker.ckpt.counters or {})
            worker.ckpt.counters.update(
                segment_idx=0,
                segment_ep_remaining=int(args.p1_cycle_dqn_before_power),
                power_episode_offset=0,
            )
        workers[wid] = worker
    return workers


def run_pbt_master(argv: Optional[List[str]] = None) -> None:
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    args = parse_args_and_validate(argv)
    compatibility = _pbt_resume_signature(args)
    resume_bundle = None
    if args.resume_run_dir:
        resume_bundle = load_pbt_recovery(
            Path(args.resume_run_dir),
            expected_worker_ids=range(int(args.n_workers)),
            map_location="cpu",
        )
        validate_signature(
            resume_bundle["manifest"].get("compatibility", {}),
            compatibility,
            strict=bool(args.resume_strict),
        )
    if resume_bundle is not None and resume_bundle["controller"].get("phase") not in {
        "adapt",
        "formal",
    }:
        raise RecoveryError("Unsupported PBT recovery phase")
    log_paths = prepare_log_dirs(args)
    writers_worker, writer_global = make_writers(
        log_paths,
        n_workers=args.n_workers,
        tb_flush_secs=args.tb_flush_secs,
        tb_max_queue=args.tb_max_queue,
        tb_p2_diag_every=args.tb_p2_diag_every,
        enable_tb=bool(getattr(args, "enable_tb", False)),
    )

    workers: Dict[int, WorkerState] = {}
    stop_state = init_stop_state()
    generation_id = int(resume_bundle["manifest"]["generation_id"]) if resume_bundle else 0
    chunk_id = int(resume_bundle["manifest"]["chunk_id"]) if resume_bundle else 0

    if resume_bundle is not None:
        controller = resume_bundle["controller"]
        stop_saved = controller.get("stop_state", {})
        stop_state = StopState(
            best_ready_score_ever=float(stop_saved.get("best_ready_score_ever", float("-inf"))),
            best_smoothed=(
                None
                if stop_saved.get("best_smoothed") is None
                else float(stop_saved["best_smoothed"])
            ),
            no_improve_ready=int(stop_saved.get("no_improve_ready", 0)),
        )
        with (log_paths.run_root / "resume.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"RESUME time={time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"generation={generation_id} chunk={chunk_id} "
                f"global_episode={resume_bundle['manifest']['global_episode']}\n"
            )

    def commit_recovery(
        phase: str,
        snapshots: Optional[Dict[int, Dict[str, Any]]],
        *,
        adapt_progress: int,
        formal_progress: int,
        global_episode: int,
    ) -> None:
        nonlocal generation_id, chunk_id
        generation_id += 1
        chunk_id += 1
        controller_state = {
            "phase": str(phase),
            "adapt_progress": int(adapt_progress),
            "formal_progress": int(formal_progress),
            "global_episode": int(global_episode),
            "ready_index": int(global_episode) // int(args.ready_every_episodes),
            "daily_index": int(global_episode) // int(args.eval_every_episodes),
            "chunk_id": int(chunk_id),
            "generation_id": int(generation_id),
            "stop_state": asdict(stop_state),
            "stop_requested": bool(
                int(args.patience_ready) > 0
                and stop_state.best_smoothed is not None
                and int(stop_state.no_improve_ready) >= int(args.patience_ready)
            ),
            "master_rng": capture_rng_state(),
        }

        def worker_payload(wid: int) -> Dict[str, Any]:
            if snapshots is not None:
                return snapshots[int(wid)]
            return _build_adapt_snapshot(workers[int(wid)], train_info=None)

        save_pbt_recovery(
            log_paths.run_root / "recovery",
            generation_id=int(generation_id),
            chunk_id=int(chunk_id),
            global_episode=int(global_episode),
            phase=str(phase),
            compatibility=compatibility,
            controller_state=controller_state,
            worker_ids=range(int(args.n_workers)),
            worker_payload=worker_payload,
        )

    try:
        print(
            f"[PBT Master] config summary: "
            f"adapt_mode={args.adapt_mode}, "
            f"adapt_episodes={int(args.adapt_episodes)}, "
            f"ready_every_episodes={int(args.ready_every_episodes)}, "
            f"cooldown_episodes={int(args.cooldown_episodes)}, "
            f"p12_flip_prob={float(args.p12_flip_prob):.4f}, "
            f"p21_flip_prob={float(args.p21_flip_prob):.4f}, "
            f"p2_phase_policy_dist={str(getattr(args, 'p2_phase_policy_dist', 'gaussian_latent'))}, "
            f"p1_phase_model_policy_dist={str(getattr(args, 'p1_phase_model_policy_dist', None))}"
        )

        resume_controller = resume_bundle["controller"] if resume_bundle else {}
        resume_phase = str(resume_controller.get("phase", ""))
        resume_snapshots = dict(resume_bundle["workers"]) if resume_bundle else {}
        saved_adapt_progress = int(resume_controller.get("adapt_progress", 0))

        adapt_snapshots: Dict[int, Dict[str, Any]] = {}
        if resume_bundle is not None and resume_phase == "formal":
            adapt_snapshots = dict(resume_snapshots)
        elif int(args.adapt_episodes) > 0:
            adapt_start = dict(resume_snapshots) if resume_phase == "adapt" else None
            adapt_done = saved_adapt_progress if resume_phase == "adapt" else 0
            adapt_remaining = int(args.adapt_episodes) - int(adapt_done)
            if adapt_remaining < 0:
                raise RecoveryError("PBT adapt recovery progress exceeds configured adapt_episodes")
            adapt_snapshots = _run_adapt_window_parallel_spawn(
                args,
                log_paths,
                window_episodes=int(adapt_remaining),
                adapt_mode=str(args.adapt_mode).upper().strip(),
                input_snapshots=adapt_start,
                window_tag="GLOBAL_ADAPT",
                on_committed_chunk=lambda snapshots, local_progress, global_ep: commit_recovery(
                    "adapt",
                    snapshots,
                    adapt_progress=int(adapt_done) + int(local_progress),
                    formal_progress=0,
                    global_episode=int(global_ep),
                ),
            )

        initial_snapshots = resume_snapshots if resume_phase == "formal" else adapt_snapshots
        workers = _initialize_formal_workers(
            args,
            log_paths,
            writers_worker,
            initial_snapshots,
            resume_formal=resume_phase == "formal",
        )

        if resume_bundle is not None:
            restore_rng_state(resume_controller.get("master_rng"))

        should_stop = int(args.patience_ready) > 0 and bool(
            resume_controller.get("stop_requested", False)
        )
        while not should_stop:
            ge_before = _get_min_global_episode_now(workers)

            if ge_before >= int(args.max_pbt_episodes):
                break

            remaining = int(args.max_pbt_episodes) - int(ge_before)
            if remaining <= 0:
                break
            chunk = int(min(int(args.chunk_len), remaining))
            chunk = int(max(1, chunk))

            _run_formal_chunk_parallel_spawn(
                args,
                workers,
                log_paths,
                chunk_len=int(chunk),
            )

            ge_after = _get_min_global_episode_now(workers)

            for ready_idx in _iter_crossed_boundary_indices(
                ge_before, ge_after, int(args.ready_every_episodes)
            ):
                ready_scores = _run_ready_eval_parallel_spawn(
                    args,
                    workers,
                    log_paths,
                    ready_idx=int(ready_idx),
                )

                _ = on_ready_boundary(
                    workers,
                    args,
                    int(ready_idx),
                    log_paths,
                    writer_global,
                    precomputed_scores=ready_scores,
                )

                _sync_master_workers_after_ready_boundary(workers)

                _pulse_tick_after_ready_boundary(workers)

                if soft_stop_update_on_ready(
                    workers,
                    args,
                    stop_state,
                    ready_idx=int(ready_idx),
                    log_paths=log_paths,
                    writer_global=writer_global,
                ):
                    should_stop = True
                    break

            commit_recovery(
                "formal",
                None,
                adapt_progress=int(args.adapt_episodes),
                formal_progress=max(
                    0,
                    int(ge_after) - int(args.adapt_episodes),
                ),
                global_episode=int(ge_after),
            )

            if hard_stop_reached(workers, args):
                break

        print(f"[PBT Master] finished. global_episode={_get_min_global_episode_now(workers)}")

    finally:
        for w in writers_worker.values():
            try:
                w.close()
            except Exception:
                pass
        try:
            writer_global.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_pbt_master()
