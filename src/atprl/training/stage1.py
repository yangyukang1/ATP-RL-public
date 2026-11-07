import json
from typing import Dict, Any, Sequence, Optional

import torch
import torch.optim as optim
import numpy as np
import random
import argparse
import time
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from atprl.envs.leo_sat import LeoSatEnv
from atprl.envs.topologies import resolve_topology
from atprl.nn.models import (
    ActorNetwork,
    CriticNetwork,
    PhaseActorGaussianLatent,
    PhaseActorVonMises,
    PhaseCritic,
)

from atprl.rl.buffers import ReplayBuffer, TrajectoryBuffer
from atprl.rl.algorithms import PPO
from atprl.rl.agent import DPRA_Agent
from atprl.rl.scheduler import build_scheduler_networks, build_dqn_learner, scheduler_log_summary
from atprl.config.kernel_signature import (
    kernel_signature_from_runtime,
    read_kernel_signature,
    validate_checkpoint_metadata,
    write_kernel_signature,
)
from atprl.training.recovery import (
    capture_rng_state,
    load_replay_buffer_state,
    load_trajectory_buffer_state,
    replay_buffer_state,
    restore_rng_state,
    trajectory_buffer_state,
)

from atprl.evaluation.utils import (
    FIXED_VALIDATION_SEEDS as VALIDATION_SEEDS,
    get_eval_config,
    summarize_metrics_by_seed,
    build_eval_curve_record,
    run_multi_seed_validation_metrics,
)

from contextlib import contextmanager


@contextmanager
def preserve_rng_state():
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _iter_crossed_boundary_indices(prev_value: int, new_value: int, every: int):
    every = int(every)
    if every <= 0:
        return
    prev_bucket = int(prev_value) // every
    new_bucket = int(new_value) // every
    for idx in range(prev_bucket + 1, new_bucket + 1):
        yield idx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an episode-aligned alternating training pipeline for the DPRA agent."
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=int(3.4e6),
        help="Total training steps for the entire pipeline.",
    )

    parser.add_argument(
        "--lr_ppo", type=float, default=3e-4, help="Default LR for PPO (as fallback)."
    )
    parser.add_argument(
        "--lr_dqn", type=float, default=5e-4, help="Learning rate for DQN Q-Network."
    )

    parser.add_argument(
        "--epsilon_start", type=float, default=0.05, help="Initial exploration rate."
    )
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final exploration rate.")
    parser.add_argument(
        "--epsilon_decay_steps",
        type=int,
        default=1,
        help="Total steps for epsilon to decay from start to end.",
    )

    parser.add_argument(
        "--dqn_batch_size", type=int, default=128, help="Batch size for DQN learning."
    )
    parser.add_argument("--dqn_update_freq", type=int, default=4, help="Update DQN every N steps.")
    parser.add_argument(
        "--target_update_freq", type=int, default=8, help="Frequency to update target network."
    )

    parser.add_argument(
        "--scheduler_mode",
        type=str,
        default="dqn",
        choices=["dqn", "random"],
        help="Discrete scheduling mode: 'dqn' keeps original epsilon-greedy/greedy DQN behavior; "
        "'random' enables Random Scheduling baseline.",
    )

    parser.add_argument(
        "--power_mode",
        type=str,
        default="ppo",
        choices=["ppo", "wmmse"],
        help=(
            "Power backend for Phase-I trainer. "
            "'ppo' keeps original Power-PPO behavior; "
            "'wmmse' replaces power generation with WMMSE while keeping DQN / phase logic unchanged."
        ),
    )

    parser.add_argument(
        "--wmmse_rb_index",
        type=int,
        default=0,
        help="RB index passed to the WMMSE solver (default: 0).",
    )
    parser.add_argument(
        "--wmmse_max_iters", type=int, default=50, help="Maximum WMMSE iterations per solve."
    )
    parser.add_argument(
        "--wmmse_tol", type=float, default=1e-6, help="Stopping tolerance for WMMSE solver."
    )
    parser.add_argument(
        "--wmmse_scipy_method",
        type=str,
        default="SLSQP",
        help="SciPy backend method for the inner WMMSE subproblem.",
    )
    parser.add_argument(
        "--wmmse_scipy_options",
        type=str,
        default=None,
        help="Optional JSON string for SciPy solver options, e.g. '{\"maxiter\": 100}'.",
    )
    parser.add_argument(
        "--wmmse_sic_repair_max_iters",
        type=int,
        default=30,
        help="Maximum SIC repair iterations after WMMSE solve.",
    )
    parser.add_argument(
        "--wmmse_sic_repair_step_ratio",
        type=float,
        default=0.05,
        help="Step ratio for common/private power repair when SIC is infeasible.",
    )

    parser.add_argument(
        "--ppo_opt_episodes",
        type=int,
        default=8,
        help="[Default] Number of episodes per PPO segment (fallback).",
    )
    parser.add_argument(
        "--ppo_buffer_size",
        type=int,
        default=1024,
        help="[Default] PPO trajectory buffer size (fallback).",
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=12, help="[Default] PPO epochs per update (fallback)."
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.012, help="[Default] PPO entropy coef (fallback)."
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="General discount factor.")

    parser.add_argument(
        "--eps_floor_start",
        type=float,
        default=0.0,
        help="Initial total minimum power share for all private streams.",
    )
    parser.add_argument(
        "--eps_floor_end", type=float, default=0.0, help="Final total minimum power share."
    )
    parser.add_argument(
        "--eps_floor_decay_steps",
        type=int,
        default=500000,
        help="Steps over which floor epsilon decays from start to end.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="alternating_episode_scheduling",
        help="Experiment name for logging.",
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Save the final model at the end of training."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--min_rate_req",
        type=float,
        default=0.1,
        help="Minimum rate requirement for each user in bps/Hz (QoS constraint).",
    )
    parser.add_argument(
        "--csi_error_variance",
        type=float,
        default=0.01,
        help="NMSE of CSI estimation error (0.01 matches你的基线).",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=0.95,
        help="Temporal correlation coefficient for small-scale fading (default: 0.95).",
    )

    parser.add_argument("--force_bs_gain_db", type=float, default=None)

    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=0,
        help="If >0, run a step-based validation every N training steps.",
    )
    parser.add_argument(
        "--eval_env_seed",
        type=int,
        default=24681357,
        help="Seed for the separate evaluation environment.",
    )
    parser.add_argument(
        "--eval_tag_steps",
        type=str,
        default="StepEval",
        help="TensorBoard tag for step-based evaluation curve.",
    )
    parser.add_argument(
        "--train_reseed_each_episode",
        action="store_true",
        help="If set, training env reseeds RNG at every env.reset().",
    )
    parser.add_argument(
        "--sat_user_gain_db_offsets",
        type=str,
        default=None,
        help='Comma-separated per-user dB offsets for LEO→GS large-scale gain, e.g. "0,-8,-12".',
    )

    parser.add_argument(
        "--num_ground_stations",
        type=int,
        default=4,
        help=(
            "Number of candidate ground users N. "
            "With use_fixed_positions=True, N must match the registered topology profile."
        ),
    )
    parser.add_argument(
        "--num_gs_per_rb",
        type=int,
        default=2,
        help=("Number of simultaneously served users K on each RB. Must satisfy 1 <= K <= N."),
    )

    parser.add_argument(
        "--cycle_dqn_before_power",
        type=int,
        default=4,
        help="Number of DQN episodes before a POWER-PPO segment.",
    )
    parser.add_argument(
        "--cycle_power",
        type=int,
        default=1,
        help="Number of POWER-PPO segments per cycle (each segment uses ppo_power_opt_episodes).",
    )
    parser.add_argument(
        "--cycle_dqn_before_phase",
        type=int,
        default=2,
        help="Number of DQN episodes before a PHASE-PPO segment.",
    )
    parser.add_argument(
        "--cycle_phase",
        type=int,
        default=1,
        help="Number of PHASE-PPO segments per cycle (each segment uses ppo_phase_opt_episodes).",
    )

    parser.add_argument(
        "--phase_delay_cycles",
        type=int,
        default=0,
        help="Delay PHASE-PPO for K full cycles (skip PHASE segments before K).",
    )
    parser.add_argument(
        "--phase_delay_episodes",
        type=int,
        default=0,
        help="Delay PHASE-PPO until E total (DQN+POWER) episodes are finished.",
    )

    parser.add_argument(
        "--eval_on_segment_end",
        action="store_true",
        help="If set, run a validation at the end of each segment (DQN/POWER/PHASE).",
    )

    parser.add_argument("--ppo_power_opt_episodes", type=int, default=None)
    parser.add_argument("--ppo_phase_opt_episodes", type=int, default=None)

    parser.add_argument("--ppo_power_epochs", type=int, default=None)
    parser.add_argument("--ppo_phase_epochs", type=int, default=None)

    parser.add_argument("--ppo_power_buffer_size", type=int, default=None)
    parser.add_argument("--ppo_phase_buffer_size", type=int, default=None)

    parser.add_argument("--entropy_coef_power", type=float, default=None)
    parser.add_argument("--entropy_coef_phase", type=float, default=None)

    parser.add_argument("--lr_power_ppo", type=float, default=None)
    parser.add_argument("--lr_phase_ppo", type=float, default=None)

    parser.add_argument(
        "--max_train_episodes",
        type=int,
        default=5000,
        help="Stop training when total finished episodes (DQN episodes + POWER PPO episodes) reach this number.",
    )
    parser.add_argument(
        "--eval_every_episodes",
        type=int,
        default=0,
        help="Run validation every N finished episodes.",
    )
    parser.add_argument(
        "--epsilon_decay_episodes",
        type=int,
        default=1000,
        help="Linearly decay DQN epsilon over this many finished episodes.",
    )

    parser.add_argument(
        "--local_epsilon_reheat_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, start a local DQN epsilon reheating schedule ONLY when a real P2->P1 "
            "mode transition happens. Default: disabled, keeps old global epsilon logic unchanged."
        ),
    )

    parser.add_argument(
        "--local_epsilon_reheat_start",
        type=float,
        default=0.50,
        help="Local reheating epsilon start value used after a real P2->P1 switch.",
    )

    parser.add_argument(
        "--local_epsilon_reheat_end",
        type=float,
        default=0.12,
        help="Local reheating epsilon floor after local decay finishes.",
    )

    parser.add_argument(
        "--local_epsilon_reheat_decay_dqn_episodes",
        type=int,
        default=64,
        help="How many P1-DQN episodes are used for local epsilon reheating decay.",
    )

    parser.add_argument(
        "--local_epsilon_reheat_log_tb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to log local reheating diagnostics to TensorBoard.",
    )

    parser.add_argument(
        "--phase1_only",
        action="store_true",
        help="Enable Phase-I: train only DQN + Power-PPO; disable Phase-PPO segments.",
    )
    parser.add_argument(
        "--phase1_fix_phase_per_episode",
        action="store_true",
        default=True,
        help="Keep RIS phase fixed within each episode.",
    )
    parser.add_argument(
        "--phase1_randomize_episode_phase",
        action="store_true",
        default=True,
        help="Randomize RIS phase once at the beginning of each episode (then keep fixed).",
    )

    parser.add_argument(
        "--phase1_phase_source",
        type=str,
        default="random_episode",
        choices=["random_episode", "phase_mean"],
        help=(
            "Phase injection source during Phase-I. "
            "'random_episode': (warmup) random phi injected at episode start; "
            "'phase_mean': (PBT P1) use frozen Phase-PPO mean(phi) deterministically."
        ),
    )

    parser.add_argument(
        "--phase1_phase_pass_every_step",
        action="store_true",
        default=False,
        help=(
            "If set, always pass phi to env.step() at every step. "
            "If not set, only pass phi at t==0 when phase is fixed within episode."
        ),
    )

    parser.add_argument(
        "--phase1_require_phase_model_when_mean",
        action="store_true",
        default=True,
        help="If set, phase1_phase_source=phase_mean requires a valid phase_actor (loaded externally by PBT).",
    )

    parser.add_argument(
        "--ris_element_amplitude",
        type=float,
        default=2000.0,
        help="Amplitude gain of each RIS element (default: 8500.0).",
    )

    parser.add_argument(
        "--uav_rx_antenna_gain_dBi",
        type=float,
        default=15.0,
        help="UAV receive antenna gain for Sat->UAV(RIS) link (dBi).",
    )
    parser.add_argument(
        "--ris_elem_max_amp_gain_dB",
        type=float,
        default=40.0,
        help="Per-element max amplification gain upper bound (dB, interpreted as POWER gain).",
    )

    parser.add_argument(
        "--active_ris_noise_power_dBm",
        type=float,
        default=-100.0,
        help="Active RIS noise power per element in dBm (default: -100.0).",
    )

    parser.add_argument(
        "--ris_reflection_power_limit_dBm",
        type=float,
        default=-5.0,
        help="Total active-RIS reflection power budget in dBm.",
    )
    parser.add_argument(
        "--dqn_scheduler_impl",
        choices=["enumerative"],
        default="enumerative",
        help="DQN architecture. The default preserves the legacy combination-enumeration scheduler.",
    )
    parser.add_argument("--topology_profile", type=str, default="paper_n4")

    parser.add_argument(
        "--disable_ris_path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "No-RIS baseline gate. If enabled, the environment physically disables the RIS-reflected path "
            "and also turns off active RIS noise. This keeps the same training pipeline but under a No-RIS PHY."
        ),
    )

    parser.add_argument(
        "--early_stop_patience_eval",
        type=int,
        default=0,
        help="If >0, stop training when eval return has not improved for this many eval rounds.",
    )
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.0,
        help="Minimal improvement in eval return to be considered better (filters noise).",
    )

    parser.set_defaults(
        phase1_only=True, phase1_fix_phase_per_episode=True, phase1_randomize_episode_phase=True
    )

    parser.add_argument(
        "--gae_lambda", type=float, default=0.95, help="GAE(lambda) for PPO advantage estimation."
    )

    parser.add_argument(
        "--fairness_alpha",
        type=float,
        default=0.0,
        help="Fairness alpha (0.0=SumRate, 1.0=Proportional Fair, 2.0=MaxMin-like).",
    )
    parser.add_argument(
        "--fairness_beta",
        type=float,
        default=0.01,
        help="EMA forgetting factor for historical average rate (0.01~0.05).",
    )
    parser.add_argument(
        "--fairness_epsilon",
        type=float,
        default=0.01,
        help="Small epsilon for fairness weights / log(historical_rates).",
    )

    parser.add_argument(
        "--fairness_lambda",
        type=float,
        default=0.0,
        help=(
            "Coefficient λ for average-deviation fairness penalty: "
            "reward = base_RSMA - λ * mean_k |R_bar_k - mean(R_bar)|."
        ),
    )
    parser.add_argument(
        "--sic_penalty",
        type=float,
        default=0.0,
        help="Additive penalty subtracted from RSMA reward when SIC fails (env_episode_lamda).",
    )

    parser.add_argument(
        "--sic_detection_threshold_dBm",
        type=float,
        default=-94.0,
        help="SIC detection threshold in dBm. Lower (more negative) is easier/looser; Higher is stricter.",
    )

    parser.add_argument(
        "--include_sr_incident_pow_groups",
        action="store_true",
        help="If set, append per-group incident power log feature (based on imperfect Sat->RIS CSI) to observation.",
    )

    parser.add_argument(
        "--learn_amp_weights",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Whether phase actor should output amplitude weights (3G latent). "
            "'true' => 3G, 'false' => 2G. "
            "'auto' keeps backward compatibility: follow include_sr_incident_pow_groups."
        ),
    )

    parser.add_argument(
        "--phase_model_include_mrt_features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether Phase-I should build MRT-appended observation when calling the frozen phase model. "
        "None means: inherit from outer controller / PBT.",
    )

    parser.add_argument(
        "--phase_model_state_mode",
        type=str,
        default="legacy_full",
        choices=["legacy_full", "compact_selected"],
        help="State mode used ONLY when Phase-I calls the frozen phase model. "
        "'legacy_full' keeps old logic: [full_obs ⊕ one-hot(selected) ⊕ power_shares]. "
        "'compact_selected' keeps only selected-user-related blocks plus power_shares.",
    )

    parser.add_argument(
        "--phase_model_compact_include_mrt_features",
        action="store_true",
        help="Only meaningful when --phase_model_state_mode compact_selected. "
        "Append selected-user MRT features into the compact frozen-phase-model state.",
    )

    parser.add_argument(
        "--phase_model_policy_dist",
        type=str,
        default="gaussian_latent",
        choices=["gaussian_latent", "von_mises"],
        help=(
            "Frozen phase-model policy distribution used by Phase-I when phase1_phase_source='phase_mean'. "
            "'gaussian_latent' keeps the original 2G/3G latent actor; "
            "'von_mises' enables theta-only circular phase model with 1G action dim."
        ),
    )

    parser.add_argument(
        "--bootstrap_phase2_final_dir",
        type=str,
        default=None,
        help=(
            "Optional Phase-II final model directory for standalone frozen-phase P1 calibration. "
            "You may pass either the Phase-II run root or its inner 'models' directory. "
            "When provided, main() will bootstrap q/power/phase from Phase-II finals and then "
            "run ONLY Phase-I (DQN + Power-PPO) under phase_mean context to measure P1 stabilization length."
        ),
    )

    parser.add_argument(
        "--bootstrap_phase2_final_strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strict load flag for external Phase-II final models. "
            "Recommended: keep True so any state-dim / architecture mismatch fails early."
        ),
    )

    parser.add_argument(
        "--bootstrap_reset_training_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, after loading external Phase-II final weights, reset P1 counters / scheduler / "
            "early-stop bookkeeping before calibration. Recommended: True."
        ),
    )

    args = parser.parse_args()

    args.num_ground_stations = int(args.num_ground_stations)
    args.num_gs_per_rb = int(args.num_gs_per_rb)

    if args.num_ground_stations <= 0:
        raise ValueError("--num_ground_stations must be a positive integer.")

    if args.num_gs_per_rb <= 0:
        raise ValueError("--num_gs_per_rb must be a positive integer.")

    if args.num_gs_per_rb > args.num_ground_stations:
        raise ValueError(
            f"Invalid topology: num_gs_per_rb={args.num_gs_per_rb} "
            f"cannot exceed num_ground_stations={args.num_ground_stations}."
        )

    resolve_topology(args.topology_profile, args.num_ground_stations)

    if args.sat_user_gain_db_offsets is not None:
        raw = str(args.sat_user_gain_db_offsets).strip()
        if raw != "":
            parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
            try:
                _ = [float(p) for p in parts]
            except Exception as e:
                raise ValueError(
                    "--sat_user_gain_db_offsets must be a comma-separated list of floats, "
                    f"got: {args.sat_user_gain_db_offsets}"
                ) from e

            if len(parts) not in (1, args.num_ground_stations):
                raise ValueError(
                    "--sat_user_gain_db_offsets must provide either exactly 1 value "
                    f"or exactly N={args.num_ground_stations} values; got {len(parts)} values."
                )

    if args.ppo_power_opt_episodes is None:
        args.ppo_power_opt_episodes = args.ppo_opt_episodes
    if args.ppo_power_buffer_size is None:
        args.ppo_power_buffer_size = args.ppo_buffer_size
    if args.ppo_power_epochs is None:
        args.ppo_power_epochs = args.ppo_epochs
    if args.entropy_coef_power is None:
        args.entropy_coef_power = args.entropy_coef
    if args.lr_power_ppo is None:
        args.lr_power_ppo = args.lr_ppo

    if args.ppo_phase_opt_episodes is None:
        args.ppo_phase_opt_episodes = args.ppo_opt_episodes
    if args.ppo_phase_buffer_size is None:
        args.ppo_phase_buffer_size = args.ppo_buffer_size
    if args.ppo_phase_epochs is None:
        args.ppo_phase_epochs = args.ppo_epochs
    if args.entropy_coef_phase is None:
        args.entropy_coef_phase = args.entropy_coef
    if args.lr_phase_ppo is None:
        args.lr_phase_ppo = args.lr_ppo

    effective_phase_model_mrt = _resolve_phase_model_include_mrt_features(args)

    if str(getattr(args, "phase_model_state_mode", "legacy_full")) not in (
        "legacy_full",
        "compact_selected",
    ):
        raise ValueError(
            f"Unsupported --phase_model_state_mode: {getattr(args, 'phase_model_state_mode', None)}"
        )

    if bool(getattr(args, "phase_model_compact_include_mrt_features", False)):
        if str(getattr(args, "phase_model_state_mode", "legacy_full")) != "compact_selected":
            raise ValueError(
                "--phase_model_compact_include_mrt_features only works with "
                "--phase_model_state_mode compact_selected."
            )
        if not bool(effective_phase_model_mrt):
            raise ValueError(
                "--phase_model_compact_include_mrt_features requires "
                "--phase_model_include_mrt_features (or inherited MRT-enabled phase model observation)."
            )

    phase_model_policy_dist = _resolve_phase_model_policy_dist(args)
    learn_amp_weights_resolved = _resolve_learn_amp_weights(args)

    if phase_model_policy_dist == "von_mises" and bool(learn_amp_weights_resolved):
        raise ValueError(
            "--phase_model_policy_dist von_mises requires --learn_amp_weights false. "
            "The frozen Von Mises phase model in this codebase is theta-only (1G) and does not use amp logits."
        )

    return args


def _build_phase_obs_for_phase_model(env, base_obs, user_ctx, power_ctx, include_mrt: bool):
    phase_obs = np.concatenate([base_obs, user_ctx, power_ctx], axis=0)

    if include_mrt:
        mrt = env.get_mrt_features()
        phase_obs = np.concatenate([phase_obs, mrt], axis=0)

    return phase_obs


def _resolve_learn_amp_weights(args) -> bool:
    v = getattr(args, "learn_amp_weights", "auto")
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    return bool(getattr(args, "include_sr_incident_pow_groups", False))


def _resolve_phase_model_policy_dist(args) -> str:
    mode = str(getattr(args, "phase_model_policy_dist", "gaussian_latent")).strip().lower()
    if mode not in ("gaussian_latent", "von_mises"):
        raise ValueError(
            f"Invalid --phase_model_policy_dist: {mode}. "
            f"Supported choices are ['gaussian_latent', 'von_mises']."
        )
    return mode


def _resolve_phase_model_include_mrt_features(args) -> bool:
    raw = getattr(args, "phase_model_include_mrt_features", None)
    if raw is None:
        return bool(getattr(args, "include_mrt_features", False))
    return bool(raw)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = float(np.sum(e))
    if s <= 1e-12:
        return np.ones_like(x, dtype=np.float64) / float(x.size)
    return (e / s).astype(np.float64)


def _wrap_phase(phi: np.ndarray, mode: str) -> np.ndarray:
    if mode is None:
        return phi
    mode = str(mode).lower()
    if mode == "pi":
        return ((phi + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    if mode == "2pi":
        return (phi % (2.0 * np.pi)).astype(np.float32)
    return phi.astype(np.float32)


def _phase_mean_to_ris_config_from_agent(
    agent,
    phase_state,
    ris_num_groups: int,
    device: str,
    *,
    force_uniform_amp: bool = False,
    phase_wrap_mode: str = "pi",
):
    if phase_state is None:
        raise ValueError("phase_state is None; cannot compute phase_mean RIS config.")

    if isinstance(phase_state, np.ndarray):
        phase_state_t = torch.as_tensor(phase_state, dtype=torch.float32, device=device)
    else:
        phase_state_t = phase_state.to(device)

    if phase_state_t.dim() == 1:
        phase_state_t = phase_state_t.unsqueeze(0)

    G = int(ris_num_groups)

    phase_policy_dist = getattr(agent, "phase_policy_dist", None)
    if phase_policy_dist is None:
        phase_policy_dist = getattr(agent.phase_actor, "policy_dist_name", "gaussian_latent")
    phase_policy_dist = str(phase_policy_dist).strip().lower()

    with torch.no_grad():
        dist = agent.phase_actor(phase_state_t)

    if phase_policy_dist == "von_mises":
        base = getattr(dist, "base_dist", None)
        if base is None or (not hasattr(base, "loc")):
            raise RuntimeError(
                "Frozen Von Mises phase model must expose base_dist.loc. "
                "Please ensure phase_actor returns Independent(VonMises(...), 1)."
            )

        theta_mean = base.loc.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if theta_mean.shape[0] != G:
            raise ValueError(
                f"Von Mises frozen phase-model mean dim mismatch: got {theta_mean.shape[0]}, expected G={G}."
            )

        phi = _wrap_phase(theta_mean, phase_wrap_mode)

        w = (np.ones((G,), dtype=np.float32) / float(G)).astype(np.float32)
        return phi, w

    mu = dist.mean.squeeze(0).detach().cpu().numpy()

    if mu.shape[0] < 2 * G:
        raise ValueError(f"PhaseActor mean dim {mu.shape[0]} < 2G={2 * G}")

    xy = mu[: 2 * G].reshape(G, 2)
    phi = np.arctan2(xy[:, 1], xy[:, 0]).astype(np.float32)
    phi = _wrap_phase(phi, phase_wrap_mode)

    w = None
    learn_amp = bool(getattr(agent, "learn_amp_weights", False))

    if bool(force_uniform_amp):
        w = (np.ones((G,), dtype=np.float32) / float(G)).astype(np.float32)
        return phi, w

    if learn_amp and (mu.shape[0] >= 3 * G):
        mu_amp = mu[2 * G : 3 * G]
        w = _softmax_np(mu_amp).astype(np.float32)

    return phi, w


def _phase_mean_to_ris_config_from_agent_local(env, agents, args, agent_id: str, info: dict):
    phase_state = info.get("phase_state", None)
    if phase_state is None:
        return None, None

    agent = agents[agent_id]

    ris_num_groups = getattr(env, "ris_num_groups", None)
    if ris_num_groups is None:
        ris_num_groups = getattr(args, "ris_num_groups", None)
    if ris_num_groups is None:
        raise ValueError("Cannot resolve ris_num_groups for phase_mean->RIS config mapping.")

    return _phase_mean_to_ris_config_from_agent(
        agent=agent,
        phase_state=phase_state,
        ris_num_groups=int(ris_num_groups),
        device=agent.device,
        force_uniform_amp=bool(getattr(args, "force_phase2_uniform_amp", False)),
        phase_wrap_mode=str(getattr(args, "phase_wrap_mode", "pi")),
    )


def _resolve_power_mode(args) -> str:
    mode = str(getattr(args, "power_mode", "ppo")).strip().lower()
    if mode not in ("ppo", "wmmse"):
        return "ppo"
    return mode


def _build_agent_power_backend_kwargs(args, env_obj) -> Dict[str, Any]:
    power_mode = _resolve_power_mode(args)

    kwargs: Dict[str, Any] = {
        "power_mode": power_mode,
    }

    if power_mode != "wmmse":
        return kwargs

    scipy_options = getattr(args, "wmmse_scipy_options", None)
    if isinstance(scipy_options, str):
        txt = scipy_options.strip()
        if txt == "":
            scipy_options = None
        else:
            try:
                scipy_options = json.loads(txt)
            except Exception:
                scipy_options = None

    kwargs.update(
        {
            "env": env_obj,
            "rb_index": int(getattr(args, "wmmse_rb_index", 0)),
            "max_wmmse_iters": int(getattr(args, "wmmse_max_iters", 50)),
            "wmmse_tol": float(getattr(args, "wmmse_tol", 1e-6)),
            "wmmse_scipy_method": str(getattr(args, "wmmse_scipy_method", "SLSQP")),
            "wmmse_scipy_options": scipy_options,
            "wmmse_sic_repair_max_iters": int(getattr(args, "wmmse_sic_repair_max_iters", 30)),
            "wmmse_sic_repair_step_ratio": float(
                getattr(args, "wmmse_sic_repair_step_ratio", 0.05)
            ),
        }
    )
    return kwargs


def _compute_phase_state_dim(
    phase_state_mode,
    phase_obs_layout,
    obs_dim_actual,
    n_users_total,
    k_users_per_rb,
    include_selected_mrt=False,
):
    phase_state_mode = str(phase_state_mode)
    obs_dim_actual = int(obs_dim_actual)
    n_users_total = int(n_users_total)
    k_users_per_rb = int(k_users_per_rb)
    power_action_dim = 1 + k_users_per_rb

    if phase_state_mode == "legacy_full":
        return int(obs_dim_actual + n_users_total + power_action_dim)

    if phase_state_mode != "compact_selected":
        raise ValueError(f"Unsupported phase_state_mode: {phase_state_mode}")

    if phase_obs_layout is None:
        raise ValueError("phase_obs_layout is required when phase_state_mode='compact_selected'.")

    n_users_layout = int(phase_obs_layout.get("n_users", n_users_total))
    if n_users_layout != n_users_total:
        raise ValueError(
            f"layout n_users mismatch: layout={n_users_layout}, n_users_total={n_users_total}"
        )

    casc_feats_per_user = int(phase_obs_layout["casc_feats_per_user"])
    direct_feats_per_user = int(phase_obs_layout["direct_feats_per_user"])
    uav_xy_feats_per_user = int(phase_obs_layout["uav_xy_feats_per_user"])
    hist_rate_feats_per_user = int(phase_obs_layout["hist_rate_feats_per_user"])

    uav_z_slice = phase_obs_layout["uav_z_slice"]
    uav_z_dim = int(uav_z_slice[1] - uav_z_slice[0])
    if uav_z_dim <= 0:
        raise ValueError(f"Invalid uav_z_dim from layout: {uav_z_dim}")

    ls_selected_dim = 1 + k_users_per_rb + k_users_per_rb

    total_dim = 0
    total_dim += k_users_per_rb * casc_feats_per_user
    total_dim += k_users_per_rb * direct_feats_per_user
    total_dim += ls_selected_dim
    total_dim += k_users_per_rb * uav_xy_feats_per_user
    total_dim += uav_z_dim
    total_dim += k_users_per_rb * hist_rate_feats_per_user

    if bool(include_selected_mrt):
        mrt_feats_per_user = int(phase_obs_layout.get("mrt_feats_per_user", 0))
        if mrt_feats_per_user <= 0:
            raise ValueError(
                "phase_model_compact_include_mrt_features=True but mrt_feats_per_user is missing/invalid in layout."
            )
        total_dim += k_users_per_rb * mrt_feats_per_user

    total_dim += power_action_dim
    return int(total_dim)


def run_power_ppo_optimization(
    env,
    agents,
    ppo_power,
    power_buffers,
    writer,
    global_step_base,
    args,
    current_eps_floor: float,
    global_episode_base: int,
    *,
    phase_source: str = "random_episode",
    phase_models: dict = None,
    fix_phase_per_episode: bool = True,
    pass_phase_every_step: bool = False,
    n_episodes: Optional[int] = None,
):
    power_mode = _resolve_power_mode(args)

    for agent in agents.values():
        if power_mode == "ppo":
            agent.power_actor.train()
            agent.power_critic.train()
        else:
            agent.power_actor.eval()
            agent.power_critic.eval()
        agent.phase_actor.eval()
        agent.phase_critic.eval()

    def _sample_random_phi_once(ep_seed_val):
        if hasattr(env, "sample_random_ris_phase"):
            return env.sample_random_ris_phase()

        G = getattr(args, "ris_num_groups", None)
        if G is None:
            any_agent = next(iter(agents.values()))
            G = int(getattr(any_agent, "ris_num_groups", 4))

        rng_local = np.random.default_rng(int(ep_seed_val) ^ 0xBADC0FFE)
        phi = rng_local.uniform(-np.pi, np.pi, size=(G,)).astype(np.float32)
        phi = _wrap_phase(phi, str(getattr(args, "phase_wrap_mode", "pi")))
        return phi

    steps_in_mode_b = 0
    episode_count = int(args.ppo_power_opt_episodes if n_episodes is None else n_episodes)
    progress_bar = tqdm(range(episode_count), desc=f"Mode B: Power ({power_mode})", leave=False)

    for ep_idx_in_segment in progress_bar:
        for buf in power_buffers.values():
            buf.clear()

        current_ep_seed = int(args.seed) + int(global_episode_base) + int(ep_idx_in_segment)
        observations = env.reset(seed=current_ep_seed)
        episode_return = 0.0

        terminated_flags = {aid: False for aid in env.agents}
        time_limit_flags = {aid: False for aid in env.agents}

        episode_phi = None
        episode_w = None
        if fix_phase_per_episode:
            if phase_source == "random_episode":
                episode_phi = _sample_random_phi_once(current_ep_seed)
                episode_w = None
            elif phase_source == "phase_mean":
                episode_phi = None
                episode_w = None

        for t in range(env.config["max_steps_per_episode"]):
            actions_to_env, action_info_map = {}, {}

            with torch.no_grad():
                for agent_id, agent in agents.items():
                    info = agent.select_action(
                        local_observation=observations[agent_id],
                        current_epsilon=0.0,
                        current_min_private_power_fraction=current_eps_floor,
                        deterministic_dqn=True,
                        deterministic_power=False if power_mode == "ppo" else True,
                        deterministic_phase=True,
                        scheduler_mode=str(getattr(args, "scheduler_mode", "dqn")).strip().lower(),
                        **_build_agent_power_backend_kwargs(args, env),
                    )
                    action_info_map[agent_id] = info

                    users = info["env_action"][0]
                    power_vec = info["env_action"][1]

                    phi_to_use = None
                    w_to_use = None

                    if phase_source == "random_episode":
                        if fix_phase_per_episode:
                            if pass_phase_every_step:
                                phi_to_use = episode_phi
                            else:
                                phi_to_use = episode_phi if (t == 0) else None
                        else:
                            phi_to_use = _sample_random_phi_once(current_ep_seed + t)
                        w_to_use = None

                    elif phase_source == "phase_mean":
                        if fix_phase_per_episode:
                            if (episode_phi is None) and (t == 0):
                                phi_mean, w_mean = _phase_mean_to_ris_config_from_agent_local(
                                    env, agents, args, agent_id, info
                                )
                                episode_phi, episode_w = phi_mean, w_mean

                            if pass_phase_every_step:
                                phi_to_use, w_to_use = episode_phi, episode_w
                            else:
                                if t == 0:
                                    phi_to_use, w_to_use = episode_phi, episode_w
                                else:
                                    phi_to_use, w_to_use = None, None
                        else:
                            phi_to_use, w_to_use = _phase_mean_to_ris_config_from_agent_local(
                                env, agents, args, agent_id, info
                            )

                    if phi_to_use is None:
                        actions_to_env[agent_id] = (users, power_vec)
                    else:
                        if w_to_use is None:
                            actions_to_env[agent_id] = (users, power_vec, phi_to_use)
                        else:
                            actions_to_env[agent_id] = (users, power_vec, phi_to_use, w_to_use)

            next_obs, rewards, dones, infos = env.step(actions_to_env)
            episode_end = bool(dones.get("__all__", False))

            steps_in_mode_b += 1
            episode_return += float(sum(rewards.values()))

            if power_mode == "ppo":
                for aid in env.agents:
                    info_env = infos.get(aid, {}) if isinstance(infos, dict) else {}
                    terminated_flags[aid] = bool(info_env.get("terminated", False))
                    time_limit_flags[aid] = bool(info_env.get("time_limit_truncated", episode_end))

                for agent_id in env.agents:
                    info = action_info_map[agent_id]
                    done_for_bootstrap = float(terminated_flags[agent_id])

                    power_buffers[agent_id].store(
                        state=info["power_state"],
                        action=info["power_action_raw"],
                        reward=rewards[agent_id],
                        done=done_for_bootstrap,
                        log_prob=info["power_log_prob"],
                        value=info["power_value"],
                    )

            if writer is not None and power_mode == "wmmse":
                for agent_id in env.agents:
                    info = action_info_map[agent_id]
                    is_feas = info.get("wmmse_is_feasible", None)
                    if is_feas is not None:
                        writer.add_scalar(
                            "WMMSE/IsFeasible",
                            1.0 if bool(is_feas) else 0.0,
                            global_step_base + steps_in_mode_b,
                        )

            observations = next_obs
            if episode_end:
                break

        if power_mode == "ppo":
            for agent_id, buffer in power_buffers.items():
                if len(buffer) == 0:
                    continue

                batch = {k: v.to(args.device) for k, v in buffer.get_batch().items()}

                with torch.no_grad():
                    nxt = agents[agent_id].select_action(
                        local_observation=observations[agent_id],
                        current_epsilon=0.0,
                        current_min_private_power_fraction=current_eps_floor,
                        deterministic_dqn=True,
                        deterministic_power=True,
                        deterministic_phase=True,
                        scheduler_mode=str(getattr(args, "scheduler_mode", "dqn")).strip().lower(),
                        **_build_agent_power_backend_kwargs(args, env),
                    )
                    next_power_state = nxt["power_state"].to(args.device)

                tl_trunc = bool(time_limit_flags[agent_id]) and (
                    not bool(terminated_flags[agent_id])
                )
                losses = ppo_power.update(batch, next_power_state, tl_trunc)

                if writer is not None and isinstance(losses, dict):
                    step_for_tb = global_episode_base + ep_idx_in_segment
                    for lk, lv in losses.items():
                        try:
                            writer.add_scalar(f"ModeB/Power_{lk}", float(lv), step_for_tb)
                        except Exception:
                            pass

        progress_bar.set_postfix({"EpRet": f"{episode_return:.2f}"})
        if writer is not None:
            writer.add_scalar(
                f"ModeB/Power_EpisodeReturn_{power_mode}",
                episode_return,
                global_episode_base + ep_idx_in_segment,
            )

    return steps_in_mode_b


def run_phase_ppo_optimization(
    env,
    agents,
    ppo_phase,
    phase_buffers,
    writer,
    global_step_base,
    args,
    current_eps_floor: float,
    global_episode_base: int = 0,
) -> int:
    for agent in agents.values():
        agent.phase_actor.train()
        agent.phase_critic.train()
        agent.power_actor.eval()
        agent.power_critic.eval()

    steps_in_mode_b = 0
    progress_bar = tqdm(range(args.ppo_phase_opt_episodes), desc="Mode B: Phase PPO", leave=False)

    for ep_idx, _ in enumerate(progress_bar):
        for buf in phase_buffers.values():
            buf.clear()

        current_ep_seed = int(args.seed) + int(global_episode_base) + int(ep_idx)
        observations = env.reset(seed=current_ep_seed)

        for _ in range(env.config["max_steps_per_episode"]):
            actions_to_env, action_info_map = {}, {}
            with torch.no_grad():
                for agent_id, agent in agents.items():
                    info = agent.select_action(
                        observations[agent_id],
                        current_epsilon=0.0,
                        current_min_private_power_fraction=current_eps_floor,
                    )
                    action_info_map[agent_id] = info
                    actions_to_env[agent_id] = info["env_action"]

            next_observations, rewards, dones, infos = env.step(actions_to_env)
            episode_end = bool(dones.get("__all__", False))

            terminated_flags = {}
            time_limit_flags = {}
            for aid in env.agents:
                info_env = infos.get(aid, {}) if isinstance(infos, dict) else {}
                terminated_flags[aid] = bool(info_env.get("terminated", False))
                time_limit_flags[aid] = bool(info_env.get("time_limit_truncated", episode_end))

            for agent_id in env.agents:
                info = action_info_map[agent_id]
                reward = rewards[agent_id]
                done_for_bootstrap = float(terminated_flags[agent_id])

                phase_buffers[agent_id].store(
                    info["phase_state"],
                    info["phase_joint_action"],
                    reward,
                    done_for_bootstrap,
                    info["phase_log_prob"],
                    info["phase_value"],
                )

            steps_in_mode_b += 1
            observations = next_observations

            buffer_is_full = (
                min(len(buf) for buf in phase_buffers.values()) >= args.ppo_phase_buffer_size
            )

            if buffer_is_full or episode_end:
                for agent_id, buffer in phase_buffers.items():
                    if len(buffer) == 0:
                        continue
                    batch = {k: v.to(args.device) for k, v in buffer.get_batch().items()}

                    with torch.no_grad():
                        nxt = agents[agent_id].select_action(
                            next_observations[agent_id], 0.0, current_eps_floor
                        )
                        next_phase_state = nxt["phase_state"].to(args.device)

                    tl_trunc = bool(time_limit_flags[agent_id]) and (
                        not bool(terminated_flags[agent_id])
                    )
                    losses = ppo_phase.update(batch, next_phase_state, tl_trunc)

                    if writer is not None:
                        writer.add_scalar(
                            "Loss/ModeB_PhasePPO_Actor",
                            losses["actor_loss"],
                            global_step_base + steps_in_mode_b,
                        )
                        writer.add_scalar(
                            "Loss/ModeB_PhasePPO_Critic",
                            losses["critic_loss"],
                            global_step_base + steps_in_mode_b,
                        )
                    buffer.clear()

            if episode_end:
                break

    return steps_in_mode_b


def run_step_based_validation(eval_env, agents, args) -> float:
    for ag in agents.values():
        ag.power_actor.eval()
        ag.power_critic.eval()
        ag.phase_actor.eval()
        ag.phase_critic.eval()

    obs = eval_env.reset()
    G = 0.0

    phase_source = getattr(args, "phase1_phase_source", "random_episode")
    pass_every_step = bool(getattr(args, "phase1_phase_pass_every_step", False))

    episode_phi = None
    episode_w = None
    if phase_source == "random_episode":
        if getattr(args, "phase1_fix_phase_per_episode", False) and getattr(
            args, "phase1_randomize_episode_phase", False
        ):
            episode_phi = np.random.uniform(-np.pi, np.pi, size=eval_env.ris_num_groups).astype(
                np.float32
            )
            episode_w = None
    elif phase_source == "phase_mean":
        episode_phi, episode_w = None, None
    else:
        raise ValueError(f"Unknown phase1_phase_source={phase_source}")

    for t in range(eval_env.config["max_steps_per_episode"]):
        actions = {}

        for aid, ag in agents.items():
            info = ag.select_action(
                local_observation=obs[aid],
                current_epsilon=0.0,
                current_min_private_power_fraction=0.0,
                deterministic_dqn=True,
                deterministic_power=True,
                deterministic_phase=True,
                scheduler_mode=str(getattr(args, "scheduler_mode", "dqn")).strip().lower(),
            )

            users = list(info["env_action"][0])
            pvec = info["env_action"][1]

            phi_to_use = None
            w_to_use = None

            if phase_source == "random_episode":
                phi_to_use = episode_phi if (t == 0 and episode_phi is not None) else None
                w_to_use = None
            elif phase_source == "phase_mean":
                if getattr(args, "phase1_fix_phase_per_episode", False) and (not pass_every_step):
                    G_groups = eval_env.config.get(
                        "ris_num_groups", getattr(args, "ris_num_groups", 0)
                    )

                    if t == 0 and episode_phi is None:
                        episode_phi, episode_w = _phase_mean_to_ris_config_from_agent(
                            ag,
                            info.get("phase_state", None),
                            G_groups,
                            args.device,
                        )
                    if t == 0:
                        phi_to_use, w_to_use = episode_phi, episode_w
                    else:
                        phi_to_use, w_to_use = None, None
                else:
                    G_groups = eval_env.config.get(
                        "ris_num_groups", getattr(args, "ris_num_groups", 0)
                    )
                    phi_to_use, w_to_use = _phase_mean_to_ris_config_from_agent(
                        ag,
                        info.get("phase_state", None),
                        G_groups,
                        args.device,
                    )

            if phi_to_use is None:
                actions[aid] = (users, pvec)
            else:
                actions[aid] = (
                    (users, pvec, phi_to_use)
                    if (w_to_use is None)
                    else (users, pvec, phi_to_use, w_to_use)
                )

        next_obs, rewards, dones, infos = eval_env.step(actions)
        G += float(sum(rewards.values()))
        obs = next_obs
        if bool(dones.get("__all__", False)):
            break

    return float(G)


def _jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    s = float(np.sum(x))
    ssq = float(np.sum(x * x))
    n = int(x.size)
    return float((s * s) / (n * ssq + 1e-12))


def run_step_based_validation_metrics(eval_env, agents, args) -> dict:
    for ag in agents.values():
        ag.power_actor.eval()
        ag.power_critic.eval()
        ag.phase_actor.eval()
        ag.phase_critic.eval()

    obs = eval_env.reset()
    episode_return = 0.0
    n_steps = 0

    sum_rate_total_over_steps = 0.0
    sum_rate_per_user_over_steps = None

    phase_source = getattr(args, "phase1_phase_source", "random_episode")
    pass_every_step = bool(getattr(args, "phase1_phase_pass_every_step", False))

    episode_phi = None
    episode_w = None
    if phase_source == "random_episode":
        if getattr(args, "phase1_fix_phase_per_episode", False) and getattr(
            args, "phase1_randomize_episode_phase", False
        ):
            current_eval_seed = int(eval_env.config.get("random_seed", args.seed))
            phi_rng = np.random.default_rng(current_eval_seed ^ 0x9E3779B9)

            episode_phi = phi_rng.uniform(-np.pi, np.pi, size=eval_env.ris_num_groups).astype(
                np.float32
            )
            episode_phi = _wrap_phase(episode_phi, str(getattr(args, "phase_wrap_mode", "pi")))
            episode_w = None
    elif phase_source == "phase_mean":
        episode_phi, episode_w = None, None
    else:
        raise ValueError(f"Unknown phase1_phase_source={phase_source}")

    for t in range(eval_env.config["max_steps_per_episode"]):
        actions = {}

        for aid, ag in agents.items():
            info = ag.select_action(
                local_observation=obs[aid],
                current_epsilon=0.0,
                current_min_private_power_fraction=0.0,
                deterministic_dqn=True,
                deterministic_power=True,
                deterministic_phase=True,
                scheduler_mode=str(getattr(args, "scheduler_mode", "dqn")).strip().lower(),
            )

            users = list(info["env_action"][0])
            pvec = info["env_action"][1]

            phi_to_use = None
            w_to_use = None

            if phase_source == "random_episode":
                phi_to_use = episode_phi if (t == 0 and episode_phi is not None) else None
                w_to_use = None

            elif phase_source == "phase_mean":
                if getattr(args, "phase1_fix_phase_per_episode", False) and (not pass_every_step):
                    G_groups = eval_env.config.get(
                        "ris_num_groups", getattr(args, "ris_num_groups", 0)
                    )

                    if t == 0 and episode_phi is None:
                        episode_phi, episode_w = _phase_mean_to_ris_config_from_agent(
                            ag,
                            info.get("phase_state", None),
                            G_groups,
                            args.device,
                            force_uniform_amp=bool(
                                getattr(args, "force_phase2_uniform_amp", False)
                            ),
                        )
                    if t == 0:
                        phi_to_use, w_to_use = episode_phi, episode_w
                    else:
                        phi_to_use, w_to_use = None, None
                else:
                    G_groups = eval_env.config.get(
                        "ris_num_groups", getattr(args, "ris_num_groups", 0)
                    )
                    phi_to_use, w_to_use = _phase_mean_to_ris_config_from_agent(
                        ag,
                        info.get("phase_state", None),
                        G_groups,
                        args.device,
                        force_uniform_amp=bool(getattr(args, "force_phase2_uniform_amp", False)),
                    )

            if phi_to_use is None:
                actions[aid] = (users, pvec)
            else:
                actions[aid] = (
                    (users, pvec, phi_to_use)
                    if (w_to_use is None)
                    else (users, pvec, phi_to_use, w_to_use)
                )

        next_obs, rewards, dones, infos = eval_env.step(actions)
        episode_return += float(sum(rewards.values()))
        n_steps += 1

        step_rate_per_user = None
        for _aid, info_env in (infos or {}).items():
            if not isinstance(info_env, dict):
                continue

            if "rates_macro_vec" in info_env:
                v = np.asarray(info_env["rates_macro_vec"], dtype=np.float64).reshape(-1)
            elif "total_rates_per_user" in info_env:
                v = np.asarray(info_env["total_rates_per_user"], dtype=np.float64).reshape(-1)
            else:
                continue

            if step_rate_per_user is None:
                step_rate_per_user = v.copy()
            else:
                step_rate_per_user += v

        if step_rate_per_user is not None:
            if sum_rate_per_user_over_steps is None:
                sum_rate_per_user_over_steps = np.zeros_like(step_rate_per_user, dtype=np.float64)
            sum_rate_per_user_over_steps += step_rate_per_user
            sum_rate_total_over_steps += float(np.sum(step_rate_per_user))

        obs = next_obs
        if bool(dones.get("__all__", False)):
            break

    T = max(1, int(n_steps))

    if sum_rate_per_user_over_steps is None:
        try:
            avg_rates = np.asarray(eval_env.historical_avg_rates, dtype=np.float64).reshape(-1)
        except Exception:
            avg_rates = np.zeros(
                int(eval_env.config.get("num_ground_stations", 1)), dtype=np.float64
            )
        avg_sum_rate = float(np.sum(avg_rates))
    else:
        avg_rates = (sum_rate_per_user_over_steps / float(T)).reshape(-1)
        avg_sum_rate = float(sum_rate_total_over_steps / float(T))

    min_user_avg_rate = float(np.min(avg_rates)) if avg_rates.size > 0 else 0.0
    jfi = _jain_index(avg_rates)

    return {
        "undiscounted_return": float(episode_return),
        "avg_sum_rate": float(avg_sum_rate),
        "min_user_avg_rate": float(min_user_avg_rate),
        "jain_index": float(jfi),
        "n_steps": int(T),
    }


def run_robust_validation_metrics(eval_env_config, agents, args, seed_list) -> dict:
    seed_list = list(map(int, list(seed_list) if seed_list is not None else []))
    metrics_all = []

    with preserve_rng_state():
        for s in seed_list:
            cfg = dict(eval_env_config)
            cfg["random_seed"] = s
            cfg["reseed_each_episode"] = True
            eval_env = LeoSatEnv(config=cfg)

            m = run_step_based_validation_metrics(eval_env, agents, args)
            metrics_all.append(m)

    metric_keys = ["undiscounted_return", "avg_sum_rate", "min_user_avg_rate", "jain_index"]
    metrics_pack = summarize_metrics_by_seed(metrics_all, metric_keys)

    return {
        "seed_list": seed_list,
        "metrics": metrics_pack,
    }


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


def run_robust_validation(eval_env_config, agents, args, seed_list):
    pack = run_robust_validation_metrics(eval_env_config, agents, args, seed_list)
    return float(pack["metrics"]["undiscounted_return"]["mean"])


def _get_phase1_eval_policy_name(args) -> str:
    scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()
    power_mode = _resolve_power_mode(args)

    if scheduler_mode == "random":
        if power_mode == "wmmse":
            return "random_sched_wmmse_power_phase1_local_phase_source"
        return "random_sched_mean_power_phase1_local_phase_source"

    if power_mode == "wmmse":
        return "greedy_dqn_wmmse_power_phase1_local_phase_source"
    return "greedy_dqn_mean_power_phase1_local_phase_source"


class Phase1TrainerPBT:
    """Train DQN and Power-PPO while holding the phase controller fixed."""

    def __init__(self, args, *, log_writer=None):
        self.args = args
        self.device = args.device
        self.writer = log_writer

        self.scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()
        if self.scheduler_mode not in ("dqn", "random"):
            self.scheduler_mode = "dqn"
        self.random_scheduling_enabled = self.scheduler_mode == "random"

        self.local_epsilon_reheat_enable = bool(getattr(args, "local_epsilon_reheat_enable", False))
        self.local_epsilon_reheat_start = float(getattr(args, "local_epsilon_reheat_start", 0.50))
        self.local_epsilon_reheat_end = float(getattr(args, "local_epsilon_reheat_end", 0.12))
        self.local_epsilon_reheat_decay_dqn_episodes = int(
            getattr(args, "local_epsilon_reheat_decay_dqn_episodes", 64)
        )
        self.local_epsilon_reheat_log_tb = bool(getattr(args, "local_epsilon_reheat_log_tb", True))

        self.local_epsilon_reheat_active = False
        self.local_epsilon_reheat_progress_dqn = 0
        self.local_epsilon_reheat_last_trigger = None

        env_config = {
            "num_ground_stations": int(args.num_ground_stations),
            "num_resource_blocks": 1,
            "num_gs_per_rb": int(args.num_gs_per_rb),
            "deployment_area_size_m": 400.0,
            "satellite_max_power_dBm": 50.0,
            "terrestrial_cell_power_dBm": 35.0,
            "noise_power_dBm": -114.0,
            "sic_detection_threshold_dBm": args.sic_detection_threshold_dBm,
            "carrier_frequency_GHz": 4.0,
            "satellite_antenna_gain_dBi": 24.3,
            "ground_station_antenna_gain_dBi": 24.3,
            "satellite_altitude_m": 600e3,
            "log_normal_shadowing_variance_dB": 8.0,
            "rician_k_factor": 10.0,
            "k_factor_ris_to_gs_linear": 10.0,
            "k_factor_sat_to_ris_linear": 30.0,
            "max_doppler_frequency_Hz": 50.0,
            "channel_update_interval_s": 0.04,
            "num_micro_steps": 20,
            "min_rate_requirement_bps_hz": args.min_rate_req,
            "max_steps_per_episode": 1024,
            "ris_boost_lambda": 0.0,
            "random_seed": args.seed,
            "use_fixed_positions": True,
            "topology_profile": args.topology_profile,
            "rho": args.rho,
            "csi_error_variance": args.csi_error_variance,
            "force_bs_gain_db": args.force_bs_gain_db,
            "reseed_each_episode": args.train_reseed_each_episode,
            "sat_user_gain_db_offsets": args.sat_user_gain_db_offsets,
            "include_angle_features": False,
            "ris_num_elem_x": int(getattr(args, "ris_num_elem_x", 16)),
            "ris_num_elem_y": int(getattr(args, "ris_num_elem_y", 16)),
            "ris_power_control_mode": "power_budget",
            "ris_reflection_power_limit_dBm": float(args.ris_reflection_power_limit_dBm),
            "ris_elem_spacing_ratio": 0.5,
            "active_ris_enabled": True,
            "active_ris_noise_power_dBm": args.active_ris_noise_power_dBm,
            "disable_ris_path": bool(getattr(args, "disable_ris_path", False)),
            "uav_initial_position_xyz": [0.0, 0.0, 80.0],
            "uav_velocity_per_step_xyz": [0.282842712, 0.282842712, 0.0],
            "uav_rx_antenna_gain_dBi": args.uav_rx_antenna_gain_dBi,
            "ris_elem_max_amp_gain_dB": args.ris_elem_max_amp_gain_dB,
            "fairness_alpha": args.fairness_alpha,
            "fairness_beta": args.fairness_beta,
            "fairness_epsilon": args.fairness_epsilon,
            "fairness_lambda": args.fairness_lambda,
            "sic_penalty": float(getattr(args, "sic_penalty", 0.0)),
            "include_sr_incident_pow_groups": args.include_sr_incident_pow_groups,
            "include_mrt_features": False,
        }

        env_config_base = dict(env_config)
        env_config_base["include_mrt_features"] = False

        mrt_tau = float(getattr(self.args, "mrt_casc_abs_tau", 1e-12))
        env_config_base["mrt_casc_abs_tau"] = mrt_tau

        include_mrt_phase_model = _resolve_phase_model_include_mrt_features(self.args)

        env_config_full = dict(env_config_base)
        env_config_full["include_mrt_features"] = include_mrt_phase_model
        env_config_full["mrt_casc_abs_tau"] = mrt_tau

        _probe_env_base = LeoSatEnv(config=env_config_base)
        _probe_base = _probe_env_base.reset()
        _any_aid = next(iter(_probe_base))

        base_obs_dim = None
        if hasattr(_probe_env_base, "get_base_obs_dim") and callable(
            getattr(_probe_env_base, "get_base_obs_dim")
        ):
            base_obs_dim = _probe_env_base.get_base_obs_dim()
        if base_obs_dim is None:
            base_obs_dim = len(_probe_base[_any_aid])
        dqn_obs_layout = _probe_env_base.get_obs_layout_spec(include_mrt_features=False)
        if dqn_obs_layout is None:
            raise RuntimeError("Base observation layout is unavailable after env.reset()")

        if hasattr(_probe_env_base, "close") and callable(getattr(_probe_env_base, "close")):
            _probe_env_base.close()

        self.env = LeoSatEnv(config=env_config_full)

        _probe_full = self.env.reset()
        _any_aid_full = next(iter(_probe_full))
        full_obs_dim = len(_probe_full[_any_aid_full])

        self.base_obs_dim = int(base_obs_dim)
        self.full_obs_dim = int(full_obs_dim)

        self.n_users_total = getattr(
            self.env, "num_ground_stations", self.env.config["num_ground_stations"]
        )
        self.k_users_per_rb = getattr(self.env, "num_gs_per_rb", self.env.config["num_gs_per_rb"])
        self.G_groups = getattr(
            self.env, "ris_num_groups", self.env.config.get("ris_num_groups", 0)
        )

        scheduler_build = build_scheduler_networks(
            state_dim=int(base_obs_dim),
            n_users=self.n_users_total,
            k_users_per_rb=self.k_users_per_rb,
            obs_layout=dqn_obs_layout,
            implementation=getattr(args, "dqn_scheduler_impl", "enumerative"),
            device=self.device,
        )
        self.scheduler_build = scheduler_build
        self.discrete_actions_map = scheduler_build.discrete_actions_map
        dqn_action_dim = scheduler_build.network_output_dim

        print(
            f"[Phase1TrainerPBT/Topology] num_ground_stations={self.n_users_total}, "
            f"num_gs_per_rb={self.k_users_per_rb}, "
            f"{scheduler_log_summary(scheduler_build, self.n_users_total, self.k_users_per_rb)}, "
            f"discrete_actions_map={self.discrete_actions_map}"
        )

        if self.writer is not None:
            self.writer.add_text(
                "Config/Topology",
                scheduler_log_summary(scheduler_build, self.n_users_total, self.k_users_per_rb),
                0,
            )
            self.writer.add_scalar("Config/NumGroundStations", float(self.n_users_total), 0)
            self.writer.add_scalar("Config/NumGsPerRb", float(self.k_users_per_rb), 0)
            self.writer.add_scalar("Config/DQNActionDim", float(dqn_action_dim), 0)

        dqn_state_dim = self.base_obs_dim
        power_state_dim = self.n_users_total + self.base_obs_dim
        power_action_dim = 1 + self.k_users_per_rb

        phase_model_state_mode = str(getattr(self.args, "phase_model_state_mode", "legacy_full"))
        phase_model_compact_include_mrt_features = bool(
            getattr(self.args, "phase_model_compact_include_mrt_features", False)
        )

        if phase_model_compact_include_mrt_features and (not include_mrt_phase_model):
            raise ValueError(
                "--phase_model_compact_include_mrt_features requires "
                "Phase-I frozen phase-model observation to include MRT features."
            )

        phase_obs_layout = None
        if phase_model_state_mode == "compact_selected":
            if not hasattr(self.env, "get_obs_layout_spec") or not callable(
                getattr(self.env, "get_obs_layout_spec")
            ):
                raise RuntimeError(
                    "Current LeoSatEnv does not provide get_obs_layout_spec(); "
                    "compact_selected phase-model state cannot be enabled."
                )

            phase_obs_layout = self.env.get_obs_layout_spec(
                include_mrt_features=include_mrt_phase_model
            )
            if phase_obs_layout is None:
                raise RuntimeError(
                    "LeoSatEnv.get_obs_layout_spec() returned None. "
                    "Please ensure env.reset() / observation building has run before probing layout."
                )

        phase_state_dim = _compute_phase_state_dim(
            phase_state_mode=phase_model_state_mode,
            phase_obs_layout=phase_obs_layout,
            obs_dim_actual=int(self.full_obs_dim),
            n_users_total=int(self.n_users_total),
            k_users_per_rb=int(self.k_users_per_rb),
            include_selected_mrt=phase_model_compact_include_mrt_features,
        )

        self.dqn_state_dim = int(dqn_state_dim)
        self.power_state_dim = int(power_state_dim)
        self.phase_state_dim = int(phase_state_dim)
        self.kernel_signature = kernel_signature_from_runtime(
            args=self.args,
            env=self.env,
            scheduler_build=scheduler_build,
            dqn_state_dim=dqn_state_dim,
            power_state_dim=power_state_dim,
            power_action_dim=power_action_dim,
            phase_state_dim=phase_state_dim,
        )

        self.phase_model_state_mode = phase_model_state_mode
        self.phase_model_compact_include_mrt_features = phase_model_compact_include_mrt_features
        self.phase_model_policy_dist = _resolve_phase_model_policy_dist(args)
        self.phase_obs_layout = None if phase_obs_layout is None else dict(phase_obs_layout)

        self.q_network = scheduler_build.q_network
        self.target_q_network = scheduler_build.target_q_network

        self.power_actor = ActorNetwork(
            state_dim=power_state_dim,
            action_dim=power_action_dim,
            max_power_watt=self.env.satellite_max_power_watt,
        ).to(self.device)
        self.power_critic = CriticNetwork(state_dim=power_state_dim).to(self.device)

        learn_amp_weights_resolved = _resolve_learn_amp_weights(args)
        phase_model_policy_dist = _resolve_phase_model_policy_dist(args)

        if phase_model_policy_dist == "gaussian_latent":
            self.phase_actor = PhaseActorGaussianLatent(
                state_dim=phase_state_dim,
                num_phase_groups=self.G_groups,
                learn_amp=learn_amp_weights_resolved,
            ).to(self.device)
        elif phase_model_policy_dist == "von_mises":
            if bool(learn_amp_weights_resolved):
                raise ValueError(
                    "Phase1TrainerPBT build error: phase_model_policy_dist='von_mises' requires learn_amp_weights=False."
                )
            self.phase_actor = PhaseActorVonMises(
                state_dim=phase_state_dim,
                num_phase_groups=self.G_groups,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported phase_model_policy_dist: {phase_model_policy_dist}")

        self.phase_critic = PhaseCritic(state_dim=phase_state_dim).to(self.device)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr_dqn)
        self.power_actor_optim = optim.Adam(self.power_actor.parameters(), lr=args.lr_power_ppo)
        self.power_critic_optim = optim.Adam(self.power_critic.parameters(), lr=args.lr_power_ppo)

        self.replay_buffer = ReplayBuffer(
            capacity=int(1e5),
            state_dim=dqn_state_dim,
            action_dim=scheduler_build.replay_action_dim,
        )
        self.power_buffers = {aid: TrajectoryBuffer() for aid in self.env.agents}

        self.dqn_learner = build_dqn_learner(
            scheduler_build, self.q_optimizer, gamma=args.gamma, tau=0.005
        )
        self.ppo_power = PPO(
            self.power_actor,
            self.power_critic,
            self.power_actor_optim,
            self.power_critic_optim,
            {
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_epsilon": 0.2,
                "ppo_epochs": args.ppo_power_epochs,
                "entropy_coef": args.entropy_coef_power,
            },
        )

        self.agents = {
            aid: DPRA_Agent(
                q_network=self.q_network,
                power_actor=self.power_actor,
                power_critic=self.power_critic,
                phase_actor=self.phase_actor,
                phase_critic=self.phase_critic,
                discrete_actions_map=self.discrete_actions_map,
                n_users=self.n_users_total,
                dqn_state_dim=dqn_state_dim,
                max_power_watt=self.env.satellite_max_power_watt,
                device=self.device,
                include_sr_incident_pow_groups=bool(args.include_sr_incident_pow_groups),
                learn_amp_weights=learn_amp_weights_resolved,
                ris_num_groups=self.G_groups,
                phase_state_mode=self.phase_model_state_mode,
                phase_obs_layout=self.phase_obs_layout,
                phase_compact_include_mrt_features=self.phase_model_compact_include_mrt_features,
                dqn_scheduler_impl=scheduler_build.implementation,
                k_users_per_rb=self.k_users_per_rb,
                phase_policy_dist=self.phase_model_policy_dist,
            )
            for aid in self.env.agents
        }

        self.total_steps = 0
        self.episodes_done = 0
        self.current_epsilon = args.epsilon_start

        self.segment_order = [("DQN", args.cycle_dqn_before_power), ("POWER", args.cycle_power)]
        self.segment_idx = 0
        self.segment_ep_remaining = self.segment_order[0][1]
        self.power_episode_offset = 0

        self.eval_env_config = get_eval_config(
            base_config=env_config_base,
            mode="standard",
            profile="phase1",
        )
        if getattr(args, "eval_env_seed", None) is not None:
            self.eval_env_config["random_seed"] = args.eval_env_seed

        writer_log_dir = getattr(self.writer, "log_dir", None) if self.writer is not None else None
        if writer_log_dir is not None:
            self.log_dir = Path(writer_log_dir)
        else:
            self.log_dir = Path(f"Try_final_episode_lamda_PBT_runs/{args.exp_name}_phase1_trainer")

        self.curve_json_path = self.log_dir / "convergence_curve_phase1.json"
        self.model_dir = self.log_dir / "models"

        self.best_eval_return = -float("inf")
        self.best_eval_episode = 0
        self.eval_no_improve_count = 0
        self.training_should_stop = False

    def set_phase_source(self, source: str):
        self.args.phase1_phase_source = source

    def attach_phase_models(
        self, phase_actor_state_dict=None, phase_critic_state_dict=None, strict: bool = True
    ):
        def _infer_first_linear_in_dim(sd):
            if sd is None:
                return None, None
            k = next(
                (
                    kk
                    for kk, vv in sd.items()
                    if kk.endswith("weight") and getattr(vv, "ndim", 0) == 2
                ),
                None,
            )
            if k is None:
                return None, None
            return k, int(sd[k].shape[1])

        if phase_actor_state_dict is not None:
            cur_sd = self.phase_actor.state_dict()
            ckpt_k, ckpt_in_dim = _infer_first_linear_in_dim(phase_actor_state_dict)
            cur_k, cur_in_dim = _infer_first_linear_in_dim(cur_sd)

            if ckpt_in_dim is not None and cur_in_dim is not None and ckpt_in_dim != cur_in_dim:
                raise RuntimeError(
                    "[Phase1TrainerPBT] Frozen phase_actor input-dim mismatch.\n"
                    f"  - checkpoint in_dim = {ckpt_in_dim} (key={ckpt_k})\n"
                    f"  - current   in_dim = {cur_in_dim} (key={cur_k})\n"
                    f"  - phase_model_state_mode = {getattr(self, 'phase_model_state_mode', None)}\n"
                    f"  - phase_model_compact_include_mrt_features = {getattr(self, 'phase_model_compact_include_mrt_features', None)}\n"
                    f"  - phase_model_policy_dist = {getattr(self, 'phase_model_policy_dist', None)}\n"
                    f"  - full_obs_dim = {getattr(self, 'full_obs_dim', None)}\n"
                    f"  - phase_state_dim = {getattr(self, 'phase_state_dim', None)}\n"
                    "\n"
                    "This usually means P1 frozen-phase observation mode does not match the Phase-II checkpoint."
                )

            self.phase_actor.load_state_dict(phase_actor_state_dict, strict=strict)

        if phase_critic_state_dict is not None:
            self.phase_critic.load_state_dict(phase_critic_state_dict, strict=strict)

        for p in self.phase_actor.parameters():
            p.requires_grad = False
        for p in self.phase_critic.parameters():
            p.requires_grad = False

        self.phase_actor.eval()
        self.phase_critic.eval()

    def reset_buffers(
        self, *, clear_replay: bool = True, clear_power: bool = True, reset_scheduler: bool = True
    ):
        if clear_replay:
            self.replay_buffer.clear()
        if clear_power:
            for b in self.power_buffers.values():
                b.clear()
        if reset_scheduler:
            self.segment_idx = 0
            self.segment_ep_remaining = self.segment_order[0][1]
            self.power_episode_offset = 0

    def checkpoint_dict(self) -> dict:
        return {
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "power_actor": self.power_actor.state_dict(),
            "power_critic": self.power_critic.state_dict(),
            "phase_actor": self.phase_actor.state_dict(),
            "phase_critic": self.phase_critic.state_dict(),
            "meta": {
                "dqn_scheduler_impl": self.scheduler_build.implementation,
                "kernel_signature": dict(self.kernel_signature),
            },
            "q_optimizer": self.q_optimizer.state_dict(),
            "power_actor_optim": self.power_actor_optim.state_dict(),
            "power_critic_optim": self.power_critic_optim.state_dict(),
            "buffers": {
                "replay": None
                if len(self.replay_buffer) == 0
                else replay_buffer_state(self.replay_buffer),
                "power": {
                    aid: trajectory_buffer_state(buffer)
                    for aid, buffer in self.power_buffers.items()
                },
            },
            "rng": capture_rng_state(env=self.env),
            "counters": {
                "total_steps": int(self.total_steps),
                "episodes_done": int(self.episodes_done),
                "current_epsilon": float(self.current_epsilon),
                "segment_idx": int(self.segment_idx),
                "segment_ep_remaining": int(self.segment_ep_remaining),
                "power_episode_offset": int(self.power_episode_offset),
                "local_reheat_active": bool(self.local_epsilon_reheat_active),
                "local_reheat_progress_dqn": int(self.local_epsilon_reheat_progress_dqn),
                "local_reheat_last_trigger": self.local_epsilon_reheat_last_trigger,
                "best_eval_return": float(self.best_eval_return),
                "best_eval_episode": int(self.best_eval_episode),
                "eval_no_improve_count": int(self.eval_no_improve_count),
                "training_should_stop": bool(self.training_should_stop),
            },
        }

    def _compute_global_dqn_epsilon(self) -> float:
        decay_eps = int(getattr(self.args, "epsilon_decay_episodes", 0))
        eps_start = float(getattr(self.args, "epsilon_start", 0.05))
        eps_end = float(getattr(self.args, "epsilon_end", 0.05))

        if decay_eps > 0 and int(self.episodes_done) < decay_eps:
            r = float(int(self.episodes_done) + 1) / float(max(1, decay_eps))
            return float(eps_start + (eps_end - eps_start) * r)
        return float(eps_end)

    def _compute_local_reheat_epsilon(self) -> float:
        M = max(1, int(self.local_epsilon_reheat_decay_dqn_episodes))
        s = float(self.local_epsilon_reheat_start)
        e = float(self.local_epsilon_reheat_end)
        m = int(self.local_epsilon_reheat_progress_dqn)

        if m >= M:
            return float(e)

        ratio = float(m) / float(M)
        return float(s + (e - s) * ratio)

    def _resolve_effective_dqn_epsilon(self) -> float:
        if bool(self.local_epsilon_reheat_enable):
            if bool(self.local_epsilon_reheat_active):
                return float(self._compute_local_reheat_epsilon())
            return float(self.local_epsilon_reheat_end)

        return float(self._compute_global_dqn_epsilon())

    def _advance_local_reheat_after_dqn_episode(self, n_dqn_episodes: int = 1) -> None:
        """Advance local exploration decay only after DQN episodes."""
        if not bool(self.local_epsilon_reheat_enable):
            return
        if not bool(self.local_epsilon_reheat_active):
            return

        self.local_epsilon_reheat_progress_dqn += int(n_dqn_episodes)

        if self.local_epsilon_reheat_progress_dqn >= int(
            self.local_epsilon_reheat_decay_dqn_episodes
        ):
            self.local_epsilon_reheat_progress_dqn = int(
                self.local_epsilon_reheat_decay_dqn_episodes
            )

    def start_local_epsilon_reheat(self, reason: str = "P2_TO_P1") -> None:
        if not bool(self.local_epsilon_reheat_enable):
            return

        self.local_epsilon_reheat_active = True
        self.local_epsilon_reheat_progress_dqn = 0
        self.local_epsilon_reheat_last_trigger = str(reason)
        self.current_epsilon = float(self.local_epsilon_reheat_start)

    def on_mode_transition(self, prev_mode: str, new_mode: str) -> None:
        pm = str(prev_mode).upper().strip()
        nm = str(new_mode).upper().strip()

        if bool(self.local_epsilon_reheat_enable) and pm == "P2" and nm == "P1":
            self.start_local_epsilon_reheat(reason="P2_TO_P1")

    def set_hyperparams(
        self,
        *,
        lr_dqn: Optional[float] = None,
        lr_power_ppo: Optional[float] = None,
        entropy_power: Optional[float] = None,
    ) -> None:
        if lr_dqn is not None:
            lr_dqn = float(lr_dqn)
            for pg in self.q_optimizer.param_groups:
                pg["lr"] = lr_dqn
            self.args.lr_dqn = lr_dqn

        if lr_power_ppo is not None:
            lr_power_ppo = float(lr_power_ppo)
            for opt in [self.power_actor_optim, self.power_critic_optim]:
                for pg in opt.param_groups:
                    pg["lr"] = lr_power_ppo
            self.args.lr_power_ppo = lr_power_ppo

        if entropy_power is not None:
            entropy_power = float(entropy_power)
            self.ppo_power.entropy_coef = entropy_power
            self.args.entropy_coef_power = entropy_power

    def load_checkpoint_dict(self, ckpt: dict, strict: bool = True):
        validate_checkpoint_metadata(
            ckpt.get("meta"),
            self.kernel_signature,
            context="Phase-I checkpoint",
        )
        self.q_network.load_state_dict(ckpt["q_network"], strict=strict)
        self.target_q_network.load_state_dict(ckpt["target_q_network"], strict=strict)
        self.power_actor.load_state_dict(ckpt["power_actor"], strict=strict)
        self.power_critic.load_state_dict(ckpt["power_critic"], strict=strict)
        self.phase_actor.load_state_dict(ckpt["phase_actor"], strict=strict)
        self.phase_critic.load_state_dict(ckpt["phase_critic"], strict=strict)

        self.q_optimizer.load_state_dict(ckpt["q_optimizer"])
        self.power_actor_optim.load_state_dict(ckpt["power_actor_optim"])
        self.power_critic_optim.load_state_dict(ckpt["power_critic_optim"])

        c = ckpt.get("counters", {})
        self.total_steps = int(c.get("total_steps", self.total_steps))
        self.episodes_done = int(c.get("episodes_done", self.episodes_done))
        self.current_epsilon = float(c.get("current_epsilon", self.current_epsilon))
        self.segment_idx = int(c.get("segment_idx", self.segment_idx))
        self.segment_ep_remaining = int(c.get("segment_ep_remaining", self.segment_ep_remaining))
        self.power_episode_offset = int(c.get("power_episode_offset", 0))

        self.local_epsilon_reheat_active = bool(
            c.get("local_reheat_active", self.local_epsilon_reheat_active)
        )
        self.local_epsilon_reheat_progress_dqn = int(
            c.get("local_reheat_progress_dqn", self.local_epsilon_reheat_progress_dqn)
        )
        self.local_epsilon_reheat_last_trigger = c.get(
            "local_reheat_last_trigger", self.local_epsilon_reheat_last_trigger
        )

        self.best_eval_return = float(c.get("best_eval_return", self.best_eval_return))
        self.best_eval_episode = int(c.get("best_eval_episode", self.best_eval_episode))
        self.eval_no_improve_count = int(c.get("eval_no_improve_count", self.eval_no_improve_count))
        self.training_should_stop = bool(c.get("training_should_stop", self.training_should_stop))

        buffers = ckpt.get("buffers", {})
        replay_state = buffers.get("replay")
        if replay_state is None:
            self.replay_buffer.clear()
        else:
            load_replay_buffer_state(self.replay_buffer, replay_state)
        for buffer in self.power_buffers.values():
            buffer.clear()
        for aid, buffer_state in buffers.get("power", {}).items():
            if aid not in self.power_buffers:
                raise ValueError(f"Unknown Power trajectory buffer in checkpoint: {aid}")
            load_trajectory_buffer_state(self.power_buffers[aid], buffer_state)

        restore_rng_state(ckpt.get("rng"), env=self.env)

    def save_final_if_needed(self):
        if not bool(getattr(self.args, "save_model", False)):
            return

        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), self.model_dir / "q_network_phase1.pth")
        torch.save(self.power_actor.state_dict(), self.model_dir / "power_actor_phase1.pth")
        torch.save(self.power_critic.state_dict(), self.model_dir / "power_critic_phase1.pth")
        write_kernel_signature(self.model_dir, self.kernel_signature)

    def _select_eval_seed_list(self):
        curve_k = int(getattr(self.args, "curve_k", 0) or 0)
        seed_list = list(VALIDATION_SEEDS)
        if curve_k > 0:
            seed_list = seed_list[:curve_k]
        return seed_list

    def _maybe_eval_on_segment_end(self, *, allow_internal_eval: bool = True):
        if not bool(allow_internal_eval):
            return
        if not bool(getattr(self.args, "eval_on_segment_end", False)):
            return

        seed_list = self._select_eval_seed_list()
        metrics_pack = run_multi_seed_validation_metrics(
            base_env_config=self.eval_env_config,
            agents=self.agents,
            args=self.args,
            seed_list=seed_list,
            profile="phase1",
            mode="standard",
            force_fixed_positions=True,
        )

        seg_val = float(metrics_pack["metrics"]["undiscounted_return"]["mean"])
        if self.writer is not None:
            self.writer.add_scalar(
                f"{self.args.eval_tag_steps}/UndiscountedReturn", seg_val, self.total_steps
            )

    def _eval_and_maybe_early_stop(self, eval_episode_index: int, *, allow_early_stop: bool = True):
        seed_list = self._select_eval_seed_list()

        metrics_pack = run_multi_seed_validation_metrics(
            base_env_config=self.eval_env_config,
            agents=self.agents,
            args=self.args,
            seed_list=seed_list,
            profile="phase1",
            mode="standard",
            force_fixed_positions=True,
        )

        pure_val = float(metrics_pack["metrics"]["undiscounted_return"]["mean"])

        if self.writer is not None:
            self.writer.add_scalar(
                f"{self.args.eval_tag_steps}/UndiscountedReturn", pure_val, eval_episode_index
            )

        record = build_eval_curve_record(
            episodes_done=int(eval_episode_index),
            metrics_pack=metrics_pack,
        )

        append_eval_curve_json(
            self.curve_json_path,
            record,
            meta={
                "exp_name": str(getattr(self.args, "exp_name", "")),
                "script": "atprl.training.stage1",
                "curve_kind": "daily_like_detailed_values",
                "scheduler_mode": str(getattr(self.args, "scheduler_mode", "dqn")).strip().lower(),
                "power_mode": _resolve_power_mode(self.args),
                "eval_policy": _get_phase1_eval_policy_name(self.args),
                "eval_phase_source": str(
                    getattr(self.args, "phase1_phase_source", "random_episode")
                ),
            },
        )

        improved = pure_val > self.best_eval_return + self.args.min_improvement

        if improved:
            self.best_eval_return = pure_val
            self.best_eval_episode = int(eval_episode_index)
            self.eval_no_improve_count = 0

            if bool(getattr(self.args, "save_model", False)):
                self.model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.q_network.state_dict(), self.model_dir / "q_network_phase1_best.pth"
                )
                torch.save(
                    self.power_actor.state_dict(), self.model_dir / "power_actor_phase1_best.pth"
                )
                torch.save(
                    self.power_critic.state_dict(), self.model_dir / "power_critic_phase1_best.pth"
                )
                write_kernel_signature(self.model_dir, self.kernel_signature)
                print(
                    f"[P1Trainer][EarlyStop] New BEST eval return = {pure_val:.4f} "
                    f"at episode {eval_episode_index} (best models saved)."
                )
        else:
            self.eval_no_improve_count += 1
            if (
                bool(allow_early_stop)
                and self.args.early_stop_patience_eval > 0
                and self.eval_no_improve_count >= self.args.early_stop_patience_eval
            ):
                print(
                    f"[P1Trainer][EarlyStop] No improvement for {self.eval_no_improve_count} eval rounds. "
                    f"Best = {self.best_eval_return:.4f} at episode {self.best_eval_episode}."
                )
                self.training_should_stop = True

    def train_for(
        self,
        num_finished_episodes: int,
        *,
        allow_internal_eval: bool = True,
        allow_early_stop: bool = True,
    ):
        allow_internal_eval = bool(allow_internal_eval)
        allow_early_stop = bool(allow_early_stop)

        target = self.episodes_done + int(num_finished_episodes)

        while self.episodes_done < target:
            if self.training_should_stop:
                break

            seg_type, _ = self.segment_order[self.segment_idx]

            current_eps_floor = 0.0

            if seg_type == "DQN":
                current_ep_seed = int(self.args.seed) + int(self.episodes_done)
                observations = self.env.reset(seed=current_ep_seed)

                done = False
                ep_return = 0.0

                self.current_epsilon = float(self._resolve_effective_dqn_epsilon())

                if self.writer is not None and bool(self.local_epsilon_reheat_log_tb):
                    self.writer.add_scalar(
                        "Epsilon/Global_Current_Epsilon",
                        float(self._compute_global_dqn_epsilon()),
                        self.total_steps,
                    )
                    self.writer.add_scalar(
                        "Epsilon/Local_Reheat_Active",
                        1.0 if bool(self.local_epsilon_reheat_active) else 0.0,
                        self.total_steps,
                    )
                    self.writer.add_scalar(
                        "Epsilon/Local_Reheat_ProgressDQN",
                        float(self.local_epsilon_reheat_progress_dqn),
                        self.total_steps,
                    )
                    if bool(self.local_epsilon_reheat_active):
                        self.writer.add_scalar(
                            "Epsilon/Local_Reheat_Current",
                            float(self._compute_local_reheat_epsilon()),
                            self.total_steps,
                        )

                phase_source = getattr(self.args, "phase1_phase_source", "random_episode")
                pass_every_step = bool(getattr(self.args, "phase1_phase_pass_every_step", False))

                episode_phi = None
                episode_w = None

                if phase_source == "random_episode":
                    if getattr(self.args, "phase1_fix_phase_per_episode", False):
                        phi_rng = np.random.default_rng(int(current_ep_seed) ^ 0x9E3779B9)
                        episode_phi = phi_rng.uniform(-np.pi, np.pi, size=self.G_groups).astype(
                            np.float32
                        )
                        episode_phi = _wrap_phase(
                            episode_phi, str(getattr(self.args, "phase_wrap_mode", "pi"))
                        )
                        episode_w = None

                elif phase_source == "phase_mean":
                    episode_phi, episode_w = None, None
                    _any_agent = next(iter(self.agents.values()))
                    if not hasattr(_any_agent, "phase_actor") or _any_agent.phase_actor is None:
                        raise RuntimeError(
                            "phase1_phase_source=phase_mean but agent.phase_actor is missing."
                        )
                else:
                    raise ValueError(f"Unknown phase1_phase_source={phase_source}")

                for t in range(self.env.config["max_steps_per_episode"]):
                    actions_to_env, cache = {}, {}

                    for agent_id, agent in self.agents.items():
                        act_info = agent.select_action(
                            local_observation=observations[agent_id],
                            current_epsilon=self.current_epsilon,
                            current_min_private_power_fraction=0.0,
                            deterministic_dqn=False,
                            deterministic_power=False
                            if _resolve_power_mode(self.args) == "ppo"
                            else True,
                            deterministic_phase=True,
                            scheduler_mode=self.scheduler_mode,
                            **_build_agent_power_backend_kwargs(self.args, self.env),
                        )
                        cache[agent_id] = act_info

                        users = list(act_info["env_action"][0])
                        power_vec = act_info["env_action"][1]

                        phi_to_use = None
                        w_to_use = None

                        if phase_source == "random_episode":
                            phi_to_use = (
                                episode_phi if (t == 0 and episode_phi is not None) else None
                            )
                            w_to_use = None

                        elif phase_source == "phase_mean":
                            if getattr(self.args, "phase1_fix_phase_per_episode", False) and (
                                not pass_every_step
                            ):
                                if t == 0 and episode_phi is None:
                                    episode_phi, episode_w = _phase_mean_to_ris_config_from_agent(
                                        agent,
                                        act_info.get("phase_state", None),
                                        self.G_groups,
                                        self.args.device,
                                        force_uniform_amp=bool(
                                            getattr(self.args, "force_phase2_uniform_amp", False)
                                        ),
                                    )
                                if t == 0:
                                    phi_to_use, w_to_use = episode_phi, episode_w
                                else:
                                    phi_to_use, w_to_use = None, None
                            else:
                                phi_to_use, w_to_use = _phase_mean_to_ris_config_from_agent(
                                    agent,
                                    act_info.get("phase_state", None),
                                    self.G_groups,
                                    self.args.device,
                                    force_uniform_amp=bool(
                                        getattr(self.args, "force_phase2_uniform_amp", False)
                                    ),
                                )

                        if phi_to_use is None:
                            actions_to_env[agent_id] = (users, power_vec)
                        else:
                            actions_to_env[agent_id] = (
                                (users, power_vec, phi_to_use)
                                if (w_to_use is None)
                                else (users, power_vec, phi_to_use, w_to_use)
                            )

                    next_obs, rewards, dones, infos = self.env.step(actions_to_env)

                    episode_end = bool(dones.get("__all__", False))
                    self.total_steps += 1

                    if not self.random_scheduling_enabled:
                        for aid in self.env.agents:
                            obs_full = np.asarray(observations[aid], dtype=np.float32)
                            next_obs_full = np.asarray(next_obs[aid], dtype=np.float32)

                            obs_as_dqn_state = obs_full[: self.dqn_state_dim].copy()
                            next_obs_as_dqn_state = next_obs_full[: self.dqn_state_dim].copy()

                            scheduler_action = cache[aid]["scheduler_action"]
                            reward_scalar = np.asarray([float(rewards[aid])], dtype=np.float32)

                            info_env = infos.get(aid, {}) if isinstance(infos, dict) else {}
                            terminated_flag = bool(info_env.get("terminated", False))

                            done_scalar = np.asarray([float(terminated_flag)], dtype=np.float32)

                            if obs_as_dqn_state.shape[0] != self.dqn_state_dim:
                                raise ValueError(
                                    f"[P1][DQN-buffer] state_dim mismatch: got {obs_as_dqn_state.shape[0]} vs {self.dqn_state_dim}"
                                )
                            if next_obs_as_dqn_state.shape[0] != self.dqn_state_dim:
                                raise ValueError(
                                    f"[P1][DQN-buffer] next_state_dim mismatch: got {next_obs_as_dqn_state.shape[0]} vs {self.dqn_state_dim}"
                                )

                            self.replay_buffer.store(
                                state=obs_as_dqn_state,
                                action=scheduler_action,
                                reward=reward_scalar,
                                next_state=next_obs_as_dqn_state,
                                done=done_scalar,
                            )

                        if (self.total_steps % self.args.dqn_update_freq == 0) and (
                            len(self.replay_buffer) >= self.args.dqn_batch_size
                        ):
                            batch = self.replay_buffer.sample(self.args.dqn_batch_size)
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                            _ = self.dqn_learner.update(batch)

                        if self.total_steps % self.args.target_update_freq == 0:
                            self.target_q_network.load_state_dict(self.q_network.state_dict())

                    ep_return += float(sum(rewards.values()))

                    observations = next_obs

                    if episode_end:
                        break

                prev_done = int(self.episodes_done)
                self.episodes_done += 1

                self._advance_local_reheat_after_dqn_episode(1)

                if allow_internal_eval and self.args.eval_every_episodes > 0:
                    for idx in _iter_crossed_boundary_indices(
                        prev_done, self.episodes_done, self.args.eval_every_episodes
                    ):
                        tag_ep = idx * self.args.eval_every_episodes
                        self._eval_and_maybe_early_stop(
                            eval_episode_index=tag_ep,
                            allow_early_stop=allow_early_stop,
                        )
                        if self.training_should_stop:
                            break

                self.segment_ep_remaining -= 1

            elif seg_type == "POWER":
                current_phase_source = getattr(self.args, "phase1_phase_source", "phase_mean")
                available = int(self.args.ppo_power_opt_episodes) - self.power_episode_offset
                episodes_to_run = min(available, target - self.episodes_done)
                steps_run = run_power_ppo_optimization(
                    self.env,
                    self.agents,
                    self.ppo_power,
                    self.power_buffers,
                    self.writer,
                    self.total_steps,
                    self.args,
                    current_eps_floor,
                    global_episode_base=self.episodes_done,
                    phase_source=current_phase_source,
                    fix_phase_per_episode=bool(
                        getattr(self.args, "phase1_fix_phase_per_episode", True)
                    ),
                    pass_phase_every_step=bool(
                        getattr(self.args, "phase1_phase_pass_every_step", False)
                    ),
                    n_episodes=episodes_to_run,
                )

                self.total_steps += int(steps_run)

                self.replay_buffer.clear()

                prev_done = int(self.episodes_done)
                self.episodes_done += episodes_to_run
                self.power_episode_offset += episodes_to_run

                if allow_internal_eval and self.args.eval_every_episodes > 0:
                    for idx in _iter_crossed_boundary_indices(
                        prev_done, self.episodes_done, self.args.eval_every_episodes
                    ):
                        tag_ep = idx * self.args.eval_every_episodes
                        self._eval_and_maybe_early_stop(
                            eval_episode_index=tag_ep,
                            allow_early_stop=allow_early_stop,
                        )
                        if self.training_should_stop:
                            break

                if self.power_episode_offset == int(self.args.ppo_power_opt_episodes):
                    self.power_episode_offset = 0
                    self.segment_ep_remaining -= 1

            else:
                raise ValueError(f"Unexpected segment {seg_type} in Phase1TrainerPBT")

            if self.segment_ep_remaining == 0:
                self._maybe_eval_on_segment_end(allow_internal_eval=allow_internal_eval)
                if self.training_should_stop:
                    break

                self.segment_idx = (self.segment_idx + 1) % len(self.segment_order)
                self.segment_ep_remaining = self.segment_order[self.segment_idx][1]

        return {
            "episodes_done": self.episodes_done,
            "total_steps": self.total_steps,
            "best_eval_return": self.best_eval_return,
            "best_eval_episode": self.best_eval_episode,
        }


def _resolve_external_models_dir(user_path: str) -> Path:
    p = Path(user_path)
    if p.is_dir() and (p / "models").is_dir():
        return p / "models"
    return p


def _load_required_state_dict(path: Path, device) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint file not found: {path}")
    obj = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like dict: {path}")
    return obj


def _load_phase2_final_assets_for_p1(args, device) -> dict:
    model_dir = _resolve_external_models_dir(args.bootstrap_phase2_final_dir)

    q_path = model_dir / "q_network_final.pth"
    pa_path = model_dir / "power_actor_final.pth"
    pc_path = model_dir / "power_critic_final.pth"
    pha_path = model_dir / "phase_actor_final.pth"
    phc_path = model_dir / "phase_critic_final.pth"
    signature = read_kernel_signature(model_dir)

    assets = {
        "model_dir": model_dir,
        "meta": None if signature is None else {"kernel_signature": signature},
        "q_network": _load_required_state_dict(q_path, device),
        "power_actor": _load_required_state_dict(pa_path, device),
        "power_critic": _load_required_state_dict(pc_path, device),
        "phase_actor": _load_required_state_dict(pha_path, device),
        "phase_critic": _load_required_state_dict(phc_path, device),
    }
    return assets


def _run_frozen_phase_p1_calibration_from_phase2_final(args):
    args.phase1_phase_source = "phase_mean"
    args.phase1_phase_pass_every_step = True
    args.phase1_fix_phase_per_episode = False
    args.phase1_randomize_episode_phase = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"Try_final_episode_lamda_PBT_runs/{args.exp_name}_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"[Frozen-P1-Calib] TensorBoard logs will be saved to: {log_dir}")

    trainer = Phase1TrainerPBT(args, log_writer=writer)

    assets = _load_phase2_final_assets_for_p1(args, trainer.device)
    strict = bool(getattr(args, "bootstrap_phase2_final_strict", True))
    validate_checkpoint_metadata(
        assets.get("meta"),
        trainer.kernel_signature,
        context=f"Phase-II final model directory {assets['model_dir']}",
    )

    print(f"[Frozen-P1-Calib] Loading external Phase-II finals from: {assets['model_dir']}")

    trainer.q_network.load_state_dict(assets["q_network"], strict=strict)
    trainer.target_q_network.load_state_dict(assets["q_network"], strict=strict)
    trainer.power_actor.load_state_dict(assets["power_actor"], strict=strict)
    trainer.power_critic.load_state_dict(assets["power_critic"], strict=strict)

    trainer.attach_phase_models(
        assets["phase_actor"],
        assets["phase_critic"],
        strict=strict,
    )

    if bool(getattr(args, "bootstrap_reset_training_state", True)):
        trainer.total_steps = 0
        trainer.episodes_done = 0
        trainer.current_epsilon = float(args.epsilon_start)
        trainer.segment_idx = 0
        trainer.segment_ep_remaining = trainer.segment_order[0][1]

        trainer.local_epsilon_reheat_active = False
        trainer.local_epsilon_reheat_progress_dqn = 0
        trainer.local_epsilon_reheat_last_trigger = None

        trainer.best_eval_return = -float("inf")
        trainer.best_eval_episode = 0
        trainer.eval_no_improve_count = 0
        trainer.training_should_stop = False

    trainer.reset_buffers(clear_replay=True, clear_power=True, reset_scheduler=True)

    print(
        "[Frozen-P1-Calib] Starting P1-only calibration under frozen phase_mean context:\n"
        f"  - phase1_phase_source={args.phase1_phase_source}\n"
        f"  - phase1_phase_pass_every_step={args.phase1_phase_pass_every_step}\n"
        f"  - phase1_fix_phase_per_episode={args.phase1_fix_phase_per_episode}\n"
        f"  - max_train_episodes={args.max_train_episodes}"
    )

    summary = trainer.train_for(
        args.max_train_episodes,
        allow_internal_eval=True,
        allow_early_stop=True,
    )

    trainer.save_final_if_needed()

    writer.close()

    print("\n" + "=" * 70)
    print("====== Frozen-Phase P1 Calibration Finished! ======")
    print("=" * 70)
    print(
        f"[Summary] episodes_done={summary['episodes_done']}, "
        f"total_steps={summary['total_steps']}, "
        f"best_eval_return={summary['best_eval_return']:.4f}, "
        f"best_eval_episode={summary['best_eval_episode']}"
    )


def main():
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    args = parse_args()

    if getattr(args, "bootstrap_phase2_final_dir", None):
        _run_frozen_phase_p1_calibration_from_phase2_final(args)
        return

    scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()
    random_scheduling_enabled = scheduler_mode == "random"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"Try_final_episode_lamda_PBT_runs/{args.exp_name}_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    curve_json_path = log_dir / "convergence_curve_phase1.json"

    model_dir = log_dir / "models"

    best_eval_return = -float("inf")
    best_eval_episode = 0
    eval_no_improve_count = 0
    training_should_stop = False

    env_config = {
        "num_ground_stations": int(args.num_ground_stations),
        "num_resource_blocks": 1,
        "num_gs_per_rb": int(args.num_gs_per_rb),
        "deployment_area_size_m": 400.0,
        "satellite_max_power_dBm": 50.0,
        "terrestrial_cell_power_dBm": 35.0,
        "noise_power_dBm": -114.0,
        "sic_detection_threshold_dBm": args.sic_detection_threshold_dBm,
        "carrier_frequency_GHz": 4.0,
        "satellite_antenna_gain_dBi": 24.3,
        "ground_station_antenna_gain_dBi": 24.3,
        "satellite_altitude_m": 600e3,
        "log_normal_shadowing_variance_dB": 8.0,
        "rician_k_factor": 10.0,
        "k_factor_ris_to_gs_linear": 10.0,
        "k_factor_sat_to_ris_linear": 30.0,
        "max_doppler_frequency_Hz": 50.0,
        "channel_update_interval_s": 0.04,
        "num_micro_steps": 20,
        "min_rate_requirement_bps_hz": args.min_rate_req,
        "max_steps_per_episode": 1024,
        "ris_boost_lambda": 0.0,
        "random_seed": args.seed,
        "use_fixed_positions": True,
        "topology_profile": args.topology_profile,
        "rho": args.rho,
        "csi_error_variance": args.csi_error_variance,
        "force_bs_gain_db": args.force_bs_gain_db,
        "reseed_each_episode": args.train_reseed_each_episode,
        "sat_user_gain_db_offsets": args.sat_user_gain_db_offsets,
        "include_angle_features": False,
        "ris_num_elem_x": int(getattr(args, "ris_num_elem_x", 16)),
        "ris_num_elem_y": int(getattr(args, "ris_num_elem_y", 16)),
        "ris_power_control_mode": "power_budget",
        "ris_reflection_power_limit_dBm": float(args.ris_reflection_power_limit_dBm),
        "ris_elem_spacing_ratio": 0.5,
        "active_ris_enabled": True,
        "active_ris_noise_power_dBm": args.active_ris_noise_power_dBm,
        "disable_ris_path": bool(getattr(args, "disable_ris_path", False)),
        "uav_initial_position_xyz": [0.0, 0.0, 80.0],
        "uav_velocity_per_step_xyz": [0.282842712, 0.282842712, 0.0],
        "uav_rx_antenna_gain_dBi": args.uav_rx_antenna_gain_dBi,
        "ris_elem_max_amp_gain_dB": args.ris_elem_max_amp_gain_dB,
        "fairness_alpha": args.fairness_alpha,
        "fairness_beta": args.fairness_beta,
        "fairness_epsilon": args.fairness_epsilon,
        "fairness_lambda": args.fairness_lambda,
        "sic_penalty": float(getattr(args, "sic_penalty", 0.0)),
        "include_sr_incident_pow_groups": args.include_sr_incident_pow_groups,
        "include_mrt_features": False,
    }

    env_config_base = dict(env_config)
    env_config_base["include_mrt_features"] = False

    mrt_tau = float(getattr(args, "mrt_casc_abs_tau", 1e-12))
    env_config_base["mrt_casc_abs_tau"] = mrt_tau

    include_mrt_phase_model = _resolve_phase_model_include_mrt_features(args)

    env_config_full = dict(env_config_base)
    env_config_full["include_mrt_features"] = include_mrt_phase_model
    env_config_full["mrt_casc_abs_tau"] = mrt_tau

    env = LeoSatEnv(config=env_config_full)

    eval_env_config = get_eval_config(
        base_config=env_config_base,
        mode="standard",
        profile="phase1",
    )
    if args.eval_env_seed is not None:
        eval_env_config["random_seed"] = args.eval_env_seed

    eval_env = LeoSatEnv(config=eval_env_config)

    _probe_env_base = LeoSatEnv(config=env_config_base)
    _probe_base = _probe_env_base.reset()
    _any_agent_id = next(iter(_probe_base))

    base_obs_dim = None
    if hasattr(_probe_env_base, "get_base_obs_dim") and callable(
        getattr(_probe_env_base, "get_base_obs_dim")
    ):
        base_obs_dim = _probe_env_base.get_base_obs_dim()
    if base_obs_dim is None:
        base_obs_dim = len(_probe_base[_any_agent_id])
    dqn_obs_layout = _probe_env_base.get_obs_layout_spec(include_mrt_features=False)
    if dqn_obs_layout is None:
        raise RuntimeError("Base observation layout is unavailable after env.reset()")

    if hasattr(_probe_env_base, "close") and callable(getattr(_probe_env_base, "close")):
        _probe_env_base.close()

    _obs_probe_full = env.reset()
    _any_agent_id_full = next(iter(_obs_probe_full))
    full_obs_dim = len(_obs_probe_full[_any_agent_id_full])

    n_users_total = getattr(env, "num_ground_stations", env.config["num_ground_stations"])
    k_users_per_rb = getattr(env, "num_gs_per_rb", env.config["num_gs_per_rb"])
    G_ris_groups = getattr(env, "ris_num_groups", env.config.get("ris_num_groups", 0))

    dqn_state_dim = int(base_obs_dim)
    power_state_dim = int(n_users_total + base_obs_dim)
    power_action_dim = int(1 + k_users_per_rb)
    phase_groups = int(G_ris_groups)

    scheduler_build = build_scheduler_networks(
        state_dim=dqn_state_dim,
        n_users=n_users_total,
        k_users_per_rb=k_users_per_rb,
        obs_layout=dqn_obs_layout,
        implementation=getattr(args, "dqn_scheduler_impl", "enumerative"),
        device=device,
    )
    q_network = scheduler_build.q_network
    target_q_network = scheduler_build.target_q_network
    discrete_actions_map = scheduler_build.discrete_actions_map
    dqn_action_dim = scheduler_build.network_output_dim

    print(
        f"[StandaloneP1/Topology] num_ground_stations={n_users_total}, "
        f"num_gs_per_rb={k_users_per_rb}, "
        f"{scheduler_log_summary(scheduler_build, n_users_total, k_users_per_rb)}, "
        f"discrete_actions_map={discrete_actions_map}"
    )

    writer.add_text(
        "Config/Topology", scheduler_log_summary(scheduler_build, n_users_total, k_users_per_rb), 0
    )
    writer.add_scalar("Config/NumGroundStations", float(n_users_total), 0)
    writer.add_scalar("Config/NumGsPerRb", float(k_users_per_rb), 0)
    writer.add_scalar("Config/DQNActionDim", float(dqn_action_dim), 0)

    phase_model_state_mode = str(getattr(args, "phase_model_state_mode", "legacy_full"))
    phase_model_compact_include_mrt_features = bool(
        getattr(args, "phase_model_compact_include_mrt_features", False)
    )

    if phase_model_compact_include_mrt_features and (not include_mrt_phase_model):
        raise ValueError(
            "--phase_model_compact_include_mrt_features requires "
            "Phase-I frozen phase-model observation to include MRT features."
        )

    phase_obs_layout = None
    if phase_model_state_mode == "compact_selected":
        if not hasattr(env, "get_obs_layout_spec") or not callable(
            getattr(env, "get_obs_layout_spec")
        ):
            raise RuntimeError(
                "Current LeoSatEnv does not provide get_obs_layout_spec(); "
                "compact_selected phase-model state cannot be enabled."
            )

        phase_obs_layout = env.get_obs_layout_spec(include_mrt_features=include_mrt_phase_model)
        if phase_obs_layout is None:
            raise RuntimeError(
                "LeoSatEnv.get_obs_layout_spec() returned None. "
                "Please ensure env.reset() / observation building has run before probing layout."
            )

    phase_state_dim = _compute_phase_state_dim(
        phase_state_mode=phase_model_state_mode,
        phase_obs_layout=phase_obs_layout,
        obs_dim_actual=int(full_obs_dim),
        n_users_total=int(n_users_total),
        k_users_per_rb=int(k_users_per_rb),
        include_selected_mrt=phase_model_compact_include_mrt_features,
    )
    kernel_signature = kernel_signature_from_runtime(
        args=args,
        env=env,
        scheduler_build=scheduler_build,
        dqn_state_dim=dqn_state_dim,
        power_state_dim=power_state_dim,
        power_action_dim=power_action_dim,
        phase_state_dim=phase_state_dim,
    )

    power_actor = ActorNetwork(
        state_dim=power_state_dim,
        action_dim=power_action_dim,
        max_power_watt=env.satellite_max_power_watt,
    ).to(device)
    power_critic = CriticNetwork(state_dim=power_state_dim).to(device)

    learn_amp_weights_resolved = _resolve_learn_amp_weights(args)
    phase_model_policy_dist = _resolve_phase_model_policy_dist(args)

    if phase_model_policy_dist == "gaussian_latent":
        phase_actor = PhaseActorGaussianLatent(
            state_dim=phase_state_dim,
            num_phase_groups=phase_groups,
            learn_amp=learn_amp_weights_resolved,
        ).to(device)
    elif phase_model_policy_dist == "von_mises":
        if bool(learn_amp_weights_resolved):
            raise ValueError(
                "Standalone Phase-I build error: phase_model_policy_dist='von_mises' requires learn_amp_weights=False."
            )
        phase_actor = PhaseActorVonMises(
            state_dim=phase_state_dim,
            num_phase_groups=phase_groups,
        ).to(device)
    else:
        raise ValueError(f"Unsupported phase_model_policy_dist: {phase_model_policy_dist}")

    phase_critic = PhaseCritic(state_dim=phase_state_dim).to(device)

    phase_latent_dim = (3 if learn_amp_weights_resolved else 2) * phase_groups

    q_optimizer = optim.Adam(q_network.parameters(), lr=args.lr_dqn)
    power_actor_optim = optim.Adam(power_actor.parameters(), lr=args.lr_power_ppo)
    power_critic_optim = optim.Adam(power_critic.parameters(), lr=args.lr_power_ppo)
    phase_actor_optim = optim.Adam(phase_actor.parameters(), lr=args.lr_phase_ppo)
    phase_critic_optim = optim.Adam(phase_critic.parameters(), lr=args.lr_phase_ppo)

    replay_buffer = ReplayBuffer(
        capacity=int(1e5), state_dim=dqn_state_dim, action_dim=scheduler_build.replay_action_dim
    )
    power_buffers = {agent_id: TrajectoryBuffer() for agent_id in env.agents}
    phase_buffers = {agent_id: TrajectoryBuffer() for agent_id in env.agents}

    dqn_learner = build_dqn_learner(scheduler_build, q_optimizer, gamma=args.gamma, tau=0.005)
    ppo_power = PPO(
        power_actor,
        power_critic,
        power_actor_optim,
        power_critic_optim,
        {
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_epsilon": 0.2,
            "ppo_epochs": args.ppo_power_epochs,
            "entropy_coef": args.entropy_coef_power,
        },
    )
    ppo_phase = PPO(
        phase_actor,
        phase_critic,
        phase_actor_optim,
        phase_critic_optim,
        {
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_epsilon": 0.2,
            "ppo_epochs": args.ppo_phase_epochs,
            "entropy_coef": args.entropy_coef_phase,
        },
    )

    agents = {
        agent_id: DPRA_Agent(
            q_network=q_network,
            power_actor=power_actor,
            power_critic=power_critic,
            phase_actor=phase_actor,
            phase_critic=phase_critic,
            discrete_actions_map=discrete_actions_map,
            n_users=n_users_total,
            dqn_state_dim=dqn_state_dim,
            max_power_watt=env.satellite_max_power_watt,
            device=device,
            include_sr_incident_pow_groups=bool(args.include_sr_incident_pow_groups),
            learn_amp_weights=learn_amp_weights_resolved,
            dqn_scheduler_impl=scheduler_build.implementation,
            k_users_per_rb=k_users_per_rb,
            ris_num_groups=phase_groups,
            phase_state_mode=phase_model_state_mode,
            phase_obs_layout=phase_obs_layout,
            phase_compact_include_mrt_features=phase_model_compact_include_mrt_features,
            phase_policy_dist=phase_model_policy_dist,
        )
        for agent_id in env.agents
    }

    total_steps = 0
    episode_count = 0
    current_epsilon = args.epsilon_start

    if getattr(args, "phase1_only", False):
        args.eps_floor_start = 0.0
        args.eps_floor_end = 0.0
        args.eps_floor_decay_steps = 1

    segment_order = [
        ("DQN", args.cycle_dqn_before_power),
        ("POWER", args.cycle_power),
    ]
    if not getattr(args, "phase1_only", False):
        segment_order += [
            ("DQN", args.cycle_dqn_before_phase),
            ("PHASE", args.cycle_phase),
        ]

    segment_idx = 0
    segment_ep_remaining = segment_order[0][1]
    cycles_done = 0
    episodes_done = 0

    progress_bar = tqdm(total=args.max_train_episodes, desc="Alternating Training (episodes)")

    def maybe_eval_on_segment_end():
        if not args.eval_on_segment_end:
            return

        curve_k = int(getattr(args, "curve_k", 0) or 0)
        seed_list = list(VALIDATION_SEEDS)
        if curve_k > 0:
            seed_list = seed_list[:curve_k]

        metrics_pack = run_robust_validation_metrics(
            eval_env_config=eval_env_config,
            agents=agents,
            args=args,
            seed_list=seed_list,
        )

        seg_val = float(metrics_pack["metrics"]["undiscounted_return"]["mean"])
        writer.add_scalar(f"{args.eval_tag_steps}/UndiscountedReturn", seg_val, total_steps)

    def eval_and_maybe_early_stop(eval_episode_index: int):
        nonlocal best_eval_return, best_eval_episode, eval_no_improve_count, training_should_stop

        curve_k = int(getattr(args, "curve_k", 0) or 0)
        seed_list = list(VALIDATION_SEEDS)
        if curve_k > 0:
            seed_list = seed_list[:curve_k]

        metrics_pack = run_robust_validation_metrics(
            eval_env_config=eval_env_config,
            agents=agents,
            args=args,
            seed_list=seed_list,
        )

        pure_val = float(metrics_pack["metrics"]["undiscounted_return"]["mean"])

        writer.add_scalar(f"{args.eval_tag_steps}/UndiscountedReturn", pure_val, eval_episode_index)

        record = build_eval_curve_record(
            episodes_done=int(eval_episode_index),
            metrics_pack=metrics_pack,
        )

        append_eval_curve_json(
            curve_json_path,
            record,
            meta={
                "exp_name": str(getattr(args, "exp_name", "")),
                "script": "train_phase1_power_only_lambda_PBT_v55.py",
                "curve_kind": "daily_like_detailed_values",
                "scheduler_mode": str(getattr(args, "scheduler_mode", "dqn")).strip().lower(),
                "eval_policy": _get_phase1_eval_policy_name(args),
                "eval_phase_source": str(getattr(args, "phase1_phase_source", "random_episode")),
            },
        )

        improved = pure_val > best_eval_return + args.min_improvement

        if improved:
            best_eval_return = pure_val
            best_eval_episode = eval_episode_index
            eval_no_improve_count = 0

            if args.save_model:
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(q_network.state_dict(), model_dir / "q_network_phase1_best.pth")
                torch.save(power_actor.state_dict(), model_dir / "power_actor_phase1_best.pth")
                torch.save(power_critic.state_dict(), model_dir / "power_critic_phase1_best.pth")
                write_kernel_signature(model_dir, kernel_signature)
                print(
                    f"[EarlyStop] New BEST eval return = {pure_val:.4f} "
                    f"at episode {eval_episode_index} (best models saved)."
                )
        else:
            eval_no_improve_count += 1
            if (
                args.early_stop_patience_eval > 0
                and eval_no_improve_count >= args.early_stop_patience_eval
            ):
                print(
                    f"[EarlyStop] No improvement for "
                    f"{eval_no_improve_count} eval rounds. "
                    f"Best = {best_eval_return:.4f} at episode {best_eval_episode}."
                )
                training_should_stop = True

    while episodes_done < args.max_train_episodes:
        if training_should_stop:
            break
        seg_type, _ = segment_order[segment_idx]

        current_eps_floor = get_eps_floor(total_steps, args)

        if seg_type == "DQN":
            episode_count += 1

            current_ep_seed = int(args.seed) + int(episodes_done)
            observations = env.reset(seed=current_ep_seed)

            episode_total_reward = 0.0
            q_network.train()

            for agent in agents.values():
                agent.power_actor.eval()
                agent.power_critic.eval()
                agent.phase_actor.eval()
                agent.phase_critic.eval()

            phase_source = getattr(args, "phase1_phase_source", "random_episode")
            pass_every_step = bool(getattr(args, "phase1_phase_pass_every_step", False))

            episode_phi = None
            episode_w = None

            if phase_source == "random_episode":
                if getattr(args, "phase1_fix_phase_per_episode", False) and getattr(
                    args, "phase1_randomize_episode_phase", False
                ):
                    phi_rng = np.random.default_rng(int(current_ep_seed) ^ 0x9E3779B9)
                    episode_phi = phi_rng.uniform(-np.pi, np.pi, size=G_ris_groups).astype(
                        np.float32
                    )
                    episode_phi = _wrap_phase(
                        episode_phi, str(getattr(args, "phase_wrap_mode", "pi"))
                    )
                    episode_w = None

            elif phase_source == "phase_mean":
                episode_phi, episode_w = None, None
                if bool(getattr(args, "phase1_require_phase_model_when_mean", True)):
                    _any_agent = next(iter(agents.values()))
                    if not hasattr(_any_agent, "phase_actor") or _any_agent.phase_actor is None:
                        raise RuntimeError(
                            "phase1_phase_source=phase_mean but agent.phase_actor is missing/None."
                        )
            else:
                raise ValueError(f"Unknown phase1_phase_source={phase_source}")

            for t in range(env.config["max_steps_per_episode"]):
                actions_to_env, cache = {}, {}
                for agent_id, agent in agents.items():
                    act_info = agent.select_action(
                        local_observation=observations[agent_id],
                        current_epsilon=current_epsilon,
                        current_min_private_power_fraction=current_eps_floor,
                        scheduler_mode=str(getattr(args, "scheduler_mode", "dqn")).strip().lower(),
                    )
                    cache[agent_id] = act_info

                    users = list(act_info["env_action"][0])
                    power_vec = act_info["env_action"][1]

                    phi_to_use = None
                    w_to_use = None

                    if phase_source == "random_episode":
                        phi_to_use = episode_phi if (t == 0 and episode_phi is not None) else None
                        w_to_use = None

                    elif phase_source == "phase_mean":
                        if getattr(args, "phase1_fix_phase_per_episode", False) and (
                            not pass_every_step
                        ):
                            if t == 0 and episode_phi is None:
                                episode_phi, episode_w = _phase_mean_to_ris_config_from_agent(
                                    agent, act_info.get("phase_state", None), G_ris_groups, device
                                )
                            if t == 0:
                                phi_to_use, w_to_use = episode_phi, episode_w
                            else:
                                phi_to_use, w_to_use = None, None
                        else:
                            phi_to_use, w_to_use = _phase_mean_to_ris_config_from_agent(
                                agent, act_info.get("phase_state", None), G_ris_groups, device
                            )

                    if phi_to_use is None:
                        actions_to_env[agent_id] = (users, power_vec)
                    else:
                        if w_to_use is None:
                            actions_to_env[agent_id] = (users, power_vec, phi_to_use)
                        else:
                            actions_to_env[agent_id] = (users, power_vec, phi_to_use, w_to_use)

                next_observations, rewards, dones, infos = env.step(actions_to_env)

                for agent_id in env.agents:
                    info = cache[agent_id]

                    obs_full = np.asarray(observations[agent_id], dtype=np.float32)
                    next_obs_full = np.asarray(next_observations[agent_id], dtype=np.float32)
                    obs_as_dqn_state = obs_full[:dqn_state_dim].copy()
                    next_obs_as_dqn_state = next_obs_full[:dqn_state_dim].copy()

                    scheduler_action = info["scheduler_action"]
                    reward_scalar = np.asarray([rewards[agent_id]], dtype=np.float32)
                    info_env = infos.get(agent_id, {})
                    terminated_flag = bool(info_env.get("terminated", False))

                    done_scalar = np.asarray([terminated_flag], dtype=np.float32)

                    if not random_scheduling_enabled:
                        replay_buffer.store(
                            state=obs_as_dqn_state,
                            action=scheduler_action,
                            reward=reward_scalar,
                            next_state=next_obs_as_dqn_state,
                            done=done_scalar,
                        )

                    power_vector_watt = info["env_action"][1]
                    common_ratio = float(power_vector_watt[0] / env.satellite_max_power_watt)
                    writer.add_scalar("Power/Common_Stream_Power_Ratio", common_ratio, total_steps)

                    info_env = infos.get(agent_id, {})
                    if "rate_total_sum" in info_env:
                        writer.add_scalar(
                            "Rate/Total", float(info_env["rate_total_sum"]), total_steps
                        )
                    if "sic_success" in info_env:
                        sic_violation = 0.0 if bool(info_env["sic_success"]) else 1.0
                        writer.add_scalar("SIC/Violation", sic_violation, total_steps)

                rates = env.historical_avg_rates
                sum_r = float(np.sum(rates))
                sum_sq_r = float(np.sum(rates**2))
                jfi = (sum_r**2) / (len(rates) * sum_sq_r + 1e-9)

                writer.add_scalar("Fairness/Jains_Index", jfi, total_steps)
                writer.add_scalar("Fairness/Min_User_Rate", float(np.min(rates)), total_steps)
                writer.add_scalar("Fairness/Sum_Rate_Physical", sum_r, total_steps)

                total_steps += 1

                writer.add_scalar(
                    "Scheduler/ModeIsRandom", 1.0 if random_scheduling_enabled else 0.0, total_steps
                )

                if (
                    (not random_scheduling_enabled)
                    and len(replay_buffer) > args.dqn_batch_size
                    and total_steps % args.dqn_update_freq == 0
                ):
                    batch = replay_buffer.sample(args.dqn_batch_size)
                    batch_on_device = {k: v.to(device) for k, v in batch.items()}
                    dqn_loss = dqn_learner.update(batch_on_device)
                    writer.add_scalar("Loss/ModeA_DQN", dqn_loss, total_steps)

                if (not random_scheduling_enabled) and total_steps % args.target_update_freq == 0:
                    dqn_learner.update_target_network()

                if episodes_done < args.epsilon_decay_episodes:
                    r = (episodes_done + 1) / max(1, args.epsilon_decay_episodes)
                    current_epsilon = (
                        args.epsilon_start + (args.epsilon_end - args.epsilon_start) * r
                    )
                else:
                    current_epsilon = args.epsilon_end

                episode_total_reward += float(sum(rewards.values()))
                observations = next_observations

                if dones["__all__"]:
                    break

            writer.add_scalar("Returns/Episode_Return", episode_total_reward, episode_count)

            if args.eval_every_episodes > 0 and (episodes_done + 1) % args.eval_every_episodes == 0:
                eval_and_maybe_early_stop(eval_episode_index=episodes_done + 1)

            progress_bar.update(1)

            writer.add_scalar("Epsilon/Current_Epsilon", current_epsilon, total_steps)
            progress_bar.set_postfix(
                {
                    "Return": f"{episode_total_reward:.2f}",
                    "Epsilon": f"{current_epsilon:.3f}",
                    "Episode": episode_count,
                }
            )

            episodes_done += 1
            segment_ep_remaining -= 1
            if segment_ep_remaining == 0:
                segment_idx = (segment_idx + 1) % len(segment_order)
                segment_ep_remaining = segment_order[segment_idx][1]
                if seg_type == "PHASE":
                    cycles_done += 1
                maybe_eval_on_segment_end()

            continue

        if seg_type == "POWER":
            q_network.eval()

            steps_run = run_power_ppo_optimization(
                env,
                agents,
                ppo_power,
                power_buffers,
                writer,
                total_steps,
                args,
                current_eps_floor,
                global_episode_base=episodes_done,
            )

            total_steps += steps_run

            replay_buffer.clear()

            episodes_done += args.ppo_power_opt_episodes
            segment_ep_remaining -= 1
            progress_bar.update(args.ppo_power_opt_episodes)
            if args.eval_every_episodes > 0 and (episodes_done % args.eval_every_episodes == 0):
                eval_and_maybe_early_stop(eval_episode_index=episodes_done)

            if segment_ep_remaining == 0:
                segment_idx = (segment_idx + 1) % len(segment_order)
                segment_ep_remaining = segment_order[segment_idx][1]
                maybe_eval_on_segment_end()

            continue

        if seg_type == "PHASE":
            if (args.phase_delay_cycles > 0 and cycles_done < args.phase_delay_cycles) or (
                args.phase_delay_episodes > 0 and episodes_done < args.phase_delay_episodes
            ):
                segment_idx = (segment_idx + 1) % len(segment_order)
                segment_ep_remaining = segment_order[segment_idx][1]
                cycles_done += 1
                maybe_eval_on_segment_end()
                continue

            steps_run = run_phase_ppo_optimization(
                env,
                agents,
                ppo_phase,
                phase_buffers,
                writer,
                total_steps,
                args,
                current_eps_floor,
                global_episode_base=episodes_done,
            )
            total_steps += steps_run
            replay_buffer.clear()
            episodes_done += args.ppo_phase_opt_episodes

            segment_ep_remaining -= 1
            if args.eval_every_steps > 0 and (total_steps % args.eval_every_steps == 0):
                with preserve_rng_state():
                    val = run_robust_validation(eval_env_config, agents, args, VALIDATION_SEEDS)
                writer.add_scalar(
                    f"{args.eval_tag_steps}/UndiscountedReturn", float(val), total_steps
                )

            if segment_ep_remaining == 0:
                segment_idx = (segment_idx + 1) % len(segment_order)
                segment_ep_remaining = segment_order[segment_idx][1]
                cycles_done += 1
                maybe_eval_on_segment_end()

            continue

    progress_bar.close()
    if args.save_model:
        model_path = model_dir
        model_path.mkdir(parents=True, exist_ok=True)

        torch.save(q_network.state_dict(), model_path / "q_network_phase1.pth")
        torch.save(power_actor.state_dict(), model_path / "power_actor_phase1.pth")
        torch.save(power_critic.state_dict(), model_path / "power_critic_phase1.pth")
        write_kernel_signature(model_path, kernel_signature)
        print(f"[Phase-I] FINAL DQN + PowerPPO models saved to {model_path}")

        if best_eval_return > -float("inf"):
            print(
                f"[Summary] Best eval return = {best_eval_return:.4f} "
                f"at episode {best_eval_episode}."
            )
            print(
                "          Best models saved as "
                "`q_network_phase1_best.pth`, `power_actor_phase1_best.pth`, "
                "`power_critic_phase1_best.pth` in the same folder."
            )
        else:
            print(
                "[Summary] No evaluation was run (or early-stop disabled); "
                "no best-* model was tracked."
            )

    writer.close()


def get_eps_floor(total_steps, args):
    if args.eps_floor_decay_steps <= 0:
        return args.eps_floor_end
    if total_steps < args.eps_floor_decay_steps:
        r = total_steps / args.eps_floor_decay_steps
        return args.eps_floor_start * (1 - r) + args.eps_floor_end * r
    return args.eps_floor_end


if __name__ == "__main__":
    main()
