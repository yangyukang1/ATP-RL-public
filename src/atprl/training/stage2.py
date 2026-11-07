from atprl.config.budgets import POST_WARMUP_EPISODES
from typing import Optional

import torch
import torch.optim as optim
import numpy as np
import random
import argparse
import time
import json
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
    KERNEL_DIMENSION_FIELDS,
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

from contextlib import contextmanager


class NullSummaryWriter:
    def __init__(self, log_dir=None):
        self.log_dir = None if log_dir is None else str(log_dir)

    def add_scalar(self, *args, **kwargs):
        return

    def add_text(self, *args, **kwargs):
        return

    def add_histogram(self, *args, **kwargs):
        return

    def add_image(self, *args, **kwargs):
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

    def get_logdir(self):
        return self.log_dir

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return

        return _noop


class _MaskedThetaOnlyDist:
    def __init__(self, base_dist, phase_dim: int):
        self._dist = base_dist
        self._phase_dim = int(phase_dim)

    @property
    def mean(self):
        return self._dist.mean

    def sample(self):
        return self._dist.sample()

    def log_prob(self, value):
        per_dim_lp = self._dist.base_dist.log_prob(value)
        return per_dim_lp[..., : self._phase_dim].sum(dim=-1)

    def entropy(self):
        per_dim_ent = self._dist.base_dist.entropy()
        return per_dim_ent[..., : self._phase_dim].sum(dim=-1)


class PhaseActorThetaOnlyWrapper(torch.nn.Module):
    def __init__(self, base_actor, num_phase_groups: int, enable_theta_only: bool, learn_amp: bool):
        super().__init__()
        self.base_actor = base_actor
        self.G = int(num_phase_groups)

        self.learn_amp = bool(learn_amp)
        self.phase_dim = 2 * self.G
        self.amp_dim = self.G if self.learn_amp else 0
        self.total_dim = self.phase_dim + self.amp_dim

        self.enable_theta_only = bool(enable_theta_only) and self.learn_amp

    def forward(self, state):
        dist = self.base_actor(state)

        if not self.enable_theta_only:
            return dist

        loc = dist.base_dist.loc
        scale = dist.base_dist.scale

        if loc.shape[-1] != self.total_dim:
            return dist

        loc2 = torch.cat([loc[..., : self.phase_dim], loc[..., self.phase_dim :].detach()], dim=-1)
        scale2 = torch.cat(
            [scale[..., : self.phase_dim], scale[..., self.phase_dim :].detach()], dim=-1
        )

        base = torch.distributions.Independent(torch.distributions.Normal(loc2, scale2), 1)
        return _MaskedThetaOnlyDist(base, phase_dim=self.phase_dim)


class _MaskedAmpOnlyDist:
    def __init__(self, base_dist, phase_dim: int):
        self._dist = base_dist
        self._phase_dim = int(phase_dim)

    @property
    def mean(self):
        return self._dist.mean

    def sample(self):
        return self._dist.sample()

    def log_prob(self, value):
        per_dim_lp = self._dist.base_dist.log_prob(value)
        return per_dim_lp[..., self._phase_dim :].sum(dim=-1)

    def entropy(self):
        per_dim_ent = self._dist.base_dist.entropy()
        return per_dim_ent[..., self._phase_dim :].sum(dim=-1)


class PhaseActorAmpOnlyWrapper(torch.nn.Module):
    def __init__(self, base_actor, num_phase_groups: int, enable_amp_only: bool, learn_amp: bool):
        super().__init__()
        self.base_actor = base_actor
        self.G = int(num_phase_groups)

        self.learn_amp = bool(learn_amp)
        self.phase_dim = 2 * self.G
        self.amp_dim = self.G if self.learn_amp else 0
        self.total_dim = self.phase_dim + self.amp_dim

        self.enable_amp_only = bool(enable_amp_only) and self.learn_amp

    def forward(self, state):
        dist = self.base_actor(state)

        if not self.enable_amp_only:
            return dist

        loc = dist.base_dist.loc
        scale = dist.base_dist.scale

        if loc.shape[-1] != self.total_dim:
            return dist

        loc2 = torch.cat([loc[..., : self.phase_dim].detach(), loc[..., self.phase_dim :]], dim=-1)
        scale2 = torch.cat(
            [scale[..., : self.phase_dim].detach(), scale[..., self.phase_dim :]], dim=-1
        )

        base = torch.distributions.Independent(torch.distributions.Normal(loc2, scale2), 1)
        return _MaskedAmpOnlyDist(base, phase_dim=self.phase_dim)


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


from atprl.evaluation.utils import (
    FIXED_VALIDATION_SEEDS as VALIDATION_SEEDS,
    get_eval_config,
    run_multi_seed_validation_metrics,
    build_eval_curve_record,
)


def _resolve_learn_amp(args) -> bool:
    v = getattr(args, "learn_amp_weights", "auto")
    if isinstance(v, bool):
        return bool(v)
    v = str(v).lower()
    if v == "auto":
        return bool(args.include_sr_incident_pow_groups)
    if v == "true":
        return True
    if v == "false":
        return False
    raise ValueError(f"Invalid --learn_amp_weights: {v}")


def _resolve_phase_policy_dist(args) -> str:
    mode = str(getattr(args, "phase_policy_dist", "gaussian_latent")).strip().lower()
    if mode not in ("gaussian_latent", "von_mises"):
        raise ValueError(
            f"Invalid --phase_policy_dist: {mode}. "
            f"Supported choices are ['gaussian_latent', 'von_mises']."
        )
    return mode


def _expected_phase_action_dim(
    phase_policy_dist: str,
    learn_amp: bool,
    num_phase_groups: int,
) -> int:
    phase_policy_dist = str(phase_policy_dist).strip().lower()
    G = int(num_phase_groups)

    if phase_policy_dist == "gaussian_latent":
        return int(2 * G + (G if bool(learn_amp) else 0))

    if phase_policy_dist == "von_mises":
        if bool(learn_amp):
            raise ValueError(
                "phase_policy_dist='von_mises' requires learn_amp=False in _expected_phase_action_dim()."
            )
        return int(G)

    raise ValueError(f"Unsupported phase_policy_dist: {phase_policy_dist}")


def _resolve_power_mode(args) -> str:
    mode = str(getattr(args, "power_mode", "ppo")).strip().lower()
    if mode not in ("ppo", "wmmse"):
        return "ppo"
    return mode


def _build_agent_power_backend_kwargs(args, env_obj) -> dict:
    power_mode = _resolve_power_mode(args)

    kwargs = {
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

    if phase_state_mode == "all_users_no_action":
        if phase_obs_layout is None:
            raise ValueError("phase_obs_layout is required for all_users_no_action")
        if int(phase_obs_layout["n_users"]) != n_users_total:
            raise ValueError("layout n_users mismatch for all_users_no_action")
        if int(phase_obs_layout["obs_dim"]) != obs_dim_actual:
            raise ValueError("layout obs_dim mismatch for all_users_no_action")
        per_user = sum(
            int(phase_obs_layout[key])
            for key in (
                "casc_feats_per_user",
                "direct_feats_per_user",
                "uav_xy_feats_per_user",
                "hist_rate_feats_per_user",
            )
        )
        z_start, z_end = phase_obs_layout["uav_z_slice"]
        total = n_users_total * per_user + 1 + 2 * n_users_total + int(z_end - z_start)
        if include_selected_mrt:
            mrt_dim = int(phase_obs_layout.get("mrt_feats_per_user", 0))
            if not phase_obs_layout.get("include_mrt_features", False) or mrt_dim <= 0:
                raise ValueError("all-users MRT state requires valid MRT observations")
            total += n_users_total * mrt_dim
        return int(total)

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
                "phase_compact_include_mrt_features=True but mrt_feats_per_user is missing/invalid in layout."
            )
        total_dim += k_users_per_rb * mrt_feats_per_user

    total_dim += power_action_dim
    return int(total_dim)


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
        default=int(5e5),
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
        "--ppo_opt_episodes",
        type=int,
        default=10,
        help="[Default] Number of episodes per PPO segment (fallback).",
    )
    parser.add_argument(
        "--ppo_buffer_size",
        type=int,
        default=1024,
        help="[Default] PPO trajectory buffer size (fallback).",
    )
    parser.add_argument(
        "--ppo_epochs", type=int, default=10, help="[Default] PPO epochs per update (fallback)."
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="[Default] PPO entropy coef (fallback)."
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="General discount factor.")
    parser.add_argument(
        "--gae_lambda", type=float, default=0.95, help="GAE lambda for PPO advantage estimation."
    )

    parser.add_argument(
        "--eps_floor_start",
        type=float,
        default=0,
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
        "--enable_tb",
        action="store_true",
        default=False,
        help="Enable TensorBoard event logging. Default: disabled to avoid generating event files.",
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
        "--scheduler_mode",
        type=str,
        default="dqn",
        choices=["dqn", "random"],
        help="Discrete scheduler source in Phase-II: 'dqn' uses frozen greedy DQN; "
        "'random' uses uniform random scheduling while keeping frozen power + trainable phase PPO.",
    )

    parser.add_argument(
        "--power_mode",
        type=str,
        default="ppo",
        choices=["ppo", "wmmse"],
        help=(
            "Frozen power backend used in Phase-II. "
            "'ppo' keeps original frozen Power-PPO kernel; "
            "'wmmse' replaces runtime frozen power generation with WMMSE while keeping DQN/random scheduling "
            "and Phase-PPO training logic unchanged."
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

    parser.add_argument(
        "--lambda_boost",
        type=float,
        default=0.0,
        help="RIS 增益 shaping 系数（仅 Phase-II 环境奖励中生效，Phase-I 固定为 0.0）。",
    )

    parser.add_argument(
        "--active_ris_noise_power_dBm",
        type=float,
        default=-100.0,
        help="Active RIS noise power in dBm (used in SINR denominator).",
    )

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

    parser.add_argument(
        "--fairness_lambda",
        type=float,
        default=0.0,
        help="Additive fairness shaping coefficient (must match Phase-I / env).",
    )

    parser.add_argument(
        "--sic_penalty",
        type=float,
        default=0.0,
        help="Penalty added when SIC fails (must match Phase-I / env).",
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
    parser.add_argument("--topology_profile", type=str, default="paper_n4")

    parser.add_argument(
        "--cycle_dqn_before_power",
        type=int,
        default=2,
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
        help="Deprecated. Internal evaluation in this script is controlled by eval_every_episodes and allow_internal_eval.",
    )

    parser.add_argument("--ppo_power_opt_episodes", type=int, default=8)
    parser.add_argument("--ppo_phase_opt_episodes", type=int, default=16)

    parser.add_argument("--ppo_power_epochs", type=int, default=12)
    parser.add_argument("--ppo_phase_epochs", type=int, default=6)

    parser.add_argument("--ppo_power_buffer_size", type=int, default=1024)
    parser.add_argument("--ppo_phase_buffer_size", type=int, default=1024)

    parser.add_argument("--entropy_coef_power", type=float, default=0.012)
    parser.add_argument("--entropy_coef_phase", type=float, default=0.006)

    parser.add_argument("--lr_power_ppo", type=float, default=3e-4)
    parser.add_argument("--lr_phase_ppo", type=float, default=6e-5)

    parser.add_argument(
        "--phase2_only",
        action="store_true",
        help="If set, treat DQN + PowerPPO as a fixed kernel (from Phase-I) and only train phase-PPO.",
    )
    parser.add_argument(
        "--phase1_model_dir",
        type=str,
        default=None,
        help="Directory containing Phase-I checkpoints: "
        "q_network_phase1.pth, power_actor_phase1.pth, power_critic_phase1.pth.",
    )
    parser.add_argument(
        "--freeze_dqn_power",
        action="store_true",
        help="Freeze DQN and PowerPPO parameters during Phase-II (recommended together with --phase2_only).",
    )

    parser.add_argument(
        "--max_train_episodes",
        type=int,
        default=POST_WARMUP_EPISODES,
        help="Stop training when total finished episodes reach this number.",
    )

    parser.add_argument(
        "--eval_every_episodes",
        type=int,
        default=32,
        help="Run validation every N finished episodes.",
    )
    parser.add_argument(
        "--curve_k",
        type=int,
        default=3,
        help="How many FIXED_VALIDATION_SEEDS to use for convergence-curve eval JSON. "
        "Set <=0 to use all seeds.",
    )
    parser.add_argument(
        "--epsilon_decay_episodes",
        type=int,
        default=1,
        help="Linearly decay DQN epsilon over this many finished episodes.",
    )

    parser.add_argument(
        "--early_stop_patience_eval",
        type=int,
        default=0,
        help="If >0, stop Phase-II early when eval has not improved for this many evaluation rounds.",
    )
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.0,
        help="Minimal improvement in eval return to be considered better (to filter evaluation noise).",
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
        "--phase1_ckpt",
        type=str,
        choices=["final", "best"],
        default="best",
        help="Which Phase-I checkpoint to load. 'best' loads *_best.pth; 'final' loads *.pth.",
    )
    parser.add_argument(
        "--phase1_ckpt_strict",
        action="store_true",
        help="If set and --phase1_ckpt=best, raise error when *_best.pth is missing (no fallback).",
    )

    parser.add_argument(
        "--sic_detection_threshold_dBm",
        type=float,
        default=-94.0,
        help="SIC detection threshold in dBm. Lower (more negative) is easier/looser; Higher is stricter.",
    )

    parser.add_argument(
        "--phase2_power_deterministic_mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In PHASE PPO segment, use dist_power.mean instead of sampling power (recommended when power is frozen). "
        "Use --no-phase2_power_deterministic_mean to fallback to sampling.",
    )

    parser.add_argument(
        "--phase_wrap_mode",
        type=str,
        default="pi",
        choices=["pi", "2pi"],
        help="Phase wrap interval used right before sending action to env. "
        "'pi' => [-pi, pi), '2pi' => [0, 2pi). Must match env expectation.",
    )

    parser.add_argument(
        "--include_sr_incident_pow_groups",
        action="store_true",
        help="If set, append per-RIS-group incident power (log compressed) "
        "computed from imperfect Sat->RIS estimate into observation. "
        "IMPORTANT: must be consistent between Phase-I and Phase-II.",
    )

    parser.add_argument(
        "--learn_amp_weights",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to learn RIS amplitude weights (amp head / 3G latent). "
        "'auto' keeps backward compatibility: learn_amp = include_sr_incident_pow_groups. "
        "'true' forces learning amp weights even if incident power feature is off. "
        "'false' disables amp learning even if incident power feature is on.",
    )

    parser.add_argument(
        "--force_phase2_uniform_amp",
        action="store_true",
        help="If set, overwrite RIS amplitude weights to uniform distribution in Phase-II training and validation.",
    )

    parser.add_argument(
        "--phase2_alt_enable",
        action="store_true",
        help="Enable Phase-II alternating optimization: theta-segment (phase-only) and amp-segment (amp-only).",
    )
    parser.add_argument(
        "--phase2_alt_theta_episodes",
        type=int,
        default=12,
        help="N_theta: number of consecutive episodes for theta-segment in Phase-II PHASE block.",
    )
    parser.add_argument(
        "--phase2_alt_amp_episodes",
        type=int,
        default=4,
        help="N_amp: number of consecutive episodes for amp-segment in Phase-II PHASE block.",
    )
    parser.add_argument(
        "--phase2_alt_warmup_theta_only_episodes",
        type=int,
        default=0,
        help="Warmup episodes: force theta-segment only for the first N episodes in Phase-II PHASE block.",
    )

    parser.add_argument(
        "--include_mrt_features",
        action="store_true",
        help="If set, append MRT-derived features (computed from imperfect CSI) into observation. "
        "Recommended to enable ONLY in Phase-II.",
    )

    parser.add_argument(
        "--mrt_casc_abs_tau",
        type=float,
        default=1e-12,
        help="Stabilizer tau used in MRT feature: denom uses max(|h_casc|, tau).",
    )

    parser.add_argument(
        "--phase_state_mode",
        type=str,
        default="legacy_full",
        choices=["legacy_full", "compact_selected"],
        help="Phase-state mode. "
        "'legacy_full' keeps old logic: [full_obs ⊕ one-hot(selected) ⊕ power_shares]. "
        "'compact_selected' keeps only selected-user-related blocks plus power_shares.",
    )

    parser.add_argument(
        "--phase_compact_include_mrt_features",
        action="store_true",
        help="If set, compact phase-state additionally appends MRT features of selected users only. "
        "Requires --phase_state_mode compact_selected and --include_mrt_features.",
    )

    parser.add_argument(
        "--phase_policy_dist",
        type=str,
        default="gaussian_latent",
        choices=["gaussian_latent", "von_mises"],
        help="Phase policy distribution. "
        "'gaussian_latent' keeps the original 2G/3G latent PPO logic; "
        "'von_mises' enables theta-only circular policy with 1G action dim and requires "
        "--learn_amp_weights false.",
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

    args.phase2_only = True
    args.freeze_dqn_power = True
    args.kernel_frozen = True

    if getattr(args, "force_phase2_uniform_amp", False) and getattr(
        args, "phase2_alt_enable", False
    ):
        args.phase2_alt_enable = False

    if getattr(args, "phase2_alt_enable", False):
        if args.phase2_alt_theta_episodes <= 0:
            raise ValueError(
                "--phase2_alt_theta_episodes must be > 0 when --phase2_alt_enable is set."
            )
        if args.phase2_alt_amp_episodes <= 0:
            raise ValueError(
                "--phase2_alt_amp_episodes must be > 0 when --phase2_alt_enable is set."
            )
        if args.phase2_alt_warmup_theta_only_episodes < 0:
            raise ValueError("--phase2_alt_warmup_theta_only_episodes must be >= 0.")

    if str(getattr(args, "phase_state_mode", "legacy_full")) not in (
        "legacy_full",
        "compact_selected",
    ):
        raise ValueError(
            f"Unsupported --phase_state_mode: {getattr(args, 'phase_state_mode', None)}"
        )

    if bool(getattr(args, "phase_compact_include_mrt_features", False)):
        if str(getattr(args, "phase_state_mode", "legacy_full")) != "compact_selected":
            raise ValueError(
                "--phase_compact_include_mrt_features only works with --phase_state_mode compact_selected."
            )
        if not bool(getattr(args, "include_mrt_features", False)):
            raise ValueError(
                "--phase_compact_include_mrt_features requires --include_mrt_features, "
                "because selected MRT features must actually exist in the observation."
            )

    phase_policy_dist = _resolve_phase_policy_dist(args)
    learn_amp_resolved = _resolve_learn_amp(args)

    if phase_policy_dist == "von_mises":
        if bool(learn_amp_resolved):
            raise ValueError(
                "--phase_policy_dist von_mises requires --learn_amp_weights false. "
                "The Von Mises path in this codebase is theta-only (1G) and does not learn amp logits."
            )

        if bool(getattr(args, "phase2_alt_enable", False)):
            raise ValueError(
                "--phase_policy_dist von_mises is not compatible with --phase2_alt_enable. "
                "The current Phase-II theta/amp alternating wrappers are written for Gaussian latent only."
            )

    return args


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
) -> int:
    for agent in agents.values():
        agent.power_actor.train()
        agent.power_critic.train()
        agent.phase_actor.eval()
        agent.phase_critic.eval()

    scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()

    steps_in_mode_b = 0
    progress_bar = tqdm(range(args.ppo_power_opt_episodes), desc="Mode B: Power PPO", leave=False)

    for ep_idx_in_segment in progress_bar:
        for buf in power_buffers.values():
            buf.clear()

        observations = env.reset()
        episode_return = 0.0

        for _ in range(env.config["max_steps_per_episode"]):
            actions_to_env, action_info_map = {}, {}
            with torch.no_grad():
                for agent_id, agent in agents.items():
                    info = agent.select_action(
                        observations[agent_id],
                        current_epsilon=0.0,
                        current_min_private_power_fraction=current_eps_floor,
                        deterministic_dqn=True,
                        deterministic_power=False,
                        deterministic_phase=True,
                        scheduler_mode=scheduler_mode,
                        force_uniform_amp=bool(args.force_phase2_uniform_amp),
                    )

                    action_info_map[agent_id] = info
                    actions_to_env[agent_id] = info["env_action"]

            next_observations, rewards, dones, infos = env.step(actions_to_env)
            episode_end = bool(dones.get("__all__", False))
            episode_return += float(sum(rewards.values()))

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

                power_buffers[agent_id].store(
                    info["power_state"],
                    info["power_action_raw"],
                    reward,
                    done_for_bootstrap,
                    info["power_log_prob"],
                    info["power_value"],
                )

                power_vector_watt = info["env_action"][1]
                common_power_ratio = power_vector_watt[0] / env.satellite_max_power_watt
                writer.add_scalar(
                    "Power/ModeB_Power_CommonRatio",
                    common_power_ratio,
                    global_step_base + steps_in_mode_b,
                )

            rates = env.historical_avg_rates
            sum_r = float(np.sum(rates))
            sum_sq_r = float(np.sum(rates**2))
            jfi = (sum_r**2) / (len(rates) * sum_sq_r + 1e-9)

            writer.add_scalar("Fairness/Jains_Index", jfi, global_step_base + steps_in_mode_b)
            writer.add_scalar(
                "Fairness/Min_User_Rate", float(np.min(rates)), global_step_base + steps_in_mode_b
            )
            writer.add_scalar(
                "Fairness/Sum_Rate_Physical", sum_r, global_step_base + steps_in_mode_b
            )

            wrap_oor = float(
                np.mean(
                    [action_info_map[aid].get("phase_wrap_oor_ratio", 0.0) for aid in env.agents]
                )
            )
            wrap_delta = float(
                np.mean(
                    [action_info_map[aid].get("phase_wrap_delta_abs", 0.0) for aid in env.agents]
                )
            )
            writer.add_scalar(
                "PhaseWrap/OutOfRangeRatio", wrap_oor, global_step_base + steps_in_mode_b
            )
            writer.add_scalar(
                "PhaseWrap/MeanAbsDelta", wrap_delta, global_step_base + steps_in_mode_b
            )

            steps_in_mode_b += 1
            observations = next_observations

            buffer_is_full = (
                min(len(buf) for buf in power_buffers.values()) >= args.ppo_power_buffer_size
            )

            if buffer_is_full or episode_end:
                for agent_id, buffer in power_buffers.items():
                    if len(buffer) == 0:
                        continue
                    batch = {k: v.to(args.device) for k, v in buffer.get_batch().items()}

                    with torch.no_grad():
                        nxt = agents[agent_id].select_action(
                            next_observations[agent_id],
                            current_epsilon=0.0,
                            current_min_private_power_fraction=current_eps_floor,
                            deterministic_dqn=True,
                            deterministic_power=True,
                            deterministic_phase=True,
                            scheduler_mode=scheduler_mode,
                            force_uniform_amp=bool(args.force_phase2_uniform_amp),
                        )
                        next_power_state = nxt["power_state"].to(args.device)

                    tl_trunc = bool(time_limit_flags[agent_id]) and (
                        not bool(terminated_flags[agent_id])
                    )
                    losses = ppo_power.update(batch, next_power_state, tl_trunc)

                    writer.add_scalar(
                        "Loss/ModeB_PowerPPO_Actor",
                        losses["actor_loss"],
                        global_step_base + steps_in_mode_b,
                    )
                    writer.add_scalar(
                        "Loss/ModeB_PowerPPO_Critic",
                        losses["critic_loss"],
                        global_step_base + steps_in_mode_b,
                    )
                    buffer.clear()

            if episode_end:
                break

        writer.add_scalar(
            "Returns/ModeB_PowerPPO_Episode_Return",
            episode_return,
            global_episode_base + ep_idx_in_segment,
        )

    return steps_in_mode_b


def run_phase_ppo_optimization(
    env,
    agents,
    ppo_phase_theta,
    ppo_phase_amp,
    phase_buffers,
    writer,
    global_step_base,
    args,
    current_eps_floor: float,
    global_episode_base: int,
    n_episodes: Optional[int] = None,
) -> int:
    for agent in agents.values():
        agent.phase_actor.train()
        agent.phase_critic.train()
        agent.power_actor.eval()
        agent.power_critic.eval()

    learn_amp = _resolve_learn_amp(args)
    phase_policy_dist = _resolve_phase_policy_dist(args)

    if phase_policy_dist == "von_mises":
        alt_enable = False
    else:
        alt_enable = (
            bool(getattr(args, "phase2_alt_enable", False))
            and bool(learn_amp)
            and (not bool(args.force_phase2_uniform_amp))
        )

    N_theta = int(getattr(args, "phase2_alt_theta_episodes", 12))
    N_amp = int(getattr(args, "phase2_alt_amp_episodes", 4))
    warmup = int(getattr(args, "phase2_alt_warmup_theta_only_episodes", 0))

    scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()

    steps_in_mode_b = 0
    episode_count = int(args.ppo_phase_opt_episodes if n_episodes is None else n_episodes)
    progress_bar = tqdm(range(episode_count), desc="Mode B: Phase PPO", leave=False)

    for ep_idx_in_segment in progress_bar:
        for buf in phase_buffers.values():
            buf.clear()
        observations = env.reset()
        episode_return = 0.0

        if phase_policy_dist == "von_mises":
            mode = "joint"
        else:
            if (not alt_enable) or (ep_idx_in_segment < warmup):
                mode = "joint" if (not alt_enable) else "theta"
            else:
                cycle = N_theta + N_amp
                pos = (ep_idx_in_segment - warmup) % cycle
                mode = "theta" if pos < N_theta else "amp"

            if bool(args.force_phase2_uniform_amp):
                mode = "joint"

        writer.add_scalar(
            "Alt/ModeIsAmp", 1.0 if mode == "amp" else 0.0, global_episode_base + ep_idx_in_segment
        )

        if phase_policy_dist == "von_mises":
            ppo_updater = ppo_phase_theta
        else:
            if mode == "amp":
                ppo_updater = ppo_phase_amp
            else:
                ppo_updater = ppo_phase_theta

        for _ in range(env.config["max_steps_per_episode"]):
            actions_to_env, action_info_map = {}, {}

            with torch.no_grad():
                use_det_power_phase = bool(getattr(args, "kernel_frozen", False)) and bool(
                    getattr(args, "phase2_power_deterministic_mean", True)
                )

                for agent_id, agent in agents.items():
                    if phase_policy_dist == "von_mises":
                        info = agent.select_action(
                            observations[agent_id],
                            current_epsilon=0.0,
                            current_min_private_power_fraction=current_eps_floor,
                            deterministic_dqn=True,
                            deterministic_power=use_det_power_phase,
                            deterministic_phase=False,
                            deterministic_amp=True,
                            scheduler_mode=scheduler_mode,
                            phase_ppo_mode="joint",
                            force_uniform_amp=bool(args.force_phase2_uniform_amp),
                            **_build_agent_power_backend_kwargs(args, env),
                        )
                    else:
                        if mode == "theta":
                            info = agent.select_action(
                                observations[agent_id],
                                current_epsilon=0.0,
                                current_min_private_power_fraction=current_eps_floor,
                                deterministic_dqn=True,
                                deterministic_power=use_det_power_phase,
                                deterministic_phase=False,
                                deterministic_amp=True,
                                scheduler_mode=scheduler_mode,
                                phase_ppo_mode="theta",
                                force_uniform_amp=bool(args.force_phase2_uniform_amp),
                                **_build_agent_power_backend_kwargs(args, env),
                            )
                        elif mode == "amp":
                            info = agent.select_action(
                                observations[agent_id],
                                current_epsilon=0.0,
                                current_min_private_power_fraction=current_eps_floor,
                                deterministic_dqn=True,
                                deterministic_power=use_det_power_phase,
                                deterministic_phase=True,
                                deterministic_amp=False,
                                scheduler_mode=scheduler_mode,
                                phase_ppo_mode="amp",
                                force_uniform_amp=bool(args.force_phase2_uniform_amp),
                                **_build_agent_power_backend_kwargs(args, env),
                            )
                        else:
                            info = agent.select_action(
                                observations[agent_id],
                                current_epsilon=0.0,
                                current_min_private_power_fraction=current_eps_floor,
                                deterministic_dqn=True,
                                deterministic_power=use_det_power_phase,
                                deterministic_phase=False,
                                deterministic_amp=True,
                                scheduler_mode=scheduler_mode,
                                phase_ppo_mode="joint",
                                force_uniform_amp=bool(args.force_phase2_uniform_amp),
                                **_build_agent_power_backend_kwargs(args, env),
                            )

                    action_info_map[agent_id] = info
                    actions_to_env[agent_id] = info["env_action"]

            next_observations, rewards, dones, infos = env.step(actions_to_env)

            episode_end = bool(dones["__all__"])
            episode_return += float(sum(rewards.values()))

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

                u = info["phase_joint_action"]
                assert u.ndim == 1, f"Expected phase_joint_action to be 1D, got {u.shape}"

                G = int(env.ris_num_groups)
                expected_dim = _expected_phase_action_dim(
                    phase_policy_dist=phase_policy_dist,
                    learn_amp=learn_amp,
                    num_phase_groups=G,
                )
                assert u.shape[0] == expected_dim, (
                    f"Expected phase action dim={expected_dim} under phase_policy_dist='{phase_policy_dist}', "
                    f"got {u.shape[0]}"
                )

            rates = env.historical_avg_rates
            sum_r = float(np.sum(rates))
            sum_sq_r = float(np.sum(rates**2))
            jfi = (sum_r**2) / (len(rates) * sum_sq_r + 1e-9)

            writer.add_scalar("Fairness/Jains_Index", jfi, global_step_base + steps_in_mode_b)
            writer.add_scalar(
                "Fairness/Min_User_Rate", float(np.min(rates)), global_step_base + steps_in_mode_b
            )
            writer.add_scalar(
                "Fairness/Sum_Rate_Physical", sum_r, global_step_base + steps_in_mode_b
            )

            wrap_oor = float(
                np.mean(
                    [action_info_map[aid].get("phase_wrap_oor_ratio", 0.0) for aid in env.agents]
                )
            )
            wrap_delta = float(
                np.mean(
                    [action_info_map[aid].get("phase_wrap_delta_abs", 0.0) for aid in env.agents]
                )
            )
            writer.add_scalar(
                "PhaseWrap/OutOfRangeRatio", wrap_oor, global_step_base + steps_in_mode_b
            )
            writer.add_scalar(
                "PhaseWrap/MeanAbsDelta", wrap_delta, global_step_base + steps_in_mode_b
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
                        use_det_power_phase = bool(getattr(args, "kernel_frozen", False)) and bool(
                            getattr(args, "phase2_power_deterministic_mean", True)
                        )

                        nxt = agents[agent_id].select_action(
                            next_observations[agent_id],
                            current_epsilon=0.0,
                            current_min_private_power_fraction=current_eps_floor,
                            deterministic_dqn=True,
                            deterministic_power=use_det_power_phase,
                            deterministic_phase=True,
                            deterministic_amp=True,
                            scheduler_mode=scheduler_mode,
                            phase_ppo_mode="joint",
                            force_uniform_amp=bool(args.force_phase2_uniform_amp),
                            **_build_agent_power_backend_kwargs(args, env),
                        )
                        next_phase_state = nxt["phase_state"].to(args.device)

                    tl_trunc = bool(time_limit_flags[agent_id]) and (
                        not bool(terminated_flags[agent_id])
                    )
                    losses = ppo_updater.update(batch, next_phase_state, tl_trunc)

                    if phase_policy_dist == "von_mises":
                        tag_prefix = "VonMises"
                    else:
                        tag_prefix = "Theta" if mode != "amp" else "Amp"

                    writer.add_scalar(
                        f"Loss/ModeB_PhasePPO_{tag_prefix}_Actor",
                        losses["actor_loss"],
                        global_step_base + steps_in_mode_b,
                    )
                    writer.add_scalar(
                        f"Loss/ModeB_PhasePPO_{tag_prefix}_Critic",
                        losses["critic_loss"],
                        global_step_base + steps_in_mode_b,
                    )

                    buffer.clear()

            if episode_end:
                break

        writer.add_scalar(
            "Returns/ModeB_PhasePPO_Episode_Return",
            episode_return,
            global_episode_base + ep_idx_in_segment,
        )

    return steps_in_mode_b


def run_validation_episode(env, agents, args) -> float:
    for agent in agents.values():
        agent.power_actor.eval()
        agent.power_critic.eval()
        agent.phase_actor.eval()
        agent.phase_critic.eval()

    scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()

    observations = env.reset()
    episode_return = 0.0

    for _ in range(env.config["max_steps_per_episode"]):
        actions_to_env = {}
        with torch.no_grad():
            for agent_id, agent in agents.items():
                act = agent.select_action(
                    observations[agent_id],
                    current_epsilon=0.0,
                    current_min_private_power_fraction=0.0,
                    deterministic_dqn=True,
                    deterministic_power=True,
                    deterministic_phase=True,
                    deterministic_amp=True,
                    scheduler_mode=scheduler_mode,
                    phase_ppo_mode="joint",
                    force_uniform_amp=bool(args.force_phase2_uniform_amp),
                    **_build_agent_power_backend_kwargs(args, env),
                )
                actions_to_env[agent_id] = act["env_action"]

        next_obs, rewards, dones, _ = env.step(actions_to_env)
        episode_return += float(sum(rewards.values()))
        observations = next_obs

        if bool(dones.get("__all__", False)):
            break

    return float(episode_return)


def run_robust_validation(eval_env_config, agents, args, seed_list) -> float:
    metrics_pack = run_robust_validation_metrics(
        eval_env_config=eval_env_config,
        agents=agents,
        args=args,
        seed_list=seed_list,
    )
    return float(metrics_pack["metrics"]["undiscounted_return"]["mean"])


def run_robust_validation_metrics(eval_env_config, agents, args, seed_list):
    profile = "phase2" if bool(getattr(args, "include_mrt_features", False)) else "phase2_no_mrt"

    return run_multi_seed_validation_metrics(
        base_env_config=eval_env_config,
        agents=agents,
        args=args,
        seed_list=seed_list,
        profile=profile,
        mode="standard",
        force_fixed_positions=True,
    )


def _get_phase2_eval_policy_name(args) -> str:
    scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()
    power_mode = _resolve_power_mode(args)

    if scheduler_mode == "random":
        if power_mode == "wmmse":
            return "random_sched_wmmse_power_phase2_eval"
        return "random_sched_frozen_power_phase2_eval"

    if power_mode == "wmmse":
        return "greedy_dqn_wmmse_power_phase2_eval"
    return "greedy_dqn_frozen_power_phase2_eval"


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


def _capture_rng_state():
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def _restore_rng_state(state):
    if state is None:
        return
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda", None) is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _iter_crossed_boundary_indices(prev_ep: int, new_ep: int, every: int):
    if every <= 0:
        return
    if new_ep <= prev_ep:
        return
    prev_idx = prev_ep // every
    new_idx = new_ep // every
    for idx in range(prev_idx + 1, new_idx + 1):
        yield idx


class Phase2Trainer:
    """Train Phase-PPO with the DQN and power-control kernel frozen."""

    def __init__(self, args, *, log_dir: Path = None, writer: SummaryWriter = None):
        self.args = args

        self.scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()
        if self.scheduler_mode not in ("dqn", "random"):
            self.scheduler_mode = "dqn"
        self.random_scheduling_enabled = self.scheduler_mode == "random"

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        self.device = torch.device(args.device)

        if log_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_dir = Path(f"Try_final_episode_lamda_early_runs/{args.exp_name}_{timestamp}")
        self.log_dir = log_dir

        if writer is not None:
            self.writer = writer
        else:
            if bool(getattr(args, "enable_tb", False)):
                self.writer = SummaryWriter(self.log_dir)
                print(f"TensorBoard logs will be saved to: {self.log_dir}")
            else:
                self.writer = NullSummaryWriter(self.log_dir)
                print(
                    f"TensorBoard event logging is disabled by default. Run with --enable_tb to enable it."
                )

        self.model_dir = self.log_dir / "models"
        if args.save_model:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        self.curve_json_path = self.log_dir / "convergence_curve_phase2.json"

        self._build_everything_from_original_main()

        self._init_training_state_from_original_main()

        self.mode = "P2"

        self.kernel_frozen = True
        self.args.kernel_frozen = True

        self.best_eval_return = -float("inf")
        self.best_eval_episode = 0
        self.eval_no_improve_count = 0
        self.training_should_stop = False

        self._last_rng_state = _capture_rng_state()

    def _get_eps_floor(self, total_steps: int) -> float:
        args = self.args
        if args.eps_floor_decay_steps <= 0:
            return args.eps_floor_end
        if total_steps < args.eps_floor_decay_steps:
            r = total_steps / args.eps_floor_decay_steps
            return args.eps_floor_start * (1 - r) + args.eps_floor_end * r
        return args.eps_floor_end

    def _build_everything_from_original_main(self):
        args = self.args
        device = self.device

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
            "ris_boost_lambda": args.lambda_boost,
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
            "ris_reflection_power_limit_dBm": float(args.ris_reflection_power_limit_dBm),
            "ris_power_control_mode": "power_budget",
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
            "include_sr_incident_pow_groups": bool(args.include_sr_incident_pow_groups),
            "include_mrt_features": bool(getattr(args, "include_mrt_features", False)),
            "mrt_casc_abs_tau": float(getattr(args, "mrt_casc_abs_tau", 1e-12)),
        }
        self.env_config = env_config

        env_config_base = dict(env_config)
        env_config_base["include_mrt_features"] = False
        probe_env_base = LeoSatEnv(config=env_config_base)
        obs_dict_base = probe_env_base.reset()

        base_aid = next(k for k in obs_dict_base.keys() if k != "__all__")

        base_obs_dim = None
        if hasattr(probe_env_base, "get_base_obs_dim") and callable(
            getattr(probe_env_base, "get_base_obs_dim")
        ):
            base_obs_dim = probe_env_base.get_base_obs_dim()
        if base_obs_dim is None:
            base_obs_dim = len(obs_dict_base[base_aid])
        dqn_obs_layout = probe_env_base.get_obs_layout_spec(include_mrt_features=False)
        if dqn_obs_layout is None:
            raise RuntimeError("Base observation layout is unavailable after env.reset()")

        if hasattr(probe_env_base, "close") and callable(getattr(probe_env_base, "close")):
            probe_env_base.close()

        self.env = LeoSatEnv(config=env_config)
        obs_dict_full = self.env.reset()
        full_aid = next(k for k in obs_dict_full.keys() if k != "__all__")
        full_obs_dim = len(obs_dict_full[full_aid])

        profile = (
            "phase2" if bool(env_config.get("include_mrt_features", False)) else "phase2_no_mrt"
        )
        eval_env_config = get_eval_config(
            base_config=env_config,
            mode="standard",
            profile=profile,
        )
        self.eval_env_config = eval_env_config
        self.eval_env = LeoSatEnv(config=eval_env_config)

        n_rbs = int(self.env.num_resource_blocks)
        n_users_total = int(self.env.num_ground_stations)
        k_users_per_rb = int(self.env.num_gs_per_rb)

        dqn_state_dim = int(base_obs_dim)
        power_state_dim = int(base_obs_dim) + n_users_total

        phase_state_mode = str(getattr(args, "phase_state_mode", "legacy_full"))
        phase_compact_include_mrt_features = bool(
            getattr(args, "phase_compact_include_mrt_features", False)
        )

        if phase_compact_include_mrt_features and not bool(
            getattr(args, "include_mrt_features", False)
        ):
            raise ValueError(
                "--phase_compact_include_mrt_features requires --include_mrt_features, "
                "because selected MRT features must exist in the observation."
            )

        phase_obs_layout = None
        if phase_state_mode == "compact_selected":
            if not hasattr(self.env, "get_obs_layout_spec") or not callable(
                getattr(self.env, "get_obs_layout_spec")
            ):
                raise RuntimeError(
                    "Current LeoSatEnv does not provide get_obs_layout_spec(); "
                    "compact_selected phase-state cannot be enabled."
                )

            phase_obs_layout = self.env.get_obs_layout_spec(
                include_mrt_features=bool(getattr(args, "include_mrt_features", False))
            )
            if phase_obs_layout is None:
                raise RuntimeError(
                    "LeoSatEnv.get_obs_layout_spec() returned None. "
                    "Please ensure env.reset() / observation building has run before probing layout."
                )

        phase_state_dim = _compute_phase_state_dim(
            phase_state_mode=phase_state_mode,
            phase_obs_layout=phase_obs_layout,
            obs_dim_actual=int(full_obs_dim),
            n_users_total=n_users_total,
            k_users_per_rb=k_users_per_rb,
            include_selected_mrt=phase_compact_include_mrt_features,
        )

        phase_groups = int(getattr(self.env, "ris_num_groups"))
        self.phase_groups = phase_groups

        self.base_obs_dim = int(base_obs_dim)
        self.full_obs_dim = int(full_obs_dim)
        self.phase_state_mode = phase_state_mode
        self.phase_compact_include_mrt_features = phase_compact_include_mrt_features
        self.phase_obs_layout = None if phase_obs_layout is None else dict(phase_obs_layout)
        self.phase_state_dim = int(phase_state_dim)

        scheduler_build = build_scheduler_networks(
            state_dim=dqn_state_dim,
            n_users=n_users_total,
            k_users_per_rb=k_users_per_rb,
            obs_layout=dqn_obs_layout,
            implementation=getattr(args, "dqn_scheduler_impl", "enumerative"),
            device=device,
            n_resource_blocks=n_rbs,
        )
        self.scheduler_build = scheduler_build
        discrete_actions_map = scheduler_build.discrete_actions_map
        dqn_action_dim = scheduler_build.network_output_dim
        self.kernel_signature = kernel_signature_from_runtime(
            args=args,
            env=self.env,
            scheduler_build=scheduler_build,
            dqn_state_dim=dqn_state_dim,
            power_state_dim=power_state_dim,
            power_action_dim=1 + k_users_per_rb,
            phase_state_dim=phase_state_dim,
        )

        print(
            f"[Phase2Trainer/Topology] num_ground_stations={n_users_total}, "
            f"num_gs_per_rb={k_users_per_rb}, "
            f"{scheduler_log_summary(scheduler_build, n_users_total, k_users_per_rb)}, "
            f"discrete_actions_map={discrete_actions_map}"
        )

        if self.writer is not None:
            self.writer.add_text(
                "Config/Topology",
                scheduler_log_summary(scheduler_build, n_users_total, k_users_per_rb),
                0,
            )
            self.writer.add_scalar("Config/NumGroundStations", float(n_users_total), 0)
            self.writer.add_scalar("Config/NumGsPerRb", float(k_users_per_rb), 0)
            self.writer.add_scalar("Config/DQNActionDim", float(dqn_action_dim), 0)

        power_action_dim = 1 + k_users_per_rb

        learn_amp = _resolve_learn_amp(args)
        phase_policy_dist = _resolve_phase_policy_dist(args)

        q_network = scheduler_build.q_network
        target_q_network = scheduler_build.target_q_network

        power_actor = ActorNetwork(
            state_dim=power_state_dim,
            action_dim=power_action_dim,
            max_power_watt=self.env.satellite_max_power_watt,
        ).to(device)
        power_critic = CriticNetwork(state_dim=power_state_dim).to(device)

        if phase_policy_dist == "gaussian_latent":
            phase_actor = PhaseActorGaussianLatent(
                state_dim=phase_state_dim,
                num_phase_groups=phase_groups,
                learn_amp=learn_amp,
            ).to(device)
        elif phase_policy_dist == "von_mises":
            if bool(learn_amp):
                raise ValueError(
                    "Phase-II build error: phase_policy_dist='von_mises' requires learn_amp=False."
                )
            phase_actor = PhaseActorVonMises(
                state_dim=phase_state_dim,
                num_phase_groups=phase_groups,
            ).to(device)
        else:
            raise ValueError(f"Unsupported phase_policy_dist: {phase_policy_dist}")

        phase_critic = PhaseCritic(state_dim=phase_state_dim).to(device)

        if phase_policy_dist == "gaussian_latent":
            theta_only_enable = bool(learn_amp) and (
                bool(args.force_phase2_uniform_amp)
                or bool(getattr(args, "phase2_alt_enable", False))
            )
            amp_only_enable = (
                bool(learn_amp)
                and bool(getattr(args, "phase2_alt_enable", False))
                and (not bool(args.force_phase2_uniform_amp))
            )

            phase_actor_theta_for_ppo = PhaseActorThetaOnlyWrapper(
                base_actor=phase_actor,
                num_phase_groups=phase_groups,
                enable_theta_only=theta_only_enable,
                learn_amp=learn_amp,
            ).to(device)

            phase_actor_amp_for_ppo = PhaseActorAmpOnlyWrapper(
                base_actor=phase_actor,
                num_phase_groups=phase_groups,
                enable_amp_only=amp_only_enable,
                learn_amp=learn_amp,
            ).to(device)
        else:
            phase_actor_theta_for_ppo = phase_actor
            phase_actor_amp_for_ppo = None

        q_optimizer = optim.Adam(q_network.parameters(), lr=args.lr_dqn)
        power_actor_optim = optim.Adam(power_actor.parameters(), lr=args.lr_power_ppo)
        power_critic_optim = optim.Adam(power_critic.parameters(), lr=args.lr_power_ppo)
        phase_actor_optim = optim.Adam(phase_actor.parameters(), lr=args.lr_phase_ppo)
        phase_critic_optim = optim.Adam(phase_critic.parameters(), lr=args.lr_phase_ppo)

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

        ppo_phase_theta = PPO(
            phase_actor_theta_for_ppo,
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

        if phase_policy_dist == "gaussian_latent":
            ppo_phase_amp = PPO(
                phase_actor_amp_for_ppo,
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
        else:
            ppo_phase_amp = None

        self.phase_policy_dist = phase_policy_dist

        skip_phase1_kernel_load = bool(getattr(args, "skip_phase1_kernel_load", False))

        if getattr(args, "phase2_only", False):
            if (not skip_phase1_kernel_load) and (args.phase1_model_dir is None):
                raise ValueError("Phase-II 模式下必须提供 --phase1_model_dir")

            if not skip_phase1_kernel_load:
                phase1_dir = Path(args.phase1_model_dir)
                phase1_signature = read_kernel_signature(phase1_dir)
                validate_checkpoint_metadata(
                    None if phase1_signature is None else {"kernel_signature": phase1_signature},
                    self.kernel_signature,
                    context=f"Phase-I model directory {phase1_dir}",
                    dimension_fields=KERNEL_DIMENSION_FIELDS,
                )

                def _pick_phase1_ckpt(
                    phase1_dir: Path, base_name: str, prefer: str, strict_best: bool
                ) -> Path:
                    final_path = phase1_dir / f"{base_name}.pth"
                    best_path = phase1_dir / f"{base_name}_best.pth"
                    if prefer == "best":
                        if best_path.is_file():
                            return best_path
                        if strict_best:
                            raise FileNotFoundError(
                                f"[Phase-II] Requested best ckpt but not found: {best_path}"
                            )
                        return final_path
                    return final_path

                q_path = _pick_phase1_ckpt(
                    phase1_dir, "q_network_phase1", args.phase1_ckpt, args.phase1_ckpt_strict
                )
                if q_path.is_file():
                    q_state = torch.load(q_path, map_location=device, weights_only=True)

                    ckpt_w_key = next(
                        (
                            k
                            for k, v in q_state.items()
                            if k.endswith("weight") and getattr(v, "ndim", 0) == 2
                        ),
                        None,
                    )
                    cur_sd = q_network.state_dict()
                    cur_w_key = next(
                        (
                            k
                            for k, v in cur_sd.items()
                            if k.endswith("weight") and getattr(v, "ndim", 0) == 2
                        ),
                        None,
                    )
                    if ckpt_w_key is None or cur_w_key is None:
                        raise RuntimeError(
                            "[Phase-II] Cannot infer first Linear layer weight key for DQN to check input dim."
                        )

                    ckpt_in_dim = int(q_state[ckpt_w_key].shape[1])
                    cur_in_dim = int(cur_sd[cur_w_key].shape[1])

                    if ckpt_in_dim != cur_in_dim:
                        raise RuntimeError(
                            "[Phase-II] Warmup DQN input-dim mismatch.\n"
                            f"  - checkpoint in_dim = {ckpt_in_dim} (from {q_path.name}, key={ckpt_w_key})\n"
                            f"  - current   in_dim = {cur_in_dim} (built from current env/args, key={cur_w_key})\n"
                            f"  - include_sr_incident_pow_groups = {bool(getattr(args, 'include_sr_incident_pow_groups', False))}\n"
                            "\n"
                            "This usually happens because you enabled/disabled 'incident power' feature compared to warmup.\n"
                            "Fix options:\n"
                            "  (1) Default recommendation: keep incident power OFF (do NOT pass --p2_include_sr_incident_pow_groups)\n"
                            "  (2) If you want incident power ON, you MUST use a warmup model trained with the same feature ON.\n"
                        )

                    q_network.load_state_dict(q_state, strict=True)
                    target_q_network.load_state_dict(q_state, strict=True)
                else:
                    print(f"[Phase-II] WARNING: Phase-I DQN checkpoint not found at {q_path}")

                pa_path = _pick_phase1_ckpt(
                    phase1_dir, "power_actor_phase1", args.phase1_ckpt, args.phase1_ckpt_strict
                )
                pc_path = _pick_phase1_ckpt(
                    phase1_dir, "power_critic_phase1", args.phase1_ckpt, args.phase1_ckpt_strict
                )

                if pa_path.is_file():
                    power_actor.load_state_dict(
                        torch.load(pa_path, map_location=device, weights_only=True)
                    )
                else:
                    print(
                        f"[Phase-II] WARNING: Phase-I power actor checkpoint not found at {pa_path}"
                    )

                if pc_path.is_file():
                    power_critic.load_state_dict(
                        torch.load(pc_path, map_location=device, weights_only=True)
                    )
                else:
                    print(
                        f"[Phase-II] WARNING: Phase-I power critic checkpoint not found at {pc_path}"
                    )

            if args.freeze_dqn_power:
                for p in q_network.parameters():
                    p.requires_grad = False
                for p in target_q_network.parameters():
                    p.requires_grad = False
                for p in power_actor.parameters():
                    p.requires_grad = False
                for p in power_critic.parameters():
                    p.requires_grad = False

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
                max_power_watt=self.env.satellite_max_power_watt,
                device=device,
                ris_num_groups=phase_groups,
                phase_wrap_mode=args.phase_wrap_mode,
                force_phase2_uniform_amp=bool(args.force_phase2_uniform_amp),
                include_sr_incident_pow_groups=bool(args.include_sr_incident_pow_groups),
                learn_amp_weights=_resolve_learn_amp(args),
                phase_state_mode=self.phase_state_mode,
                phase_obs_layout=self.phase_obs_layout,
                phase_compact_include_mrt_features=self.phase_compact_include_mrt_features,
                phase_policy_dist=self.phase_policy_dist,
                dqn_scheduler_impl=scheduler_build.implementation,
                k_users_per_rb=k_users_per_rb,
            )
            for agent_id in self.env.agents
        }

        replay_buffer = ReplayBuffer(
            capacity=int(2e5),
            state_dim=dqn_state_dim,
            action_dim=scheduler_build.replay_action_dim,
        )

        power_buffers = {aid: TrajectoryBuffer() for aid in self.env.agents}
        phase_buffers = {aid: TrajectoryBuffer() for aid in self.env.agents}

        self.dqn_state_dim = dqn_state_dim
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.power_actor = power_actor
        self.power_critic = power_critic
        self.phase_actor = phase_actor
        self.phase_critic = phase_critic

        self.q_optimizer = q_optimizer
        self.power_actor_optim = power_actor_optim
        self.power_critic_optim = power_critic_optim
        self.phase_actor_optim = phase_actor_optim
        self.phase_critic_optim = phase_critic_optim

        self.dqn_learner = dqn_learner
        self.ppo_power = ppo_power
        self.ppo_phase_theta = ppo_phase_theta
        self.ppo_phase_amp = ppo_phase_amp

        self.agents = agents
        self.replay_buffer = replay_buffer
        self.power_buffers = power_buffers
        self.phase_buffers = phase_buffers

    def _init_training_state_from_original_main(self):
        args = self.args
        segment_order = [("PHASE", int(args.cycle_phase))]

        self.segment_order = segment_order
        self.segment_idx = 0
        self.segment_ep_remaining = segment_order[self.segment_idx][1]

        self.total_steps = 0
        self.episode_count = 0
        self.current_epsilon = args.epsilon_start

        self.cycles_done = 0
        self.episodes_done = 0

    def reset_buffers(
        self, *, clear_replay: bool = True, clear_power: bool = True, clear_phase: bool = True
    ):
        if clear_replay and hasattr(self, "replay_buffer") and self.replay_buffer is not None:
            self.replay_buffer.clear()

        if clear_power and hasattr(self, "power_buffers") and self.power_buffers is not None:
            for b in self.power_buffers.values():
                if hasattr(b, "clear"):
                    b.clear()

        if clear_phase and hasattr(self, "phase_buffers") and self.phase_buffers is not None:
            for b in self.phase_buffers.values():
                if hasattr(b, "clear"):
                    b.clear()

    def set_mode(self, mode: str, *, clear_buffers_on_switch: bool = True):
        mode = str(mode).upper().strip()

        if mode != "P2":
            raise ValueError(
                "[Phase2Trainer.set_mode] atprl.training.stage2 is Phase-II-only. "
                "Only mode='P2' is supported. Use the dedicated Phase-I trainer for P1."
            )

        if getattr(self, "mode", None) == "P2" and getattr(self, "kernel_frozen", None) is True:
            if (
                (not isinstance(getattr(self, "segment_order", None), list))
                or len(self.segment_order) != 1
                or self.segment_order[0][0] != "PHASE"
            ):
                self.segment_order = [("PHASE", int(getattr(self.args, "cycle_phase", 1)))]
                self.segment_idx = 0
                self.segment_ep_remaining = self.segment_order[0][1]
            return

        self.mode = "P2"
        self.kernel_frozen = True
        self.args.kernel_frozen = True

        def _set_req_grad(module, flag: bool):
            for p in module.parameters():
                p.requires_grad = bool(flag)

        _set_req_grad(self.q_network, False)
        _set_req_grad(self.target_q_network, False)
        _set_req_grad(self.power_actor, False)
        _set_req_grad(self.power_critic, False)

        if clear_buffers_on_switch:
            self.reset_buffers(clear_replay=True, clear_power=True, clear_phase=True)

        self.segment_order = [("PHASE", int(getattr(self.args, "cycle_phase", 1)))]
        self.segment_idx = 0
        self.segment_ep_remaining = self.segment_order[0][1]

    def load_kernel_state(
        self, kernel_state: dict, *, strict: bool = True, load_optim: bool = False
    ):
        """Load DQN and power-control state without replacing the phase controller."""
        validate_checkpoint_metadata(
            kernel_state.get("meta"),
            self.kernel_signature,
            context="Phase-II kernel checkpoint",
            dimension_fields=KERNEL_DIMENSION_FIELDS,
        )
        m = kernel_state.get("models", kernel_state)
        self.q_network.load_state_dict(m["q_network"], strict=strict)
        self.target_q_network.load_state_dict(m["target_q_network"], strict=strict)
        self.power_actor.load_state_dict(m["power_actor"], strict=strict)
        self.power_critic.load_state_dict(m["power_critic"], strict=strict)

        if load_optim:
            o = kernel_state.get("optims", {})
            if o:
                self.q_optimizer.load_state_dict(o["q_optimizer"])
                self.power_actor_optim.load_state_dict(o["power_actor_optim"])
                self.power_critic_optim.load_state_dict(o["power_critic_optim"])

    def _eval_step_robust(self, global_step_index: int):
        args = self.args

        seed_list = list(VALIDATION_SEEDS)
        k = int(getattr(args, "curve_k", 0) or 0)
        if k > 0:
            seed_list = seed_list[:k]

        profile = (
            "phase2" if bool(getattr(args, "include_mrt_features", False)) else "phase2_no_mrt"
        )

        metrics_pack = run_multi_seed_validation_metrics(
            base_env_config=self.eval_env_config,
            agents=self.agents,
            args=args,
            seed_list=seed_list,
            profile=profile,
            mode="standard",
            force_fixed_positions=True,
        )

        pure_val = float(metrics_pack["metrics"]["undiscounted_return"]["mean"])

        if self.writer is not None:
            self.writer.add_scalar(
                f"{args.eval_tag_steps}_Pure/UndiscountedReturn", pure_val, global_step_index
            )

    def _eval_and_maybe_early_stop(self, eval_episode_index: int, *, allow_early_stop: bool = True):
        args = self.args

        seed_list = list(VALIDATION_SEEDS)
        k = int(getattr(args, "curve_k", 0) or 0)
        if k > 0:
            seed_list = seed_list[:k]

        profile = (
            "phase2" if bool(getattr(args, "include_mrt_features", False)) else "phase2_no_mrt"
        )

        pure_pack = run_multi_seed_validation_metrics(
            base_env_config=self.eval_env_config,
            agents=self.agents,
            args=args,
            seed_list=seed_list,
            profile=profile,
            mode="standard",
            force_fixed_positions=True,
        )

        pure_val = float(pure_pack["metrics"]["undiscounted_return"]["mean"])

        record = build_eval_curve_record(
            episodes_done=int(eval_episode_index),
            metrics_pack=pure_pack,
        )

        append_eval_curve_json(
            self.curve_json_path,
            record,
            meta={
                "exp_name": str(getattr(args, "exp_name", "")),
                "script": "train_phase2_alternating_PBT_v45.py",
                "curve_kind": "daily_like_detailed_values",
                "scheduler_mode": str(getattr(args, "scheduler_mode", "dqn")).strip().lower(),
                "power_mode": _resolve_power_mode(args),
                "phase_policy_dist": _resolve_phase_policy_dist(args),
                "eval_policy": _get_phase2_eval_policy_name(args),
                "phase2_only": bool(getattr(args, "phase2_only", True)),
                "freeze_dqn_power": bool(getattr(args, "freeze_dqn_power", True)),
            },
        )

        if self.writer is not None:
            self.writer.add_scalar(
                "EpisodeEval_Pure/UndiscountedReturn", float(pure_val), eval_episode_index
            )

        improved = pure_val > self.best_eval_return + args.min_improvement
        if improved:
            self.best_eval_return = pure_val
            self.best_eval_episode = eval_episode_index
            self.eval_no_improve_count = 0
            if args.save_model:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.phase_actor.state_dict(), self.model_dir / "phase_actor_phase2_best.pth"
                )
                torch.save(
                    self.phase_critic.state_dict(), self.model_dir / "phase_critic_phase2_best.pth"
                )
                write_kernel_signature(self.model_dir, self.kernel_signature)
        else:
            self.eval_no_improve_count += 1
            if (
                allow_early_stop
                and args.early_stop_patience_eval > 0
                and self.eval_no_improve_count >= args.early_stop_patience_eval
            ):
                self.training_should_stop = True

    def train_for(
        self,
        num_episodes: int,
        *,
        allow_internal_eval: bool = True,
        allow_early_stop: bool = True,
    ):
        args = self.args
        device = self.device

        allow_internal_eval = bool(allow_internal_eval)
        allow_early_stop = bool(allow_early_stop)

        target_episodes = self.episodes_done + int(num_episodes)
        if target_episodes <= self.episodes_done:
            return {"episodes_done": self.episodes_done, "total_steps": self.total_steps}

        env = self.env
        agents = self.agents
        writer = self.writer

        segment_order = self.segment_order

        self.segment_order = [("PHASE", int(getattr(args, "cycle_phase", 1)))]
        self.segment_idx = 0

        if (
            (not isinstance(self.segment_idx, int))
            or (self.segment_idx < 0)
            or (self.segment_idx >= len(self.segment_order))
        ):
            self.segment_idx = int(self.segment_idx) % len(self.segment_order)

        seg_len = int(self.segment_order[self.segment_idx][1])
        if (
            (not isinstance(self.segment_ep_remaining, int))
            or (self.segment_ep_remaining <= 0)
            or (self.segment_ep_remaining > seg_len)
        ):
            self.segment_ep_remaining = seg_len

        segment_order = self.segment_order

        while self.episodes_done < target_episodes:
            if self.training_should_stop:
                break

            seg_type, _ = segment_order[self.segment_idx]
            current_eps_floor = self._get_eps_floor(self.total_steps)

            if seg_type == "DQN":
                self.episode_count += 1
                observations = env.reset()
                episode_total_reward = 0.0

                for agent in agents.values():
                    agent.power_actor.eval()
                    agent.power_critic.eval()
                    agent.phase_actor.eval()
                    agent.phase_critic.eval()

                for _ in range(env.config["max_steps_per_episode"]):
                    actions_to_env, action_info_map = {}, {}
                    with torch.no_grad():
                        for agent_id, agent in agents.items():
                            local_obs = observations[agent_id]
                            act_info = agent.select_action(
                                local_observation=local_obs,
                                current_epsilon=self.current_epsilon,
                                current_min_private_power_fraction=current_eps_floor,
                                deterministic_dqn=False,
                                deterministic_power=False,
                                deterministic_phase=True,
                                scheduler_mode=self.scheduler_mode,
                            )
                            action_info_map[agent_id] = act_info
                            actions_to_env[agent_id] = act_info["env_action"]

                    next_observations, rewards, dones, infos = env.step(actions_to_env)

                    for agent_id in env.agents:
                        obs_full = observations[agent_id]
                        next_obs_full = next_observations[agent_id]

                        obs_as_dqn_state = obs_full[: self.dqn_state_dim]
                        next_obs_as_dqn_state = next_obs_full[: self.dqn_state_dim]

                        reward = rewards[agent_id]
                        info = action_info_map[agent_id]

                        info_env = infos.get(agent_id, {})
                        terminated_flag = bool(info_env.get("terminated", False))

                        self.replay_buffer.store(
                            obs_as_dqn_state,
                            info["scheduler_action"],
                            reward,
                            next_obs_as_dqn_state,
                            float(terminated_flag),
                        )

                        power_vector_watt = info["env_action"][1]
                        common_ratio = power_vector_watt[0] / env.satellite_max_power_watt
                        writer.add_scalar(
                            "Power/Common_Stream_Power_Ratio", common_ratio, self.total_steps
                        )

                        info_env = infos.get(agent_id, {})
                        if len(info["env_action"]) >= 3:
                            phi_vec = info["env_action"][2]
                            writer.add_scalar(
                                "RIS/MeanAbsPhi", float(np.mean(np.abs(phi_vec))), self.total_steps
                            )
                        if "rate_total_sum" in info_env:
                            writer.add_scalar(
                                "Rate/Total", float(info_env["rate_total_sum"]), self.total_steps
                            )
                        if "sic_success" in info_env:
                            sic_violation = 0.0 if bool(info_env["sic_success"]) else 1.0
                            writer.add_scalar("SIC/Violation", sic_violation, self.total_steps)

                    rates = env.historical_avg_rates
                    sum_r = float(np.sum(rates))
                    sum_sq_r = float(np.sum(rates**2))
                    jfi = (sum_r**2) / (len(rates) * sum_sq_r + 1e-9)
                    writer.add_scalar("Fairness/Jains_Index", jfi, self.total_steps)
                    writer.add_scalar(
                        "Fairness/Min_User_Rate", float(np.min(rates)), self.total_steps
                    )
                    writer.add_scalar("Fairness/Sum_Rate_Physical", sum_r, self.total_steps)

                    self.total_steps += 1

                    if args.eval_every_steps > 0 and (
                        self.total_steps % args.eval_every_steps == 0
                    ):
                        self._eval_step_robust(self.total_steps)

                    if (
                        len(self.replay_buffer) > args.dqn_batch_size
                        and self.total_steps % args.dqn_update_freq == 0
                    ):
                        batch = self.replay_buffer.sample(args.dqn_batch_size)
                        batch_on_device = {k: v.to(device) for k, v in batch.items()}
                        dqn_loss = self.dqn_learner.update(batch_on_device)
                        writer.add_scalar("Loss/ModeA_DQN", dqn_loss, self.total_steps)

                    if self.total_steps % args.target_update_freq == 0:
                        self.dqn_learner.update_target_network()

                    if self.episodes_done < args.epsilon_decay_episodes:
                        r = (self.episodes_done + 1) / max(1, args.epsilon_decay_episodes)
                        self.current_epsilon = (
                            args.epsilon_start + (args.epsilon_end - args.epsilon_start) * r
                        )
                    else:
                        self.current_epsilon = args.epsilon_end

                    episode_total_reward += float(sum(rewards.values()))
                    observations = next_observations
                    if dones["__all__"]:
                        break

                writer.add_scalar(
                    "Returns/Episode_Return", episode_total_reward, self.episode_count
                )
                writer.add_scalar("Epsilon/Current_Epsilon", self.current_epsilon, self.total_steps)

                prev_done = int(self.episodes_done)
                self.episodes_done += 1

                if allow_internal_eval and args.eval_every_episodes > 0:
                    for idx in _iter_crossed_boundary_indices(
                        prev_done, self.episodes_done, args.eval_every_episodes
                    ):
                        tag_ep = idx * args.eval_every_episodes
                        self._eval_and_maybe_early_stop(
                            eval_episode_index=tag_ep,
                            allow_early_stop=allow_early_stop,
                        )
                        if self.training_should_stop:
                            break

                self.segment_ep_remaining -= 1
                if self.segment_ep_remaining == 0:
                    self.segment_idx = (self.segment_idx + 1) % len(segment_order)
                    self.segment_ep_remaining = segment_order[self.segment_idx][1]
                continue

            if seg_type == "POWER":
                steps_run = run_power_ppo_optimization(
                    env,
                    agents,
                    self.ppo_power,
                    self.power_buffers,
                    writer,
                    self.total_steps,
                    args,
                    current_eps_floor,
                    global_episode_base=self.episodes_done,
                )
                self.total_steps += steps_run
                self.replay_buffer.clear()

                prev_done = int(self.episodes_done)
                self.episodes_done += int(args.ppo_power_opt_episodes)

                if allow_internal_eval and args.eval_every_episodes > 0:
                    for idx in _iter_crossed_boundary_indices(
                        prev_done, self.episodes_done, args.eval_every_episodes
                    ):
                        tag_ep = idx * args.eval_every_episodes
                        self._eval_and_maybe_early_stop(
                            eval_episode_index=tag_ep,
                            allow_early_stop=allow_early_stop,
                        )
                        if self.training_should_stop:
                            break

                self.segment_ep_remaining -= 1
                if self.segment_ep_remaining == 0:
                    self.segment_idx = (self.segment_idx + 1) % len(segment_order)
                    self.segment_ep_remaining = segment_order[self.segment_idx][1]
                continue

            if seg_type == "PHASE":
                if (args.phase_delay_cycles > 0 and self.cycles_done < args.phase_delay_cycles) or (
                    args.phase_delay_episodes > 0 and self.episodes_done < args.phase_delay_episodes
                ):
                    self.segment_idx = (self.segment_idx + 1) % len(segment_order)
                    self.segment_ep_remaining = segment_order[self.segment_idx][1]
                    self.cycles_done += 1
                    continue

                episodes_to_run = min(
                    int(args.ppo_phase_opt_episodes), target_episodes - self.episodes_done
                )
                steps_run = run_phase_ppo_optimization(
                    env,
                    agents,
                    self.ppo_phase_theta,
                    self.ppo_phase_amp,
                    self.phase_buffers,
                    writer,
                    self.total_steps,
                    args,
                    current_eps_floor,
                    global_episode_base=self.episodes_done,
                    n_episodes=episodes_to_run,
                )
                self.total_steps += steps_run
                self.replay_buffer.clear()

                prev_done = int(self.episodes_done)
                self.episodes_done += episodes_to_run

                if allow_internal_eval and args.eval_every_episodes > 0:
                    for idx in _iter_crossed_boundary_indices(
                        prev_done, self.episodes_done, args.eval_every_episodes
                    ):
                        tag_ep = idx * args.eval_every_episodes
                        self._eval_and_maybe_early_stop(
                            eval_episode_index=tag_ep,
                            allow_early_stop=allow_early_stop,
                        )
                        if self.training_should_stop:
                            break

                self.segment_ep_remaining -= 1
                if self.segment_ep_remaining == 0:
                    self.segment_idx = (self.segment_idx + 1) % len(segment_order)
                    self.segment_ep_remaining = segment_order[self.segment_idx][1]
                    self.cycles_done += 1
                continue

        self._last_rng_state = _capture_rng_state()

        return {
            "episodes_done": self.episodes_done,
            "total_steps": self.total_steps,
            "best_eval_return": self.best_eval_return,
            "best_eval_episode": self.best_eval_episode,
        }

    def state_dict(self) -> dict:
        return {
            "models": {
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "power_actor": self.power_actor.state_dict(),
                "power_critic": self.power_critic.state_dict(),
                "phase_actor": self.phase_actor.state_dict(),
                "phase_critic": self.phase_critic.state_dict(),
            },
            "optims": {
                "q_optimizer": self.q_optimizer.state_dict(),
                "power_actor_optim": self.power_actor_optim.state_dict(),
                "power_critic_optim": self.power_critic_optim.state_dict(),
                "phase_actor_optim": self.phase_actor_optim.state_dict(),
                "phase_critic_optim": self.phase_critic_optim.state_dict(),
            },
            "counters": {
                "total_steps": self.total_steps,
                "episode_count": self.episode_count,
                "current_epsilon": self.current_epsilon,
                "segment_idx": self.segment_idx,
                "segment_ep_remaining": self.segment_ep_remaining,
                "cycles_done": self.cycles_done,
                "episodes_done": self.episodes_done,
                "best_eval_return": self.best_eval_return,
                "best_eval_episode": self.best_eval_episode,
                "eval_no_improve_count": self.eval_no_improve_count,
                "training_should_stop": self.training_should_stop,
            },
            "buffers": {
                "replay": None
                if len(self.replay_buffer) == 0
                else replay_buffer_state(self.replay_buffer),
                "power": {
                    aid: trajectory_buffer_state(buffer)
                    for aid, buffer in self.power_buffers.items()
                },
                "phase": {
                    aid: trajectory_buffer_state(buffer)
                    for aid, buffer in self.phase_buffers.items()
                },
            },
            "rng": capture_rng_state(env=self.env),
            "meta": {
                "exp_name": getattr(self.args, "exp_name", ""),
                "dqn_scheduler_impl": self.scheduler_build.implementation,
                "kernel_signature": dict(self.kernel_signature),
            },
        }

    def load_state_dict(self, state: dict, *, strict: bool = True):
        m = state.get("models", {})
        o = state.get("optims", {})
        c = state.get("counters", {})
        r = state.get("rng", None)
        validate_checkpoint_metadata(
            state.get("meta"),
            self.kernel_signature,
            context="Phase-II checkpoint",
        )

        self.q_network.load_state_dict(m["q_network"], strict=strict)
        self.target_q_network.load_state_dict(m["target_q_network"], strict=strict)
        self.power_actor.load_state_dict(m["power_actor"], strict=strict)
        self.power_critic.load_state_dict(m["power_critic"], strict=strict)
        self.phase_actor.load_state_dict(m["phase_actor"], strict=strict)
        self.phase_critic.load_state_dict(m["phase_critic"], strict=strict)

        self.q_optimizer.load_state_dict(o["q_optimizer"])
        self.power_actor_optim.load_state_dict(o["power_actor_optim"])
        self.power_critic_optim.load_state_dict(o["power_critic_optim"])
        self.phase_actor_optim.load_state_dict(o["phase_actor_optim"])
        self.phase_critic_optim.load_state_dict(o["phase_critic_optim"])

        self.total_steps = int(c.get("total_steps", self.total_steps))
        self.episode_count = int(c.get("episode_count", self.episode_count))
        self.current_epsilon = float(c.get("current_epsilon", self.current_epsilon))
        self.segment_idx = int(c.get("segment_idx", self.segment_idx))
        self.segment_ep_remaining = int(c.get("segment_ep_remaining", self.segment_ep_remaining))
        self.cycles_done = int(c.get("cycles_done", self.cycles_done))
        self.episodes_done = int(c.get("episodes_done", self.episodes_done))

        self.best_eval_return = float(c.get("best_eval_return", self.best_eval_return))
        self.best_eval_episode = int(c.get("best_eval_episode", self.best_eval_episode))
        self.eval_no_improve_count = int(c.get("eval_no_improve_count", self.eval_no_improve_count))
        self.training_should_stop = bool(c.get("training_should_stop", self.training_should_stop))

        buffers = state.get("buffers", {})
        replay_state = buffers.get("replay")
        if replay_state is None:
            self.replay_buffer.clear()
        else:
            load_replay_buffer_state(self.replay_buffer, replay_state)
        for name, runtime_buffers in (
            ("power", self.power_buffers),
            ("phase", self.phase_buffers),
        ):
            for buffer in runtime_buffers.values():
                buffer.clear()
            for aid, buffer_state in buffers.get(name, {}).items():
                if aid not in runtime_buffers:
                    raise ValueError(f"Unknown {name} trajectory buffer in checkpoint: {aid}")
                load_trajectory_buffer_state(runtime_buffers[aid], buffer_state)

        self._last_rng_state = r
        restore_rng_state(r, env=self.env)

    def set_hyperparams(
        self,
        *,
        lr_power_ppo: Optional[float] = None,
        entropy_power: Optional[float] = None,
        lr_phase_ppo: Optional[float] = None,
        entropy_phase: Optional[float] = None,
    ) -> None:
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

        if lr_phase_ppo is not None:
            lr_phase_ppo = float(lr_phase_ppo)
            for opt in [self.phase_actor_optim, self.phase_critic_optim]:
                for pg in opt.param_groups:
                    pg["lr"] = lr_phase_ppo
            self.args.lr_phase_ppo = lr_phase_ppo

        if entropy_phase is not None:
            entropy_phase = float(entropy_phase)

            self.ppo_phase_theta.entropy_coef = entropy_phase

            if self.ppo_phase_amp is not None:
                self.ppo_phase_amp.entropy_coef = entropy_phase

            self.args.entropy_coef_phase = entropy_phase

    def save_checkpoint(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: Path, *, strict: bool = True):
        ckpt = torch.load(Path(path), map_location=self.device)
        self.load_state_dict(ckpt, strict=strict)

    def save_final_if_needed(self):
        args = self.args
        if not args.save_model:
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), self.model_dir / "q_network_final.pth")
        torch.save(self.power_actor.state_dict(), self.model_dir / "power_actor_final.pth")
        torch.save(self.power_critic.state_dict(), self.model_dir / "power_critic_final.pth")
        torch.save(self.phase_actor.state_dict(), self.model_dir / "phase_actor_final.pth")
        torch.save(self.phase_critic.state_dict(), self.model_dir / "phase_critic_final.pth")
        write_kernel_signature(self.model_dir, self.kernel_signature)

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass


def main():
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    args = parse_args()

    trainer = Phase2Trainer(args)
    trainer.train_for(
        args.max_train_episodes,
        allow_internal_eval=True,
        allow_early_stop=True,
    )
    trainer.save_final_if_needed()
    trainer.close()


def get_eps_floor(total_steps, args):
    if args.eps_floor_decay_steps <= 0:
        return args.eps_floor_end
    if total_steps < args.eps_floor_decay_steps:
        r = total_steps / args.eps_floor_decay_steps
        return args.eps_floor_start * (1 - r) + args.eps_floor_end * r
    return args.eps_floor_end


if __name__ == "__main__":
    main()
