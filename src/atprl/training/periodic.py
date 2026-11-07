from atprl.config.budgets import TOTAL_TRAINING_EPISODES
from typing import Dict, Any, Sequence, Optional, Tuple, List

import torch
import torch.optim as optim
import numpy as np
import random
import argparse
import time
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
import json
from atprl.evaluation.utils import (
    get_eval_config,
    FIXED_VALIDATION_SEEDS,
)
import copy


from torch.utils.tensorboard import SummaryWriter

from atprl.envs.leo_sat import LeoSatEnv
from atprl.nn.models import (
    QNetwork,
    ActorNetwork,
    CriticNetwork,
    PhaseActorGaussianLatent,
    PhaseCritic,
)

from atprl.rl.buffers import ReplayBuffer, TrajectoryBuffer
from atprl.rl.algorithms import DQN, PPO
from atprl.rl.agent import DPRA_Agent


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
        "--epsilon_start", type=float, default=1.0, help="Initial exploration rate."
    )
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final exploration rate.")
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
        "--scheduler_mode",
        type=str,
        default="dqn",
        choices=["dqn", "random"],
        help="Discrete scheduling mode: 'dqn' keeps original epsilon-greedy/greedy DQN behavior; "
        "'random' enables Random Scheduling baseline.",
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
        "--eps_floor_start",
        type=float,
        default=0.02,
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
        help="Number of ground stations/users. Current fixed-topology periodic baseline supports 3, 4, or 5.",
    )
    parser.add_argument(
        "--num_gs_per_rb",
        type=int,
        default=2,
        help="Number of users served per RB. Current periodic baseline is designed for K=2.",
    )
    parser.add_argument("--topology_profile", type=str, default="paper_n4")

    parser.add_argument(
        "--cycle_dqn_before_power",
        type=int,
        default=6,
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
        default=TOTAL_TRAINING_EPISODES,
        help="Stop training when total finished episodes (DQN episodes + POWER PPO episodes) reach this number.",
    )
    parser.add_argument(
        "--eval_every_episodes",
        type=int,
        default=16,
        help="Run validation every N finished episodes.",
    )
    parser.add_argument(
        "--epsilon_decay_episodes",
        type=int,
        default=1000,
        help="Linearly decay DQN epsilon over this many finished episodes.",
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

    parser.add_argument("--deployment_area_size_m", type=float, default=400.0)
    parser.add_argument("--satellite_max_power_dBm", type=float, default=50.0)
    parser.add_argument("--terrestrial_cell_power_dBm", type=float, default=35.0)
    parser.add_argument("--noise_power_dBm", type=float, default=-114.0)

    parser.add_argument("--carrier_frequency_GHz", type=float, default=4.0)
    parser.add_argument("--satellite_antenna_gain_dBi", type=float, default=24.3)
    parser.add_argument("--ground_station_antenna_gain_dBi", type=float, default=24.3)
    parser.add_argument("--satellite_altitude_m", type=float, default=600e3)

    parser.add_argument("--rician_k_factor", type=float, default=10.0)
    parser.add_argument("--k_factor_ris_to_gs_linear", type=float, default=10.0)
    parser.add_argument("--k_factor_sat_to_ris_linear", type=float, default=30.0)

    parser.add_argument("--max_doppler_frequency_Hz", type=float, default=50.0)

    parser.add_argument("--channel_update_interval_s", type=float, default=0.04)
    parser.add_argument("--num_micro_steps", type=int, default=20)

    parser.add_argument("--ris_num_elem_x", type=int, default=16)
    parser.add_argument("--ris_num_elem_y", type=int, default=16)
    parser.add_argument("--ris_elem_spacing_ratio", type=float, default=0.5)
    parser.add_argument("--ris_reflection_power_limit_dBm", type=float, default=-5.0)

    parser.add_argument(
        "--active_ris_noise_power_dBm",
        type=float,
        default=-100.0,
        help="Active RIS noise power per element in dBm (align with PBT v50 default).",
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
        help="UAV Rx antenna gain for Sat->UAV(RIS) link (dBi).",
    )
    parser.add_argument(
        "--ris_elem_max_amp_gain_dB",
        type=float,
        default=40.0,
        help="Per-element max amplification gain upper bound (dB, POWER gain).",
    )

    parser.add_argument(
        "--ris_element_amplitude",
        type=float,
        default=2000.0,
        help="[Deprecated] kept for backward compatibility; not used by env_config.",
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
        "--curve_k",
        type=int,
        default=0,
        help="Number of seeds to use for convergence curve evaluation (0 = all seeds).",
    )

    parser.add_argument(
        "--eval_profile",
        type=str,
        default="phase1",
        choices=["phase1", "phase2", "phase2_no_mrt"],
        help="Eval feature profile. Phase-I warmup should use 'phase1'.",
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

    if args.num_ground_stations not in (3, 4, 5):
        raise ValueError(
            "Current periodic baseline only supports --num_ground_stations in {3, 4, 5}, "
            "because no periodic service/target plan is defined for other user populations."
        )

    if args.num_gs_per_rb != 2:
        raise ValueError(
            "Current periodic baseline only supports --num_gs_per_rb=2, "
            "because the built-in periodic plan is defined on user pairs."
        )

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
    if args.ppo_phase_opt_episodes is None:
        args.ppo_phase_opt_episodes = args.ppo_opt_episodes

    if args.ppo_power_epochs is None:
        args.ppo_power_epochs = args.ppo_epochs
    if args.ppo_phase_epochs is None:
        args.ppo_phase_epochs = args.ppo_epochs

    if args.ppo_power_buffer_size is None:
        args.ppo_power_buffer_size = args.ppo_buffer_size
    if args.ppo_phase_buffer_size is None:
        args.ppo_phase_buffer_size = args.ppo_buffer_size

    if args.entropy_coef_power is None:
        args.entropy_coef_power = args.entropy_coef
    if args.entropy_coef_phase is None:
        args.entropy_coef_phase = args.entropy_coef

    if args.lr_power_ppo is None:
        args.lr_power_ppo = args.lr_ppo
    if args.lr_phase_ppo is None:
        args.lr_phase_ppo = args.lr_ppo

    return args


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


def run_power_ppo_optimization(
    env,
    agents,
    ppo_power,
    power_buffers,
    writer,
    global_step_base,
    args,
    current_eps_floor: float,
    pair_to_action_idx: Dict[Tuple[int, ...], int],
    n_episodes: Optional[int] = None,
) -> int:
    seg_eps = int(n_episodes) if (n_episodes is not None) else int(args.ppo_power_opt_episodes)
    seg_eps = max(0, seg_eps)
    print(f"\n--- [Mode B: POWER] PPO intensive optimization for {seg_eps} episodes ---")

    for ag in agents.values():
        ag.power_actor.train()
        ag.power_critic.train()
        ag.phase_actor.eval()
        ag.phase_critic.eval()

    steps_in_mode_b = 0
    progress_bar = tqdm(range(seg_eps), desc="Mode B: Power PPO", leave=False)

    for _ in progress_bar:
        for buf in power_buffers.values():
            buf.clear()

        observations = env.reset()

        terminated_flags = {aid: False for aid in env.agents}
        time_limit_flags = {aid: False for aid in env.agents}

        for t in range(env.config["max_steps_per_episode"]):
            actions_to_env, cache = {}, {}

            for agent_id, agent in agents.items():
                info = _build_periodic_action_info(
                    agent=agent,
                    env=env,
                    local_observation=observations[agent_id],
                    pair_to_action_idx=pair_to_action_idx,
                    step_in_episode=t,
                    deterministic_power=False,
                )
                cache[agent_id] = info
                actions_to_env[agent_id] = info["env_action"]

            next_obs, rewards, dones, infos = env.step(actions_to_env)
            episode_end = bool(dones["__all__"])
            steps_in_mode_b += 1

            for aid in env.agents:
                info_env = infos.get(aid, {}) if isinstance(infos, dict) else {}
                terminated_flags[aid] = bool(info_env.get("terminated", False))
                time_limit_flags[aid] = bool(info_env.get("time_limit_truncated", episode_end))

            for agent_id in env.agents:
                info = cache[agent_id]
                done_for_bootstrap = float(terminated_flags[agent_id])

                power_buffers[agent_id].store(
                    info["power_state"],
                    info["power_action_raw"],
                    float(rewards[agent_id]),
                    done_for_bootstrap,
                    info["power_log_prob"],
                    info["power_value"],
                )

                pw = np.asarray(info["env_action"][1], dtype=np.float32).reshape(-1)
                writer.add_scalar(
                    "Power/ModeB_Power_CommonRatio",
                    float(pw[0] / env.satellite_max_power_watt),
                    global_step_base + steps_in_mode_b,
                )

            buffer_full = min(len(b) for b in power_buffers.values()) >= args.ppo_power_buffer_size
            if buffer_full or episode_end:
                for agent_id, buf in power_buffers.items():
                    if len(buf) == 0:
                        continue

                    batch = {k: v.to(args.device) for k, v in buf.get_batch().items()}

                    with torch.no_grad():
                        next_pair, _ = _get_periodic_pair_and_target(
                            step_in_episode=t + 1,
                            num_ground_stations=int(
                                getattr(
                                    env, "num_ground_stations", env.config["num_ground_stations"]
                                )
                            ),
                            num_gs_per_rb=int(
                                getattr(env, "num_gs_per_rb", env.config["num_gs_per_rb"])
                            ),
                        )
                        next_power_state = _build_periodic_power_state_tensor(
                            agent=agents[agent_id],
                            local_observation=next_obs[agent_id],
                            selected_users=next_pair,
                        ).to(args.device)

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
                    buf.clear()

            observations = next_obs
            if episode_end:
                break

    print(f"--- [Mode B: POWER] finished. Ran for {steps_in_mode_b} steps. ---\n")
    return steps_in_mode_b


def run_phase_ppo_optimization(
    env, agents, ppo_phase, phase_buffers, writer, global_step_base, args, current_eps_floor: float
) -> int:
    print(
        f"\n--- [Mode B: PHASE] PPO intensive optimization for {args.ppo_phase_opt_episodes} episodes ---"
    )

    for agent in agents.values():
        agent.phase_actor.train()
        agent.phase_critic.train()
        agent.power_actor.eval()
        agent.power_critic.eval()

    steps_in_mode_b = 0
    progress_bar = tqdm(range(args.ppo_phase_opt_episodes), desc="Mode B: Phase PPO", leave=False)

    for _ in progress_bar:
        for buf in phase_buffers.values():
            buf.clear()
        observations = env.reset()

        for _ in range(env.config["max_steps_per_episode"]):
            actions_to_env, action_info_map = {}, {}
            with torch.no_grad():
                for agent_id, agent in agents.items():
                    info = agent.select_action(
                        observations[agent_id],
                        current_epsilon=0.0,
                        current_min_private_power_fraction=current_eps_floor,
                        scheduler_mode=str(getattr(args, "scheduler_mode", "dqn")).strip().lower(),
                    )
                    action_info_map[agent_id] = info
                    actions_to_env[agent_id] = info["env_action"]

            next_observations, rewards, dones, infos = env.step(actions_to_env)
            done_flag = dones["__all__"]

            for agent_id in env.agents:
                info = action_info_map[agent_id]
                reward = rewards[agent_id]
                phase_buffers[agent_id].store(
                    info["phase_state"],
                    info["phase_joint_action"],
                    reward,
                    done_flag,
                    info["phase_log_prob"],
                    info["phase_value"],
                )

            steps_in_mode_b += 1
            observations = next_observations

            buffer_is_full = (
                min(len(buf) for buf in phase_buffers.values()) >= args.ppo_phase_buffer_size
            )
            if buffer_is_full or done_flag:
                for agent_id, buffer in phase_buffers.items():
                    if len(buffer) == 0:
                        continue
                    batch = {k: v.to(args.device) for k, v in buffer.get_batch().items()}

                    with torch.no_grad():
                        nxt = agents[agent_id].select_action(
                            next_observations[agent_id],
                            0.0,
                            current_eps_floor,
                            scheduler_mode=str(getattr(args, "scheduler_mode", "dqn"))
                            .strip()
                            .lower(),
                        )
                        next_phase_state = nxt["phase_state"].to(args.device)

                    losses = ppo_phase.update(batch, next_phase_state, done_flag)
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

            if done_flag:
                break

    print(f"--- [Mode B: PHASE] finished. Ran for {steps_in_mode_b} steps. ---\n")
    return steps_in_mode_b


DEFAULT_METRIC_KEYS = (
    "undiscounted_return",
    "avg_sum_rate",
    "min_user_avg_rate",
    "jain_index",
)

PERIODIC_PAIR_CYCLE_12_4U: List[Tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (0, 3),
    (1, 2),
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (0, 3),
    (1, 2),
]

PERIODIC_TARGET_CYCLE_12_4U: List[int] = [
    0,
    2,
    0,
    1,
    3,
    1,
    1,
    3,
    2,
    3,
    0,
    2,
]

PERIODIC_PAIR_CYCLE_12_3U: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 2),
]

PERIODIC_TARGET_CYCLE_12_3U: List[int] = [
    0,
    2,
    1,
    1,
    0,
    2,
    0,
    2,
    1,
    1,
    0,
    2,
]

PERIODIC_PAIR_CYCLE_20_5U: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
]

PERIODIC_TARGET_CYCLE_20_5U: List[int] = [
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    3,
    1,
    2,
    3,
    4,
    2,
    3,
    4,
    3,
    4,
    4,
]


def _get_periodic_cycles_for_topology(
    num_ground_stations: int,
    num_gs_per_rb: int,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    n = int(num_ground_stations)
    k = int(num_gs_per_rb)

    if k != 2:
        raise ValueError(f"Periodic baseline currently only supports K=2, got num_gs_per_rb={k}.")

    if n == 3:
        return list(PERIODIC_PAIR_CYCLE_12_3U), list(PERIODIC_TARGET_CYCLE_12_3U)

    if n == 4:
        return list(PERIODIC_PAIR_CYCLE_12_4U), list(PERIODIC_TARGET_CYCLE_12_4U)

    if n == 5:
        return list(PERIODIC_PAIR_CYCLE_20_5U), list(PERIODIC_TARGET_CYCLE_20_5U)

    raise ValueError(
        f"Unsupported periodic topology: num_ground_stations={n}, num_gs_per_rb={k}. "
        "Supported topologies are only (3,2), (4,2), and (5,2)."
    )


def _validate_periodic_cycles_for_topology(
    num_ground_stations: int,
    num_gs_per_rb: int,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    n = int(num_ground_stations)
    k = int(num_gs_per_rb)

    pair_cycle, target_cycle = _get_periodic_cycles_for_topology(
        num_ground_stations=n,
        num_gs_per_rb=k,
    )

    if len(pair_cycle) != len(target_cycle) or len(pair_cycle) <= 0:
        raise ValueError(
            "Periodic pair/target cycles must have the same positive length. "
            f"Got len(pair_cycle)={len(pair_cycle)}, len(target_cycle)={len(target_cycle)}."
        )

    cycle_len = int(len(pair_cycle))
    service_counts = {u: 0 for u in range(n)}
    target_counts = {u: 0 for u in range(n)}

    pair_visit_counts: Dict[Tuple[int, int], int] = {}
    pair_target_sets: Dict[Tuple[int, int], set] = {}

    for step_idx, (pair_raw, target_raw) in enumerate(zip(pair_cycle, target_cycle)):
        pair = tuple(sorted(int(u) for u in pair_raw))
        target = int(target_raw)

        if len(pair) != k:
            raise ValueError(
                f"Periodic pair length mismatch at step={step_idx}: pair={pair}, expected K={k}."
            )

        if len(set(pair)) != k:
            raise ValueError(
                f"Periodic pair contains duplicate users at step={step_idx}: pair={pair}."
            )

        for u in pair:
            if u < 0 or u >= n:
                raise ValueError(
                    f"Periodic pair user out of range at step={step_idx}: pair={pair}, N={n}."
                )

        if target < 0 or target >= n:
            raise ValueError(
                f"Periodic target out of range at step={step_idx}: target={target}, N={n}."
            )

        if target not in pair:
            raise ValueError(
                f"Periodic target {target} is not in its serving pair {pair} at step={step_idx}."
            )

        for u in pair:
            service_counts[u] += 1
        target_counts[target] += 1

        pair_visit_counts[pair] = pair_visit_counts.get(pair, 0) + 1
        pair_target_sets.setdefault(pair, set()).add(target)

    if len(set(service_counts.values())) != 1:
        raise ValueError(
            f"Periodic pair cycle is not fair for topology (N={n},K={k},L={cycle_len}): {service_counts}"
        )

    if len(set(target_counts.values())) != 1:
        raise ValueError(
            f"Periodic target cycle is not fair for topology (N={n},K={k},L={cycle_len}): {target_counts}"
        )

    if n == 5 and k == 2:
        expected_pairs = {tuple(sorted(p)) for p in combinations(range(n), 2)}

        if set(pair_visit_counts.keys()) != expected_pairs:
            raise ValueError(
                "5-user periodic plan does not cover exactly all C(5,2)=10 pairs. "
                f"Observed pairs={sorted(pair_visit_counts.keys())}"
            )

        bad_visit = {pair: cnt for pair, cnt in pair_visit_counts.items() if cnt != 2}
        if bad_visit:
            raise ValueError(
                "5-user periodic plan requires every pair to appear exactly twice. "
                f"Got bad_visit={bad_visit}"
            )

        bad_target_cover = {
            pair: sorted(list(ts)) for pair, ts in pair_target_sets.items() if ts != set(pair)
        }
        if bad_target_cover:
            raise ValueError(
                "5-user periodic plan requires each pair's two visits to target its two endpoints exactly once. "
                f"Got bad_target_cover={bad_target_cover}"
            )

    return pair_cycle, target_cycle


_validate_periodic_cycles_for_topology(3, 2)
_validate_periodic_cycles_for_topology(4, 2)
_validate_periodic_cycles_for_topology(5, 2)


def _get_phase1_eval_policy_name(args) -> str:
    n = int(getattr(args, "num_ground_stations", -1))
    k = int(getattr(args, "num_gs_per_rb", -1))
    pair_cycle, _ = _get_periodic_cycles_for_topology(
        num_ground_stations=n,
        num_gs_per_rb=k,
    )
    L = int(len(pair_cycle))
    return f"periodic_sched_mean_power_singleuser_mrt_phase_N{n}K{k}L{L}"


def _wrap_phase_to_pi(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float32)
    return ((phi + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _get_periodic_pair_and_target(
    step_in_episode: int,
    num_ground_stations: int,
    num_gs_per_rb: int,
) -> Tuple[Tuple[int, int], int]:
    pair_cycle, target_cycle = _get_periodic_cycles_for_topology(
        num_ground_stations=num_ground_stations,
        num_gs_per_rb=num_gs_per_rb,
    )

    idx = int(step_in_episode) % len(pair_cycle)
    pair = tuple(sorted(int(u) for u in pair_cycle[idx]))
    target_user = int(target_cycle[idx])

    if target_user not in pair:
        raise ValueError(
            f"Periodic target mismatch at step={step_in_episode}: pair={pair}, target={target_user}"
        )

    return pair, target_user


def _build_pair_to_action_idx(
    discrete_actions_map: Dict[int, Tuple[int, ...]],
) -> Dict[Tuple[int, ...], int]:
    out: Dict[Tuple[int, ...], int] = {}
    for idx, combo in discrete_actions_map.items():
        key = tuple(sorted(int(x) for x in combo))
        out[key] = int(idx)
    return out


def _build_periodic_power_state_tensor(
    agent: DPRA_Agent,
    local_observation: np.ndarray,
    selected_users: Tuple[int, ...],
) -> torch.Tensor:
    obs_full = np.asarray(local_observation, dtype=np.float32).reshape(-1)
    dqn_state = obs_full[: agent.dqn_state_dim].astype(np.float32, copy=True)

    alpha = np.zeros(agent.n_users, dtype=np.float32)
    alpha[list(selected_users)] = 1.0

    power_state = np.concatenate([alpha, dqn_state], axis=0).astype(np.float32)
    return torch.from_numpy(power_state).float().to(agent.device)


def _build_periodic_single_user_mrt_phase(
    env: LeoSatEnv,
    target_user: int,
) -> np.ndarray:
    G = int(getattr(env, "ris_num_groups", 0))
    K = int(getattr(env, "num_ground_stations", 0))

    if G <= 0:
        return np.zeros((0,), dtype=np.float32)

    if bool(getattr(env, "disable_ris_path", False)) or bool(
        getattr(env, "config", {}).get("disable_ris_path", False)
    ):
        return np.zeros((G,), dtype=np.float32)

    target_user = int(target_user)
    if target_user < 0 or target_user >= K:
        raise IndexError(f"target_user out of range: {target_user}, K={K}")

    mrt_feat = np.asarray(env.get_mrt_features(), dtype=np.float32).reshape(K, G, 2)
    cos_th = mrt_feat[target_user, :, 0]
    sin_th = mrt_feat[target_user, :, 1]
    phi = np.arctan2(sin_th, cos_th).astype(np.float32)
    return _wrap_phase_to_pi(phi)


def _attach_periodic_phase_to_env_action(
    env: LeoSatEnv,
    selected_users: Tuple[int, ...],
    power_vec: np.ndarray,
    target_user: int,
):
    phi = _build_periodic_single_user_mrt_phase(env, target_user)
    return (list(selected_users), np.asarray(power_vec, dtype=np.float32).copy(), phi)


def _build_periodic_action_info(
    agent: DPRA_Agent,
    env: LeoSatEnv,
    local_observation: np.ndarray,
    pair_to_action_idx: Dict[Tuple[int, ...], int],
    step_in_episode: int,
    *,
    deterministic_power: bool,
) -> Dict[str, Any]:
    selected_users, target_user = _get_periodic_pair_and_target(
        step_in_episode=step_in_episode,
        num_ground_stations=int(
            getattr(env, "num_ground_stations", env.config["num_ground_stations"])
        ),
        num_gs_per_rb=int(getattr(env, "num_gs_per_rb", env.config["num_gs_per_rb"])),
    )
    discrete_action_idx = int(pair_to_action_idx[tuple(sorted(selected_users))])

    power_state_tensor = _build_periodic_power_state_tensor(
        agent=agent,
        local_observation=local_observation,
        selected_users=selected_users,
    )
    power_state_batch = power_state_tensor.unsqueeze(0)

    with torch.no_grad():
        dist_power = agent.power_actor(power_state_batch)
        value_power = agent.power_critic(power_state_batch)

        if bool(deterministic_power):
            power_action_raw = dist_power.mean
        else:
            power_action_raw = dist_power.sample()

        log_prob_power = dist_power.log_prob(power_action_raw)

    power_shares = torch.softmax(power_action_raw, dim=-1)
    final_power_watt = power_shares * float(agent.max_power_watt)
    final_power_watt_np = final_power_watt.detach().cpu().numpy().squeeze(0).astype(np.float32)

    env_action = _attach_periodic_phase_to_env_action(
        env=env,
        selected_users=selected_users,
        power_vec=final_power_watt_np,
        target_user=target_user,
    )

    return {
        "dqn_state": np.asarray(local_observation, dtype=np.float32)
        .reshape(-1)[: agent.dqn_state_dim]
        .astype(np.float32, copy=True),
        "discrete_action_idx": int(discrete_action_idx),
        "selected_users": tuple(sorted(selected_users)),
        "scheduler_mode": "periodic",
        "power_state": power_state_tensor.detach().cpu(),
        "power_action_raw": power_action_raw.squeeze(0).detach().cpu(),
        "power_action": final_power_watt_np,
        "power_log_prob": log_prob_power.squeeze(0).detach().cpu(),
        "power_value": value_power.squeeze(0).detach().cpu(),
        "target_user": int(target_user),
        "env_action": env_action,
    }


class preserve_rng_state_local:
    def __enter__(self):
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)


def _set_agents_eval_mode(agents: Dict[str, Any]) -> None:
    for agent in agents.values():
        for net_name in ("q_network", "power_actor", "power_critic", "phase_actor", "phase_critic"):
            if hasattr(agent, net_name):
                net = getattr(agent, net_name)
                if net is not None and hasattr(net, "eval"):
                    net.eval()


def _set_agents_train_mode(agents: Dict[str, Any]) -> None:
    for agent in agents.values():
        for net_name in ("q_network", "power_actor", "power_critic", "phase_actor", "phase_critic"):
            if hasattr(agent, net_name):
                net = getattr(agent, net_name)
                if net is not None and hasattr(net, "train"):
                    net.train()


def _jain_index_local(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    denom = float(x.size) * float(np.sum(np.square(x)))
    if denom <= 0.0:
        return 0.0
    return float((np.sum(x) ** 2) / denom)


def _infer_num_users_from_env_local(eval_env: LeoSatEnv) -> int:
    try:
        n = int(eval_env.config.get("num_ground_stations", 0))
        if n > 0:
            return n
    except Exception:
        pass

    try:
        n = int(len(np.asarray(eval_env.historical_avg_rates).reshape(-1)))
        if n > 0:
            return n
    except Exception:
        pass

    return 1


def _extract_step_rate_vector_from_infos_local(
    infos: Dict[str, Dict[str, Any]],
    eval_env: LeoSatEnv,
) -> np.ndarray:
    if infos is None:
        infos = {}

    for _, info in infos.items():
        if not isinstance(info, dict):
            continue
        v = info.get("rates_macro_vec", None)
        if v is not None:
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size > 0:
                return arr

    for _, info in infos.items():
        if not isinstance(info, dict):
            continue
        v = info.get("total_rates_per_user", None)
        if v is not None:
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size > 0:
                return arr

    try:
        arr = np.asarray(eval_env.historical_avg_rates, dtype=np.float64).reshape(-1)
        if arr.size > 0:
            return arr
    except Exception:
        pass

    return np.zeros(_infer_num_users_from_env_local(eval_env), dtype=np.float64)


def _make_local_eval_env(eval_env_config: Dict[str, Any], seed: int) -> LeoSatEnv:
    cfg = copy.deepcopy(eval_env_config)
    cfg["random_seed"] = int(seed)
    cfg["reseed_each_episode"] = True
    return LeoSatEnv(config=cfg)


def _run_single_seed_validation_phase1_mrt(
    eval_env_config: Dict[str, Any],
    agents: Dict[str, Any],
    args,
    seed: int,
    pair_to_action_idx: Dict[Tuple[int, ...], int],
) -> Dict[str, float]:
    eval_env = _make_local_eval_env(eval_env_config, seed=int(seed))

    reset_ret = eval_env.reset(seed=int(seed))
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2 and isinstance(reset_ret[1], dict):
        obs = reset_ret[0]
    else:
        obs = reset_ret

    episode_return = 0.0
    n_steps = 0
    sum_rate_per_user_over_steps = None
    sum_rate_total_over_steps = 0.0

    while True:
        actions_to_env = {}

        for agent_id in list(getattr(eval_env, "agents", list(agents.keys()))):
            if agent_id not in agents:
                raise KeyError(f"[Phase1Eval-Periodic] Missing agent for agent_id={agent_id}")

            act_info = _build_periodic_action_info(
                agent=agents[agent_id],
                env=eval_env,
                local_observation=obs[agent_id],
                pair_to_action_idx=pair_to_action_idx,
                step_in_episode=n_steps,
                deterministic_power=True,
            )
            actions_to_env[agent_id] = act_info["env_action"]

        next_obs, rewards, dones, infos = eval_env.step(actions_to_env)

        if isinstance(rewards, dict):
            episode_return += float(sum(float(v) for v in rewards.values()))
        else:
            episode_return += float(rewards)

        step_rate_per_user = _extract_step_rate_vector_from_infos_local(infos, eval_env)
        step_rate_per_user = np.asarray(step_rate_per_user, dtype=np.float64).reshape(-1)

        if sum_rate_per_user_over_steps is None:
            sum_rate_per_user_over_steps = np.zeros_like(step_rate_per_user, dtype=np.float64)

        sum_rate_per_user_over_steps += step_rate_per_user
        sum_rate_total_over_steps += float(np.sum(step_rate_per_user))

        n_steps += 1
        obs = next_obs

        done_flag = bool(dones["__all__"]) if isinstance(dones, dict) else bool(dones)
        if done_flag:
            break

    T = max(1, int(n_steps))
    if sum_rate_per_user_over_steps is None:
        avg_rates = np.zeros(_infer_num_users_from_env_local(eval_env), dtype=np.float64)
        avg_sum_rate = 0.0
    else:
        avg_rates = (sum_rate_per_user_over_steps / float(T)).reshape(-1)
        avg_sum_rate = float(sum_rate_total_over_steps / float(T))

    min_user_avg_rate = float(np.min(avg_rates)) if avg_rates.size > 0 else 0.0
    jfi = _jain_index_local(avg_rates)

    return {
        "undiscounted_return": float(episode_return),
        "avg_sum_rate": float(avg_sum_rate),
        "min_user_avg_rate": float(min_user_avg_rate),
        "jain_index": float(jfi),
        "n_steps": int(n_steps),
    }


def _summarize_metrics_by_seed_local(
    metrics_all: Sequence[Dict[str, float]],
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k in metric_keys:
        vals = [float(m.get(k, 0.0)) for m in metrics_all]
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            out[k] = {"mean": 0.0, "std": 0.0, "values": []}
        else:
            out[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "values": [float(x) for x in arr.tolist()],
            }
    return out


def build_eval_curve_record_local(
    episodes_done: int,
    metrics_pack: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "episodes_done": int(episodes_done),
        "metrics": copy.deepcopy(metrics_pack.get("metrics", {})),
        "seed_list": list(metrics_pack.get("seed_list", [])),
    }


def run_robust_validation_metrics(
    eval_env_config, agents, args, seed_list, pair_to_action_idx
) -> dict:
    seed_list = list(map(int, list(seed_list) if seed_list is not None else []))
    metrics_all = []

    _set_agents_eval_mode(agents)
    try:
        with preserve_rng_state_local(), torch.no_grad():
            for seed in seed_list:
                m = _run_single_seed_validation_phase1_mrt(
                    eval_env_config=eval_env_config,
                    agents=agents,
                    args=args,
                    seed=int(seed),
                    pair_to_action_idx=pair_to_action_idx,
                )
                metrics_all.append(m)
    finally:
        _set_agents_train_mode(agents)

    summary = _summarize_metrics_by_seed_local(
        metrics_all=metrics_all,
        metric_keys=DEFAULT_METRIC_KEYS,
    )

    return {
        "seed_list": seed_list,
        "metrics": summary,
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


def run_robust_validation(eval_env_config, agents, args, seed_list, pair_to_action_idx) -> float:
    m = run_robust_validation_metrics(eval_env_config, agents, args, seed_list, pair_to_action_idx)
    return float(m["metrics"]["undiscounted_return"]["mean"])


def _maybe_eval_between(prev_done: int, new_done: int, args, eval_fn):
    k = int(getattr(args, "eval_every_episodes", 0) or 0)
    if k <= 0:
        return
    prev_done = int(prev_done)
    new_done = int(new_done)

    start = ((prev_done // k) + 1) * k
    for ep in range(start, new_done + 1, k):
        eval_fn(eval_episode_index=ep)


def main():
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    args = parse_args()
    scheduler_mode = "periodic"
    random_scheduling_enabled = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"Try_final_episode_lamda_PBT_runs/{args.exp_name}_{timestamp}")
    if bool(getattr(args, "enable_tb", False)):
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    else:
        writer = NullSummaryWriter(log_dir)
        print(
            f"TensorBoard event logging is disabled by default. Run with --enable_tb to enable it."
        )

    writer.add_text("Config/SchedulerMode", scheduler_mode, 0)
    writer.add_scalar(
        "Scheduler/RandomSchedulingEnabled", 1.0 if random_scheduling_enabled else 0.0, 0
    )
    writer.add_scalar("Config/DisableRISPath", 1.0 if bool(args.disable_ris_path) else 0.0, 0)

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
        "deployment_area_size_m": args.deployment_area_size_m,
        "satellite_max_power_dBm": args.satellite_max_power_dBm,
        "terrestrial_cell_power_dBm": args.terrestrial_cell_power_dBm,
        "noise_power_dBm": args.noise_power_dBm,
        "sic_detection_threshold_dBm": args.sic_detection_threshold_dBm,
        "carrier_frequency_GHz": args.carrier_frequency_GHz,
        "satellite_antenna_gain_dBi": args.satellite_antenna_gain_dBi,
        "ground_station_antenna_gain_dBi": args.ground_station_antenna_gain_dBi,
        "satellite_altitude_m": args.satellite_altitude_m,
        "num_micro_steps": int(args.num_micro_steps),
        "log_normal_shadowing_variance_dB": 8.0,
        "rician_k_factor": args.rician_k_factor,
        "k_factor_ris_to_gs_linear": args.k_factor_ris_to_gs_linear,
        "k_factor_sat_to_ris_linear": args.k_factor_sat_to_ris_linear,
        "max_doppler_frequency_Hz": float(args.max_doppler_frequency_Hz),
        "channel_update_interval_s": float(args.channel_update_interval_s),
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
        "ris_num_elem_x": int(args.ris_num_elem_x),
        "ris_num_elem_y": int(args.ris_num_elem_y),
        "ris_power_control_mode": "power_budget",
        "ris_reflection_power_limit_dBm": float(args.ris_reflection_power_limit_dBm),
        "ris_elem_spacing_ratio": float(args.ris_elem_spacing_ratio),
        "active_ris_enabled": True,
        "active_ris_noise_power_dBm": float(args.active_ris_noise_power_dBm),
        "disable_ris_path": bool(args.disable_ris_path),
        "uav_initial_position_xyz": [0.0, 0.0, 80.0],
        "uav_velocity_per_step_xyz": [0.282842712, 0.282842712, 0.0],
        "fairness_alpha": args.fairness_alpha,
        "fairness_beta": args.fairness_beta,
        "fairness_epsilon": args.fairness_epsilon,
        "fairness_lambda": args.fairness_lambda,
        "sic_penalty": args.sic_penalty,
        "include_sr_incident_pow_groups": args.include_sr_incident_pow_groups,
        "include_mrt_features": False,
        "uav_rx_antenna_gain_dBi": float(args.uav_rx_antenna_gain_dBi),
        "ris_elem_max_amp_gain_dB": float(args.ris_elem_max_amp_gain_dB),
    }
    env = LeoSatEnv(config=env_config)
    eval_env_config = get_eval_config(
        base_config=env_config,
        mode="standard",
        profile=str(getattr(args, "eval_profile", "phase1")),
    )

    eval_env_config["random_seed"] = int(args.eval_env_seed)
    eval_env_config["reseed_each_episode"] = True

    _obs_probe = env.reset()
    _any_agent_id = next(iter(_obs_probe))
    obs_dim = len(_obs_probe[_any_agent_id])

    n_users_total = getattr(env, "num_ground_stations", env.config["num_ground_stations"])
    k_users_per_rb = getattr(env, "num_gs_per_rb", env.config["num_gs_per_rb"])
    G_ris_groups = getattr(env, "ris_num_groups", env.config.get("ris_num_groups", 0))

    dqn_state_dim = obs_dim

    ppo_state_dim = n_users_total + obs_dim

    ppo_action_dim = 1 + k_users_per_rb + G_ris_groups

    from itertools import combinations

    discrete_actions_map = {
        i: combo for i, combo in enumerate(combinations(range(n_users_total), k_users_per_rb))
    }
    pair_to_action_idx = _build_pair_to_action_idx(discrete_actions_map)
    dqn_action_dim = len(discrete_actions_map)

    periodic_pair_cycle, periodic_target_cycle = _validate_periodic_cycles_for_topology(
        num_ground_stations=int(n_users_total),
        num_gs_per_rb=int(k_users_per_rb),
    )

    for pair in periodic_pair_cycle:
        key = tuple(sorted(int(u) for u in pair))
        if key not in pair_to_action_idx:
            raise KeyError(
                f"Periodic pair {key} is not present in discrete_actions_map={discrete_actions_map}."
            )

    print(
        f"[Topology] num_ground_stations={n_users_total}, "
        f"num_gs_per_rb={k_users_per_rb}, "
        f"dqn_action_dim={dqn_action_dim}, "
        f"discrete_actions_map={discrete_actions_map}"
    )
    print(
        f"[PeriodicPlan] cycle_len={len(periodic_pair_cycle)}, "
        f"pair_cycle={periodic_pair_cycle}, "
        f"target_cycle={periodic_target_cycle}"
    )

    writer.add_text(
        "Config/Topology",
        f"N={n_users_total}, K={k_users_per_rb}, |A|={dqn_action_dim}, map={discrete_actions_map}",
        0,
    )
    writer.add_scalar("Config/NumGroundStations", float(n_users_total), 0)
    writer.add_scalar("Config/NumGsPerRb", float(k_users_per_rb), 0)
    writer.add_scalar("Config/DQNActionDim", float(dqn_action_dim), 0)

    writer.add_text(
        "Config/PeriodicPlan",
        f"cycle_len={len(periodic_pair_cycle)}, pair_cycle={periodic_pair_cycle}, target_cycle={periodic_target_cycle}",
        0,
    )
    writer.add_scalar("Config/PeriodicCycleLen", float(len(periodic_pair_cycle)), 0)

    power_state_dim = n_users_total + obs_dim
    power_action_dim = 1 + k_users_per_rb

    phase_state_dim = obs_dim + n_users_total + (1 + k_users_per_rb)
    phase_groups = G_ris_groups

    q_network = QNetwork(state_dim=dqn_state_dim, action_dim=dqn_action_dim).to(device)
    target_q_network = QNetwork(state_dim=dqn_state_dim, action_dim=dqn_action_dim).to(device)
    target_q_network.load_state_dict(q_network.state_dict())

    power_actor = ActorNetwork(
        state_dim=power_state_dim,
        action_dim=power_action_dim,
        max_power_watt=env.satellite_max_power_watt,
    ).to(device)
    power_critic = CriticNetwork(state_dim=power_state_dim).to(device)

    learn_amp_weights_resolved = _resolve_learn_amp_weights(args)

    phase_actor = PhaseActorGaussianLatent(
        state_dim=phase_state_dim,
        num_phase_groups=phase_groups,
        learn_amp=learn_amp_weights_resolved,
    ).to(device)

    phase_critic = PhaseCritic(state_dim=phase_state_dim).to(device)

    phase_latent_dim = (3 if learn_amp_weights_resolved else 2) * phase_groups

    q_optimizer = optim.Adam(q_network.parameters(), lr=args.lr_dqn)
    power_actor_optim = optim.Adam(power_actor.parameters(), lr=args.lr_power_ppo)
    power_critic_optim = optim.Adam(power_critic.parameters(), lr=args.lr_power_ppo)
    phase_actor_optim = optim.Adam(phase_actor.parameters(), lr=args.lr_phase_ppo)
    phase_critic_optim = optim.Adam(phase_critic.parameters(), lr=args.lr_phase_ppo)

    replay_buffer = ReplayBuffer(capacity=int(1e5), state_dim=dqn_state_dim, action_dim=1)
    power_buffers = {agent_id: TrajectoryBuffer() for agent_id in env.agents}
    phase_buffers = {agent_id: TrajectoryBuffer() for agent_id in env.agents}

    dqn_learner = DQN(q_network, target_q_network, q_optimizer, {"gamma": args.gamma, "tau": 0.005})
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
            ris_num_groups=phase_groups,
        )
        for agent_id in env.agents
    }

    print("\n" + "=" * 70)
    print("====== Starting Alternating Training (Episode-Aligned Scheduling) ======")
    print("=" * 70)

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

    def _advance_to_next_active_segment(curr_idx: int) -> Tuple[int, int]:
        n_seg = len(segment_order)
        for _ in range(n_seg):
            curr_idx = (curr_idx + 1) % n_seg
            seg_len = int(segment_order[curr_idx][1])
            if seg_len > 0:
                return curr_idx, seg_len
        raise ValueError("All segment lengths are <= 0. Please enable at least one active segment.")

    def _normalize_active_segment(curr_idx: int) -> Tuple[int, int]:
        n_seg = len(segment_order)
        for _ in range(n_seg):
            seg_len = int(segment_order[curr_idx][1])
            if seg_len > 0:
                return curr_idx, seg_len
            curr_idx = (curr_idx + 1) % n_seg
        raise ValueError("All segment lengths are <= 0. Please enable at least one active segment.")

    segment_idx, segment_ep_remaining = _normalize_active_segment(segment_idx)
    cycles_done = 0
    episodes_done = 0

    progress_bar = tqdm(total=args.max_train_episodes, desc="Alternating Training (episodes)")

    def maybe_eval_on_segment_end():
        if not args.eval_on_segment_end:
            return

        seed_list = list(FIXED_VALIDATION_SEEDS)

        metrics_pack = run_robust_validation_metrics(
            eval_env_config=eval_env_config,
            agents=agents,
            args=args,
            seed_list=seed_list,
            pair_to_action_idx=pair_to_action_idx,
        )

        seg_val = float(metrics_pack["metrics"]["undiscounted_return"]["mean"])
        writer.add_scalar(f"{args.eval_tag_steps}/UndiscountedReturn", seg_val, total_steps)

    def eval_and_maybe_early_stop(eval_episode_index: int):
        nonlocal best_eval_return, best_eval_episode, eval_no_improve_count, training_should_stop

        curve_k = int(getattr(args, "curve_k", 0) or 0)
        seed_list = list(FIXED_VALIDATION_SEEDS)
        if curve_k > 0:
            seed_list = seed_list[:curve_k]

        metrics_pack = run_robust_validation_metrics(
            eval_env_config=eval_env_config,
            agents=agents,
            args=args,
            seed_list=seed_list,
            pair_to_action_idx=pair_to_action_idx,
        )

        pure_val = float(metrics_pack["metrics"]["undiscounted_return"]["mean"])

        writer.add_scalar(f"{args.eval_tag_steps}/UndiscountedReturn", pure_val, eval_episode_index)

        record = build_eval_curve_record_local(
            episodes_done=int(eval_episode_index),
            metrics_pack=metrics_pack,
        )

        append_eval_curve_json(
            curve_json_path,
            record,
            meta={
                "exp_name": str(getattr(args, "exp_name", "")),
                "script": "train_phase1_power_only_lambda_MRT_v6_periodic12.py",
                "curve_kind": "daily_like_detailed_values",
                "scheduler_mode": "periodic",
                "power_mode": "ppo",
                "eval_policy": _get_phase1_eval_policy_name(args),
                "eval_phase_source": (
                    f"periodic_single_user_target_cycle_"
                    f"N{int(n_users_total)}K{int(k_users_per_rb)}L{int(len(periodic_pair_cycle))}"
                ),
                "num_seeds": len(seed_list),
                "num_ground_stations": int(n_users_total),
                "num_gs_per_rb": int(k_users_per_rb),
                "periodic_cycle_len": int(len(periodic_pair_cycle)),
                "periodic_pair_cycle": [list(map(int, p)) for p in periodic_pair_cycle],
                "periodic_target_cycle": [int(x) for x in periodic_target_cycle],
            },
        )

        improved = pure_val > best_eval_return + args.min_improvement
        if improved:
            best_eval_return = pure_val
            best_eval_episode = int(eval_episode_index)
            eval_no_improve_count = 0

            if args.save_model:
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(q_network.state_dict(), model_dir / "q_network_phase1_best.pth")
                torch.save(power_actor.state_dict(), model_dir / "power_actor_phase1_best.pth")
                torch.save(power_critic.state_dict(), model_dir / "power_critic_phase1_best.pth")
        else:
            eval_no_improve_count += 1
            if (
                args.early_stop_patience_eval > 0
                and eval_no_improve_count >= args.early_stop_patience_eval
            ):
                training_should_stop = True

    while episodes_done < args.max_train_episodes:
        if training_should_stop:
            print("[EarlyStop] Flag set, breaking out of training loop.")
            break
        seg_type, _ = segment_order[segment_idx]

        current_eps_floor = get_eps_floor(total_steps, args)

        if seg_type == "DQN":
            episode_count += 1
            observations = env.reset()
            episode_total_reward = 0.0
            q_network.train()

            for agent in agents.values():
                agent.power_actor.eval()
                agent.power_critic.eval()
                agent.phase_actor.eval()
                agent.phase_critic.eval()

            for step_in_episode in range(env.config["max_steps_per_episode"]):
                actions_to_env = {}
                action_cache = {}

                for agent_id, agent in agents.items():
                    act_info = _build_periodic_action_info(
                        agent=agent,
                        env=env,
                        local_observation=observations[agent_id],
                        pair_to_action_idx=pair_to_action_idx,
                        step_in_episode=step_in_episode,
                        deterministic_power=False,
                    )
                    action_cache[agent_id] = act_info
                    actions_to_env[agent_id] = act_info["env_action"]

                next_observations, rewards, dones, infos = env.step(actions_to_env)

                for agent_id in env.agents:
                    info = action_cache[agent_id]

                    obs_as_dqn_state = np.asarray(observations[agent_id], dtype=np.float32)
                    next_obs_as_dqn_state = np.asarray(
                        next_observations[agent_id], dtype=np.float32
                    )
                    discrete_idx = np.asarray([info["discrete_action_idx"]], dtype=np.int64)
                    reward_scalar = np.asarray([rewards[agent_id]], dtype=np.float32)
                    done_scalar = np.asarray([dones["__all__"]], dtype=np.float32)

                    if not random_scheduling_enabled:
                        replay_buffer.store(
                            state=obs_as_dqn_state,
                            action=discrete_idx,
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

                total_steps += 1
                writer.add_scalar(
                    "Scheduler/ModeIsRandom", 1.0 if random_scheduling_enabled else 0.0, total_steps
                )

                if (
                    (not random_scheduling_enabled)
                    and (len(replay_buffer) > args.dqn_batch_size)
                    and (total_steps % args.dqn_update_freq == 0)
                ):
                    batch = replay_buffer.sample(args.dqn_batch_size)
                    batch_on_device = {k: v.to(device) for k, v in batch.items()}
                    dqn_loss = dqn_learner.update(batch_on_device)
                    writer.add_scalar("Loss/ModeA_DQN", dqn_loss, total_steps)

                if (not random_scheduling_enabled) and (total_steps % args.target_update_freq == 0):
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
                if seg_type == "PHASE":
                    cycles_done += 1
                segment_idx, segment_ep_remaining = _advance_to_next_active_segment(segment_idx)
                maybe_eval_on_segment_end()

            continue

        if seg_type == "POWER":
            remaining = int(args.max_train_episodes) - int(episodes_done)
            if remaining <= 0:
                break

            seg_eps = min(int(args.ppo_power_opt_episodes), remaining)

            q_network.eval()

            prev_done = int(episodes_done)

            steps_run = run_power_ppo_optimization(
                env,
                agents,
                ppo_power,
                power_buffers,
                writer,
                total_steps,
                args,
                current_eps_floor,
                pair_to_action_idx=pair_to_action_idx,
                n_episodes=seg_eps,
            )
            total_steps += steps_run
            replay_buffer.clear()

            episodes_done = prev_done + seg_eps
            segment_ep_remaining -= 1
            progress_bar.update(seg_eps)

            _maybe_eval_between(prev_done, episodes_done, args, eval_and_maybe_early_stop)

            if segment_ep_remaining == 0:
                segment_idx, segment_ep_remaining = _advance_to_next_active_segment(segment_idx)
                maybe_eval_on_segment_end()

            continue

        if seg_type == "PHASE":
            if (args.phase_delay_cycles > 0 and cycles_done < args.phase_delay_cycles) or (
                args.phase_delay_episodes > 0 and episodes_done < args.phase_delay_episodes
            ):
                cycles_done += 1
                segment_idx, segment_ep_remaining = _advance_to_next_active_segment(segment_idx)
                maybe_eval_on_segment_end()
                continue

            steps_run = run_phase_ppo_optimization(
                env, agents, ppo_phase, phase_buffers, writer, total_steps, args, current_eps_floor
            )
            total_steps += steps_run
            replay_buffer.clear()
            episodes_done += args.ppo_phase_opt_episodes

            segment_ep_remaining -= 1
            if args.eval_every_steps > 0 and (total_steps % args.eval_every_steps == 0):
                val = run_robust_validation(
                    eval_env_config, agents, args, FIXED_VALIDATION_SEEDS, pair_to_action_idx
                )
                writer.add_scalar(
                    f"{args.eval_tag_steps}/UndiscountedReturn", float(val), total_steps
                )

            if segment_ep_remaining == 0:
                cycles_done += 1
                segment_idx, segment_ep_remaining = _advance_to_next_active_segment(segment_idx)
                maybe_eval_on_segment_end()

            continue

    progress_bar.close()
    if args.save_model:
        model_path = model_dir
        model_path.mkdir(parents=True, exist_ok=True)

        torch.save(q_network.state_dict(), model_path / "q_network_phase1.pth")
        torch.save(power_actor.state_dict(), model_path / "power_actor_phase1.pth")
        torch.save(power_critic.state_dict(), model_path / "power_critic_phase1.pth")
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
    print("\n" + "=" * 70)
    print("====== Alternating Training (Strict On-Policy) Finished! ======")
    print("=" * 70)


def get_eps_floor(total_steps, args):
    if args.eps_floor_decay_steps <= 0:
        return args.eps_floor_end
    if total_steps < args.eps_floor_decay_steps:
        r = total_steps / args.eps_floor_decay_steps
        return args.eps_floor_start * (1 - r) + args.eps_floor_end * r
    return args.eps_floor_end


if __name__ == "__main__":
    main()
