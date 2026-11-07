from __future__ import annotations

from argparse import Namespace
from typing import Any

import numpy as np
import torch

from atprl.envs.leo_sat import LeoSatEnv
from atprl.nn.models import ActorNetwork, CriticNetwork, PhaseActorGaussianLatent, PhaseCritic
from atprl.rl.agent import DPRA_Agent
from atprl.rl.algorithms import PPO
from atprl.rl.buffers import ReplayBuffer, TrajectoryBuffer
from atprl.rl.scheduler import build_dqn_learner, build_scheduler_networks
from atprl.training.stage2 import _compute_phase_state_dim
from atprl.config.budgets import TOTAL_TRAINING_EPISODES


DEFAULTS = dict(
    seed=42,
    device="cpu",
    num_threads=1,
    max_train_episodes=TOTAL_TRAINING_EPISODES,
    num_ground_stations=4,
    num_gs_per_rb=2,
    topology_profile="paper_n4",
    dqn_scheduler_impl="enumerative",
    scheduler_mode="dqn",
    power_mode="ppo",
    rollout_length=1024,
    max_steps_per_episode=1024,
    replay_capacity=200000,
    dqn_batch_size=128,
    learning_starts=128,
    dqn_update_freq=4,
    target_update_freq=8,
    lr_dqn=0.0005,
    epsilon_start=0.05,
    epsilon_end=0.05,
    epsilon_decay_episodes=1000,
    lr_power_ppo=0.0003,
    lr_phase_ppo=0.00006,
    ppo_power_epochs=12,
    ppo_phase_epochs=6,
    entropy_coef_power=0.012,
    entropy_coef_phase=0.006,
    gamma=0.9,
    gae_lambda=0.95,
    fairness_lambda=3.0,
    fairness_beta=0.05,
    fairness_alpha=0.0,
    fairness_epsilon=0.01,
    csi_error_variance=0.01,
    sic_detection_threshold_dBm=-94.0,
    sic_penalty=0.0,
    ris_reflection_power_limit_dBm=-5.0,
    active_ris_noise_power_dBm=-100.0,
    uav_rx_antenna_gain_dBi=15.0,
    ris_elem_max_amp_gain_dB=40.0,
    ris_num_elem_x=16,
    ris_num_elem_y=16,
    min_rate_req=0.1,
    rho=0.95,
    force_bs_gain_db=None,
    sat_user_gain_db_offsets=None,
    lambda_boost=0.0,
    train_reseed_each_episode=False,
    include_sr_incident_pow_groups=False,
    include_mrt_features=True,
    mrt_casc_abs_tau=1e-12,
    phase_state_mode="compact_selected",
    phase_compact_include_mrt_features=True,
    phase_policy_dist="gaussian_latent",
    phase_wrap_mode="pi",
    learn_amp_weights=False,
    eval_every_episodes=32,
    selection_every_episodes=224,
    k_daily=3,
    k_ready=15,
    num_eval_seeds=500,
    eval_seed_base=10001,
    eval_seed_stride=37,
    checkpoint_every_episodes=16,
    logging_interval=1,
)


def make_args(**overrides: Any) -> Namespace:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown Joint runtime options: {sorted(unknown)}")
    return Namespace(**(DEFAULTS | overrides))


def validate_joint_args(args: Namespace) -> None:
    if (args.dqn_scheduler_impl, args.scheduler_mode, args.power_mode) != (
        "enumerative",
        "dqn",
        "ppo",
    ):
        raise ValueError("Joint requires Enumerative DQN and Power PPO")
    if args.phase_policy_dist != "gaussian_latent" or args.learn_amp_weights:
        raise ValueError(
            "Joint paper baseline requires Gaussian latent phase and fixed amplitude weights"
        )
    length, horizon = int(args.rollout_length), int(args.max_steps_per_episode)
    if length < 2 or horizon < 2 or horizon % length == 1:
        raise ValueError("Joint segmentation would produce a single-sample PPO trajectory")
    for name in (
        "dqn_batch_size",
        "learning_starts",
        "dqn_update_freq",
        "target_update_freq",
        "ppo_power_epochs",
        "ppo_phase_epochs",
        "checkpoint_every_episodes",
        "logging_interval",
        "num_threads",
        "max_train_episodes",
        "num_eval_seeds",
        "eval_seed_stride",
        "selection_every_episodes",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_train_episodes < args.selection_every_episodes:
        raise ValueError("Training horizon must include a scheduled model-selection event")
    if args.replay_capacity < max(args.dqn_batch_size, args.learning_starts):
        raise ValueError("Replay capacity cannot be smaller than the learning gate")
    if not 1 <= args.num_gs_per_rb <= args.num_ground_stations:
        raise ValueError("Topology must satisfy 1 <= K <= N")
    if args.phase_state_mode not in ("legacy_full", "compact_selected", "all_users_no_action"):
        raise ValueError(f"Unsupported phase_state_mode: {args.phase_state_mode}")
    if args.phase_compact_include_mrt_features and (
        not args.include_mrt_features
        or args.phase_state_mode not in ("compact_selected", "all_users_no_action")
    ):
        raise ValueError(
            "Phase MRT state requires compact_selected or all_users_no_action and MRT observations"
        )
    if args.lambda_boost != 0 or args.include_sr_incident_pow_groups:
        raise ValueError("Joint uses the paper's unshaped reward and observation features")


def build_env_config(args: Namespace) -> dict[str, Any]:
    return dict(
        num_ground_stations=args.num_ground_stations,
        num_resource_blocks=1,
        num_gs_per_rb=args.num_gs_per_rb,
        deployment_area_size_m=400.0,
        satellite_max_power_dBm=50.0,
        terrestrial_cell_power_dBm=35.0,
        noise_power_dBm=-114.0,
        sic_detection_threshold_dBm=args.sic_detection_threshold_dBm,
        carrier_frequency_GHz=4.0,
        satellite_antenna_gain_dBi=24.3,
        ground_station_antenna_gain_dBi=24.3,
        satellite_altitude_m=600e3,
        log_normal_shadowing_variance_dB=8.0,
        rician_k_factor=10.0,
        k_factor_ris_to_gs_linear=10.0,
        k_factor_sat_to_ris_linear=30.0,
        max_doppler_frequency_Hz=50.0,
        channel_update_interval_s=0.04,
        num_micro_steps=20,
        min_rate_requirement_bps_hz=args.min_rate_req,
        max_steps_per_episode=args.max_steps_per_episode,
        ris_boost_lambda=args.lambda_boost,
        random_seed=args.seed,
        use_fixed_positions=True,
        topology_profile=args.topology_profile,
        rho=args.rho,
        csi_error_variance=args.csi_error_variance,
        force_bs_gain_db=args.force_bs_gain_db,
        reseed_each_episode=args.train_reseed_each_episode,
        sat_user_gain_db_offsets=args.sat_user_gain_db_offsets,
        include_angle_features=False,
        ris_num_elem_x=args.ris_num_elem_x,
        ris_num_elem_y=args.ris_num_elem_y,
        ris_reflection_power_limit_dBm=args.ris_reflection_power_limit_dBm,
        ris_power_control_mode="power_budget",
        ris_elem_spacing_ratio=0.5,
        active_ris_enabled=True,
        active_ris_noise_power_dBm=args.active_ris_noise_power_dBm,
        disable_ris_path=False,
        uav_initial_position_xyz=[0.0, 0.0, 80.0],
        uav_velocity_per_step_xyz=[0.282842712, 0.282842712, 0.0],
        uav_rx_antenna_gain_dBi=args.uav_rx_antenna_gain_dBi,
        ris_elem_max_amp_gain_dB=args.ris_elem_max_amp_gain_dB,
        fairness_alpha=args.fairness_alpha,
        fairness_beta=args.fairness_beta,
        fairness_epsilon=args.fairness_epsilon,
        fairness_lambda=args.fairness_lambda,
        sic_penalty=args.sic_penalty,
        include_sr_incident_pow_groups=args.include_sr_incident_pow_groups,
        include_mrt_features=args.include_mrt_features,
        mrt_casc_abs_tau=args.mrt_casc_abs_tau,
    )


class JointRuntime:
    model_names = (
        "q_network",
        "target_q_network",
        "power_actor",
        "power_critic",
        "phase_actor",
        "phase_critic",
    )
    optimizer_names = (
        "q_optimizer",
        "power_actor_optim",
        "power_critic_optim",
        "phase_actor_optim",
        "phase_critic_optim",
    )

    def __init__(self, args: Namespace):
        validate_joint_args(args)
        self.args, self.device = args, torch.device(args.device)
        self.env_config = build_env_config(args)
        self.env = LeoSatEnv(self.env_config)
        observations = self.env.reset()
        base_dim = self.env.get_base_obs_dim()
        layout = self.env.get_obs_layout_spec(include_mrt_features=args.include_mrt_features)
        self.dqn_state_dim = int(base_dim)
        full_dim = len(next(iter(observations.values())))
        phase_dim = _compute_phase_state_dim(
            phase_state_mode=args.phase_state_mode,
            phase_obs_layout=layout,
            obs_dim_actual=full_dim,
            n_users_total=args.num_ground_stations,
            k_users_per_rb=args.num_gs_per_rb,
            include_selected_mrt=args.phase_compact_include_mrt_features,
        )
        self.phase_state_dim = phase_dim
        self.phase_state_spec = dict(
            phase_state_mode=args.phase_state_mode,
            phase_state_dim=phase_dim,
            phase_compact_include_mrt_features=args.phase_compact_include_mrt_features,
            include_mrt_features=args.include_mrt_features,
        )
        scheduler = build_scheduler_networks(
            state_dim=base_dim,
            n_users=args.num_ground_stations,
            k_users_per_rb=args.num_gs_per_rb,
            obs_layout=layout,
            implementation="enumerative",
            device=self.device,
        )
        self.q_network, self.target_q_network = scheduler.q_network, scheduler.target_q_network
        power_dim = base_dim + args.num_ground_stations
        self.power_actor = ActorNetwork(
            power_dim, 1 + args.num_gs_per_rb, self.env.satellite_max_power_watt
        ).to(self.device)
        self.power_critic = CriticNetwork(power_dim).to(self.device)
        self.phase_actor = PhaseActorGaussianLatent(
            phase_dim, self.env.ris_num_groups, learn_amp=False
        ).to(self.device)
        self.phase_critic = PhaseCritic(phase_dim).to(self.device)
        for name, model, lr in zip(
            self.optimizer_names,
            (
                self.q_network,
                self.power_actor,
                self.power_critic,
                self.phase_actor,
                self.phase_critic,
            ),
            (
                args.lr_dqn,
                args.lr_power_ppo,
                args.lr_power_ppo,
                args.lr_phase_ppo,
                args.lr_phase_ppo,
            ),
        ):
            setattr(self, name, torch.optim.Adam(model.parameters(), lr=lr))
        self.dqn_learner = build_dqn_learner(scheduler, self.q_optimizer, gamma=args.gamma)
        common = dict(gamma=args.gamma, gae_lambda=args.gae_lambda, clip_epsilon=0.2)
        self.ppo_power = PPO(
            self.power_actor,
            self.power_critic,
            self.power_actor_optim,
            self.power_critic_optim,
            common | dict(ppo_epochs=args.ppo_power_epochs, entropy_coef=args.entropy_coef_power),
        )
        self.ppo_phase = PPO(
            self.phase_actor,
            self.phase_critic,
            self.phase_actor_optim,
            self.phase_critic_optim,
            common | dict(ppo_epochs=args.ppo_phase_epochs, entropy_coef=args.entropy_coef_phase),
        )
        self.agents = {
            aid: DPRA_Agent(
                self.q_network,
                self.power_actor,
                self.power_critic,
                self.phase_actor,
                self.phase_critic,
                scheduler.discrete_actions_map,
                args.num_ground_stations,
                base_dim,
                self.env.satellite_max_power_watt,
                device=self.device,
                ris_num_groups=self.env.ris_num_groups,
                phase_wrap_mode=args.phase_wrap_mode,
                learn_amp_weights=False,
                phase_state_mode=args.phase_state_mode,
                phase_obs_layout=layout,
                phase_compact_include_mrt_features=args.phase_compact_include_mrt_features,
                phase_policy_dist=args.phase_policy_dist,
                dqn_scheduler_impl="enumerative",
                k_users_per_rb=args.num_gs_per_rb,
            )
            for aid in self.env.agents
        }
        self.replay_buffer = ReplayBuffer(args.replay_capacity, base_dim, 1)
        self.power_buffer, self.phase_buffer = TrajectoryBuffer(), TrajectoryBuffer()

    def model_state(self) -> dict:
        return {name: getattr(self, name).state_dict() for name in self.model_names}

    def load_models(self, state: dict) -> None:
        for name in self.model_names:
            getattr(self, name).load_state_dict(state[name], strict=True)

    def select(self, observation: np.ndarray, epsilon: float = 0.0, *, deterministic=False) -> dict:
        agent = next(iter(self.agents.values()))
        return agent.select_action(
            observation,
            current_epsilon=epsilon,
            deterministic=deterministic,
            deterministic_amp=True,
            phase_ppo_mode="joint",
            scheduler_mode="dqn",
            power_mode="ppo",
        )
