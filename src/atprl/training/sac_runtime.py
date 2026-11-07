import copy, random
from argparse import Namespace
import numpy as np
import torch
from atprl.config.sac import SACRunConfig
from atprl.envs.leo_sat import LeoSatEnv
from atprl.training.joint_runtime import make_args, build_env_config
from atprl.training.stage2 import _compute_phase_state_dim
from atprl.rl.agent import DPRA_Agent
from atprl.rl.scheduler import build_scheduler_networks, build_dqn_learner
from atprl.rl.buffers import ReplayBuffer
from atprl.rl.sac import SACConfig, SACController
from atprl.training.recovery import capture_rng_state, restore_rng_state


class SACActionAdapter:
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, agent, obs, context, power_state, det_power, det_phase):
        rt = self.runtime
        power = rt.power.act(power_state, deterministic=det_power)
        shares = power.action[0].numpy()
        watts = shares * rt.env.satellite_max_power_watt
        info = dict(
            context, power_state=power_state, power_policy_action=shares, power_sample=power
        )
        if rt.phase is not None:
            ps = agent._build_phase_state(obs, context["selected_users"], shares)
            phase = rt.phase.act(ps, deterministic=det_phase)
            theta = phase.theta[0].numpy()
            info.update(
                phase_state=ps, phase_policy_action=phase.action[0].numpy(), phase_sample=phase
            )
            action = (context["selected_users"], watts, theta, np.full(rt.g, 1 / rt.g))
        else:
            action = (context["selected_users"], watts)
        info["env_action"] = action
        return info


class SACSTPRuntime:
    def __init__(self, config, stage="warmup"):
        self.config = config
        self.stage = stage
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        args = make_args(
            seed=config.seed,
            device="cpu",
            num_ground_stations=config.num_ground_stations,
            num_gs_per_rb=config.num_gs_per_rb,
            topology_profile=config.topology_profile,
            ris_num_elem_x=config.ris_num_elem_x,
            ris_num_elem_y=config.ris_num_elem_y,
            max_steps_per_episode=config.macro_steps_per_episode,
            fairness_lambda=config.fairness_lambda,
            csi_error_variance=config.csi_error_variance,
            include_mrt_features=stage == "stage2",
            phase_compact_include_mrt_features=stage == "stage2",
        )
        self.args = args
        self.env_config = build_env_config(args)
        self.env_config["num_micro_steps"] = config.num_micro_steps
        self.env = LeoSatEnv(self.env_config)
        obs = self.env.reset()
        self.aid = self.env.agents[0]
        if len(self.env.agents) != 1:
            raise ValueError("SAC-STP formal protocol is single RB")
        self.base_dim = self.env.get_base_obs_dim()
        self.g = self.env.ris_num_groups
        self.layout = self.env.get_obs_layout_spec(include_mrt_features=stage == "stage2")
        self.scheduler = build_scheduler_networks(
            state_dim=self.base_dim,
            n_users=config.num_ground_stations,
            k_users_per_rb=config.num_gs_per_rb,
            obs_layout=self.layout,
            implementation=config.dqn_scheduler_impl,
            device=torch.device("cpu"),
        )
        self.q_network = self.scheduler.q_network
        self.target_q_network = self.scheduler.target_q_network
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=5e-4)
        self.dqn_learner = build_dqn_learner(self.scheduler, self.q_optimizer, gamma=0.9, tau=0.005)
        self.dqn_replay = ReplayBuffer(
            config.dqn_replay_capacity, self.base_dim, self.scheduler.replay_action_dim
        )
        self.power = SACController(
            self.base_dim + config.num_ground_stations,
            config.num_gs_per_rb,
            SACConfig(**config.power_sac),
            seed=config.seed + 1000,
        )
        self.phase = None
        if stage == "stage2":
            dim = _compute_phase_state_dim(
                "compact_selected",
                self.layout,
                len(obs[self.aid]),
                config.num_ground_stations,
                config.num_gs_per_rb,
                True,
            )
            self.phase = SACController(
                dim,
                self.g,
                SACConfig(**config.phase_sac),
                phase=True,
                seed=config.seed + 2000,
                phase_distribution=config.phase_distribution or "tanh_normal_v1",
            )
        self.agent = DPRA_Agent(
            self.q_network,
            self.power.actor,
            None,
            self.phase.actor if self.phase else None,
            None,
            self.scheduler.discrete_actions_map,
            config.num_ground_stations,
            self.base_dim,
            self.env.satellite_max_power_watt,
            ris_num_groups=self.g,
            learn_amp_weights=False,
            phase_state_mode="compact_selected",
            phase_obs_layout=self.layout,
            phase_compact_include_mrt_features=stage == "stage2",
            dqn_scheduler_impl=config.dqn_scheduler_impl,
            k_users_per_rb=config.num_gs_per_rb,
            continuous_backend=SACActionAdapter(self),
        )
        self.dqn_updates = self.dqn_target_updates = 0
        if stage == "stage2":
            self.freeze_kernel()

    def freeze_kernel(self):
        for model in [self.q_network, self.target_q_network] + [
            getattr(self.power, n) for n in self.power.model_names
        ]:
            model.requires_grad_(False)
        self.power.log_temperature.requires_grad_(False)

    def power_state(self, obs):
        ctx = self.agent.select_scheduler_context(obs, deterministic_dqn=True)
        mask = np.zeros(self.config.num_ground_stations, np.float32)
        mask[list(ctx["selected_users"])] = 1
        return np.concatenate((mask, ctx["dqn_state"])).astype(np.float32)

    def phase_state(self, obs):
        ctx = self.agent.select_scheduler_context(obs, deterministic_dqn=True)
        shares = self.power.act(self.power_state(obs), deterministic=True).action[0].numpy()
        return self.agent._build_phase_state(obs, ctx["selected_users"], shares)

    def select(self, obs, *, mode="POWER", epsilon=0.0, deterministic=False):
        return self.agent.select_action(
            obs,
            current_epsilon=epsilon,
            deterministic_dqn=mode != "DQN",
            deterministic_power=deterministic or self.stage == "stage2",
            deterministic_phase=deterministic,
        )
