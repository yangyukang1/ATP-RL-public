from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Dict, Optional

import torch

from atprl.nn.models import QNetwork
from atprl.rl.algorithms import DQN


VALID_DQN_SCHEDULER_IMPLS = ("enumerative",)


def normalize_dqn_scheduler_impl(value: Any) -> str:
    impl = str(value if value is not None else "enumerative").strip().lower()
    if impl not in VALID_DQN_SCHEDULER_IMPLS:
        raise ValueError(
            f"dqn_scheduler_impl must be one of {list(VALID_DQN_SCHEDULER_IMPLS)}, got {value!r}"
        )
    return impl


@dataclass(frozen=True)
class SchedulerBuild:
    q_network: torch.nn.Module
    target_q_network: torch.nn.Module
    discrete_actions_map: Optional[Dict[int, tuple]]
    network_output_dim: int
    replay_action_dim: int
    implementation: str


def build_scheduler_networks(
    *,
    state_dim: int,
    n_users: int,
    k_users_per_rb: int,
    obs_layout: Dict[str, Any],
    implementation: str,
    device: torch.device,
    n_resource_blocks: int = 1,
) -> SchedulerBuild:
    impl = normalize_dqn_scheduler_impl(implementation)
    n_users = int(n_users)
    k_users_per_rb = int(k_users_per_rb)
    if impl == "enumerative":
        user_combos = list(combinations(range(n_users), k_users_per_rb))
        actions = (
            user_combos
            if int(n_resource_blocks) == 1
            else list(product(user_combos, repeat=int(n_resource_blocks)))
        )
        action_map = {idx: action for idx, action in enumerate(actions)}
        output_dim = len(action_map)
        q_network = QNetwork(state_dim=int(state_dim), action_dim=output_dim).to(device)
        target_q_network = QNetwork(state_dim=int(state_dim), action_dim=output_dim).to(device)
        replay_action_dim = 1
    target_q_network.load_state_dict(q_network.state_dict())
    return SchedulerBuild(
        q_network=q_network,
        target_q_network=target_q_network,
        discrete_actions_map=action_map,
        network_output_dim=output_dim,
        replay_action_dim=replay_action_dim,
        implementation=impl,
    )


def build_dqn_learner(
    build: SchedulerBuild,
    optimizer: torch.optim.Optimizer,
    *,
    gamma: float,
    tau: float = 0.005,
):
    config = {"gamma": float(gamma), "tau": float(tau)}
    learner_cls = DQN
    return learner_cls(build.q_network, build.target_q_network, optimizer, config)


def scheduler_log_summary(build: SchedulerBuild, n_users: int, k_users_per_rb: int) -> str:
    return (
        f"SchedulerImpl={build.implementation}, N={int(n_users)}, K={int(k_users_per_rb)}, "
        f"SchedulerOutputDim={build.network_output_dim}, "
        f"SequentialSteps={int(k_users_per_rb) if build.implementation == 'sequential' else 0}"
    )
