import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from torch.distributions import Normal, VonMises, Independent
from typing import Any, Dict, Tuple

HIDDEN_DIM = 64
LOG_STD_MIN = -20
LOG_STD_MAX = 2
PHASE_HIDDEN_DIM = 128
KAPPA_MIN = 1e-3
KAPPA_MAX = 50.0
AMP_LOG_STD_MIN = -5.0
AMP_LOG_STD_MAX = 2.0


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ActorNetwork(nn.Module):
    """Gaussian power logits; the agent maps samples to a power simplex."""

    def __init__(self, state_dim: int, action_dim: int, max_power_watt: float):
        super(ActorNetwork, self).__init__()
        self.max_power_watt = max_power_watt

        self.body = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(HIDDEN_DIM, action_dim)

        self.log_std_head = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, state: torch.Tensor) -> Normal:
        x = self.body(state)

        mean_logits = self.mean_head(x)

        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        return Normal(mean_logits, std)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PhaseActorVonMises(nn.Module):
    def __init__(self, state_dim: int, num_phase_groups: int):
        super().__init__()
        self.num_phase_groups = int(num_phase_groups)

        self.learn_amp = False
        self.phase_dim = self.num_phase_groups
        self.amp_dim = 0
        self.action_dim = self.phase_dim
        self.policy_dist_name = "von_mises"

        self.body = nn.Sequential(
            nn.Linear(state_dim, PHASE_HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(PHASE_HIDDEN_DIM),
            nn.Linear(PHASE_HIDDEN_DIM, PHASE_HIDDEN_DIM),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(PHASE_HIDDEN_DIM, self.num_phase_groups)

        self.kappa_head = nn.Linear(PHASE_HIDDEN_DIM, self.num_phase_groups)

    def _compute_phase_params(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.body(state)

        mu = torch.tanh(self.mu_head(x)) * np.pi

        raw_kappa = self.kappa_head(x)
        kappa = torch.nn.functional.softplus(raw_kappa) + KAPPA_MIN
        kappa = torch.clamp(kappa, min=KAPPA_MIN, max=KAPPA_MAX)

        return mu, kappa

    def forward(self, state: torch.Tensor) -> Independent:
        mu, kappa = self._compute_phase_params(state)
        base_dist = VonMises(loc=mu, concentration=kappa)
        return Independent(base_dist, 1)

    def get_mean_direction(self, state: torch.Tensor) -> torch.Tensor:
        mu, _ = self._compute_phase_params(state)
        return mu

    def get_concentration(self, state: torch.Tensor) -> torch.Tensor:
        _, kappa = self._compute_phase_params(state)
        return kappa

    def forward_with_amp(self, state: torch.Tensor):
        raise NotImplementedError(
            "PhaseActorVonMises is theta-only (1G) in this codebase. "
            "Amplitude is not produced by the actor; it must be computed outside the model."
        )


class PhaseCritic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, PHASE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(PHASE_HIDDEN_DIM, PHASE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(PHASE_HIDDEN_DIM, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PhaseActorGaussianLatent(nn.Module):
    """Gaussian phase coordinates ordered as (x1, y1, ...), with optional amplitude logits."""

    def __init__(self, state_dim: int, num_phase_groups: int, learn_amp: bool = True):
        super().__init__()
        self.num_phase_groups = int(num_phase_groups)
        self.learn_amp = bool(learn_amp)

        self.phase_dim = 2 * self.num_phase_groups
        self.amp_dim = self.num_phase_groups if self.learn_amp else 0
        self.action_dim = self.phase_dim + self.amp_dim

        self.body = nn.Sequential(
            nn.Linear(state_dim, PHASE_HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(PHASE_HIDDEN_DIM),
            nn.Linear(PHASE_HIDDEN_DIM, PHASE_HIDDEN_DIM),
            nn.ReLU(),
        )

        self.mean_phase_head = nn.Linear(PHASE_HIDDEN_DIM, self.phase_dim)
        self.log_std_phase_head = nn.Linear(PHASE_HIDDEN_DIM, self.phase_dim)

        if self.learn_amp:
            self.mean_amp_head = nn.Linear(PHASE_HIDDEN_DIM, self.num_phase_groups)
            self.log_std_amp_head = nn.Linear(PHASE_HIDDEN_DIM, self.num_phase_groups)

    def forward(self, state: torch.Tensor) -> Independent:
        x = self.body(state)

        mean_phase = self.mean_phase_head(x)
        log_std_phase = self.log_std_phase_head(x)
        log_std_phase = torch.clamp(log_std_phase, LOG_STD_MIN, LOG_STD_MAX)
        std_phase = torch.exp(log_std_phase)

        if self.learn_amp:
            mean_amp = self.mean_amp_head(x)
            log_std_amp = self.log_std_amp_head(x)
            log_std_amp = torch.clamp(log_std_amp, AMP_LOG_STD_MIN, AMP_LOG_STD_MAX)
            std_amp = torch.exp(log_std_amp)

            mean = torch.cat([mean_phase, mean_amp], dim=-1)
            std = torch.cat([std_phase, std_amp], dim=-1)
        else:
            mean, std = mean_phase, std_phase

        base = Normal(loc=mean, scale=std)
        return Independent(base, 1)
