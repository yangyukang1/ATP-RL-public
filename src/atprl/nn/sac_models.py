import torch
from atprl.rl.sac_distributions import (
    SimplexLogRatioTransform,
    PowerSimplexPolicy,
    PhaseAnglePolicy,
)
from atprl.rl.sac_distributions import WrappedNormalPhasePolicy, TANH_NORMAL, WRAPPED_NORMAL


class SACActor(torch.nn.Module):
    def __init__(self, state_dim, freedom, phase=False, phase_distribution=TANH_NORMAL):
        super().__init__()
        self.phase = phase
        h = 128 if phase else 64
        if phase_distribution not in [TANH_NORMAL, WRAPPED_NORMAL]:
            raise ValueError("Unknown Phase distribution")
        if not phase and phase_distribution != TANH_NORMAL:
            raise ValueError("Phase distribution cannot change Power policy")
        self.phase_distribution = phase_distribution
        layers = [torch.nn.Linear(state_dim, h), torch.nn.ReLU()]
        if phase:
            layers.append(torch.nn.LayerNorm(h))
        self.body = torch.nn.Sequential(*layers, torch.nn.Linear(h, h), torch.nn.ReLU())
        self.mean_head = torch.nn.Linear(h, freedom)
        self.log_std_head = torch.nn.Linear(h, freedom)
        torch.nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        torch.nn.init.zeros_(self.mean_head.bias)
        torch.nn.init.zeros_(self.log_std_head.weight)
        torch.nn.init.zeros_(self.log_std_head.bias)
        if not phase:
            self.transform = SimplexLogRatioTransform(freedom)

    def forward(self, state):
        x = self.body(state)
        m = self.mean_head(x)
        ls = self.log_std_head(x).clamp(-5, 1)
        if self.phase:
            return (
                WrappedNormalPhasePolicy
                if self.phase_distribution == WRAPPED_NORMAL
                else PhaseAnglePolicy
            )(m, ls)
        return PowerSimplexPolicy(m, ls, self.transform)


class PowerSACActor(SACActor):
    def __init__(self, state_dim, k):
        super().__init__(state_dim, k)


class PhaseSACActor(SACActor):
    def __init__(self, state_dim, g, phase_distribution=TANH_NORMAL):
        super().__init__(state_dim, g, True, phase_distribution)


class SACQNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, phase=False):
        super().__init__()
        h = 128 if phase else 64
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, 1),
        )

    def forward(self, state, action):
        return self.network(torch.cat((state, action), -1))
