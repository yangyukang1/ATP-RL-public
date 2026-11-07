from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F

TANH_NORMAL = "tanh_normal_v1"
WRAPPED_NORMAL = "wrapped_normal_v1"
WRAPPED_DENSITY_VERSION = "centered_logsumexp_k5_float64_v1"


def wrap_to_pi(theta):
    return torch.remainder(theta + math.pi, 2 * math.pi) - math.pi


class WrappedNormalPhasePolicy:
    """Wrapped-normal angles with density relative to the uniform torus."""

    def __init__(self, mean, log_std):
        self.mean, self.log_std = mean.double(), log_std.double()
        finite(self.mean, self.log_std)
        if bool(((self.log_std < -5) | (self.log_std > 1)).any()):
            raise ValueError("Wrapped-normal density requires actor log_std in [-5,1]")
        self.base = torch.distributions.Normal(self.mean, self.log_std.exp())

    def log_prob(self, theta, *, reference=True):
        theta = theta.double()
        finite(theta)
        delta = wrap_to_pi(theta - self.mean)
        k = torch.arange(-5, 6, dtype=torch.float64, device=delta.device)
        standardized = (delta.unsqueeze(-1) + 2 * math.pi * k) / self.log_std.exp().unsqueeze(-1)
        images = (
            -0.5 * standardized.square() - self.log_std.unsqueeze(-1) - 0.5 * math.log(2 * math.pi)
        )
        lp = torch.logsumexp(images, dim=-1).sum(-1, keepdim=True)
        if reference:
            lp = lp + self.mean.shape[-1] * math.log(2 * math.pi)
        finite(lp)
        return lp

    def _from_latent(self, z):
        theta = wrap_to_pi(z)
        action = torch.stack((z.cos(), z.sin()), -1).flatten(-2).float()
        coordinate = self.log_prob(theta, reference=False)
        reference = coordinate + self.mean.shape[-1] * math.log(2 * math.pi)
        validate_action(action, True)
        return PolicySample(action, coordinate, reference, z, theta)

    def rsample(self, generator=None):
        epsilon = torch.randn(
            self.mean.shape, dtype=torch.float64, device=self.mean.device, generator=generator
        )
        return self._from_latent(self.mean + self.log_std.exp() * epsilon)

    def sample(self, generator=None):
        with torch.no_grad():
            return self.rsample(generator)

    def deterministic(self):
        return self._from_latent(self.mean)


def finite(*values):
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError("Non-finite SAC policy/value")


def validate_action(a, phase=False):
    finite(a)
    if phase:
        if a.shape[-1] % 2 or not torch.allclose(
            a.reshape(*a.shape[:-1], -1, 2).square().sum(-1),
            torch.ones_like(a[..., ::2]),
            atol=2e-6,
            rtol=0,
        ):
            raise ValueError("Phase action must be unit XY pairs")
    elif bool((a < 0).any()) or not torch.allclose(
        a.sum(-1), torch.ones_like(a.sum(-1)), atol=2e-6, rtol=0
    ):
        raise ValueError("Power action must be simplex shares")


@dataclass
class PolicySample:
    action: torch.Tensor
    log_prob_coordinate: torch.Tensor
    log_prob_reference: torch.Tensor
    latent: torch.Tensor
    theta: torch.Tensor | None = None


class SimplexLogRatioTransform(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        if k < 1:
            raise ValueError("K must be positive")
        b = torch.zeros(k + 1, k, dtype=torch.float64)
        for j in range(k):
            b[: j + 1, j] = 1 / math.sqrt((j + 1) * (j + 2))
            b[j + 1, j] = -(j + 1) / math.sqrt((j + 1) * (j + 2))
        self.register_buffer("basis", b)

    def log_shares(self, z):
        return F.log_softmax(z.double() @ self.basis.T, dim=-1)

    def forward(self, z):
        return self.log_shares(z).exp()

    def inverse(self, a):
        validate_action(a)
        if bool((a <= 0).any()):
            raise ValueError("Inverse requires interior simplex")
        return a.double().log() @ self.basis

    def log_abs_det_jacobian(self, z):
        return 0.5 * math.log(self.basis.shape[0]) + self.log_shares(z).sum(-1, keepdim=True)


class PhysicalPolicy:
    def __init__(self, mean, log_std, transform=None, phase=False):
        self.mean, self.log_std = mean.double(), log_std.double()
        finite(self.mean, self.log_std)
        self.transform, self.phase = transform, phase
        self.base = torch.distributions.Normal(self.mean, self.log_std.exp())

    def _from_latent(self, z):
        base_lp = self.base.log_prob(z).sum(-1, keepdim=True)
        if self.phase:
            t = z.tanh()
            if bool((t.abs() >= 1).any()):
                raise FloatingPointError(
                    "Angular transform reached numerical cut; no silent clipping"
                )
            theta = math.pi * t
            log_j = (math.log(math.pi) + 2 * (math.log(2) - z - F.softplus(-2 * z))).sum(
                -1, keepdim=True
            )
            coordinate = base_lp - log_j
            reference = coordinate + self.mean.shape[-1] * math.log(2 * math.pi)
            action = torch.stack((theta.cos(), theta.sin()), -1).flatten(-2).float()
        else:
            log_a = self.transform.log_shares(z)
            action = log_a.exp().float()
            if bool((action <= 0).any()):
                raise FloatingPointError("Simplex transform underflow; no clipping")
            coordinate = base_lp - (0.5 * math.log(action.shape[-1]) + log_a.sum(-1, keepdim=True))
            reference = coordinate - math.lgamma(action.shape[-1])
            theta = None
        validate_action(action, self.phase)
        finite(coordinate, reference)
        return PolicySample(action, coordinate, reference, z, theta)

    def rsample(self, generator=None):
        eps = torch.randn(
            self.mean.shape, dtype=torch.float64, device=self.mean.device, generator=generator
        )
        return self._from_latent(self.mean + self.log_std.exp() * eps)

    def sample(self, generator=None):
        with torch.no_grad():
            return self.rsample(generator)

    def deterministic(self):
        return self._from_latent(self.mean)

    def log_prob(self, physical, *, reference=True):
        if self.phase:
            theta = (physical.double() + math.pi) % (2 * math.pi) - math.pi
            if bool((theta.abs() >= math.pi).any()):
                raise ValueError("Angular cut has zero density")
            z = torch.atanh(theta / math.pi)
        else:
            z = self.transform.inverse(physical)
        s = self._from_latent(z)
        return s.log_prob_reference if reference else s.log_prob_coordinate


class PowerSimplexPolicy(PhysicalPolicy):
    def __init__(self, mean, log_std, transform):
        super().__init__(mean, log_std, transform)


class PhaseAnglePolicy(PhysicalPolicy):
    def __init__(self, mean, log_std):
        super().__init__(mean, log_std, phase=True)
