import math
import unittest
import torch
from atprl.rl.sac_distributions import (
    SimplexLogRatioTransform,
    PowerSimplexPolicy,
    PhaseAnglePolicy,
    validate_action,
)

torch.set_num_threads(1)


class DistributionTests(unittest.TestCase):
    def test_simplex_inverse_jacobian_and_gradcheck(self):
        for k in [1, 2, 5]:
            tr = SimplexLogRatioTransform(k)
            z = torch.linspace(-0.5, 0.4, k, dtype=torch.double, requires_grad=True)
            a = tr(z)
            validate_action(a)
            torch.testing.assert_close(tr.inverse(a), z)
            jac = torch.autograd.functional.jacobian(lambda x: tr(x)[:k], z)
            torch.testing.assert_close(jac.det().abs().log(), tr.log_abs_det_jacobian(z).squeeze())
            self.assertTrue(torch.autograd.gradcheck(tr, (z,)))

    def test_power_physical_density_and_sample(self):
        m = torch.zeros(2, 2, requires_grad=True)
        ls = torch.zeros_like(m, requires_grad=True)
        p = PowerSimplexPolicy(m, ls, SimplexLogRatioTransform(2))
        r = p.rsample(torch.Generator().manual_seed(9))
        self.assertEqual(r.log_prob_reference.shape, (2, 1))
        validate_action(r.action)
        exact = p.base.log_prob(r.latent).sum(-1, keepdim=True) - p.transform.log_abs_det_jacobian(
            r.latent
        )
        torch.testing.assert_close(r.log_prob_coordinate, exact)
        torch.testing.assert_close(r.log_prob_reference, exact - math.log(2))
        (-r.log_prob_reference.mean() + r.action[:, 1].mean()).backward()
        self.assertTrue(torch.isfinite(m.grad).all())
        self.assertGreater(m.grad.abs().sum().item(), 0)
        self.assertFalse(p.sample().action.requires_grad)
        torch.testing.assert_close(
            p.log_prob(r.action.double()), r.log_prob_reference, atol=1e-6, rtol=1e-6
        )

    def test_phase_density_period_and_gradient(self):
        for g in [1, 4, 8]:
            m = torch.zeros(2, g, dtype=torch.double, requires_grad=True)
            ls = torch.full_like(m, -0.5)
            p = PhaseAnglePolicy(m, ls)
            r = p.rsample(torch.Generator().manual_seed(6))
            validate_action(r.action, True)
            torch.testing.assert_close(p.log_prob(r.theta + 4 * math.pi), r.log_prob_reference)
            expected = p.base.log_prob(r.latent).sum(-1, keepdim=True) - (
                math.log(math.pi) + torch.log1p(-r.latent.tanh().square())
            ).sum(-1, keepdim=True)
            torch.testing.assert_close(r.log_prob_coordinate, expected)
            (r.action[:, 0].mean() + r.log_prob_reference.mean()).backward()
            self.assertTrue(torch.isfinite(m.grad).all())

    def test_density_integrates(self):
        tr = SimplexLogRatioTransform(1)
        p = PowerSimplexPolicy(
            torch.zeros(1, dtype=torch.double), torch.zeros(1, dtype=torch.double), tr
        )
        x = torch.linspace(0.00001, 0.99999, 20000, dtype=torch.double)
        lp = p.log_prob(torch.stack((x, 1 - x), -1), reference=False).squeeze()
        self.assertAlmostEqual(torch.trapezoid(lp.exp(), x).item(), 1, places=5)
        ph = PhaseAnglePolicy(
            torch.zeros(1, dtype=torch.double), torch.zeros(1, dtype=torch.double)
        )
        theta = torch.linspace(-math.pi + 1e-6, math.pi - 1e-6, 20000, dtype=torch.double)
        lp = ph.log_prob(theta[:, None], reference=False).squeeze()
        self.assertAlmostEqual(torch.trapezoid(lp.exp(), theta).item(), 1, places=5)

    def test_unrepresentable_no_silent_clip(self):
        with self.assertRaises(FloatingPointError):
            PhaseAnglePolicy(torch.tensor([[100.0]]), torch.zeros(1, 1)).deterministic()
        with self.assertRaises(FloatingPointError):
            PowerSimplexPolicy(
                torch.tensor([[1000.0]]), torch.zeros(1, 1), SimplexLogRatioTransform(1)
            ).deterministic()

    def test_deterministic_does_not_consume_rng(self):
        torch.manual_seed(8)
        s = torch.get_rng_state().clone()
        PhaseAnglePolicy(torch.zeros(1, 4), torch.zeros(1, 4)).deterministic()
        self.assertTrue(torch.equal(s, torch.get_rng_state()))
