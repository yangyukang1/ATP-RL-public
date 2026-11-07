import copy, importlib.util, json, math, os, subprocess, unittest
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import logsumexp
from atprl.rl.sac_distributions import (
    WrappedNormalPhasePolicy as WN,
    wrap_to_pi,
    WRAPPED_NORMAL,
    TANH_NORMAL,
    PhaseAnglePolicy,
)
from atprl.rl.sac import SACController, SACConfig
from atprl.config.sac import SACRunConfig
from atprl.training.sac_recovery import digest

torch.set_num_threads(1)
AUDIT = {}


def policy(mu, ls):
    return WN(torch.as_tensor(mu, dtype=torch.double), torch.as_tensor(ls, dtype=torch.double))


def reference(theta, mu, ls):
    d = (theta - mu + np.pi) % (2 * np.pi) - np.pi
    x = (d + 2 * np.pi * np.arange(-40, 41)) / np.exp(ls)
    return logsumexp(-0.5 * x * x - ls - 0.5 * np.log(2 * np.pi))


class WrappedMathTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        folder = os.environ.get("WN_EVIDENCE_DIR")
        if folder:
            p = Path(folder)
            p.mkdir(parents=True, exist_ok=True)
            (p / "mathematics.json").write_text(json.dumps(AUDIT, indent=2), encoding="utf-8")

    def test_reference_density_and_gradients(self):
        maxlp = maxgrad = 0.0
        mp.mp.dps = 90
        for ls in [-5.0, -2.0, 0.0, 1.0]:
            for mu in [0.0, 2.9, -3.1, 2 * np.pi * 1000 + 0.37]:
                for th in [-np.pi, -np.pi + 1e-7, -0.2, 0.37, np.pi - 1e-7]:
                    m = torch.tensor([mu], dtype=torch.double, requires_grad=True)
                    l = torch.tensor([ls], dtype=torch.double, requires_grad=True)
                    p = WN(m, l)
                    theta = torch.tensor([th], dtype=torch.double)
                    y = p.log_prob(theta, reference=False).sum()
                    gm, gl = torch.autograd.grad(y, (m, l))
                    expected = reference(th, mu, ls)
                    maxlp = max(maxlp, abs(y.item() - expected))
                    d = (th - mu + np.pi) % (2 * np.pi) - np.pi
                    k = np.arange(-40, 41)
                    v = d + 2 * np.pi * k
                    logs = -0.5 * (v / np.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi)
                    w = np.exp(logs - logsumexp(logs))
                    em = np.sum(w * v / np.exp(2 * ls))
                    el = np.sum(w * ((v / np.exp(ls)) ** 2 - 1))
                    maxgrad = max(maxgrad, abs(gm.item() - em), abs(gl.item() - el))
                    np.testing.assert_allclose(
                        [gm.item(), gl.item()], [em, el], rtol=2e-11, atol=2e-8
                    )
        mperr = 0.0
        for mu, th, ls in [(0.0, np.pi, -5.0), (0.7, -np.pi + 1e-9, 1.0), (-3.0, 3.12, 0.0)]:
            mm, tt, ss = map(mp.mpf, [mu, th, ls])
            sig = mp.exp(ss)
            terms = [
                -0.5 * ((tt + 2 * mp.pi * k - mm) / sig) ** 2 - ss - mp.log(2 * mp.pi) / 2
                for k in range(-40, 41)
            ]
            largest = max(terms)
            ref = largest + mp.log(sum(mp.exp(x - largest) for x in terms))
            got = (
                policy([mu], [ls])
                .log_prob(torch.tensor([th], dtype=torch.double), reference=False)
                .item()
            )
            mperr = max(mperr, abs(got - float(ref)))
        self.assertLess(maxlp, 3e-10)
        self.assertLess(mperr, 3e-10)
        AUDIT.update(
            max_logdensity_error_vs_81_images=maxlp,
            max_gradient_absolute_error=maxgrad,
            max_logdensity_error_vs_90digit_reference=mperr,
            relative_density_tail_bound=4e-35,
            sigma_range=[math.exp(-5), math.e],
        )

    def test_adaptive_normalization_and_circular_moment(self):
        normerr = momerr = 0.0
        for ls in [-5.0, -3.0, -1.0, 0.0, 1.0]:
            for mu in [0.0, math.pi - 1e-6, 4 * math.pi + 0.7]:
                p = policy([mu], [ls])
                center = (mu + np.pi) % (2 * np.pi) - np.pi
                sig = np.exp(ls)
                points = sorted(
                    set(
                        [
                            x
                            for offset in [0, -2 * np.pi, 2 * np.pi]
                            for t in [-12, -6, -2, 0, 2, 6, 12]
                            if -np.pi < (x := center + offset + t * sig) < np.pi
                        ]
                    )
                )

                def density(t):
                    return float(
                        p.log_prob(torch.tensor([t], dtype=torch.double), reference=False).exp()
                    )

                kw = dict(points=points, epsabs=2e-11, epsrel=2e-11, limit=300)
                mass = quad(density, -np.pi, np.pi, **kw)[0]
                moment = complex(
                    quad(lambda t: np.cos(t) * density(t), -np.pi, np.pi, **kw)[0],
                    quad(lambda t: np.sin(t) * density(t), -np.pi, np.pi, **kw)[0],
                )
                normerr = max(normerr, abs(mass - 1))
                momerr = max(momerr, abs(moment - np.exp(-(sig**2) / 2 + 1j * mu)))
        self.assertLess(normerr, 2e-9)
        self.assertLess(momerr, 2e-9)
        AUDIT.update(max_adaptive_integral_error=normerr, max_first_moment_error=momerr)

    def test_periods_actions_relative_entropy_and_float32(self):
        x = torch.tensor([[math.pi, -math.pi, 3.13, -3.13]], dtype=torch.double)
        p = policy([[0.2, -0.3, 1.0, -0.7]], [[1.0, -5.0, -0.4, 0.0]])
        torch.testing.assert_close(
            p.log_prob(x), p.log_prob(x + 2 * math.pi), atol=3e-10, rtol=3e-12
        )
        q = policy(p.mean + 200 * math.pi, p.log_std)
        torch.testing.assert_close(p.log_prob(x), q.log_prob(x), atol=3e-8, rtol=3e-11)
        torch.testing.assert_close(
            p.log_prob(x) - p.log_prob(x, reference=False),
            torch.full((1, 1), 4 * math.log(2 * math.pi), dtype=torch.double),
            atol=1e-10,
            rtol=1e-10,
        )
        rng = torch.Generator().manual_seed(7)
        r = p.rsample(rng)
        self.assertTrue(((r.theta >= -math.pi) & (r.theta < math.pi)).all())
        torch.testing.assert_close(
            r.action.reshape(-1, 4, 2).square().sum(-1), torch.ones(1, 4), atol=2e-6, rtol=0
        )
        torch.testing.assert_close(
            r.action,
            torch.stack((r.theta.cos(), r.theta.sin()), -1).flatten(-2).float(),
            atol=2e-7,
            rtol=0,
        )
        state = torch.get_rng_state()
        a = p.deterministic()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.testing.assert_close(a.theta, wrap_to_pi(p.mean))
        torch.testing.assert_close(
            a.action, torch.stack((p.mean.cos(), p.mean.sin()), -1).flatten(-2).float()
        )
        a32 = WN(p.mean.float(), p.log_std.float()).log_prob(x.float())
        ref32 = WN(p.mean.float().double(), p.log_std.float().double()).log_prob(x.float().double())
        torch.testing.assert_close(a32, ref32, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            policy([0.0], [1.01])
        AUDIT["float32_inputs_density_arithmetic"] = (
            "exactly float64 on the represented float32 inputs; XY tolerance 2e-6"
        )

    def test_fixed_action_and_reparameterized_gradcheck_and_cut(self):
        th = torch.tensor([[math.pi - 1e-4, -math.pi + 1e-4]], dtype=torch.double)
        mu = torch.tensor([[0.7, -2.0]], dtype=torch.double, requires_grad=True)
        ls = torch.tensor([[-0.4, 0.6]], dtype=torch.double, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda m, l: WN(m, l).log_prob(th), (mu, ls), eps=1e-6, atol=2e-6, rtol=2e-5
            )
        )
        eps = torch.tensor([[0.4, -0.8]], dtype=torch.double)

        def objective(m, l):
            z = m + l.exp() * eps
            p = WN(m, l)
            return 0.13 * p.log_prob(wrap_to_pi(z)) - (z.cos() + 0.3 * z.sin()).sum(
                -1, keepdim=True
            )

        self.assertTrue(
            torch.autograd.gradcheck(objective, (mu, ls), eps=1e-6, atol=2e-6, rtol=2e-5)
        )
        results = []
        for sign in [-1, 1]:
            m = torch.tensor([[math.pi + sign * 1e-7]], dtype=torch.double, requires_grad=True)
            p = WN(m, torch.zeros_like(m))
            s = p._from_latent(m)
            val = 0.1 * s.log_prob_reference - s.action[:, 0:1]
            g = torch.autograd.grad(val.sum(), m)[0]
            results.append((val.item(), g.item(), s.action.detach(), s.theta.item()))
        self.assertGreater(results[0][3], 0)
        self.assertLess(results[1][3], 0)
        self.assertLess(abs(results[0][0] - results[1][0]), 1e-6)
        self.assertLess(abs(results[0][1] - results[1][1]), 1e-6)
        torch.testing.assert_close(results[0][2], results[1][2], atol=3e-7, rtol=0)

    def test_temperature_sign_and_initial_entropy(self):
        for h, sign in [(-3.0, -1), (-1.0, 1)]:
            loga = torch.tensor(math.log(0.1), requires_grad=True)
            (loga.exp() * (h + 2)).backward()
            self.assertEqual(int(loga.grad.sign()), sign)
        x, w = np.polynomial.hermite.hermgauss(128)
        z = torch.tensor(np.sqrt(2) * x, dtype=torch.double)[:, None].repeat(1, 4)
        p = policy(torch.zeros(1, 4), torch.zeros(1, 4))
        t = PhaseAnglePolicy(torch.zeros(1, 4), torch.zeros(1, 4))
        wn = float(np.sum(w * (-p.log_prob(wrap_to_pi(z)).numpy()[:, 0])) / np.sqrt(np.pi))
        logj = (
            math.log(math.pi) + 2 * (math.log(2) - z - torch.nn.functional.softplus(-2 * z))
        ).sum(-1)
        lp = t.base.log_prob(z).sum(-1) - logj + 4 * math.log(2 * math.pi)
        tanh = float(np.sum(w * (-lp.numpy())) / np.sqrt(np.pi))
        AUDIT.update(
            initial_mu_zero_sigma_one_relative_entropy_WN=wn,
            initial_mu_zero_sigma_one_relative_entropy_Tanh=tanh,
        )

    def test_sample_first_moment(self):
        p = policy(
            torch.tensor([[0.4, -2.0, math.pi - 1e-5, 6 * math.pi + 0.7]]).repeat(40000, 1),
            torch.tensor([[-5.0, -0.5, 0.0, 1.0]]).repeat(40000, 1),
        )
        sample = p.rsample(torch.Generator().manual_seed(963))
        moment = torch.complex(sample.theta.cos(), sample.theta.sin()).mean(0)
        expected = torch.exp(-p.log_std[0].exp().square() / 2) * torch.complex(
            p.mean[0].cos(), p.mean[0].sin()
        )
        self.assertLess(float((moment - expected).abs().max()), 0.012)
        AUDIT["sample_first_moment_max_error_n40000"] = float((moment - expected).abs().max())


class WrappedCompatibilityTests(unittest.TestCase):
    def test_controller_identity_update_and_exact_restore(self):
        c = SACController(
            3,
            4,
            SACConfig(capacity=32, batch_size=4, learning_starts=4),
            phase=True,
            seed=3,
            phase_distribution=WRAPPED_NORMAL,
        )
        before = {n: digest(getattr(c, n).state_dict()) for n in c.model_names}
        for i in range(16):
            s = np.array([i / 20, 0.1, -0.2], np.float32)
            a = c.act(s).action[0].numpy()
            m = c.observe(s, a, 0.5, s + 0.01, False, False)
        self.assertTrue(all(before[n] != digest(getattr(c, n).state_dict()) for n in c.model_names))
        self.assertGreater(c.actor.mean_head.weight.grad.abs().sum().item(), 0)
        state = c.state_dict()
        self.assertEqual(state["phase_distribution"], WRAPPED_NORMAL)
        self.assertEqual(state["version"], 2)
        b = SACController(3, 4, c.config, phase=True, seed=3, phase_distribution=WRAPPED_NORMAL)
        b.load_state_dict(state)
        torch.testing.assert_close(
            c.act(np.zeros(3), True).action, b.act(np.zeros(3), True).action, atol=0, rtol=0
        )
        self.assertEqual(digest(c.update()), digest(b.update()))
        self.assertEqual(digest(c.state_dict()), digest(b.state_dict()))
        old = SACController(3, 4, c.config, phase=True, seed=3)
        with self.assertRaisesRegex(ValueError, "distribution"):
            old.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "distribution"):
            c.load_state_dict(old.state_dict())
        bad = copy.deepcopy(state)
        bad["density_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "density"):
            c.load_state_dict(bad)
        self.assertEqual(old.state_dict()["version"], 1)
        self.assertNotIn("phase_distribution", old.state_dict())
        self.assertNotIn("phase_distribution", SACRunConfig().to_dict())
        self.assertNotIn(
            "phase_distribution", SACRunConfig(phase_distribution=TANH_NORMAL).to_dict()
        )
