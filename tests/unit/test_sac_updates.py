import unittest
import numpy as np
import torch
from atprl.rl.sac import SACController, SACConfig, soft_bellman


class UpdateTests(unittest.TestCase):
    def make(self, phase=False):
        return SACController(
            3, 2, SACConfig(capacity=32, batch_size=4, learning_starts=4), phase=phase, seed=7
        )

    def fill(self, c, n=16):
        metrics = []
        for i in range(n):
            s = np.array([i / 30, 0.2, -0.1], np.float32)
            a = c.act(s).action[0].numpy()
            metrics += c.observe(
                s, a, 0.5, s + 0.01, False, i == n - 1, episode_id=i // 4, macro_step=i % 4
            )
        return metrics

    def test_bellman_flags_and_shapes(self):
        r = torch.ones(4, 1)
        term = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        y = soft_bellman(r, term, 2 * r, 3 * r, -r, 0.1, 0.9)
        torch.testing.assert_close(y, torch.tensor([[2.89], [1.0], [2.89], [1.0]]))
        with self.assertRaises(ValueError):
            soft_bellman(r[:, 0], term, r, r, r, 0.1, 0.9)

    def test_network_updates_and_targets(self):
        for phase in [False, True]:
            c = self.make(phase)
            old = {
                n: {k: v.clone() for k, v in getattr(c, n).state_dict().items()}
                for n in c.model_names
            }
            self.fill(c, 4)
            self.assertEqual(
                (c.updates, c.actor_updates, c.target_updates, c.temperature_updates), (1, 1, 1, 1)
            )
            for online, target in [("q1", "q1_target"), ("q2", "q2_target")]:
                for k, v in getattr(c, target).state_dict().items():
                    torch.testing.assert_close(
                        v,
                        old[target][k] * (1 - c.config.tau)
                        + getattr(c, online).state_dict()[k] * c.config.tau,
                    )
            self.assertFalse(hasattr(c, "actor_target"))
            self.assertTrue(
                any(not torch.equal(v, old["actor"][k]) for k, v in c.actor.state_dict().items())
            )
            self.assertFalse(any(p.grad is not None for p in c.q1_target.parameters()))

    def test_exact_resume_including_adam_and_independent_rng(self):
        a = self.make()
        self.fill(a, 8)
        b = self.make()
        b.load_state_dict(a.state_dict())
        self.fill(a, 8)
        self.fill(b, 8)
        for n in a.model_names:
            for k, v in getattr(a, n).state_dict().items():
                self.assertTrue(torch.equal(v, getattr(b, n).state_dict()[k]))
        self.assertTrue(torch.equal(a.log_temperature, b.log_temperature))
        self.assertEqual(a.replay.rng.bit_generator.state, b.replay.rng.bit_generator.state)
        for n in a.generators:
            self.assertTrue(torch.equal(a.generators[n].get_state(), b.generators[n].get_state()))

    def test_temperature_gradient_direction_and_detachment(self):
        for entropy, sign in [(-3.0, -1), (-0.1, 1)]:
            c = self.make()
            loss = c.temperature * (entropy - c.target_entropy)
            loss.backward()
            self.assertEqual(int(torch.sign(c.log_temperature.grad)), sign)
            self.assertTrue(all(p.grad is None for p in c.actor.parameters()))

    def test_reject_cross_mechanism(self):
        c = self.make()
        s = c.state_dict()
        s["reference"] = "gaussian_latent"
        with self.assertRaises(ValueError):
            c.load_state_dict(s)
