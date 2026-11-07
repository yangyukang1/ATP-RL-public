import unittest
import numpy as np
import torch
from atprl.rl.sac_replay import SACReplayBuffer


class ReplayTests(unittest.TestCase):
    def test_ring_and_rng_restore_and_outage(self):
        b = SACReplayBuffer(4, 2, 3, seed=3)
        for i in range(7):
            b.store(
                [i, i], [1, 0, 0], -i, [i + 1, i + 1], False, i == 6, episode_id=i, macro_step=1
            )
        self.assertEqual((b.ptr, b.size), (3, 4))
        self.assertEqual(sorted(b.episode_ids), [3, 4, 5, 6])
        c = SACReplayBuffer(4, 2, 3)
        c.load_state_dict(b.state_dict())
        x, y = b.sample(4), c.sample(4)
        for k in x:
            torch.testing.assert_close(x[k], y[k])
        b.clear_context("new")
        self.assertEqual(len(b), 0)
        self.assertTrue((b.episode_ids == -1).all())
        self.assertEqual(b.store_count, 7)

    def test_reject_invalid_without_partial_write(self):
        b = SACReplayBuffer(4, 2, 3)
        for a in [[0.2, 0.2, 0.2], [1, -1, 1], [float("nan"), 0, 0]]:
            with self.assertRaises((ValueError, FloatingPointError)):
                b.store([0, 0], a, 0, [0, 0], 0, 0)
        self.assertEqual(len(b), 0)
