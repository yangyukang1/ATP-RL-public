import copy, tempfile, unittest, importlib.util, sys, json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tests.integration.test_sac_stp import smoke_config
from atprl.training.sac_stp import SACWarmupTrainer, SACPhase2Trainer, run_stage, run_seed
from atprl.training.sac_recovery import (
    kernel_bundle,
    write_torch,
    load_torch,
    file_sha,
    full_state,
    digest,
    save_recovery,
    restore_checkpoint,
)
from atprl.evaluation.sac_stp import load_policy, evaluate_episode


class WrappedPhaseTests(unittest.TestCase):
    def setup_phase(self):
        c = replace(
            smoke_config(),
            stage="stage2",
            phase_distribution="wrapped_normal_v1",
            auto_final_evaluation=False,
            evaluation_seeds=[10001],
            checkpoint_every_episodes=1,
            eval_every_episodes=1,
        )
        k = kernel_bundle(SACWarmupTrainer(c))
        return c, k, SACPhase2Trainer(c, k)

    def test_initialization_same_weights_and_executed_xy_replay_freeze(self):
        c, k, tr = self.setup_phase()
        old = SACPhase2Trainer(replace(c, phase_distribution=None), k)
        self.assertEqual(
            digest(tr.runtime.phase.state_dict()["models"]),
            digest(old.runtime.phase.state_dict()["models"]),
        )

        def frozen(t):
            s = full_state(t)
            return digest(
                {
                    k: s[k]
                    for k in [
                        "q",
                        "target_q",
                        "q_optimizer",
                        "dqn_replay",
                        "dqn_updates",
                        "dqn_target_updates",
                        "power",
                    ]
                }
            )

        initial = frozen(tr)
        seen = []
        step = tr.runtime.env.step

        def traced(actions):
            out = step(actions)
            seen.append((copy.deepcopy(actions[tr.runtime.aid]), copy.deepcopy(out)))
            return out

        tr.runtime.env.step = traced
        tr.train_for(2)
        tr.runtime.env.step = step
        self.assertEqual(initial, frozen(tr))
        rp = tr.runtime.phase.replay
        for i, (action, out) in enumerate(seen):
            expected = (
                np.stack([np.cos(action[2]), np.sin(action[2])], -1).reshape(-1).astype("float32")
            )
            np.testing.assert_allclose(rp.arrays["actions"][i], expected, atol=2e-7, rtol=0)
            self.assertEqual(rp.arrays["rewards"][i, 0], np.float32(out[1][tr.runtime.aid]))
            np.testing.assert_array_equal(
                rp.arrays["next_states"][i], tr.runtime.phase_state(out[0][tr.runtime.aid])
            )

    def test_policy_roundtrip_density_and_evaluation_rng(self):
        c, k, tr = self.setup_phase()
        tr.train_for(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernel = root / "kernel.ckpt"
            write_torch(kernel, k)
            cfg = replace(
                c, kernel_checkpoint=str(kernel), kernel_checkpoint_sha256=file_sha(kernel)
            )
            t, done = run_stage(cfg, root / "run", "stage2", k)
            self.assertTrue(done)
            p = root / "run/policy_best.ckpt"
            rt, b = load_policy(p)
            self.assertEqual(rt.phase.phase_distribution, "wrapped_normal_v1")
            bank = t.runtime.phase.replay.arrays["states"][:16]
            s = torch.from_numpy(bank)
            a = t.runtime.phase.actor(s).deterministic()
            loaded = load_policy(root / "run/endpoint_18.ckpt")[0]
            other = loaded.phase.actor(s).deterministic()
            torch.testing.assert_close(a.action, other.action, rtol=0, atol=0)
            torch.testing.assert_close(
                a.log_prob_reference, other.log_prob_reference, rtol=0, atol=0
            )
            before = digest(loaded.phase.state_dict())
            evaluate_episode(loaded, 10001)
            self.assertEqual(before, digest(loaded.phase.state_dict()))
            payload = load_torch(p)
            payload["phase"]["phase_distribution"] = "tanh_normal_v1"
            write_torch(root / "bad.ckpt", payload)
            with self.assertRaisesRegex(ValueError, "distribution"):
                load_policy(root / "bad.ckpt")
            with patch(
                "atprl.evaluation.sac_stp.evaluate_final500",
                side_effect=AssertionError("No final500"),
            ):
                run_seed(replace(cfg, output_dir=str(root / "seed")), root / "seed")

    def test_strict_recovery_config_source_and_power_rejection(self):
        c, k, tr = self.setup_phase()
        tr.train_for(1)
        with tempfile.TemporaryDirectory() as tmp:
            save_recovery(tr, tmp)
            with self.assertRaises(ValueError):
                restore_checkpoint(Path(tmp) / "latest.ckpt", replace(c, phase_distribution=None))
            with patch("atprl.training.sac_recovery.source_signature", return_value="bad"):
                with self.assertRaises(ValueError):
                    restore_checkpoint(Path(tmp) / "latest.ckpt", c)
        with self.assertRaises(ValueError):
            SACPhase2Trainer(replace(c, power_sac=dict(c.power_sac, kappa=1.5)), k)
        bad = copy.deepcopy(k)
        bad["training_seed"] += 1
        with self.assertRaises(ValueError):
            SACPhase2Trainer(c, bad)
