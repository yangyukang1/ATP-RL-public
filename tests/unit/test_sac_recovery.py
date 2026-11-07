import copy, tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import torch
from tests.integration.test_sac_stp import smoke_config
from atprl.training.sac_stp import SACWarmupTrainer, SACPhase2Trainer, run_stage
from atprl.training.sac_recovery import (
    save_recovery,
    restore_checkpoint,
    full_state,
    digest,
    kernel_bundle,
    write_json,
)


class RecoveryTests(unittest.TestCase):
    def equal_runtime(self, a, b):
        for key in [
            "q",
            "target_q",
            "q_optimizer",
            "dqn_replay",
            "dqn_updates",
            "dqn_target_updates",
            "power",
            "phase",
            "progress",
            "rng",
        ]:
            self.assertEqual(digest(full_state(a)[key]), digest(full_state(b)[key]), key)

    def test_stage1_cross_segment_and_nonempty_adam_exact_resume(self):
        c = smoke_config()
        a = SACWarmupTrainer(c)
        a.train_for(7)
        with tempfile.TemporaryDirectory() as d:
            save_recovery(a, d)
            a.train_for(3)
            b = restore_checkpoint(Path(d) / "latest.ckpt", c)
            b.train_for(3)
            self.equal_runtime(a, b)
            self.assertEqual(len(list((Path(d) / "recovery_sources").glob("*.ckpt"))), 1)

    def test_stage2_full_environment_rng_and_all_frozen_state_resume(self):
        c = replace(smoke_config(), stage2_episodes=3)
        w = SACWarmupTrainer(c)
        w.train_for(7)
        a = SACPhase2Trainer(c, kernel_bundle(w))
        a.train_for(1)
        with tempfile.TemporaryDirectory() as d:
            save_recovery(a, d)
            a.train_for(2)
            b = restore_checkpoint(Path(d) / "latest.ckpt", c)
            b.train_for(2)
            self.equal_runtime(a, b)

    def test_strict_config_and_source_rejection(self):
        c = smoke_config()
        t = SACWarmupTrainer(c)
        with tempfile.TemporaryDirectory() as d:
            save_recovery(t, d)
            with self.assertRaises(ValueError):
                restore_checkpoint(Path(d) / "latest.ckpt", replace(c, fairness_lambda=0))
            with patch("atprl.training.sac_recovery.source_signature", return_value="different"):
                with self.assertRaises(ValueError):
                    restore_checkpoint(Path(d) / "latest.ckpt", c)

    def test_atomic_write_failure_preserves_prior_file_and_retries(self):
        from atprl.training.sac_recovery import retry

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            write_json(p, dict(x=1))
            old = p.read_bytes()
            with (
                patch("atprl.training.recovery.os.replace", side_effect=PermissionError),
                patch("atprl.training.sac_recovery.time.sleep"),
            ):
                with self.assertRaises(PermissionError):
                    write_json(p, dict(x=2))
            self.assertEqual(old, p.read_bytes())
            op = unittest.mock.Mock(side_effect=[PermissionError, None])
            with patch("atprl.training.sac_recovery.time.sleep"):
                retry(op)
            self.assertEqual(op.call_count, 2)

    def test_run_journal_restore_no_repeat_and_real_best_identity(self):
        c = replace(
            smoke_config(),
            eval_every_episodes=4,
            evaluation_seeds=[10001],
            checkpoint_every_episodes=1,
        )
        with tempfile.TemporaryDirectory() as d:
            t, done = run_stage(c, d, "warmup", max_episodes=7)
            self.assertFalse(done)
            with (Path(d) / "training.jsonl").open("a") as f:
                f.write("uncommitted garbage\n")
            t, done = run_stage(replace(c, resume=True), d, "warmup")
            self.assertTrue(done)
            self.assertEqual(len((Path(d) / "training.jsonl").read_text().splitlines()), 16)
            self.assertTrue((Path(d) / "kernel_best.ckpt").exists())
            self.assertTrue((Path(d) / "kernel_endpoint_16.ckpt").exists())
            from atprl.training.sac_recovery import file_sha

            h = file_sha(Path(d) / "kernel_best.ckpt")
            run_stage(replace(c, resume=True), d, "warmup")
            self.assertEqual(h, file_sha(Path(d) / "kernel_best.ckpt"))
