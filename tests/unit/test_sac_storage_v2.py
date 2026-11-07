import pytest

pytest.importorskip("fcntl", reason="SAC v2 durable storage requires POSIX file locking")
import copy, json, os, tempfile, unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from tests.integration.test_sac_stp import smoke_config
from atprl.training.sac_stp import SACWarmupTrainer, SACPhase2Trainer, run_stage
from atprl.training.sac_recovery import (
    kernel_bundle,
    full_state,
    digest,
    source_signature,
    policy_bundle,
)
from atprl.training.sac_v2 import (
    Store,
    rebuild,
    run_stage_v2,
    thin_policy,
    policy_runtime,
    enter_phase,
)
from atprl.training.sac_v2_migrate import migrate, scientific_state_identity, semantic_digest
from atprl.training.sac_v2_diagnostics import Diagnostics, CompactRecorder
from atprl.training.sac_v2_io import atomic_bytes, atomic_json, append_json, StoragePressure
from atprl.evaluation.sac_stp import evaluate_episode


def state(tr):
    s = full_state(tr)
    for row in s["progress"]["history"]:
        row.pop("episode_cpu_seconds", None)
        row.pop("episode_wall_seconds", None)
    return scientific_state_identity(s)


class StorageV2Tests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SAC_V2_FREE_MARGIN_BYTES": "0"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def seed_store(self, tr, root):
        st = Store(root)
        sd = "stage1" if tr.runtime.stage == "warmup" else "stage2"
        for row in tr.history:
            append_json(st.root / sd / "training.jsonl", row)
        d = Diagnostics()
        d.state["consumed_update"] = (tr.runtime.phase or tr.runtime.power).updates
        st.publish(tr, d)
        return st, d

    def test_warmup_split_complete_state_and_history(self):
        c = replace(smoke_config(), eval_every_episodes=100, checkpoint_every_episodes=1)
        a = SACWarmupTrainer(c)
        a.train_for(7)
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            saved = full_state(a)
            a.train_for(3)
            b, _ = st.restore()
            b.train_for(3)
            self.assertEqual(state(a), state(b))
            self.assertEqual(
                scientific_state_identity(saved), scientific_state_identity(st.load()[0])
            )

    def test_wrapped_phase_split_and_frozen_kernel_de_dup(self):
        c = replace(
            smoke_config(),
            stage="stage2",
            stage2_episodes=4,
            phase_distribution="wrapped_normal_v1",
        )
        warm = SACWarmupTrainer(replace(c, stage="warmup", phase_distribution=None))
        warm.train_for(7)
        a = SACPhase2Trainer(c, kernel_bundle(warm))
        a.train_for(1)
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            s, m = st.load()
            self.assertEqual(scientific_state_identity(full_state(a)), scientific_state_identity(s))
            packed = torch.load(st.root / m["checkpoint"]["path"], weights_only=False)
            self.assertNotIn("power", packed)
            self.assertNotIn("dqn_replay", packed)
            a.train_for(2)
            b, _ = st.restore()
            b.train_for(2)
            self.assertEqual(state(a), state(b))

    def test_frozen_differences_not_discarded(self):
        c = replace(smoke_config(), stage="stage2", phase_distribution="wrapped_normal_v1")
        w = SACWarmupTrainer(replace(c, stage="warmup", phase_distribution=None))
        a = SACPhase2Trainer(c, kernel_bundle(w))
        a.runtime.power.generators["behavior"].manual_seed(98765)
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            s, _ = st.load()
            self.assertEqual(digest(s["power"]), digest(a.runtime.power.state_dict()))

    def test_thin_policy_same_actions_and_evaluation(self):
        c = replace(smoke_config(), stage="stage2", phase_distribution="wrapped_normal_v1")
        w = SACWarmupTrainer(replace(c, stage="warmup", phase_distribution=None))
        a = SACPhase2Trainer(c, kernel_bundle(w))
        a.train_for(1)
        p = thin_policy(policy_bundle(a, "strict_best"))
        rt = policy_runtime(p)
        self.assertNotIn("replay", p["phase"])
        self.assertNotIn("optimizers", p["phase"])
        x = evaluate_episode(a.runtime, 10001)
        y = evaluate_episode(rt, 10001)
        self.assertEqual(x, y)
        wrong = copy.deepcopy(p)
        wrong["physics"]["fairness_lambda"] += 1
        with self.assertRaises(ValueError):
            policy_runtime(wrong)
        wrong = copy.deepcopy(p)
        wrong["phase"]["config"]["gamma"] = 0.1
        with self.assertRaises(ValueError):
            policy_runtime(wrong)

    def test_current_failure_rolls_back_previous_and_journal(self):
        c = replace(smoke_config(), checkpoint_every_episodes=1, eval_every_episodes=100)
        a = SACWarmupTrainer(c)
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            run_stage_v2(a, st, d, max_episodes=2)
            pointer = json.loads(st.current_path.read_text())
            m = json.loads((st.root / pointer["current"]["path"]).read_text())
            (st.root / m["checkpoint"]["path"]).write_bytes(b"corrupt")
            b, m = st.restore()
            self.assertEqual(b.episodes_done, 1)
            self.assertEqual(len((st.root / "stage1/training.jsonl").read_text().splitlines()), 1)

    def test_failed_pointer_publish_keeps_prior_and_gc_dependencies(self):
        c = replace(
            smoke_config(),
            checkpoint_every_episodes=1,
            eval_every_episodes=1,
            evaluation_seeds=[10001],
        )
        a = SACWarmupTrainer(c)
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            original = st.current_path.read_bytes()
            real = os.replace

            def fail(src, dst):
                if str(dst) == str(st.current_path):
                    raise OSError("injected CURRENT publication failure")
                return real(src, dst)

            with patch("os.replace", side_effect=fail):
                with self.assertRaises(OSError):
                    run_stage_v2(a, st, d, max_episodes=1)
            self.assertEqual(st.current_path.read_bytes(), original)
            b, m = st.restore()
            self.assertEqual(b.episodes_done, 0)
            st.gc()
            self.assertFalse(list((st.root / "objects").glob("*.ckpt")))

    def test_diagnostic_checkpoint_roundtrip_no_double_count(self):
        row = lambda i: {
            "updates": i,
            "collected_steps": 4 * i,
            "q1_gradient_norm": i * 0.1,
            "q1_parameter_delta_l2": i * 0.01,
            "q1_loss": i * 2.0,
        }
        d = Diagnostics()
        d.begin(1)
        for i in range(1, 9):
            d.consume(row(i))
        d.finish()
        with tempfile.TemporaryDirectory() as tmp:
            s = d.preview(tmp, "stage2", "one", 1)
            d.published(s)
            e = Diagnostics(s)
            with self.assertRaises(ValueError):
                e.consume(row(8))
            for obj in [d, e]:
                obj.begin(2)
                for i in range(9, 17):
                    obj.consume(row(i))
                obj.finish()
            self.assertEqual(d.state, e.state)
            self.assertEqual(d.state["count"], 16)

    def test_legacy_actual_conversion_early_no_best(self):
        c = replace(
            smoke_config(), seeds=[42], eval_every_episodes=100, checkpoint_every_episodes=1
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            tr, done = run_stage(c, p / "old", "warmup", max_episodes=7)
            job = {"key": "test", "seed": c.seed, "lambda": c.fairness_lambda}
            rec = migrate(
                p / "old/latest.ckpt", p / "new", job, allowed_sources={source_signature()}
            )
            self.assertTrue(rec["passed"])
            b, _ = Store(p / "new").restore(job)
            self.assertEqual(state(tr), state(b))

    def test_space_failure_does_not_touch_prior_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x"
            atomic_bytes(p, b"prior")
            with patch.dict(os.environ, {"SAC_V2_FREE_MARGIN_BYTES": str(2**62)}):
                with self.assertRaises(StoragePressure):
                    atomic_bytes(p, b"new")
            self.assertEqual(p.read_bytes(), b"prior")

    def test_gc_keeps_independent_evaluation_model(self):
        a = SACWarmupTrainer(smoke_config())
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            ref = st.put_object(thin_policy(policy_bundle(a, "strict_best")), "policy")
            atomic_json(
                st.root / "evaluation/strict_best/manifest.json", {"policy_sha256": ref["sha256"]}
            )
            st.gc()
            self.assertTrue((st.root / ref["path"]).exists())

    def test_full_stage_transition_and_phase_no_best_resume(self):
        c = replace(
            smoke_config(),
            stage="warmup",
            seeds=[42],
            eval_every_episodes=4,
            checkpoint_every_episodes=1,
            evaluation_seeds=[10001],
            stage2_episodes=4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, _ = run_stage(c, root / "old", "warmup")
            new = SACWarmupTrainer(c)
            st, d = self.seed_store(new, root / "new")
            new, done = run_stage_v2(new, st, d)
            self.assertTrue(done)
            self.assertEqual(state(old), state(new))
            oldkernel = torch.load(root / "old/kernel_best.ckpt", weights_only=False)
            a = SACPhase2Trainer(
                replace(c, stage="stage2", phase_distribution="wrapped_normal_v1"), oldkernel
            )
            b = enter_phase(st, c)
            saved = state(b)
            resumed, _ = st.restore()
            self.assertEqual(saved, state(resumed))
            a.train_for(1)
            resumed.train_for(1)
            self.assertEqual(state(a), state(resumed))

    def test_worker_lock_rejects_second_writer(self):
        from atprl.training.sac_v2_io import lock

        with tempfile.TemporaryDirectory() as tmp:
            with lock(Path(tmp) / "worker.lock", False):
                pid = os.fork()
                if pid == 0:
                    try:
                        with lock(Path(tmp) / "worker.lock", False):
                            os._exit(1)
                    except BlockingIOError:
                        os._exit(0)
                _, code = os.waitpid(pid, 0)
                self.assertEqual(os.waitstatus_to_exitcode(code), 0)

    def test_process_exit_before_current_publication(self):
        c = replace(smoke_config(), checkpoint_every_episodes=1, eval_every_episodes=100)
        a = SACWarmupTrainer(c)
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            original = st.current_path.read_bytes()
            pid = os.fork()
            if pid == 0:
                real = os.replace

                def die(src, dst):
                    if str(dst) == str(st.current_path):
                        os._exit(99)
                    return real(src, dst)

                with patch("os.replace", side_effect=die):
                    run_stage_v2(a, st, d, max_episodes=1)
                os._exit(1)
            _, code = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(code), 99)
            self.assertEqual(st.current_path.read_bytes(), original)
            b, m = st.restore()
            self.assertEqual(b.episodes_done, 0)

    def test_exact_resume_at_dqn_to_power_boundary(self):
        c = replace(smoke_config(), stage="warmup")
        a = SACWarmupTrainer(c)
        a.train_for(6)
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            a.train_for(1)
            b, _ = st.restore()
            b.train_for(1)
            self.assertEqual(state(a), state(b))
            self.assertEqual(b.last_mode, "POWER")

    def test_phase16_no_best_and_partially_filled_formal_replay(self):
        c = replace(
            smoke_config(),
            stage="stage2",
            stage2_episodes=20,
            phase_distribution="wrapped_normal_v1",
            phase_sac=dict(
                capacity=100000, batch_size=8, learning_starts=8, actor_lr=6e-5, critic_lr=6e-5
            ),
        )
        w = SACWarmupTrainer(replace(c, stage="warmup", phase_distribution=None))
        w.train_for(7)
        a = SACPhase2Trainer(c, kernel_bundle(w))
        a.train_for(16)
        with tempfile.TemporaryDirectory() as tmp:
            st, d = self.seed_store(a, tmp)
            a.train_for(1)
            b, _ = st.restore()
            b.train_for(1)
            self.assertEqual(state(a), state(b))
            self.assertIsNone(b.selection.get("best_episode"))
            self.assertLess(len(b.runtime.phase.replay), b.runtime.phase.replay.capacity)

    def test_bounded_gc_and_rolling_window_across_many_best_and_resume(self):
        import contextlib, io

        c = replace(
            smoke_config(),
            stage="stage2",
            stage2_episodes=700,
            macro_steps_per_episode=4,
            phase_distribution="wrapped_normal_v1",
            checkpoint_every_episodes=16,
            eval_every_episodes=1,
        )
        w = SACWarmupTrainer(replace(c, stage="warmup", phase_distribution=None))
        a = SACPhase2Trainer(c, kernel_bundle(w))

        def curve(t):
            return {
                "episodes_done": t.episodes_done,
                "training_seed": c.seed,
                "stage": "stage2",
                "per_seed": [],
                "metrics": {"undiscounted_return": {"mean": float(t.episodes_done)}},
                "synthetic_capacity_test_only": True,
            }

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("atprl.evaluation.sac_stp.evaluate_curve", side_effect=curve),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            st, d = self.seed_store(a, tmp)
            extra = {"stage1_complete": {"episode": c.stage1_episodes}}
            st.manifest["extra"] = extra
            a, done = run_stage_v2(a, st, d, max_episodes=350)
            b, m = st.restore()
            b, done = run_stage_v2(b, st, st.diagnostics)
            self.assertTrue(done)
            self.assertLessEqual(len(list((st.root / "generations").iterdir())), 2)
            self.assertLessEqual(len(list((st.root / "objects").glob("*.ckpt"))), 4)
            self.assertLessEqual(
                len(st.manifest["diagnostics_by_stage"]["stage2"]["raw_refs"]), 640
            )
            current = st.current_path.read_bytes()
            count = st.diagnostics.state["count"]
            resumed, m = st.restore()
            run_stage_v2(resumed, st, st.diagnostics)
            self.assertEqual(st.current_path.read_bytes(), current)
            self.assertEqual(st.diagnostics.state["count"], count)
            from atprl.evaluation.sac_v2_analysis import raw_diagnostics, RawHistoryUnavailable

            with self.assertRaises(RawHistoryUnavailable):
                raw_diagnostics(tmp, "stage2", 1, 32)
            tail = raw_diagnostics(tmp, "stage2", 61, 700)
            self.assertEqual(len(tail["episode"]), 640)


if __name__ == "__main__":
    unittest.main()
