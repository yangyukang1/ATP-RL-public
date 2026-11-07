import tempfile, unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from tests.integration.test_sac_stp import smoke_config
from atprl.training.sac_stp import SACWarmupTrainer
from atprl.training.sac_recovery import policy_bundle, write_torch, digest
from atprl.evaluation.sac_stp import resume_evaluation, evaluate_episode, METRICS


class EvaluationTests(unittest.TestCase):
    def test_legacy_runner_rejects_mislabeled_ris_geometry(self):
        from argparse import Namespace
        from atprl.evaluation.sac_stp import evaluate_from_runner

        cfg = smoke_config()
        rt = Namespace(config=cfg)
        args = Namespace(
            sac_checkpoint="unused",
            num_ground_stations=4,
            num_gs_per_rb=2,
            topology_profile="paper_n4",
            fairness_lambda=3.0,
            csi_error_variance=0.01,
            ris_num_elem_x=16,
            ris_num_elem_y=16,
        )
        with patch("atprl.evaluation.sac_stp.load_policy", return_value=(rt, {})):
            with self.assertRaisesRegex(ValueError, "ris_num_elem_x"):
                evaluate_from_runner(args)

    def test_500_seed_protocol_interrupt_resume_and_no_model_mixing(self):
        t = SACWarmupTrainer(smoke_config())
        calls = []

        def mock(seed):
            calls.append(seed)
            return dict(zip(METRICS, [1.0, 2.0, 0.1, 0.8]), n_steps=1024)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cp = root / "policy.ckpt"
            write_torch(cp, policy_bundle(t, "strict_best"))
            seeds = [10001 + 37 * i for i in range(500)]
            a = resume_evaluation(cp, root / "eval", seeds, evaluator=mock, max_new=7)
            self.assertFalse(a["complete"])
            b = resume_evaluation(cp, root / "eval", seeds, evaluator=mock)
            self.assertTrue(b["complete"])
            self.assertEqual(calls, seeds)
            resume_evaluation(cp, root / "eval", seeds, evaluator=mock)
            self.assertEqual(len(calls), 500)
            write_torch(cp, policy_bundle(t, "endpoint"))
            with self.assertRaises(ValueError):
                resume_evaluation(cp, root / "eval", seeds, evaluator=mock)

    def test_real_metrics_and_no_policy_rng_mutation(self):
        t = SACWarmupTrainer(smoke_config())
        before = digest(t.runtime.power.state_dict())
        a = evaluate_episode(t.runtime, 10001)
        b = evaluate_episode(t.runtime, 10001)
        self.assertEqual(a, b)
        self.assertEqual(a["n_steps"], 24)
        self.assertEqual(before, digest(t.runtime.power.state_dict()))
