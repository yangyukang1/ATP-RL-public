from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from atprl.config.loader import cli_args, load_config
from atprl.evaluation.joint import aggregate_runs, evaluate_artifact
from atprl.evaluation.utils import DEFAULT_METRIC_KEYS
from atprl.rl.algorithms import PPO
from atprl.training.joint import JointTrainer, build_parser, evaluation_protocol
from atprl.training.joint_runtime import JointRuntime, make_args, validate_joint_args
from atprl.training.recovery import (
    RecoveryError,
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    load_joint_recovery,
    replay_buffer_state,
    save_joint_recovery,
)


ROOT = Path(__file__).resolve().parents[2]


def small_args(**overrides):
    return make_args(
        **(
            dict(
                device="cpu",
                max_train_episodes=3,
                max_steps_per_episode=8,
                rollout_length=4,
                replay_capacity=64,
                dqn_batch_size=4,
                learning_starts=4,
                ris_num_elem_x=2,
                ris_num_elem_y=2,
                ppo_power_epochs=1,
                ppo_phase_epochs=1,
                eval_every_episodes=1,
                selection_every_episodes=1,
                k_daily=1,
                k_ready=2,
                num_eval_seeds=2,
                checkpoint_every_episodes=1,
            )
            | overrides
        )
    )


class JointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left.cpu(), right.cpu()))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_all_five_optimizers_and_models_update(self):
        trainer = JointTrainer(small_args(), self.root / "seed_42")
        runtime = trainer.runtime
        before = copy.deepcopy(runtime.model_state())
        trainer.train_episode()
        for name in runtime.model_names:
            self.assertTrue(
                any(
                    not torch.equal(value, before[name][key])
                    for key, value in runtime.model_state()[name].items()
                ),
                name,
            )
        for name in runtime.optimizer_names:
            state = getattr(runtime, name).state_dict()["state"]
            self.assertTrue(state, name)
            self.assertTrue(all(value["step"] > 0 for value in state.values()), name)
        self.assertEqual(len(runtime.replay_buffer), 8)
        self.assertEqual(trainer.dqn_updates, 2)
        self.assertEqual(trainer.target_updates, 1)
        self.assertEqual(trainer.power_updates, 2)
        self.assertEqual(trainer.phase_updates, 2)
        self.assertEqual(len(runtime.power_buffer), 0)
        self.assertEqual(len(runtime.phase_buffer), 0)
        self.assertTrue(all(np.isfinite(value) for value in trainer.last_losses.values()))

    def test_real_agent_raw_actions_and_context_survive_upstream_changes(self):
        trainer = JointTrainer(small_args(), self.root / "context")
        runtime = trainer.runtime
        obs = next(iter(runtime.env.reset().values()))
        info = runtime.select(obs, epsilon=1.0)
        trainer.store_transition(info, obs, 1.0, False)
        tail = trainer.build_tail_context(obs)
        old_tail = copy.deepcopy(tail)
        old_power, old_phase = (
            copy.deepcopy(runtime.power_buffer.get_batch()),
            copy.deepcopy(runtime.phase_buffer.get_batch()),
        )
        self.assert_state_equal(old_power["actions"][0], info["power_action_raw"])
        self.assert_state_equal(old_phase["actions"][0], info["phase_joint_action"])
        self.assertEqual(tuple(info["env_action"][0]), tuple(info["selected_users"]))
        self.assertAlmostEqual(
            float(np.sum(info["env_action"][1])), runtime.env.satellite_max_power_watt, places=4
        )
        np.testing.assert_allclose(info["env_action"][3], np.full(4, 0.25))
        self.assertEqual(old_phase["actions"].shape[-1], 2 * runtime.env.ris_num_groups)
        with torch.no_grad():
            for model in (runtime.q_network, runtime.power_actor):
                for parameter in model.parameters():
                    parameter.add_(3)
        self.assert_state_equal(old_power, runtime.power_buffer.get_batch())
        self.assert_state_equal(old_phase, runtime.phase_buffer.get_batch())
        self.assert_state_equal(old_tail, tail)

    def test_event_gate_is_evaluated_at_collection_time_and_order_is_preserved(self):
        trainer = JointTrainer(
            small_args(learning_starts=6, target_update_freq=4), self.root / "events"
        )
        runtime = trainer.runtime
        state = np.zeros(runtime.dqn_state_dim, dtype=np.float32)
        for step in range(1, 13):
            runtime.replay_buffer.store(state, 0, 0, state, False)
            trainer.global_env_steps = step
            trainer.queue_learning_events()
        expected = [(4, "target"), (8, "dqn"), (8, "target"), (12, "dqn"), (12, "target")]
        self.assertEqual(trainer.pending_events, expected)
        observed = []
        original_sync = runtime.target_q_network.load_state_dict

        def update(batch):
            observed.append("dqn")
            return 0.0

        def sync(*args, **kwargs):
            observed.append("target")
            return original_sync(*args, **kwargs)

        with (
            patch.object(runtime.dqn_learner, "update", side_effect=update),
            patch.object(runtime.target_q_network, "load_state_dict", side_effect=sync),
        ):
            self.assertEqual(trainer.replay_learning_events(), expected)
        self.assertEqual(observed, [kind for _, kind in expected])
        self.assertEqual(trainer.pending_events, [])

    def test_episode_boundary_recovery_is_bitwise_equal_on_cpu(self):
        args = small_args(epsilon_start=0.8, epsilon_end=0.1, epsilon_decay_episodes=10)
        continuous = JointTrainer(args, self.root / "continuous")
        continuous.train_episode()
        continuous.train_episode()
        continuous_state = copy.deepcopy(continuous.runtime.model_state())
        continuous_optim = {
            name: copy.deepcopy(getattr(continuous.runtime, name).state_dict())
            for name in continuous.runtime.optimizer_names
        }
        continuous_replay = replay_buffer_state(continuous.runtime.replay_buffer)
        continuous_rng = capture_rng_state(env=continuous.runtime.env)
        split_dir = self.root / "split"
        split = JointTrainer(args, split_dir)
        split.train_episode()
        saved = split_dir / "recovery/latest.ckpt"
        resumed = JointTrainer(args, split_dir, resume_checkpoint=saved)
        self.assertEqual(resumed.episodes_done, 1)
        self.assertEqual(resumed.global_env_steps, 8)
        self.assertEqual(resumed.epsilon(), split.epsilon())
        resumed.train_episode()
        self.assert_state_equal(continuous_state, resumed.runtime.model_state())
        self.assert_state_equal(
            continuous_replay, replay_buffer_state(resumed.runtime.replay_buffer)
        )
        self.assert_state_equal(continuous_rng, capture_rng_state(env=resumed.runtime.env))
        for name in resumed.runtime.optimizer_names:
            self.assert_state_equal(
                continuous_optim[name], getattr(resumed.runtime, name).state_dict()
            )
        self.assertEqual(continuous.history, resumed.history)
        self.assertEqual(continuous.dqn_updates, resumed.dqn_updates)
        self.assertEqual(continuous.rollout_index, resumed.rollout_index)

    def test_resume_rejects_legacy_wrong_kind_and_incompatible_config(self):
        trainer = JointTrainer(small_args(), self.root / "reject")
        trainer.train_episode()
        checkpoint = trainer.output_dir / "recovery/latest.ckpt"
        atomic_torch_save(trainer.runtime.model_state(), self.root / "weights.pth")
        with self.assertRaises(RecoveryError):
            trainer.load_checkpoint(self.root / "weights.pth")
        for changes in (
            dict(fairness_lambda=2.0),
            dict(rollout_length=2),
            dict(seed=9),
            dict(replay_capacity=128),
        ):
            with self.subTest(changes=changes), self.assertRaises(RecoveryError):
                JointTrainer(
                    small_args(**changes), trainer.output_dir, resume_checkpoint=checkpoint
                )
        payload = load_joint_recovery(checkpoint)
        payload.pop("replay")
        save_joint_recovery(payload, self.root / "incomplete.ckpt")
        with self.assertRaises(RecoveryError):
            trainer.load_checkpoint(self.root / "incomplete.ckpt")
        payload = load_joint_recovery(checkpoint)
        payload["rng"].pop("environment")
        save_joint_recovery(payload, self.root / "missing_env_rng.ckpt")
        with self.assertRaisesRegex(RecoveryError, "environment RNG"):
            trainer.load_checkpoint(self.root / "missing_env_rng.ckpt")

    def test_schedule_remainder_survives_nondivisible_episode_boundary(self):
        args = small_args(max_steps_per_episode=10, target_update_freq=6)
        trainer = JointTrainer(args, self.root / "remainder")
        trainer.train_episode()
        self.assertEqual(trainer.global_env_steps, 10)
        self.assertEqual(trainer.dqn_updates, 2)
        self.assertEqual(trainer.target_updates, 1)
        resumed = JointTrainer(
            args, trainer.output_dir, resume_checkpoint=trainer.output_dir / "recovery/latest.ckpt"
        )
        resumed.train_episode()
        self.assertEqual(resumed.global_env_steps, 20)
        self.assertEqual(resumed.dqn_updates, 5)
        self.assertEqual(resumed.target_updates, 3)

    def test_training_endpoint_does_not_add_selection_event(self):
        trainer = JointTrainer(
            small_args(max_train_episodes=3, selection_every_episodes=2), self.root / "selection"
        )
        trainer.train()
        selection_episodes = [
            row["episode"]
            for row in trainer.history
            if row["kind"] == "evaluation" and row["role"] == "selection"
        ]
        self.assertEqual(selection_episodes, [2])
        self.assertEqual(trainer.best_artifact["episode"], 2)

    def test_crash_reconciliation_excludes_uncommitted_logs_and_best(self):
        args = small_args(checkpoint_every_episodes=3, selection_every_episodes=1)
        trainer = JointTrainer(args, self.root / "crash")
        trainer.train_episode()
        trainer.save_checkpoint()
        committed = copy.deepcopy(trainer.history)
        best_step = trainer.best_artifact["global_env_steps"]
        trainer.train_episode()
        trainer.best_artifact["global_env_steps"] = trainer.global_env_steps
        trainer.publish_committed_outputs()
        resumed = JointTrainer(
            args, trainer.output_dir, resume_checkpoint=trainer.output_dir / "recovery/latest.ckpt"
        )
        records = json.loads((trainer.output_dir / "committed_metrics.json").read_text())["records"]
        self.assertEqual(records, committed)
        self.assertEqual(len(records), len(resumed.history))
        self.assertGreater(
            len((trainer.output_dir / "events.jsonl").read_text().splitlines()), len(records)
        )
        self.assertEqual(resumed.best_artifact["global_env_steps"], best_step)
        resumed.train_episode()
        resumed.save_checkpoint()
        self.assertEqual(len(resumed.runtime.replay_buffer), 16)
        curve = [
            row for row in resumed.history if row["kind"] == "evaluation" and row["role"] == "curve"
        ]
        self.assertEqual([row["global_env_steps"] for row in curve], [8, 16])

    def test_preflight_and_runtime_singleton_guards(self):
        for horizon, length in ((1, 4), (8, 1), (9, 4), (1024, 1023)):
            with self.subTest(horizon=horizon, length=length), self.assertRaises(ValueError):
                validate_joint_args(
                    small_args(max_steps_per_episode=horizon, rollout_length=length)
                )
        trainer = JointTrainer(small_args(), self.root / "singleton")
        original_step = trainer.runtime.env.step

        def premature(actions):
            obs, rewards, dones, infos = original_step(actions)
            dones["__all__"] = True
            for info in infos.values():
                info.update(terminated=True, time_limit_truncated=False)
            return obs, rewards, dones, infos

        with (
            patch.object(trainer.runtime.env, "step", side_effect=premature),
            self.assertRaisesRegex(RuntimeError, "single-sample"),
        ):
            trainer.train_episode()
        with self.assertRaises(RecoveryError):
            trainer.save_checkpoint()

    def test_gae_short_rollout_termination_and_truncation(self):
        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mean = torch.nn.Parameter(torch.zeros(1))

            def forward(self, states):
                return torch.distributions.Normal(
                    self.mean.expand_as(states), torch.ones_like(states)
                )

        for terminal, truncation, expected in (
            (True, False, 2.5),
            (False, True, 49.705),
            (False, False, 49.705),
        ):
            with self.subTest(terminal=terminal, truncation=truncation):
                actor, critic = Actor(), torch.nn.Linear(1, 1)
                with torch.no_grad():
                    critic.weight.fill_(1)
                    critic.bias.zero_()
                learner = PPO(
                    actor,
                    critic,
                    torch.optim.Adam(actor.parameters(), lr=0),
                    torch.optim.Adam(critic.parameters(), lr=0),
                    dict(gamma=0.9, gae_lambda=1.0, ppo_epochs=1),
                )
                states = torch.tensor([[1.0], [2.0]])
                actions = torch.zeros_like(states)
                batch = dict(
                    states=states,
                    actions=actions,
                    rewards=torch.zeros(2, 1),
                    dones=torch.tensor([[0.0], [float(terminal)]]),
                    log_probs=actor(states).log_prob(actions).detach(),
                    values=critic(states).detach(),
                )
                losses = learner.update(batch, torch.tensor([10.0]), truncation)
                self.assertAlmostEqual(losses["critic_loss"], expected, places=4)

    def test_formal_runtime_models_physics_and_input_states_match(self):
        from atprl.training import stage2

        args = make_args(device="cpu", ris_num_elem_x=2, ris_num_elem_y=2, replay_capacity=256)
        import sys

        with patch.object(sys, "argv", ["stage2"]):
            formal_args = stage2.parse_args()
        for name, value in vars(args).items():
            setattr(formal_args, name, value)
        formal_args.phase2_only = False
        formal = SimpleNamespace(args=formal_args, device=torch.device("cpu"), writer=None)
        torch.manual_seed(123)
        stage2.Phase2Trainer._build_everything_from_original_main(formal)
        torch.manual_seed(123)
        joint = JointRuntime(args)
        self.assertEqual(formal.env_config, joint.env_config)
        for name in joint.model_names:
            self.assert_state_equal(
                getattr(formal, name).state_dict(), getattr(joint, name).state_dict()
            )
        observation = next(iter(joint.env.reset(seed=9).values()))
        formal_info = next(iter(formal.agents.values())).select_action(
            observation, deterministic=True
        )
        joint_info = joint.select(observation, deterministic=True)
        for key in (
            "dqn_state",
            "power_state",
            "phase_state",
            "power_action",
            "phase_action_theta",
        ):
            self.assert_state_equal(formal_info[key], joint_info[key])

    def test_seed_protocol_matches_formal_pbt_and_keeps_existing_overlap(self):
        from atprl.training.pbt.controller import make_daily_seed_list, make_ready_seed_list

        args = make_args()
        protocol = evaluation_protocol(args)
        self.assertEqual(protocol["curve"], make_daily_seed_list(args, 4))
        self.assertEqual(protocol["selection"], make_ready_seed_list(args, 7))
        self.assertEqual(protocol["final"], [10001 + i * 37 for i in range(500)])
        overlapping = evaluation_protocol(make_args(k_daily=0, k_ready=2))
        self.assertIn(10001, overlapping["selection_final_overlap"])
        self.assertEqual(protocol["selection_final_overlap"], [])
        self.assertEqual(protocol["curve_final_overlap"], [10001])

    def test_multiseed_final_statistics(self):
        finals = []
        for seed in (3, 6):
            trainer = JointTrainer(
                small_args(seed=seed, max_train_episodes=1), self.root / f"seed_{seed}"
            )
            trainer.train()
            before_rng = capture_rng_state(env=trainer.runtime.env)
            pack = evaluate_artifact(
                trainer.output_dir / "models/best.ckpt",
                device="cpu",
                seeds=trainer.protocol["final"],
            )
            self.assert_state_equal(before_rng, capture_rng_state(env=trainer.runtime.env))
            atomic_json_save(pack, trainer.output_dir / "final_evaluation.json")
            finals.append(pack)
        summary = aggregate_runs(self.root, [3, 6])
        for name in DEFAULT_METRIC_KEYS:
            means = [pack["metrics"][name]["mean"] for pack in finals]
            self.assertAlmostEqual(summary["metrics"][name]["mean"], float(np.mean(means)))
            self.assertAlmostEqual(summary["metrics"][name]["std"], float(np.std(means)))
        self.assertEqual(summary["common_curve_grid"], [8])
        self.assertNotEqual(finals[0]["train_seed"], finals[1]["train_seed"])


if __name__ == "__main__":
    unittest.main()
