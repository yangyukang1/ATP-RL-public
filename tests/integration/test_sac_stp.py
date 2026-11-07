import copy, importlib.util, subprocess, random
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from atprl.config.sac import SACRunConfig
from atprl.training.sac_stp import SACWarmupTrainer
from atprl.training.recovery import capture_rng_state, restore_rng_state


def smoke_config(**kw):
    return SACRunConfig(
        profile="smoke",
        stage1_episodes=16,
        stage2_episodes=2,
        macro_steps_per_episode=24,
        num_micro_steps=2,
        dqn_replay_capacity=512,
        ris_num_elem_x=2,
        ris_num_elem_y=2,
        power_sac=dict(capacity=128, batch_size=8, learning_starts=8),
        phase_sac=dict(
            capacity=128,
            batch_size=8,
            learning_starts=8,
            actor_lr=6e-5,
            critic_lr=6e-5,
            temperature_lr=6e-5,
        ),
        **kw,
    )


class StageOneTests(unittest.TestCase):
    def test_real_replay_transition_matches_executed_physics(self):
        tr = SACWarmupTrainer(replace(smoke_config(), cycle_dqn_episodes=1, cycle_power_episodes=1))
        rt = tr.runtime
        tr.train_for(1)
        seen = []
        original = rt.env.step

        def step(actions):
            result = original(actions)
            seen.append((copy.deepcopy(actions[rt.aid]), copy.deepcopy(result)))
            return result

        rt.env.step = step
        tr.train_for(1)
        rep = rt.power.replay
        for j, (action, result) in enumerate(seen):
            np.testing.assert_allclose(
                rep.arrays["actions"][j],
                np.asarray(action[1], np.float32) / rt.env.satellite_max_power_watt,
                atol=2e-6,
                rtol=0,
            )
            self.assertEqual(rep.arrays["rewards"][j, 0], np.float32(result[1][rt.aid]))
            np.testing.assert_array_equal(
                rep.arrays["next_states"][j], rt.power_state(result[0][rt.aid])
            )
            self.assertEqual(
                rep.arrays["truncated"][j, 0], result[3][rt.aid]["time_limit_truncated"]
            )

    def test_nondefault_n_k_runtime_shapes(self):
        from atprl.training.sac_runtime import SACSTPRuntime

        for n, k, profile in [(3, 1, "legacy_n3"), (5, 3, "paper_n5"), (20, 5, "paper_n20")]:
            cfg = replace(
                smoke_config(), num_ground_stations=n, num_gs_per_rb=k, topology_profile=profile
            )
            rt = SACSTPRuntime(cfg, "stage2")
            obs = rt.env.reset()[rt.aid]
            info = rt.select(obs, deterministic=True)
            self.assertEqual(len(info["power_policy_action"]), k + 1)
            self.assertEqual(len(info["phase_state"]), 29 * k + 3)
            self.assertEqual(info["phase_policy_action"].shape, (8,))
            self.assertEqual(len(info["selected_users"]), k)

    def test_formal_environment_dictionary_parity(self):
        import ast, sys
        from unittest.mock import patch
        from atprl.training import warmup
        import yaml

        with patch.object(sys, "argv", ["warmup"]):
            args = warmup.parse_args()
        for k, v in yaml.safe_load(Path("configs/base/warmup.yaml").read_text())["args"].items():
            setattr(args, k, v)
        tree = ast.parse(Path(warmup.__file__).read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Dict)
            and any(
                isinstance(k, ast.Constant) and k.value == "satellite_max_power_dBm" for k in n.keys
            )
        )
        expected = eval(
            compile(ast.Expression(node), "baseline_config", "eval"), dict(args=args, np=np)
        )
        tr = SACWarmupTrainer(smoke_config())
        actual = tr.runtime.env_config
        for key in expected:
            if key not in [
                "max_steps_per_episode",
                "num_micro_steps",
                "ris_num_elem_x",
                "ris_num_elem_y",
                "random_seed",
            ]:
                self.assertEqual(actual.get(key), expected[key], key)

    def test_real_six_ten_and_replay_actions(self):
        tr = SACWarmupTrainer(smoke_config())
        tr.train_for(6)
        rt = tr.runtime
        self.assertEqual(rt.power.updates, 0)
        self.assertGreater(rt.dqn_updates, 0)
        self.assertEqual(len(rt.power.replay), 0)
        dq = {k: v.clone() for k, v in rt.q_network.state_dict().items()}
        tr.train_for(10)
        self.assertEqual([r["mode"] for r in tr.history], ["DQN"] * 6 + ["POWER"] * 10)
        self.assertEqual(tr.total_steps, 16 * 24)
        self.assertGreater(rt.power.updates, 0)
        self.assertTrue(all(torch.equal(v, rt.q_network.state_dict()[k]) for k, v in dq.items()))
        self.assertEqual(len(rt.dqn_replay), 0)
        a = rt.power.replay.arrays["actions"]
        self.assertTrue(np.allclose(a[: len(rt.power.replay)].sum(1), 1))

    def test_state_builders_do_not_sample(self):
        tr = SACWarmupTrainer(smoke_config())
        rt = tr.runtime
        obs = rt.env.reset()[rt.aid]
        states = {n: g.get_state().clone() for n, g in rt.power.generators.items()}
        rt.power_state(obs)
        rt.power_state(obs)
        self.assertTrue(
            all(torch.equal(g.get_state(), states[n]) for n, g in rt.power.generators.items())
        )


class StageTwoTests(unittest.TestCase):
    def kernel(self):
        from atprl.training.sac_recovery import kernel_bundle

        t = SACWarmupTrainer(smoke_config())
        t.train_for(16)
        return kernel_bundle(t)

    def test_actual_handoff_and_complete_kernel_freeze(self):
        from atprl.training.sac_stp import SACPhase2Trainer
        from atprl.training.sac_recovery import digest, kernel_bundle

        kernel = self.kernel()
        tr = SACPhase2Trainer(smoke_config(), kernel)
        rt = tr.runtime
        before = digest(
            dict(
                power=rt.power.state_dict(),
                q=rt.q_network.state_dict(),
                target=rt.target_q_network.state_dict(),
                optimizer=rt.q_optimizer.state_dict(),
            )
        )
        initial = digest(rt.phase.actor.state_dict())
        tr.train_for(2)
        after = digest(
            dict(
                power=rt.power.state_dict(),
                q=rt.q_network.state_dict(),
                target=rt.target_q_network.state_dict(),
                optimizer=rt.q_optimizer.state_dict(),
            )
        )
        self.assertEqual(before, after)
        self.assertNotEqual(initial, digest(rt.phase.actor.state_dict()))
        self.assertEqual(rt.phase.state_dim, 61)
        self.assertEqual(rt.phase.action_dim, 8)
        self.assertGreater(rt.phase.updates, 0)
        self.assertEqual(rt.phase.updates, rt.phase.actor_updates)
        self.assertEqual(rt.phase.updates, rt.phase.temperature_updates)
        self.assertEqual(rt.phase.collected_steps, 48)

    def test_phase_next_state_does_not_consume_any_policy_rng(self):
        from atprl.training.sac_stp import SACPhase2Trainer

        tr = SACPhase2Trainer(smoke_config(), self.kernel())
        rt = tr.runtime
        obs = rt.env.reset()[rt.aid]
        before = {
            k: {n: g.get_state().clone() for n, g in c.generators.items()}
            for k, c in [("p", rt.power), ("f", rt.phase)]
        }
        rt.phase_state(obs)
        for k, c in [("p", rt.power), ("f", rt.phase)]:
            for n, g in c.generators.items():
                self.assertTrue(torch.equal(before[k][n], g.get_state()))

    def test_kernel_rejects_ppo_wrong_seed_and_physics(self):
        from atprl.training.sac_stp import SACPhase2Trainer

        kernel = self.kernel()
        for name, value in [("kind", "ppo"), ("training_seed", 8)]:
            bad = copy.deepcopy(kernel)
            bad[name] = value
            with self.assertRaises(ValueError):
                SACPhase2Trainer(smoke_config(), bad)
        with self.assertRaises(ValueError):
            SACPhase2Trainer(replace(smoke_config(), fairness_lambda=0), kernel)
