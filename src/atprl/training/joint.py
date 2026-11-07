from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from atprl.evaluation.utils import (
    DEFAULT_METRIC_KEYS,
    FIXED_VALIDATION_SEEDS,
    run_multi_seed_validation_metrics,
)
from atprl.training.joint_runtime import DEFAULTS, JointRuntime, make_args, validate_joint_args
from atprl.training.recovery import (
    RecoveryError,
    _atomic_write_bytes,
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    replay_buffer_state,
    load_replay_buffer_state,
    save_joint_recovery,
    load_joint_recovery,
    signature,
    validate_signature,
)


METHOD = "joint_dqn_ppo_ppo"
POLICY_KIND = "joint_policy"


def evaluation_protocol(args) -> dict:
    if args.k_daily < 0 or args.k_ready <= 0:
        raise ValueError("k_daily must be nonnegative and k_ready must be positive")
    curve = list(FIXED_VALIDATION_SEEDS)[: args.k_daily]
    selection = list(FIXED_VALIDATION_SEEDS)[args.k_daily : args.k_daily + args.k_ready]
    if len(selection) < args.k_ready:
        selection = list(FIXED_VALIDATION_SEEDS)[: args.k_ready]
    final = [args.eval_seed_base + i * args.eval_seed_stride for i in range(args.num_eval_seeds)]
    return dict(
        curve=curve,
        selection=selection,
        final=final,
        selection_final_overlap=sorted(set(selection) & set(final)),
        curve_final_overlap=sorted(set(curve) & set(final)),
        selection_metric="undiscounted_return",
        seed_source="formal_pbt_and_final_evaluator",
    )


def evaluate_runtime(runtime: JointRuntime, seeds: list[int]) -> dict:
    args = runtime.args
    profile = "phase2" if args.include_mrt_features else "phase2_no_mrt"
    pack = run_multi_seed_validation_metrics(
        runtime.env_config, runtime.agents, args, seeds, profile=profile, mode="standard"
    )
    pack["per_seed_metrics"] = [
        dict(
            eval_seed=seed,
            **{name: pack["metrics"][name]["values"][index] for name in DEFAULT_METRIC_KEYS},
        )
        for index, seed in enumerate(seeds)
    ]
    return pack


def cpu_model_snapshot(runtime) -> dict:
    return {
        name: {key: value.detach().cpu().clone() for key, value in state.items()}
        for name, state in runtime.model_state().items()
    }


class JointTrainer:
    """Update all three controllers from one environment trajectory without stage freezing."""

    def __init__(self, args, output_dir: Path, *, resume_checkpoint: Path | None = None):
        validate_joint_args(args)
        self.args, self.output_dir = args, Path(output_dir).resolve()
        if (
            resume_checkpoint is None
            and self.output_dir.exists()
            and any(self.output_dir.iterdir())
        ):
            raise FileExistsError(f"Refusing to overwrite an existing Joint run: {self.output_dir}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.num_threads)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        self.runtime = JointRuntime(args)
        self.protocol = evaluation_protocol(args)
        ignored = {
            "device",
            "num_threads",
            "max_train_episodes",
            "checkpoint_every_episodes",
            "logging_interval",
        }
        self.compatibility = signature(
            dict(
                method=METHOD,
                env=self.runtime.env_config,
                training={k: v for k, v in vars(args).items() if k not in ignored},
            )
        )
        self.global_env_steps = self.episodes_done = self.rollout_index = 0
        self.dqn_updates = self.target_updates = self.power_updates = self.phase_updates = 0
        self.pending_events: list[tuple[int, str]] = []
        self.history: list[dict] = []
        self.best_artifact = None
        self.best_score = -float("inf")
        self.training_wall_clock = self.evaluation_wall_clock = 0.0
        self.evaluation_interactions = 0
        self.last_losses = {}
        self.at_episode_boundary = True
        self.attempt_id = uuid.uuid4().hex
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if resume_checkpoint is not None:
            self.load_checkpoint(resume_checkpoint)
        else:
            atomic_json_save(
                dict(
                    method=METHOD,
                    args=vars(args),
                    environment=self.runtime.env_config,
                    phase_state=self.runtime.phase_state_spec,
                    evaluation_protocol=self.protocol,
                ),
                self.output_dir / "joint_config.json",
            )

    def epsilon(self) -> float:
        args = self.args
        if args.epsilon_decay_episodes > 0 and self.episodes_done < args.epsilon_decay_episodes:
            ratio = (self.episodes_done + 1) / max(1, args.epsilon_decay_episodes)
            return args.epsilon_start + (args.epsilon_end - args.epsilon_start) * ratio
        return args.epsilon_end

    def record(self, kind: str, **fields) -> None:
        row = dict(
            kind=kind,
            method=METHOD,
            train_seed=self.args.seed,
            global_env_steps=self.global_env_steps,
            episode=self.episodes_done,
            **fields,
        )
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row | dict(attempt_id=self.attempt_id), allow_nan=False) + "\n")
        self.history.append(row)

    def queue_learning_events(self) -> None:
        args, replay = self.args, self.runtime.replay_buffer
        step = self.global_env_steps
        if step % args.dqn_update_freq == 0 and len(replay) >= max(
            args.learning_starts, args.dqn_batch_size
        ):
            self.pending_events.append((step, "dqn"))
        if step % args.target_update_freq == 0:
            self.pending_events.append((step, "target"))

    def replay_learning_events(self) -> list[tuple[int, str]]:
        executed = list(self.pending_events)
        for _, kind in executed:
            if kind == "dqn":
                batch = self.runtime.replay_buffer.sample(self.args.dqn_batch_size)
                loss = self.runtime.dqn_learner.update(
                    {k: v.to(self.runtime.device) for k, v in batch.items()}
                )
                self.last_losses["dqn_loss"] = float(loss)
                self.dqn_updates += 1
            else:
                self.runtime.target_q_network.load_state_dict(self.runtime.q_network.state_dict())
                self.target_updates += 1
        self.pending_events.clear()
        return executed

    def store_transition(
        self, info: dict, next_observation, reward: float, terminated: bool
    ) -> None:
        runtime = self.runtime
        runtime.replay_buffer.store(
            info["dqn_state"],
            info["scheduler_action"],
            reward,
            next_observation[: runtime.dqn_state_dim],
            float(terminated),
        )
        for buffer, prefix, action in (
            (runtime.power_buffer, "power", "power_action_raw"),
            (runtime.phase_buffer, "phase", "phase_joint_action"),
        ):
            buffer.store(
                info[f"{prefix}_state"].clone(),
                info[action].clone(),
                reward,
                float(terminated),
                info[f"{prefix}_log_prob"].clone(),
                info[f"{prefix}_value"].clone(),
            )

    def build_tail_context(self, observation) -> dict:
        tail = self.runtime.select(observation, deterministic=True)
        return {name: tail[name].clone() for name in ("power_state", "phase_state")}

    def update_rollout(self, tail: dict, *, truncated: bool) -> list:
        runtime = self.runtime
        if len(runtime.power_buffer) < 2 or len(runtime.phase_buffer) < 2:
            raise RuntimeError(
                "Actual termination produced a single-sample PPO trajectory; cannot update unchanged PPO"
            )
        events = self.replay_learning_events()
        for prefix, learner, buffer in (
            ("power", runtime.ppo_power, runtime.power_buffer),
            ("phase", runtime.ppo_phase, runtime.phase_buffer),
        ):
            batch = {k: v.to(runtime.device) for k, v in buffer.get_batch().items()}
            losses = learner.update(batch, tail[f"{prefix}_state"].to(runtime.device), truncated)
            self.last_losses.update(
                {f"{prefix}_{key}": float(value) for key, value in losses.items()}
            )
            buffer.clear()
        self.power_updates += 1
        self.phase_updates += 1
        self.rollout_index += 1
        if not all(np.isfinite(value) for value in self.last_losses.values()):
            raise FloatingPointError(f"Nonfinite Joint loss: {self.last_losses}")
        return events

    def evaluate(self, role: str) -> dict:
        seeds = self.protocol[role]
        started = time.perf_counter()
        pack = evaluate_runtime(self.runtime, seeds)
        self.evaluation_wall_clock += time.perf_counter() - started
        self.evaluation_interactions += len(seeds) * self.args.max_steps_per_episode
        self.record("evaluation", role=role, metrics=pack["metrics"], seed_list=seeds)
        if role == "selection":
            score = pack["metrics"]["undiscounted_return"]["mean"]
            if self.best_artifact is None or score > self.best_score:
                self.best_score = score
                self.best_artifact = dict(
                    checkpoint_kind=POLICY_KIND,
                    schema_version=1,
                    method=METHOD,
                    args=copy.deepcopy(vars(self.args)),
                    models=cpu_model_snapshot(self.runtime),
                    compatibility=self.compatibility,
                    global_env_steps=self.global_env_steps,
                    episode=self.episodes_done,
                    selection_score=score,
                    evaluation_protocol=self.protocol,
                )
        return pack

    def train_episode(self) -> dict:
        runtime, args = self.runtime, self.args
        if not self.at_episode_boundary:
            raise RuntimeError(
                "A failed partial episode must be resumed from the last committed boundary"
            )
        self.at_episode_boundary = False
        started = time.perf_counter()
        observations = runtime.env.reset()
        aid = runtime.env.agents[0]
        episode_return = 0.0
        epsilon = self.epsilon()
        episode_end = False
        while not episode_end:
            for _ in range(
                min(args.rollout_length, args.max_steps_per_episode - runtime.env.current_step)
            ):
                info = runtime.select(observations[aid], epsilon)
                next_obs, rewards, dones, infos = runtime.env.step({aid: info["env_action"]})
                flags = infos[aid]
                terminated = bool(flags.get("terminated", False))
                truncated = bool(flags.get("time_limit_truncated", False)) and not terminated
                episode_end = bool(dones["__all__"])
                self.store_transition(info, next_obs[aid], rewards[aid], terminated)
                self.global_env_steps += 1
                self.queue_learning_events()
                episode_return += float(rewards[aid])
                observations = next_obs
                if episode_end:
                    break
            tail = self.build_tail_context(observations[aid])
            trajectory_length = len(runtime.power_buffer)
            events = self.update_rollout(tail, truncated=truncated)
            self.record(
                "rollout",
                rollout_index=self.rollout_index,
                trajectory_length=trajectory_length,
                epsilon=epsilon,
                dqn_events=sum(kind == "dqn" for _, kind in events),
                target_events=sum(kind == "target" for _, kind in events),
                **self.last_losses,
            )
        self.episodes_done += 1
        self.training_wall_clock += time.perf_counter() - started
        self.at_episode_boundary = True
        self.record("episode", episode_return=episode_return, epsilon=epsilon)
        if (
            args.eval_every_episodes > 0
            and self.episodes_done % args.eval_every_episodes == 0
            and self.protocol["curve"]
        ):
            self.evaluate("curve")
        if (
            args.selection_every_episodes > 0
            and self.episodes_done % args.selection_every_episodes == 0
        ):
            self.evaluate("selection")
        if self.episodes_done % args.checkpoint_every_episodes == 0:
            self.save_checkpoint()
        if self.episodes_done % args.logging_interval == 0:
            print(
                "[Joint] "
                + json.dumps(
                    dict(
                        seed=args.seed,
                        episode=self.episodes_done,
                        global_env_steps=self.global_env_steps,
                        return_=episode_return,
                        **self.last_losses,
                    ),
                    allow_nan=False,
                )
            )
        return dict(episode_return=episode_return, **self.last_losses)

    def train(self) -> None:
        while self.episodes_done < self.args.max_train_episodes:
            self.train_episode()
        if self.best_artifact is None:
            raise ValueError(
                "No scheduled selection event: set selection_every_episodes within the training horizon"
            )
        self.save_checkpoint()

    def state_dict(self) -> dict:
        if (
            not self.at_episode_boundary
            or self.episodes_done == 0
            or self.pending_events
            or len(self.runtime.power_buffer)
            or len(self.runtime.phase_buffer)
        ):
            raise RecoveryError(
                "Joint checkpoint requires a completed, updated episode boundary before reset"
            )
        return dict(
            method=METHOD,
            args=vars(self.args),
            run_dir=str(self.output_dir),
            compatibility=self.compatibility,
            models=self.runtime.model_state(),
            optimizers={
                name: getattr(self.runtime, name).state_dict()
                for name in self.runtime.optimizer_names
            },
            replay=replay_buffer_state(self.runtime.replay_buffer),
            rng=capture_rng_state(env=self.runtime.env),
            progress={
                name: getattr(self, name)
                for name in (
                    "global_env_steps",
                    "episodes_done",
                    "rollout_index",
                    "dqn_updates",
                    "target_updates",
                    "power_updates",
                    "phase_updates",
                    "training_wall_clock",
                    "evaluation_wall_clock",
                    "evaluation_interactions",
                )
            },
            pending_events=[],
            epsilon=self.epsilon(),
            history=copy.deepcopy(self.history),
            best_artifact=self.best_artifact,
            best_score=self.best_score,
            last_losses=self.last_losses,
            evaluation_protocol=self.protocol,
            boundary="episode_complete_before_reset",
        )

    def save_checkpoint(self, path: Path | None = None) -> None:
        payload = self.state_dict()
        save_joint_recovery(payload, path or self.output_dir / "recovery/latest.ckpt")
        self.publish_committed_outputs()

    def publish_committed_outputs(self) -> None:
        """Publish only records covered by the committed recovery checkpoint."""
        atomic_json_save(
            dict(committed_env_steps=self.global_env_steps, records=self.history),
            self.output_dir / "committed_metrics.json",
        )
        rows = []
        for record in self.history:
            if record["kind"] == "evaluation":
                for name, stats in record["metrics"].items():
                    rows.append(
                        dict(
                            run_id=self.output_dir.name,
                            method=METHOD,
                            train_seed=self.args.seed,
                            global_env_steps=record["global_env_steps"],
                            episode=record["episode"],
                            role=record["role"],
                            metric=name,
                            value=stats["mean"],
                            eval_std=stats["std"],
                        )
                    )
        columns = [
            "run_id",
            "method",
            "train_seed",
            "global_env_steps",
            "episode",
            "role",
            "metric",
            "value",
            "eval_std",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        _atomic_write_bytes(
            self.output_dir / "metrics.csv",
            lambda handle: handle.write(stream.getvalue().encode("utf-8")),
        )
        if self.best_artifact is not None:
            atomic_torch_save(self.best_artifact, self.output_dir / "models/best.ckpt")
        atomic_json_save(
            dict(
                available=self.best_artifact is not None,
                global_env_steps=None
                if self.best_artifact is None
                else self.best_artifact["global_env_steps"],
            ),
            self.output_dir / "models/best.json",
        )
        atomic_json_save(
            dict(
                method=METHOD,
                train_seed=self.args.seed,
                episodes=self.episodes_done,
                per_run_environment_interactions=self.global_env_steps,
                training_wall_clock=self.training_wall_clock,
                evaluation_wall_clock=self.evaluation_wall_clock,
                evaluation_interactions=self.evaluation_interactions,
                worker_count=1,
                dqn_gradient_updates=self.dqn_updates,
                target_updates=self.target_updates,
                update_to_data_ratio=self.dqn_updates / max(1, self.global_env_steps),
            ),
            self.output_dir / "cost.json",
        )

    def load_checkpoint(self, path: Path) -> None:
        payload = load_joint_recovery(path)
        required = {
            "compatibility",
            "models",
            "optimizers",
            "replay",
            "rng",
            "progress",
            "history",
            "best_artifact",
            "best_score",
            "last_losses",
            "boundary",
            "pending_events",
        }
        if required - payload.keys():
            raise RecoveryError(
                f"Joint recovery is incomplete: {sorted(required - payload.keys())}"
            )
        if payload["boundary"] != "episode_complete_before_reset" or payload["pending_events"]:
            raise RecoveryError("Joint recovery is not at a legal committed episode boundary")
        if payload["rng"].get("environment") is None:
            raise RecoveryError("Joint recovery is missing the independent environment RNG")
        validate_signature(payload["compatibility"], self.compatibility, strict=True)
        self.runtime.load_models(payload["models"])
        for name in self.runtime.optimizer_names:
            getattr(self.runtime, name).load_state_dict(payload["optimizers"][name])
        load_replay_buffer_state(self.runtime.replay_buffer, payload["replay"])
        for name, value in payload["progress"].items():
            setattr(self, name, value)
        if self.episodes_done <= 0 or self.episodes_done > self.args.max_train_episodes:
            raise RecoveryError("Recovery episode is outside the configured training horizon")
        self.history = payload["history"]
        self.best_artifact, self.best_score = payload["best_artifact"], payload["best_score"]
        self.last_losses = payload["last_losses"]
        self.publish_committed_outputs()
        restore_rng_state(payload["rng"], env=self.runtime.env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent Joint DQN-PPO-PPO baseline")
    for name, value in DEFAULTS.items():
        if isinstance(value, bool):
            parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(
                "--" + name,
                default=value,
                type=float
                if name == "force_bs_gain_db"
                else (str if value is None else type(value)),
            )
    parser.add_argument("--seeds", default=None, help="Comma-separated independent training seeds")
    parser.add_argument("--output_root", default="joint_dqn_ppo_ppo")
    parser.add_argument("--resume_run_dir", default=None)
    return parser


def main() -> None:
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    parsed = build_parser().parse_args()
    seeds = (
        [parsed.seed] if parsed.seeds is None else [int(value) for value in parsed.seeds.split(",")]
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    root = Path(parsed.resume_run_dir or parsed.output_root).resolve()
    for seed in seeds:
        args = make_args(**({key: getattr(parsed, key) for key in DEFAULTS} | dict(seed=seed)))
        output = root / f"seed_{seed}"
        resume = output / "recovery/latest.ckpt" if parsed.resume_run_dir else None
        trainer = JointTrainer(args, output, resume_checkpoint=resume)
        trainer.train()
        from atprl.evaluation.joint import evaluate_artifact

        endpoint = dict(
            checkpoint_kind=POLICY_KIND,
            schema_version=1,
            method=METHOD,
            args=copy.deepcopy(vars(args)),
            models=cpu_model_snapshot(trainer.runtime),
            compatibility=trainer.compatibility,
            global_env_steps=trainer.global_env_steps,
            episode=trainer.episodes_done,
            selection_score=None,
            artifact_role="fixed_training_endpoint",
            evaluation_protocol=trainer.protocol,
        )
        atomic_torch_save(endpoint, output / "models/endpoint.ckpt")
        pack = evaluate_artifact(
            output / "models/endpoint.ckpt", device=args.device, seeds=trainer.protocol["final"]
        )
        pack["artifact_role"] = "fixed_training_endpoint"
        atomic_json_save(pack, output / "final_evaluation.json")
    from atprl.evaluation.joint import aggregate_runs

    aggregate_runs(root, seeds)
    print(f"[Joint] completed seeds={seeds}; outputs={root}")


if __name__ == "__main__":
    main()
