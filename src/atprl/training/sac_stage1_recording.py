from pathlib import Path
import time
import numpy as np
from atprl.training.recovery import _atomic_write_bytes
from atprl.training.sac_recovery import retry, write_json, digest


class Trajectory:
    def __init__(self, path, identity, *, evaluation=False):
        self.path = Path(path)
        self.identity = identity
        self.evaluation = evaluation
        self.rows = []

    def observe(
        self, info, action, next_obs, z, reward, *, epsilon=0.0, dqn_loss=None, center=None
    ):
        row = dict(
            selected_users=np.asarray(info["selected_users"], np.int16),
            shares=np.asarray(info["power_policy_action"], np.float32).copy(),
            executed_watts=np.asarray(action[1], np.float32).copy(),
            actor_center=np.asarray(
                center if center is not None else info["power_policy_action"], np.float32
            ).copy(),
            policy_latent=info["power_sample"].latent[0].detach().numpy().copy(),
            log_prob_reference=float(info["power_sample"].log_prob_reference),
            reward=float(reward),
            rates=np.asarray(z["rates_macro_vec"], np.float64).copy(),
            positive_rate=bool(np.sum(z["rates_macro_vec"]) > 0),
            sic_success_last_micro=bool(z["sic_success_last_micro"]),
            terminated=bool(z["terminated"]),
            truncated=bool(z["time_limit_truncated"]),
            epsilon=float(epsilon),
            dqn_update=dqn_loss is not None,
            dqn_loss=0.0 if dqn_loss is None else float(dqn_loss),
        )
        if self.evaluation:
            row.update(
                power_state=np.asarray(info["power_state"], np.float32).copy(),
                next_observation=np.asarray(next_obs, np.float32).copy(),
            )
        if not self.rows:
            self.identity = dict(
                self.identity,
                episode_phase=np.asarray(action[2]).tolist() if len(action) > 2 else None,
            )
        self.rows.append(row)

    def finish(self, metrics):
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise FloatingPointError("Non-finite Stage-I trace")
        retry(_atomic_write_bytes, self.path, lambda f: np.savez_compressed(f, **arrays))
        from atprl.training.sac_recovery import file_sha

        write_json(
            self.path.with_suffix(".json"),
            dict(
                identity=self.identity,
                metrics=metrics,
                macrosteps=len(self.rows),
                trajectory=self.path.name,
                sha256=file_sha(self.path),
            ),
        )


class StageOneRecorder:
    def __init__(self, trainer, directory):
        self.trainer = trainer
        self.directory = Path(directory)
        self.active = None
        if trainer.config.resume:
            for path in list((self.directory / "traces").glob("training/ep*.npz")) + list(
                (self.directory / "traces").glob("evaluation/ep*/seed*.npz")
            ):
                ep = (
                    int(path.stem[2:])
                    if path.parent.name == "training"
                    else int(path.parent.name[2:])
                )
                if ep > trainer.episodes_done:
                    target = (
                        self.directory
                        / "uncommitted_trace_attempts"
                        / str(time.time_ns())
                        / path.relative_to(self.directory / "traces")
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    retry(path.replace, target)
                    if path.with_suffix(".json").exists():
                        retry(path.with_suffix(".json").replace, target.with_suffix(".json"))

    def start(self, mode, revision, replay_before, cleared):
        tr = self.trainer
        c = tr.config
        ep = tr.episodes_done + 1
        self.started_cpu = time.process_time()
        self.started_wall = time.perf_counter()
        self.segment = dict(
            cycle=(ep - 1) // (c.cycle_dqn_episodes + c.cycle_power_episodes) + 1,
            segment=mode,
            dqn_revision_start=revision,
            power_replay_before=replay_before,
            power_replay_clear_event=cleared,
        )
        self.active = Trajectory(
            self.directory / f"traces/training/ep{ep}.npz",
            dict(training_seed=c.seed, episode=ep, behavior="stochastic_training", **self.segment),
        )

    def finish(self, row):
        rt = self.trainer.runtime
        counts = {
            ",".join(map(str, pair)): int(count)
            for pair, count in zip(
                *np.unique(
                    np.asarray([x["selected_users"] for x in self.active.rows]),
                    axis=0,
                    return_counts=True,
                )
            )
        }
        losses = [x["dqn_loss"] for x in self.active.rows if x["dqn_update"]]
        row.update(
            self.segment,
            dqn_revision=self.trainer.dqn_revision,
            dqn_target_updates=rt.dqn_target_updates,
            dqn_replay_size=len(rt.dqn_replay),
            dqn_epsilon_first=self.active.rows[0]["epsilon"],
            dqn_epsilon_last=self.active.rows[-1]["epsilon"],
            dqn_q_loss_mean=float(np.mean(losses)) if losses else None,
            dqn_updates_this_episode=len(losses),
            selected_user_combination_counts=counts,
            episode_cpu_seconds=time.process_time() - self.started_cpu,
            episode_wall_seconds=time.perf_counter() - self.started_wall,
        )
        self.active.finish(row)
        self.active = None

    def evaluation_sink(self, seed):
        tr = self.trainer
        rt = tr.runtime
        return Trajectory(
            self.directory / f"traces/evaluation/ep{tr.episodes_done}/seed{seed}.npz",
            dict(
                training_seed=tr.config.seed,
                environment_seed=int(seed),
                episode=tr.episodes_done,
                behavior="deterministic_evaluation",
                model_identity=digest(
                    dict(q=rt.q_network.state_dict(), power=rt.power.actor.state_dict())
                ),
            ),
            evaluation=True,
        )
