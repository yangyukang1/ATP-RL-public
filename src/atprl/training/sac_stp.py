from dataclasses import asdict
import copy
import os

for _thread_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[_thread_var] = "1"
import numpy as np
import torch
from atprl.training.sac_runtime import SACSTPRuntime


class SACWarmupTrainer:
    def __init__(self, config):
        self.config = config
        self.runtime = SACSTPRuntime(config, "warmup")
        if config.stage == "fixed_power":
            self.runtime.q_network.requires_grad_(False)
            self.runtime.target_q_network.requires_grad_(False)
        self.episodes_done = self.total_steps = self.dqn_revision = 0
        self.history = []
        self.last_mode = None

    def mode(self):
        if self.config.stage == "fixed_power":
            return "POWER"
        n = self.config.cycle_dqn_episodes + self.config.cycle_power_episodes
        return "DQN" if self.episodes_done % n < self.config.cycle_dqn_episodes else "POWER"

    def train_episode(self):
        rt = self.runtime
        c = self.config
        mode = self.mode()
        recorder = getattr(self, "stage1_recorder", None)
        replay_before = len(rt.power.replay)
        replay_cleared = mode == "POWER" and rt.power.replay.context_revision != self.dqn_revision
        if mode == "POWER" and rt.power.replay.context_revision != self.dqn_revision:
            rt.power.replay.clear_context(self.dqn_revision)
        if recorder is not None:
            recorder.start(mode, self.dqn_revision, replay_before, replay_cleared)
        obs = rt.env.reset()
        phi = np.random.uniform(-np.pi, np.pi, rt.g).astype(np.float32)
        reward_sum = 0.0
        positive = sic = 0
        updates = []
        actor_shares = []
        behavior_shares = []
        jacobians = []
        entropies = []
        rates_sum = np.zeros(c.num_ground_stations, np.float64)
        policy_means = []
        policy_log_stds = []
        for step in range(c.macro_steps_per_episode):
            epsilon = 1.0 + (0.01 - 1.0) * min((self.episodes_done + 1) / 1440, 1.0)
            if step == 0:
                epsilon = getattr(self, "current_epsilon", 1.0)
            info = rt.select(obs[rt.aid], mode=mode, epsilon=epsilon)
            with torch.no_grad():
                distribution = rt.power.actor(
                    torch.as_tensor(info["power_state"], dtype=torch.float32).reshape(1, -1)
                )
                center = distribution.deterministic().action[0].numpy().astype(np.float64)
                policy_means.append(distribution.mean[0].numpy())
                policy_log_stds.append(distribution.log_std[0].numpy())
            actor_shares.append(center)
            behavior_shares.append(info["power_policy_action"])
            jacobians.append(float(np.linalg.norm(np.diag(center) - np.outer(center, center))))
            entropies.append(float(-info["power_sample"].log_prob_reference))
            action = info["env_action"]
            if step == 0:
                action = (*action, phi)
            ns, rewards, dones, infos = rt.env.step({rt.aid: action})
            z = infos[rt.aid]
            reward = float(rewards[rt.aid])
            self.total_steps += 1
            term = bool(z["terminated"])
            trunc = bool(z["time_limit_truncated"])
            dqn_loss = None
            if mode == "DQN":
                rt.dqn_replay.store(
                    info["dqn_state"],
                    info["scheduler_action"],
                    reward,
                    ns[rt.aid][: rt.base_dim],
                    float(term),
                )
                if len(rt.dqn_replay) > 128 and self.total_steps % 4 == 0:
                    dqn_loss = rt.dqn_learner.update(rt.dqn_replay.sample(128))
                    rt.dqn_updates += 1
                    self.dqn_revision += 1
                if self.total_steps % 8 == 0:
                    rt.dqn_learner.update_target_network()
                    rt.dqn_target_updates += 1
                self.current_epsilon = 1.0 + (0.01 - 1.0) * min(
                    (self.episodes_done + 1) / 1440, 1.0
                )
            else:
                next_state = rt.power_state(ns[rt.aid])
                updates += rt.power.observe(
                    info["power_state"],
                    info["power_policy_action"],
                    reward,
                    next_state,
                    term,
                    trunc,
                    episode_id=self.episodes_done + 1,
                    macro_step=step + 1,
                )
            if recorder is not None:
                recorder.active.observe(
                    info,
                    action,
                    ns[rt.aid],
                    z,
                    reward,
                    epsilon=epsilon,
                    dqn_loss=dqn_loss,
                    center=center,
                )
            rates_sum += np.asarray(z["rates_macro_vec"], np.float64)
            reward_sum += reward
            positive += np.sum(z["rates_macro_vec"]) > 0
            sic += bool(z["sic_success_last_micro"])
            live_hook = getattr(self, "live_hook", None)
            if live_hook is not None and (
                (step + 1) % 128 == 0 or step + 1 == c.macro_steps_per_episode
            ):
                live_hook(step + 1, mode)
            obs = ns
            if dones["__all__"]:
                break
        self.episodes_done += 1
        self.last_mode = mode
        if (
            mode == "POWER"
            and self.episodes_done % (c.cycle_dqn_episodes + c.cycle_power_episodes) == 0
        ):
            rt.dqn_replay.clear()
        row = dict(
            episode=self.episodes_done,
            mode=mode,
            macro_steps=step + 1,
            undiscounted_return=reward_sum,
            positive_rate_macro_fraction=float(positive / (step + 1)),
            sic_success_last_micro_fraction=sic / (step + 1),
            sac_updates=rt.power.updates,
            dqn_updates=rt.dqn_updates,
            learner_metrics=updates[-1] if updates else None,
        )
        row.update(
            actor_shares_mean=np.mean(actor_shares, 0).tolist(),
            behavior_shares_mean=np.mean(behavior_shares, 0).tolist(),
            actor_softmax_jacobian_fro_mean=float(np.mean(jacobians)),
            behavior_reference_entropy=float(np.mean(entropies)),
        )
        from atprl.evaluation.utils import _jain_index

        avg_rates = rates_sum / (step + 1)
        centers = np.asarray(actor_shares)
        row.update(
            avg_sum_rate=float(avg_rates.sum()),
            min_user_avg_rate=float(avg_rates.min()),
            jain_index=float(_jain_index(avg_rates)),
            per_user_avg_rate=avg_rates.tolist(),
            actor_shares_std=centers.std(0).tolist(),
            actor_common_ge_0995_fraction=float((centers[:, 0] >= 0.995).mean()),
            actor_private_total_gt_001_fraction=float((centers[:, 1:].sum(1) > 0.01).mean()),
            policy_mean_mean=np.mean(policy_means, 0).tolist(),
            policy_mean_abs_max=float(np.abs(policy_means).max()),
            policy_log_std_mean=np.mean(policy_log_stds, 0).tolist(),
            policy_log_std_min=float(np.min(policy_log_stds)),
            policy_log_std_max=float(np.max(policy_log_stds)),
        )
        rp = rt.power.replay
        actions = rp.arrays["actions"][: len(rp)]
        if len(rp):
            coords = np.clip((actions[:, 1:] / 0.05).astype(np.int64), 0, 19)
            coverage = {
                ",".join(map(str, k)): int(n)
                for k, n in zip(*np.unique(coords, axis=0, return_counts=True))
            }
        else:
            coverage = {}
        row.update(
            replay_size=len(rp),
            power_collected_steps=rt.power.collected_steps,
            replay_store_count=rp.store_count,
            replay_sample_count=rp.sample_count,
            replay_action_bin_width=0.05,
            replay_action_bins=coverage,
            q1_updates=rt.power.updates,
            q2_updates=rt.power.updates,
            actor_updates=rt.power.actor_updates,
            temperature_updates=rt.power.temperature_updates,
            target_updates=rt.power.target_updates,
        )
        if recorder is not None:
            recorder.finish(row)
        self.history.append(row)
        return row

    def train_for(self, episodes):
        target = min(self.episodes_done + episodes, self.config.stage1_episodes)
        while self.episodes_done < target:
            self.train_episode()
        return self.history


class SACPhase2Trainer:
    def __init__(self, config, kernel):
        from atprl.training.sac_recovery import load_kernel, load_torch, digest

        self.config = config
        self.runtime = SACSTPRuntime(config, "stage2")
        if isinstance(kernel, (str, bytes)) or hasattr(kernel, "__fspath__"):
            kernel = load_torch(kernel)
        load_kernel(self.runtime, kernel)
        self.kernel_identity = digest(kernel)
        self.kernel_source = copy.deepcopy(kernel)
        self.episodes_done = self.total_steps = 0
        self.history = []

    def train_episode(self):
        rt = self.runtime
        c = self.config
        obs = rt.env.reset()
        updates = []
        reward_sum = 0.0
        positive = sic = 0
        phase_entropies = []
        for step in range(c.macro_steps_per_episode):
            info = rt.select(obs[rt.aid], mode="PHASE")
            ns, rewards, dones, infos = rt.env.step({rt.aid: info["env_action"]})
            phase_entropies.append(float(-info["phase_sample"].log_prob_reference))
            z = infos[rt.aid]
            r = float(rewards[rt.aid])
            next_state = rt.phase_state(ns[rt.aid])
            updates += rt.phase.observe(
                info["phase_state"],
                info["phase_policy_action"],
                r,
                next_state,
                bool(z["terminated"]),
                bool(z["time_limit_truncated"]),
                episode_id=self.episodes_done + 1,
                macro_step=step + 1,
            )
            self.total_steps += 1
            reward_sum += r
            positive += np.sum(z["rates_macro_vec"]) > 0
            sic += bool(z["sic_success_last_micro"])
            obs = ns
            if dones["__all__"]:
                break
        self.episodes_done += 1
        row = dict(
            episode=self.episodes_done,
            cumulative_episode=c.stage1_episodes + self.episodes_done,
            mode="PHASE",
            macro_steps=step + 1,
            undiscounted_return=reward_sum,
            positive_rate_macro_fraction=float(positive / (step + 1)),
            sic_success_last_micro_fraction=sic / (step + 1),
            sac_updates=rt.phase.updates,
            learner_metrics=updates[-1] if updates else None,
        )
        row["phase_behavior_reference_entropy"] = float(np.mean(phase_entropies))
        self.history.append(row)
        return row

    def train_for(self, episodes):
        target = min(self.episodes_done + episodes, self.config.stage2_episodes)
        while self.episodes_done < target:
            self.train_episode()
        return self.history


def run_stage(config, directory, stage, kernel=None, *, max_episodes=None):
    """Commit complete episodes; return false after a cooperative pause."""
    import json, os
    from pathlib import Path
    from atprl.training.sac_recovery import (
        save_recovery,
        restore_checkpoint,
        write_json,
        write_torch,
        policy_bundle,
        digest,
        load_torch,
        file_sha,
        copy_artifact,
    )
    from atprl.evaluation.sac_stp import evaluate_curve

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.ckpt"
    if latest.exists():
        if not config.resume:
            raise FileExistsError("Existing run requires explicit Resume")
        tr = restore_checkpoint(latest, config)
        old_path = tr.selection.get("best_path")
        if old_path:
            old = out / old_path
            if file_sha(old) != tr.selection["best_sha256"]:
                raise ValueError("Committed best artifact missing/changed")
            copy_artifact(
                old, out / ("kernel_best.ckpt" if stage == "warmup" else "policy_best.ckpt")
            )
    else:
        if config.resume:
            raise FileNotFoundError("No committed checkpoint to resume")
        tr = SACWarmupTrainer(config) if stage == "warmup" else SACPhase2Trainer(config, kernel)
        if config.stage == "fixed_power":
            if not config.dqn_checkpoint:
                raise ValueError("Fixed POWER requires explicit DQN-A checkpoint")
            if (
                config.dqn_checkpoint_sha256
                and file_sha(config.dqn_checkpoint) != config.dqn_checkpoint_sha256
            ):
                raise ValueError("DQN-A input SHA mismatch")
            q = load_torch(config.dqn_checkpoint)
            tr.runtime.q_network.load_state_dict(q)
            tr.runtime.target_q_network.load_state_dict(q)
            tr.runtime.q_network.requires_grad_(False)
            tr.runtime.target_q_network.requires_grad_(False)
        tr.selection = dict(best_score=None, best_episode=None)
        save_recovery(tr, out)
        if stage == "stage2" and config.phase_distribution == "wrapped_normal_v1":
            phase = tr.runtime.phase
            write_json(
                out / "phase_initialization.json",
                dict(
                    phase_distribution=phase.phase_distribution,
                    density_version=phase.state_dict()["density_version"],
                    phase_seed=config.seed + 2000,
                    models_sha256=digest(
                        {n: getattr(phase, n).state_dict() for n in phase.model_names}
                    ),
                    counters={
                        n: getattr(phase, n)
                        for n in [
                            "collected_steps",
                            "updates",
                            "actor_updates",
                            "target_updates",
                            "temperature_updates",
                        ]
                    },
                    replay_size=len(phase.replay),
                    temperature=float(phase.temperature.detach()),
                    kernel_identity=tr.kernel_identity,
                ),
            )
    if config.stage == "warmup" or config.experiment_group is not None:
        from atprl.training.sac_stage1_recording import StageOneRecorder

        tr.stage1_recorder = StageOneRecorder(tr, out)

    def learner_log(metrics):
        payload = json.dumps(dict(metrics, stage=stage), allow_nan=False) + "\n"
        from atprl.training.sac_recovery import retry

        log = out / "learner_updates.jsonl"
        offset = log.stat().st_size if log.exists() else 0

        def write():
            with log.open("r+b" if log.exists() else "w+b") as f:
                f.seek(offset)
                f.truncate()
                f.write(payload.encode())
                f.flush()

        retry(write)

    (tr.runtime.phase or tr.runtime.power).metrics_hook = learner_log

    def live_progress(macro, mode):
        learner = tr.runtime.phase or tr.runtime.power
        write_json(
            out / "progress.json",
            dict(
                stage=stage,
                status="training",
                pid=os.getpid(),
                mode=mode,
                episode=tr.episodes_done,
                episode_current=tr.episodes_done + 1,
                macro_step=macro,
                train_macro_steps=tr.total_steps,
                replay_size=len(learner.replay),
                replay_store_count=learner.replay.store_count,
                q1_updates=learner.updates,
                q2_updates=learner.updates,
                power_updates=tr.runtime.power.updates,
                actor_updates=learner.actor_updates,
                target_updates=learner.target_updates,
                temperature_updates=learner.temperature_updates,
                dqn_updates=tr.runtime.dqn_updates,
                dqn_target_updates=tr.runtime.dqn_target_updates,
                dqn_revision=getattr(tr, "dqn_revision", 0),
                dqn_replay_size=len(tr.runtime.dqn_replay),
                cycle=tr.episodes_done // (config.cycle_dqn_episodes + config.cycle_power_episodes)
                + 1,
                checkpoint=str(latest),
                live_may_be_uncommitted=True,
            ),
        )

    tr.live_hook = live_progress
    budget = (
        config.fixed_power_episodes
        if config.stage == "fixed_power"
        else config.stage1_episodes
        if stage == "warmup"
        else config.stage2_episodes
    )
    if (out / "complete.json").exists() and tr.episodes_done == budget:
        return tr, True
    begin = tr.episodes_done
    while tr.episodes_done < budget:
        if (out.parent / "stop.request").exists() or (out / "stop.request").exists():
            save_recovery(tr, out)
            return tr, False
        row = tr.train_episode()
        with (out / "training.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
        due = (
            tr.episodes_done in config.evaluation_episodes
            if config.stage == "fixed_power"
            else tr.episodes_done % config.eval_every_episodes == 0 or tr.episodes_done == budget
        )
        if due:
            ev = evaluate_curve(tr)
            write_json(out / f"evaluation_ep{tr.episodes_done}.json", ev)
            score = ev["metrics"]["undiscounted_return"]["mean"]
            if (config.stage != "fixed_power" or config.experiment_group is not None) and (
                tr.selection["best_score"] is None or score > tr.selection["best_score"]
            ):
                bundle = policy_bundle(tr, "strict_best")
                name = f"policies/best_ep{tr.episodes_done}.ckpt"
                write_torch(out / name, bundle)
                tr.selection = dict(
                    best_score=score,
                    best_episode=tr.episodes_done,
                    best_path=name,
                    best_sha256=file_sha(out / name),
                )
                copy_artifact(
                    out / name,
                    out / ("kernel_best.ckpt" if stage == "warmup" else "policy_best.ckpt"),
                )
                write_json(out / "best_meta.json", tr.selection)
        if (
            tr.episodes_done % config.checkpoint_every_episodes == 0
            or tr.episodes_done == budget
            or (config.stage == "fixed_power" and due)
        ):
            save_recovery(tr, out)
        write_json(
            out / "progress.json",
            dict(
                stage=stage,
                status="training",
                pid=os.getpid(),
                episode=tr.episodes_done,
                budget=budget,
                train_macro_steps=tr.total_steps,
                power_updates=tr.runtime.power.updates,
                phase_updates=tr.runtime.phase.updates if tr.runtime.phase else 0,
                actor_updates=(tr.runtime.phase or tr.runtime.power).actor_updates,
                target_updates=(tr.runtime.phase or tr.runtime.power).target_updates,
                temperature_updates=(tr.runtime.phase or tr.runtime.power).temperature_updates,
                checkpoint=str(latest),
                live_may_be_uncommitted=True,
                episode_current=tr.episodes_done,
                macro_step=row["macro_steps"],
                replay_size=len((tr.runtime.phase or tr.runtime.power).replay),
                mode=row["mode"],
                dqn_updates=tr.runtime.dqn_updates,
                dqn_target_updates=tr.runtime.dqn_target_updates,
                dqn_revision=getattr(tr, "dqn_revision", 0),
                dqn_replay_size=len(tr.runtime.dqn_replay),
                q1_updates=(tr.runtime.phase or tr.runtime.power).updates,
                q2_updates=(tr.runtime.phase or tr.runtime.power).updates,
            ),
        )
        print(
            f"SAC seed={config.seed} stage={stage} episode={tr.episodes_done}/{budget} macro={tr.total_steps} updates={(tr.runtime.phase or tr.runtime.power).updates}",
            flush=True,
        )
        if max_episodes is not None and tr.episodes_done - begin >= max_episodes:
            save_recovery(tr, out)
            return tr, False
    name = (
        f"power_endpoint_{budget}.ckpt"
        if config.stage == "fixed_power"
        else f"kernel_endpoint_{config.stage1_episodes}.ckpt"
        if stage == "warmup"
        else f"endpoint_{config.stage1_episodes + config.stage2_episodes}.ckpt"
    )
    write_torch(out / name, policy_bundle(tr, "endpoint"))
    save_recovery(tr, out)
    write_json(
        out / "complete.json",
        dict(
            stage=stage,
            episodes=tr.episodes_done,
            train_macro_steps=tr.total_steps,
            endpoint=name,
            best=tr.selection,
        ),
    )
    return tr, True


def run_seed(config, directory):
    from pathlib import Path
    from dataclasses import replace
    from atprl.training.sac_recovery import write_json, load_torch
    from atprl.evaluation.sac_stp import evaluate_final500

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    if config.stage == "evaluate":
        if not config.checkpoint:
            raise ValueError("Evaluation requires checkpoint")
        return evaluate_final500(config.checkpoint, out, config)
    if config.stage == "fixed_power":
        return run_stage(config, out / "fixed_power", "warmup")
    if config.stage in ["pipeline", "warmup"]:
        _, done = run_stage(config, out / "stage1", "warmup")
        if not done:
            return None
        if config.stage == "warmup":
            return True
        kernel = out / "stage1/kernel_best.ckpt"
    else:
        if not config.kernel_checkpoint:
            raise ValueError("Stage-II requires strict SAC kernel")
        kernel = Path(config.kernel_checkpoint)
        if config.kernel_checkpoint_sha256:
            from atprl.training.sac_recovery import file_sha

            if file_sha(kernel) != config.kernel_checkpoint_sha256:
                raise ValueError("Stage-II kernel SHA mismatch")
    phase_config = (
        replace(config, resume=(out / "stage2/latest.ckpt").exists()) if config.resume else config
    )
    _, done = run_stage(phase_config, out / "stage2", "stage2", kernel)
    if not done:
        return None
    if config.auto_final_evaluation is False:
        write_json(
            out / "training_complete.json",
            dict(
                complete=True,
                scope="stage2_training_only",
                training_seed=config.seed,
                stage2_episodes=config.stage2_episodes,
                total_training_episodes=config.stage1_episodes + config.stage2_episodes,
                final_evaluation=False,
            ),
        )
        return True
    result = evaluate_final500(
        out / "stage2/policy_best.ckpt", out / "evaluation/strict_best", config
    )
    if result["complete"]:
        write_json(
            out / "complete.json",
            dict(
                complete=True,
                training_seed=config.seed,
                total_training_episodes=config.stage1_episodes + config.stage2_episodes,
            ),
        )
    return result


def main():
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    import argparse, json, os
    from pathlib import Path
    from dataclasses import replace
    from atprl.config.sac import SACRunConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-json", required=True)
    args = parser.parse_args()
    config = SACRunConfig(**json.loads(args.settings_json))
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = "1"
    if config.resume_checkpoint:
        from atprl.training.sac_recovery import load_torch

        cp = Path(config.resume_checkpoint).resolve()
        if cp.name != "latest.ckpt":
            raise ValueError(
                "Explicit Resume requires the committed latest.ckpt; fork old checkpoints separately"
            )
        saved = load_torch(cp)
        if saved.get("kind") != "sac_stp_recovery":
            raise ValueError("Not SAC recovery")
        run_seed(replace(config, seed=saved["config"]["seed"], resume=True), cp.parent.parent)
        return
    if config.stage == "evaluate":
        run_seed(config, config.output_dir)
        return
    for seed in config.seeds:
        run_seed(replace(config, seed=seed), Path(config.output_dir) / f"seed{seed}")


if __name__ == "__main__":
    main()
