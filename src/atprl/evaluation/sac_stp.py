import copy, json
from pathlib import Path
import numpy as np
from atprl.envs.leo_sat import LeoSatEnv
from atprl.training.recovery import capture_rng_state, restore_rng_state
from atprl.training.warmup import _sample_random_episode_phase
from atprl.evaluation.utils import _jain_index, summarize_metrics_by_seed
from atprl.training.sac_recovery import load_torch, digest, file_sha, write_json, source_signature

METRICS = ["undiscounted_return", "avg_sum_rate", "min_user_avg_rate", "jain_index"]


def evaluate_episode(rt, seed, observer=None):
    saved = capture_rng_state()
    cfg = dict(rt.env_config, random_seed=int(seed))
    env = LeoSatEnv(cfg)
    try:
        obs = env.reset(seed=int(seed))
        aid = env.agents[0]
        n = 0
        ret = 0.0
        rates = np.zeros(rt.config.num_ground_stations)
        positive = sic = private = saturated = 0
        shares_sum = np.zeros(rt.config.num_gs_per_rb + 1)
        shares_squared = np.zeros_like(shares_sum)
        jac_sum = 0.0
        phi = _sample_random_episode_phase(env, int(seed)) if rt.stage == "warmup" else None
        while True:
            info = rt.select(obs[aid], deterministic=True)
            action = info["env_action"]
            if phi is not None and n == 0:
                action = (*action, phi)
            obs, rewards, dones, infos = env.step({aid: action})
            z = infos[aid]
            ret += float(rewards[aid])
            rv = np.asarray(z["rates_macro_vec"])
            if observer is not None:
                observer.observe(info, action, obs[aid], z, float(rewards[aid]))
            rates += rv
            positive += rv.sum() > 0
            sic += bool(z["sic_success_last_micro"])
            n += 1
            a = info["power_policy_action"].astype(np.float64)
            shares_sum += a
            private += a[1:].sum() > 0.01
            shares_squared += a * a
            saturated += a[0] >= 0.995
            jac_sum += np.linalg.norm(np.diag(a) - np.outer(a, a))
            if dones["__all__"]:
                break
        rates /= n
        result = dict(
            undiscounted_return=ret,
            avg_sum_rate=float(rates.sum()),
            min_user_avg_rate=float(rates.min()),
            jain_index=float(_jain_index(rates)),
            n_steps=n,
            environment_seed=int(seed),
            training_seed=rt.config.seed,
            diagnostics=dict(
                positive_rate_macro_fraction=float(positive / n),
                sic_success_last_micro_fraction=sic / n,
                power_shares_mean=(shares_sum / n).tolist(),
                actor_private_total_gt_001_fraction=private / n,
                softmax_jacobian_fro_mean=jac_sum / n,
                actor_common_ge_0995_fraction=saturated / n,
                power_shares_std=np.sqrt(
                    np.maximum(shares_squared / n - (shares_sum / n) ** 2, 0.0)
                ).tolist(),
            ),
        )
        if not np.isfinite([result[k] for k in METRICS]).all():
            raise FloatingPointError("Non-finite evaluation")
        if observer is not None:
            observer.finish(result)
        return result
    finally:
        restore_rng_state(saved)


def evaluate_curve(trainer):
    recorder = getattr(trainer, "stage1_recorder", None)
    rows = [
        evaluate_episode(trainer.runtime, seed, observer=recorder.evaluation_sink(seed))
        if recorder is not None
        else evaluate_episode(trainer.runtime, seed)
        for seed in trainer.config.evaluation_seeds
    ]
    result = dict(
        episodes_done=trainer.episodes_done,
        stage=trainer.runtime.stage,
        training_seed=trainer.config.seed,
        cumulative_episode=trainer.episodes_done
        + (trainer.config.stage1_episodes if trainer.runtime.stage == "stage2" else 0),
        seed_list=list(trainer.config.evaluation_seeds),
        per_seed=rows,
        metrics=summarize_metrics_by_seed(rows, METRICS),
    )
    if trainer.config.phase_distribution == "wrapped_normal_v1":
        result["phase_distribution"] = "wrapped_normal_v1"
    return result


def load_policy(path):
    from atprl.config.sac import SACRunConfig
    from atprl.training.sac_stp import SACWarmupTrainer, SACPhase2Trainer
    from atprl.training.sac_recovery import load_kernel

    p = load_torch(path)
    if p.get("kind") == "sac_stp_policy_v2":
        from atprl.training.sac_v2 import policy_runtime

        return policy_runtime(p), p
    if p.get("algorithm") != "sac_stp" or p.get("kind") not in [
        "sac_stage1_kernel",
        "sac_stp_policy",
    ]:
        raise ValueError("Not SAC policy")
    cfg = SACRunConfig(**p["config"])
    if p["kind"] == "sac_stage1_kernel":
        tr = SACWarmupTrainer(cfg)
        load_kernel(tr.runtime, p)
    else:
        tr = SACPhase2Trainer(cfg, p["kernel"])
        tr.runtime.phase.load_state_dict(p["phase"])
    return tr.runtime, p


def resume_evaluation(checkpoint, directory, seeds, *, evaluator=None, max_new=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seeds = list(map(int, seeds))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate evaluation seeds")
    rt, p = load_policy(checkpoint) if evaluator is None else (None, load_torch(checkpoint))
    identity = dict(
        policy_sha256=file_sha(checkpoint),
        artifact=p["artifact_identity"],
        episode=p["episode"],
        training_seed=p["training_seed"],
        config_identity=digest(p["config"]),
        seed_list=seeds,
        source_signature=source_signature(),
        metric_version="original_stp_v1",
        deterministic="logratio_center_tanh_angle_v1",
    )
    if p["config"].get("phase_distribution") == "wrapped_normal_v1":
        identity.update(
            deterministic="logratio_center_wrapped_mu_v1", phase_distribution="wrapped_normal_v1"
        )
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
        raise ValueError("Evaluation identity mismatch; no mixed models")
    if not manifest.exists():
        write_json(manifest, identity)
    rows = []
    new = 0
    for seed in seeds:
        path = directory / f"seed_{seed}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["identity"] != digest(identity) or row["environment_seed"] != seed:
                raise ValueError("Seed identity mismatch")
        else:
            if max_new is not None and new >= max_new:
                break
            if (directory / "stop.request").exists() or (
                directory.parent.parent / "stop.request"
            ).exists():
                break
            values = evaluate_episode(rt, seed) if evaluator is None else evaluator(seed)
            row = dict(values, identity=digest(identity), environment_seed=seed)
            if not np.isfinite([row[k] for k in METRICS]).all():
                raise ValueError("Invalid evaluation metrics")
            write_json(path, row)
            new += 1
        rows.append(row)
        if not np.isfinite([row[k] for k in METRICS]).all():
            raise ValueError("Corrupt saved evaluation values")
        if evaluator is None and row.get("n_steps") != p["config"]["macro_steps_per_episode"]:
            raise ValueError("Evaluation macro budget mismatch")
        write_json(
            directory / "progress.json",
            dict(
                completed=len(rows),
                total=len(seeds),
                complete=len(rows) == len(seeds),
                identity=digest(identity),
            ),
        )
    result = dict(
        complete=len(rows) == len(seeds),
        seed_list=[r["environment_seed"] for r in rows],
        metrics=summarize_metrics_by_seed(rows, METRICS),
        training_seed=p["training_seed"],
        artifact=p["artifact_identity"],
        policy_sha256=file_sha(checkpoint),
        per_seed=rows,
    )
    write_json(directory / "metrics_pack.json", result)
    if result["complete"]:
        write_json(
            directory / "complete.json", dict(identity=digest(identity), completed=len(rows))
        )
    return result


def evaluate_final500(checkpoint, directory, config):
    return resume_evaluation(
        checkpoint,
        directory,
        [config.eval_seed_base + config.eval_seed_stride * i for i in range(config.num_eval_seeds)],
    )


def evaluate_from_runner(args):
    rt, p = load_policy(args.sac_checkpoint)
    for key in [
        "num_ground_stations",
        "num_gs_per_rb",
        "topology_profile",
        "fairness_lambda",
        "csi_error_variance",
        "ris_num_elem_x",
        "ris_num_elem_y",
    ]:
        if getattr(args, key) != getattr(rt.config, key):
            raise ValueError("SAC evaluation physics mismatch: " + key)
    if float(args.ris_reflection_power_limit_dBm) != float(
        rt.env_config["ris_reflection_power_limit_dBm"]
    ):
        raise ValueError("SAC RIS budget mismatch")
    seeds = [args.eval_seed_base + args.eval_seed_stride * i for i in range(args.num_eval_seeds)]
    result = resume_evaluation(args.sac_checkpoint, args.output_dir, seeds)
    return dict(
        baseline_name="sac_stp",
        source_path=str(args.sac_checkpoint),
        complete=result["complete"],
        training_seed=p["training_seed"],
        artifact=p["artifact_identity"],
        **{k: result["metrics"][k]["mean"] for k in METRICS},
    )
