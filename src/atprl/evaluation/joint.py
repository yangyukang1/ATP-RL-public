from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import torch
import yaml

from atprl.evaluation.utils import DEFAULT_METRIC_KEYS, summarize_values
from atprl.training.joint import METHOD, POLICY_KIND, evaluate_runtime
from atprl.training.joint_runtime import JointRuntime, make_args
from atprl.training.recovery import (
    RecoveryError,
    _atomic_write_bytes,
    atomic_json_save,
    capture_rng_state,
    restore_rng_state,
)


def evaluate_artifact(path: Path, *, device: str, seeds: list[int]) -> dict:
    path = Path(path).resolve()
    rng = capture_rng_state()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("checkpoint_kind") != POLICY_KIND
            or payload.get("schema_version") != 1
        ):
            raise RecoveryError("Evaluation requires a complete Joint policy artifact")
        if path.name == "best.ckpt":
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if (
                not metadata.get("available")
                or metadata["global_env_steps"] != payload["global_env_steps"]
            ):
                raise RecoveryError(
                    "Joint best artifact does not match the committed selection snapshot"
                )
        runtime = JointRuntime(make_args(**(payload["args"] | dict(device=device))))
        runtime.load_models(payload["models"])
        pack = evaluate_runtime(runtime, seeds)
        return dict(
            method=METHOD,
            train_seed=runtime.args.seed,
            selected_global_env_steps=payload["global_env_steps"],
            selected_episode=payload["episode"],
            selection_score=payload["selection_score"],
            evaluation_protocol=payload["evaluation_protocol"],
            phase_state=runtime.phase_state_spec,
            environment=runtime.env_config,
            **pack,
        )
    finally:
        restore_rng_state(rng)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_bytes(path, lambda handle: handle.write(stream.getvalue().encode("utf-8")))


def aggregate_runs(root: Path, train_seeds: list[int]) -> dict:
    """Average environment seeds within each model, then aggregate across training seeds."""
    root = Path(root).resolve()
    finals, curves, costs, phase_states = [], [], [], []
    for seed in train_seeds:
        directory = root / f"seed_{seed}"
        final = json.loads((directory / "final_evaluation.json").read_text(encoding="utf-8"))
        if final["train_seed"] != seed:
            raise ValueError("Final evaluation train_seed does not match its output directory")
        best = json.loads((directory / "models/best.json").read_text(encoding="utf-8"))
        if final.get("artifact_role") == "fixed_training_endpoint":
            endpoint = torch.load(
                directory / "models/endpoint.ckpt", map_location="cpu", weights_only=False
            )
            valid = endpoint["global_env_steps"] == final["selected_global_env_steps"]
        else:
            valid = (
                best["available"] and best["global_env_steps"] == final["selected_global_env_steps"]
            )
        if not valid:
            raise ValueError("Final evaluation is stale relative to the committed best artifact")
        finals.append(final)
        phase_state = final.get("phase_state")
        if phase_state is None:
            payload = torch.load(
                directory / "models/best.ckpt", map_location="cpu", weights_only=False
            )
            phase_state = {
                key: payload["args"][key]
                for key in (
                    "phase_state_mode",
                    "phase_compact_include_mrt_features",
                    "include_mrt_features",
                )
            }
            phase_state["phase_state_dim"] = int(
                payload["models"]["phase_actor"]["body.0.weight"].shape[1]
            )
        phase_states.append(phase_state)
        records = json.loads((directory / "committed_metrics.json").read_text(encoding="utf-8"))[
            "records"
        ]
        curve = {}
        for record in records:
            if record["kind"] == "evaluation" and record["role"] == "curve":
                x = record["global_env_steps"]
                if x in curve:
                    raise ValueError(f"Duplicate committed curve point: train_seed={seed}, x={x}")
                curve[x] = record["metrics"]
        curves.append(curve)
        costs.append(json.loads((directory / "cost.json").read_text(encoding="utf-8")))
    if not finals or len(set(train_seeds)) != len(train_seeds):
        raise ValueError("Aggregation requires nonempty unique training seeds")
    if any(pack["seed_list"] != finals[0]["seed_list"] for pack in finals):
        raise ValueError("All trained models must use the same final-report seed set")
    if any(state != phase_states[0] for state in phase_states):
        raise ValueError("Cannot aggregate different Phase state modes or feature configurations")
    environments = [
        {k: v for k, v in pack["environment"].items() if k != "random_seed"} for pack in finals
    ]
    if any(environment != environments[0] for environment in environments):
        raise ValueError("Cannot aggregate different physical experiment configurations")
    if any(pack["evaluation_protocol"] != finals[0]["evaluation_protocol"] for pack in finals):
        raise ValueError("Cannot aggregate different evaluation protocols")
    fields = {
        name: summarize_values([pack["metrics"][name]["mean"] for pack in finals])
        for name in DEFAULT_METRIC_KEYS
    }
    shared = sorted(set.intersection(*(set(curve) for curve in curves)))
    union = set.union(*(set(curve) for curve in curves))
    run_id = f"{root.parent.name}/{root.name}"
    summary = dict(
        method=METHOD,
        run_id=run_id,
        environment=environments[0],
        phase_state=phase_states[0],
        train_seeds=train_seeds,
        eval_seeds=finals[0]["seed_list"],
        metrics=fields,
        std_ddof=0,
        curve_grid_rule="intersection_without_interpolation",
        common_curve_grid=shared,
        omitted_noncommon_x=sorted(union - set(shared)),
        evaluation_protocol=finals[0]["evaluation_protocol"],
    )
    rows = []
    for x in shared:
        row = dict(run_id=run_id, method=METHOD, global_env_steps=x, n_train_seeds=len(train_seeds))
        for name in DEFAULT_METRIC_KEYS:
            stats = summarize_values([curve[x][name]["mean"] for curve in curves])
            row[f"{name}_mean"], row[f"{name}_std"] = stats["mean"], stats["std"]
        rows.append(row)
    columns = ["run_id", "method", "global_env_steps", "n_train_seeds"] + [
        f"{name}_{suffix}" for name in DEFAULT_METRIC_KEYS for suffix in ("mean", "std")
    ]
    write_csv(root / "aggregate_curve.csv", rows, columns)
    final_row = dict(
        run_id=run_id,
        method=METHOD,
        n_train_seeds=len(train_seeds),
        num_eval_seeds=len(finals[0]["seed_list"]),
        **phase_states[0],
        **{
            key: environments[0][key]
            for key in (
                "num_ground_stations",
                "num_gs_per_rb",
                "topology_profile",
                "fairness_lambda",
                "csi_error_variance",
                "ris_reflection_power_limit_dBm",
            )
        },
    )
    for name, stats in fields.items():
        final_row[f"{name}_mean"], final_row[f"{name}_std"] = stats["mean"], stats["std"]
    write_csv(root / "aggregate_final.csv", [final_row], list(final_row))
    atomic_json_save(summary, root / "aggregate_summary.json")
    final_interactions = sum(
        len(pack["seed_list"]) * pack["environment"]["max_steps_per_episode"] for pack in finals
    )
    atomic_json_save(
        dict(
            method=METHOD,
            train_seeds=train_seeds,
            total_method_environment_interactions=sum(
                c["per_run_environment_interactions"] for c in costs
            ),
            total_evaluation_interactions=sum(c["evaluation_interactions"] for c in costs)
            + final_interactions,
            total_training_wall_clock=sum(c["training_wall_clock"] for c in costs),
            worker_count_per_train_seed=1,
            comparison="actual_cost_not_budget_matched",
        ),
        root / "cost.json",
    )
    manifest = dict(
        schema_version=1,
        figure=dict(
            kind="line",
            output="joint_convergence.pdf",
            x_label="Training environment interactions",
            y_label="Undiscounted return",
        ),
        series=[
            dict(
                label="Joint DQN-PPO-PPO",
                csv="aggregate_curve.csv",
                x_column="global_env_steps",
                y_column="undiscounted_return_mean",
                std_column="undiscounted_return_std",
                require_distinct_run_per_x=False,
            )
        ],
    )
    _atomic_write_bytes(
        root / "plot_manifest.yaml",
        lambda handle: handle.write(yaml.safe_dump(manifest, sort_keys=False).encode("utf-8")),
    )
    return summary


def evaluate_from_runner(args) -> dict:
    seeds = [args.eval_seed_base + i * args.eval_seed_stride for i in range(args.num_eval_seeds)]
    pack = evaluate_artifact(Path(args.joint_checkpoint), device=args.device, seeds=seeds)
    if (
        args.run_batch
        and float(args.ris_reflection_power_limit_dBm)
        != pack["environment"]["ris_reflection_power_limit_dBm"]
    ):
        raise ValueError(
            "Joint batch source must match the requested RIS budget; use an independently trained artifact"
        )
    if args.run_batch and (args.num_ground_stations, args.num_gs_per_rb) != (
        pack["environment"]["num_ground_stations"],
        pack["environment"]["num_gs_per_rb"],
    ):
        raise ValueError("Joint batch source must match the requested N/K topology")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_save(pack, output / "eval_metrics_pack.json")
    atomic_json_save(
        dict(per_seed_metrics=pack["per_seed_metrics"]), output / "per_seed_metrics.json"
    )
    row = dict(
        baseline_name=METHOD,
        eval_source="joint_checkpoint",
        source_path=str(Path(args.joint_checkpoint).resolve()),
        train_seed=pack["train_seed"],
        num_eval_seeds=len(seeds),
        **pack["phase_state"],
    )
    row.update(
        {
            key: pack["environment"][key]
            for key in (
                "num_ground_stations",
                "num_gs_per_rb",
                "topology_profile",
                "fairness_lambda",
                "csi_error_variance",
                "ris_reflection_power_limit_dBm",
            )
        }
    )
    for name, stats in pack["metrics"].items():
        row[f"{name}_mean"], row[f"{name}_std"] = stats["mean"], stats["std"]
    write_csv(output / args.csv_name, [row], list(row))
    print(f"[Eval] Joint results written to {output}")
    return row
