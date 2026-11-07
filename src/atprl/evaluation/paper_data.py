from __future__ import annotations

import hashlib
import json
from pathlib import Path

CORE = ("undiscounted_return", "avg_sum_rate", "min_user_avg_rate", "jain_index")
FILES = dict(
    summary_json="eval_metrics_pack.json",
    summary_csv="eval_summary.csv",
    metadata="eval_meta.json",
    per_seed_jsonl="per_seed_metrics.jsonl",
    per_seed_json="per_seed_metrics.json",
    diagnostics_jsonl="solver_diagnostics.jsonl",
    diagnostics_summary="solver_diagnostics_summary.json",
)


def publish_result_index(directory):
    from atprl.evaluation.conventional import write_json

    directory = Path(directory).resolve()
    pack = json.loads((directory / FILES["summary_json"]).read_text(encoding="utf-8"))
    from atprl.evaluation.conventional_config import resolve_method_phase

    resolve_method_phase(pack.get("baseline_name"), pack.get("phase_solver"))
    if pack.get("method") != pack.get("baseline_name"):
        raise ValueError("Result method identity mismatch")
    if (
        pack.get("method_type") != "non_learning"
        or pack.get("has_training_trajectory") is not False
    ):
        raise ValueError("Traditional result must not contain a training trajectory")
    forbidden = {"training_seed", "training_episode", "training_checkpoint"}
    if forbidden & set(pack):
        raise ValueError("Fake training fields in traditional result")
    for metric in CORE:
        for stat in ("mean", "std"):
            if metric + "_" + stat not in pack:
                raise ValueError("Missing canonical metric")
    rows = [
        json.loads(line)
        for line in (directory / FILES["per_seed_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    expected = [seed for repeat in pack["seed_matrix"] for seed in repeat]
    if [row["eval_seed"] for row in rows] != expected:
        raise ValueError("Per-seed identity/order mismatch")
    for row in rows:
        if not set(CORE).issubset(row) or not all(
            name in row for name in ("seed_index", "repeat_index", "n_steps")
        ):
            raise ValueError("Incomplete per-seed metric row")
    checksums = {
        key: hashlib.sha256((directory / file).read_bytes()).hexdigest()
        for key, file in FILES.items()
    }
    index = dict(
        schema_version=1,
        method=pack["method"],
        baseline_name=pack["baseline_name"],
        phase_solver=pack["phase_solver"],
        ris_init=pack["ris_init"],
        method_type="non_learning",
        has_training_trajectory=False,
        scenario_id=pack["scenario_id"],
        scenario_label=f"N{pack['num_ground_stations']}K{pack['num_gs_per_rb']}",
        axes={
            key: pack[key]
            for key in (
                "fairness_lambda",
                "csi_error_variance",
                "ris_reflection_power_limit_dBm",
                "num_ground_stations",
                "num_gs_per_rb",
                "topology_profile",
            )
        },
        physical_protocol_hash=pack["physical_protocol_hash"],
        seed_matrix_hash=pack["seed_matrix_hash"],
        statistic_axis="environment_seed",
        std_ddof=0,
        files=FILES,
        checksums=checksums,
        convergence_reference=dict(
            kind="horizontal_non_learning_performance_reference",
            summary_file=FILES["summary_json"],
            metrics=pack["reference_metrics"],
        ),
    )
    write_json(directory / "result_index.json", index)
    return index


def write_result_registry(run_directories, path):
    from atprl.evaluation.conventional import write_json

    path = Path(path).resolve()
    entries = []
    seen = set()
    for directory in run_directories:
        directory = Path(directory).resolve()
        index = publish_result_index(directory)
        key = (index["method"], index["scenario_id"], index["seed_matrix_hash"])
        if key in seen:
            raise ValueError("Duplicate experiment identity in paper registry")
        seen.add(key)
        entries.append(dict(index=index, index_path=str(directory / "result_index.json")))
    payload = dict(schema_version=1, kind="paper_result_registry", entries=entries)
    write_json(path, payload)
    return payload


def validate_comparison_protocol(run_directories, varying_axes=()):
    packs = [
        json.loads((Path(directory) / "eval_metrics_pack.json").read_text(encoding="utf-8"))
        for directory in run_directories
    ]
    if len(packs) < 2:
        raise ValueError("Comparison requires at least two results")
    allowed = set(varying_axes)
    reference = packs[0]
    physics = reference.get("resolved_physical_config")
    if not isinstance(physics, dict):
        raise ValueError("Missing resolved physical protocol; cannot infer it")
    shared = (
        "ao_max_iters",
        "ao_tol",
        "wmmse_max_iters",
        "wmmse_tol",
        "wmmse_scipy_maxiter",
        "wmmse_scipy_ftol",
        "phase1_max_iters",
        "sca_trust_radius",
        "sca_scipy_maxiter",
        "sca_scipy_ftol",
        "sic_guard",
        "feasibility_tol",
        "stopping_patience",
        "decision_time_limit_s",
    )
    for pack in packs[1:]:
        actual = pack.get("resolved_physical_config")
        if not isinstance(actual, dict) or set(actual) != set(physics):
            raise ValueError("Incomplete physical protocol")
        differences = [key for key in physics if key not in allowed and actual[key] != physics[key]]
        if differences:
            raise ValueError("Physical comparison mismatch: " + ",".join(differences))
        for key in (
            "seed_matrix",
            "statistic_axis",
            "scheduling_policy",
            "num_eval_seeds",
            "evaluation_repeats",
        ):
            if key not in reference or key not in pack or pack[key] != reference[key]:
                raise ValueError("Comparison identity mismatch: " + key)
        for key in shared:
            if key not in pack.get("solver_options", {}) or pack["solver_options"][
                key
            ] != reference.get("solver_options", {}).get(key):
                raise ValueError("Shared solver budget mismatch: " + key)
        for directory in run_directories:
            meta = json.loads((Path(directory) / "eval_meta.json").read_text(encoding="utf-8"))
            if meta.get("std_ddof") != 0 or meta.get("rate_unit") != "bps/Hz":
                raise ValueError("Comparison statistics mismatch")
    return dict(
        compatible=True,
        methods=[pack["method"] for pack in packs],
        varying_axes=sorted(allowed),
        statistic_axis="environment_seed",
    )
