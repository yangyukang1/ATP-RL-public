from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import struct
import subprocess
from pathlib import Path
import numpy as np
import scipy

from atprl.training.recovery import _atomic_write_bytes, RecoveryError

MAGIC = b"ATP-RR-EVAL\x00"
SCHEMA = 1
KIND = "conventional_evaluation_recovery"
ENV_FIELDS = (
    "current_step",
    "random_seed",
    "time_limit_truncated",
    "uav_position_xyz",
    "small_scale_coeffs_sat_to_gs",
    "small_scale_coeffs_bs_to_gs",
    "small_scale_sat_to_ris",
    "small_scale_ris_to_gs",
    "_nlos_sat_to_ris",
    "_nlos_ris_to_gs",
    "_macro_err_sg",
    "_macro_err_sr",
    "_macro_err_rg",
    "historical_avg_rates",
    "historical_sinrs_per_user",
    "last_power_actions_per_user",
    "ris_group_phase_vector",
    "ris_phase_response",
    "_last_ris_power_weights",
    "_last_boost_ratios_all_users",
    "_last_boost_ratios_direct_only",
    "_last_active_ris_noise_per_user",
    "sic_trials",
    "sic_failures",
)
REQUIRED_ENV_FIELDS = set(ENV_FIELDS) - {"_last_ris_power_weights", "time_limit_truncated"}
ROOT = Path(__file__).resolve().parents[1]


def source_revision():
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def compatibility(args):
    nonsemantic = {"output_dir", "resume_run_dir", "checkpoint_every_steps", "show_progress"}
    config = {key: value for key, value in vars(args).items() if key not in nonsemantic}
    config["dependencies"] = dict(numpy=np.__version__, scipy=scipy.__version__)
    if args.phase_solver == "sdr":
        from atprl.solvers.sdr_phase import backend_identity

        config["sdr_runtime"] = backend_identity()
    paths = [
        "envs/leo_sat.py",
        "envs/actuator.py",
        "solvers/optimization_context.py",
        "solvers/mrt_initialization.py",
        "solvers/wmmse.py",
        "solvers/wmmse_constrained.py",
        "solvers/sca_phase.py",
        "solvers/conventional_options.py",
        "baselines/round_robin.py",
        "baselines/rr_ao.py",
        "evaluation/conventional_config.py",
        "evaluation/conventional_metrics.py",
        "evaluation/conventional.py",
        "evaluation/conventional_recovery.py",
        "evaluation/utils.py",
    ]
    if args.phase_solver == "sdr":
        paths.append("solvers/sdr_phase.py")
    config["source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
    }
    return dict(
        fields=config,
        sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def pack_env_state(env):
    missing = REQUIRED_ENV_FIELDS - set(env.__dict__)
    if missing:
        raise RecoveryError(f"Environment recovery fields missing: {sorted(missing)}")
    fields = {name: copy.deepcopy(getattr(env, name)) for name in ENV_FIELDS if hasattr(env, name)}
    fixed = {
        name: hashlib.sha256(np.asarray(getattr(env, name)).tobytes()).hexdigest()
        for name in (
            "_fixed_shadowing_db",
            "_fixed_initial_sat_to_gs_coeffs",
            "_fixed_bs_to_gs_nlos_component",
        )
    }
    return dict(
        schema_version=SCHEMA,
        fields=fields,
        fixed_constructor_hashes=fixed,
        rng=copy.deepcopy(env._rng.bit_generator.state),
    )


def unpack_env_state(env, state):
    if state.get("schema_version") != SCHEMA:
        raise RecoveryError("Wrong environment snapshot schema")
    fields = state.get("fields", {})
    if REQUIRED_ENV_FIELDS - set(fields) or set(fields) - set(ENV_FIELDS):
        raise RecoveryError("Incomplete/unknown environment recovery fields")
    for name, expected in state["fixed_constructor_hashes"].items():
        actual = hashlib.sha256(np.asarray(getattr(env, name)).tobytes()).hexdigest()
        if expected != actual:
            raise RecoveryError("Constructor seed/physical configuration mismatch")
    for name, value in fields.items():
        if isinstance(value, np.ndarray):
            current = np.asarray(getattr(env, name))
            if current.shape != value.shape or not np.all(np.isfinite(value)):
                raise RecoveryError(f"Invalid environment field {name}")
        setattr(env, name, copy.deepcopy(value))
    env._update_uav_geometry_large_scale()
    env._apply_fixed_error_to_build_estimates()
    for i, group in enumerate(env.ris_groups):
        env.ris_phase_vector[group] = env.ris_group_phase_vector[i]
    env._update_twohop_group_csi_observation()
    for aid in env.agents:
        env._get_next_observation_for_rb(aid)
    env._rng.bit_generator.state = copy.deepcopy(state["rng"])


def save_checkpoint(path, payload):
    body = pickle.dumps(payload, protocol=5)
    envelope = MAGIC + struct.pack(">Q", len(body)) + hashlib.sha256(body).digest() + body
    _atomic_write_bytes(Path(path), lambda handle: handle.write(envelope))


def load_checkpoint(path):
    try:
        data = Path(path).read_bytes()
        header = len(MAGIC) + 8 + 32
        if len(data) < header or not data.startswith(MAGIC):
            raise RecoveryError("Wrong recovery kind/header")
        length = struct.unpack(">Q", data[len(MAGIC) : len(MAGIC) + 8])[0]
        checksum = data[len(MAGIC) + 8 : header]
        body = data[header:]
        if len(body) != length or hashlib.sha256(body).digest() != checksum:
            raise RecoveryError("Recovery checksum/length mismatch")
        payload = pickle.loads(body)
        if payload.get("checkpoint_kind") != KIND or payload.get("schema_version") != SCHEMA:
            raise RecoveryError("Wrong recovery schema/kind")
        return payload
    except RecoveryError:
        raise
    except Exception as exc:
        raise RecoveryError(f"Cannot read evaluation recovery: {exc}") from exc


def stream_watermark(path):
    path = Path(path)
    if not path.exists():
        return dict(bytes=0, sha256=hashlib.sha256(b"").hexdigest())
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())
        content = handle.read()
    return dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def repair_result_stream(path, watermark):
    path = Path(path)
    size = int(watermark["bytes"])
    if not path.exists():
        if size:
            raise RecoveryError("Committed result stream is missing")
        return []
    with path.open("r+b") as handle:
        committed = handle.read(size)
        if len(committed) != size or hashlib.sha256(committed).hexdigest() != watermark["sha256"]:
            raise RecoveryError("Committed result stream is corrupt or shorter than its watermark")
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())
    if size and not committed.endswith(b"\n"):
        raise RecoveryError("Committed stream contains a partial record")
    return [json.loads(line) for line in committed.decode("utf-8").splitlines()]


def save_experiment(experiment):
    if not experiment.safe_boundary:
        raise RecoveryError("Cannot checkpoint a partially executed decision")
    payload = dict(
        checkpoint_kind=KIND,
        schema_version=SCHEMA,
        source_revision=source_revision(),
        run_id=experiment.output.name,
        compatibility_signature=compatibility(experiment.args),
        scenario_id=experiment.scenario_id,
        seed_matrix=experiment.seeds,
        seed_index=experiment.seed_index,
        execution_state=experiment.execution_state,
        rr=experiment.policy.scheduler.state_dict(),
        environment=pack_env_state(experiment.env) if experiment.env is not None else None,
        metrics=experiment.metrics.state_dict() if experiment.metrics is not None else None,
        diagnostics=copy.deepcopy(experiment.diagnostics.__dict__)
        if experiment.diagnostics is not None
        else None,
        result_watermark=stream_watermark(experiment.stream),
        diagnostic_watermark=stream_watermark(experiment.diagnostic_stream),
        completed_seed_keys=[
            (row["eval_seed"], row["seed_index"], row["repeat_index"])
            for row in experiment.completed
        ],
        debug_saved=experiment.debug_saved,
    )
    save_checkpoint(experiment.output / "recovery/latest.ckpt", payload)


def restore_experiment(experiment):
    from atprl.evaluation.conventional import SicCountingEnv
    from atprl.evaluation.conventional_config import build_env_config
    from atprl.evaluation.conventional_metrics import (
        EpisodeMetricAccumulator,
        DiagnosticAccumulator,
    )

    payload = load_checkpoint(experiment.output / "recovery/latest.ckpt")
    expected = compatibility(experiment.args)
    if payload["compatibility_signature"] != expected:
        raise RecoveryError("Evaluation configuration/source/budget mismatch")
    if payload["seed_matrix"] != experiment.seeds or payload["run_id"] != experiment.output.name:
        raise RecoveryError("Evaluation identity mismatch")
    index = payload["seed_index"]
    if not isinstance(index, int) or not 0 <= index <= len(experiment.flat_seeds):
        raise RecoveryError("Invalid seed progress")
    experiment.completed = repair_result_stream(experiment.stream, payload["result_watermark"])
    experiment.completed_diagnostics = repair_result_stream(
        experiment.diagnostic_stream, payload["diagnostic_watermark"]
    )
    keys = [
        (row["eval_seed"], row["seed_index"], row["repeat_index"]) for row in experiment.completed
    ]
    if (
        keys != payload["completed_seed_keys"]
        or len(keys) != index
        or len(set(keys)) != len(keys)
        or len(experiment.completed_diagnostics) != index
    ):
        raise RecoveryError("Completed seed ledger mismatch")
    experiment.seed_index = index
    experiment.execution_state = payload["execution_state"]
    experiment.debug_saved = payload["debug_saved"]
    experiment.policy.scheduler.load_state_dict(payload["rr"])
    experiment.scenario_id = payload["scenario_id"]
    if payload["environment"] is not None:
        if index >= len(experiment.flat_seeds) or experiment.execution_state == "complete":
            raise RecoveryError("Invalid completed environment snapshot")
        seed = experiment.flat_seeds[index]
        experiment.env = SicCountingEnv(build_env_config(experiment.args, seed))
        experiment.env.reset(seed=seed)
        unpack_env_state(experiment.env, payload["environment"])
        experiment.metrics = EpisodeMetricAccumulator(experiment.args.num_ground_stations)
        experiment.metrics.load_state_dict(payload["metrics"])
        experiment.diagnostics = DiagnosticAccumulator(experiment.args.diagnostic_sample_limit)
        experiment.diagnostics.__dict__.update(copy.deepcopy(payload["diagnostics"]))
        if experiment.metrics.steps != experiment.env.current_step:
            raise RecoveryError("Decision/metric watermark mismatch")
        expected_decisions = experiment.env.current_step
        if not experiment.args.rr_reset_each_episode:
            expected_decisions += index * experiment.args.max_steps_per_episode
        expected_rr = (
            experiment.args.rr_initial_offset + expected_decisions
        ) % experiment.policy.scheduler.cycle_length
        if (
            experiment.policy.scheduler.decisions != expected_decisions
            or experiment.policy.scheduler.pointer != expected_rr
        ):
            raise RecoveryError("RR watermark mismatch")
    if experiment.execution_state not in ("in_progress", "complete") or (
        experiment.execution_state == "complete" and index != len(experiment.flat_seeds)
    ):
        raise RecoveryError("Invalid completion marker")
    experiment.safe_boundary = True
