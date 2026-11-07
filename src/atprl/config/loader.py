from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml

from atprl.config.budgets import POST_WARMUP_EPISODES


class ConfigError(ValueError):
    pass


ENTRYPOINTS = {
    "warmup",
    "stage1",
    "stage2",
    "pbt",
    "periodic",
    "evaluate",
    "joint",
    "sac_stp",
    "conventional",
    "no_ris_evaluate",
}
METHODS = {
    "atp_rl",
    "stp_rl",
    "rnd_stp",
    "wmmse_stp",
    "periodic",
    "no_ris",
    "joint_dqn_ppo_ppo",
    "sac_stp",
    "rr_ao_wmmse_sdr",
}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"Configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")

    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ConfigError(f"'extends' must be a string or list: {path}")

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / str(parent)).resolve()
        merged = _deep_merge(merged, _load_recursive(parent_path, (*stack, path)))
    return _deep_merge(merged, data)


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ConfigError(f"Override must use dotted.path=value: {text}")
    dotted, raw = text.split("=", 1)
    dotted = dotted.strip()
    if not dotted:
        raise ConfigError(f"Override path is empty: {text}")
    return dotted, yaml.safe_load(raw)


def set_dotted(config: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, MutableMapping):
            raise ConfigError(f"Cannot set {dotted}: {part} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def stable_config_hash(config: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(config))
    payload.pop("_meta", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    source = Path(path).resolve()
    config = _load_recursive(source, ())
    for item in overrides:
        dotted, value = _parse_override(item)
        set_dotted(config, dotted, value)
    validate_config(config, for_run=False)
    config["_meta"] = {
        "source": str(source),
        "sha256": stable_config_hash(config),
    }
    return config


def _topology_args(config: Mapping[str, Any]) -> tuple[int | None, int | None]:
    args = config.get("args", {})
    if config.get("entrypoint") == "pbt":
        return args.get("p2_num_ground_stations"), args.get("p2_num_gs_per_rb")
    return args.get("num_ground_stations"), args.get("num_gs_per_rb")


def validate_config(config: Mapping[str, Any], *, for_run: bool) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ConfigError("schema_version must be 1")
    if config.get("kind", "run") != "run":
        raise ConfigError("Run configurations must set kind: run")
    entrypoint = str(config.get("entrypoint", ""))
    if entrypoint not in ENTRYPOINTS:
        raise ConfigError(
            f"Unsupported entrypoint {entrypoint!r}; choose from {sorted(ENTRYPOINTS)}"
        )
    method = str(config.get("method", ""))
    if method not in METHODS:
        raise ConfigError(f"Unsupported method {method!r}; choose from {sorted(METHODS)}")
    args = config.get("args")
    if not isinstance(args, Mapping):
        raise ConfigError("args must be a mapping")
    if entrypoint == "conventional":
        from atprl.evaluation.conventional_config import make_args

        try:
            conventional = make_args(**dict(args))
            if conventional.baseline_name != method:
                raise ValueError("Configuration method differs from baseline_name")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    if entrypoint == "sac_stp":
        if method != "sac_stp":
            raise ConfigError("SAC entrypoint requires SAC method")
        from atprl.config.sac import SACRunConfig

        try:
            s = SACRunConfig(**args)
            if for_run and s.stage == "stage2" and not s.kernel_checkpoint and not s.resume:
                raise ValueError("Stage-II requires SAC kernel")
            if for_run and s.stage == "evaluate" and not s.checkpoint:
                raise ValueError("Evaluation requires SAC checkpoint")
            if for_run and s.stage == "fixed_power" and not s.dqn_checkpoint:
                raise ValueError("Fixed POWER requires explicit original DQN-A checkpoint")
        except (ValueError, TypeError) as exc:
            raise ConfigError(str(exc)) from exc
        return
    if entrypoint == "joint":
        if method != "joint_dqn_ppo_ppo":
            raise ConfigError("Joint entrypoint requires method: joint_dqn_ppo_ppo")
        from atprl.training.joint_runtime import DEFAULTS, make_args, validate_joint_args

        try:
            validate_joint_args(
                make_args(**{key: value for key, value in args.items() if key in DEFAULTS})
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    n_users, k_users = _topology_args(config)
    if n_users is not None or k_users is not None:
        if n_users is None or k_users is None:
            raise ConfigError("Both N and K must be provided")
        n_users, k_users = int(n_users), int(k_users)
        if n_users < 1:
            raise ConfigError("Topology must satisfy N >= 1")
        if not 1 <= k_users <= n_users:
            raise ConfigError("Topology must satisfy 1 <= K <= N")

    if entrypoint == "pbt":
        p1_n = args.get("p1_num_ground_stations")
        p1_k = args.get("p1_num_gs_per_rb")
        if p1_n is not None and int(p1_n) != int(n_users):
            raise ConfigError("P1 and P2 must use the same N")
        if p1_k is not None and int(p1_k) != int(k_users):
            raise ConfigError("P1 and P2 must use the same K")
        p1_ris = args.get("p1_ris_reflection_power_limit_dBm")
        p2_ris = args.get("p2_ris_reflection_power_limit_dBm")
        if p1_ris is not None and p2_ris is not None and float(p1_ris) != float(p2_ris):
            raise ConfigError("P1 and P2 must use the same RIS reflection budget")
        p1_impl = str(args.get("p1_dqn_scheduler_impl", "enumerative")).strip().lower()
        p2_impl = str(args.get("p2_dqn_scheduler_impl", "enumerative")).strip().lower()
        if p1_impl != p2_impl:
            raise ConfigError("P1 and P2 must use the same dqn_scheduler_impl")

    if entrypoint == "pbt":
        scheduler_impls = (
            args.get("p1_dqn_scheduler_impl", "enumerative"),
            args.get("p2_dqn_scheduler_impl", "enumerative"),
        )
    else:
        scheduler_impls = (args.get("dqn_scheduler_impl", "enumerative"),)
    invalid_impls = [
        value for value in scheduler_impls if str(value).strip().lower() not in {"enumerative"}
    ]
    if invalid_impls:
        raise ConfigError(f"Unsupported dqn_scheduler_impl value(s): {invalid_impls}")

    from atprl.envs.topologies import resolve_topology

    if n_users is not None:
        if entrypoint == "pbt":
            p1_profile = args.get("p1_topology_profile") or args.get("p2_topology_profile")
            p2_profile = args.get("p2_topology_profile")
            resolve_topology(p1_profile, int(n_users))
            resolve_topology(p2_profile, int(n_users))
        else:
            resolve_topology(args.get("topology_profile"), int(n_users))

    if entrypoint == "pbt":
        if args.get("p2_phase_state_mode") == "compact_selected":
            if bool(args.get("p2_phase_compact_include_mrt_features")) and not bool(
                args.get("p2_include_mrt_features")
            ):
                raise ConfigError("Compact MRT phase state requires p2_include_mrt_features=true")
    elif entrypoint in {"stage2", "evaluate"}:
        if args.get("phase_state_mode") == "compact_selected":
            if bool(args.get("phase_compact_include_mrt_features")) and not bool(
                args.get("include_mrt_features")
            ):
                raise ConfigError("Compact MRT phase state requires include_mrt_features=true")

    if config.get("profile") == "paper" and entrypoint == "pbt":
        expected = {
            "adapt_episodes": 2464,
            "ready_every_episodes": 224,
            "cooldown_episodes": 448,
            "p12_flip_prob": 0.33,
            "p21_flip_prob": 0.25,
            "max_pbt_episodes": POST_WARMUP_EPISODES,
            "p2_num_ground_stations": 4,
            "p2_num_gs_per_rb": 2,
            "p2_ris_reflection_power_limit_dBm": -5.0,
            "p2_fairness_lambda": 3.0,
        }
        mismatches = {
            key: (args.get(key), value) for key, value in expected.items() if args.get(key) != value
        }
        if mismatches:
            raise ConfigError(f"Paper profile defaults were changed: {mismatches}")

    offsets = args.get("sat_user_gain_db_offsets")
    if offsets not in (None, "") and n_users is not None:
        values = offsets if isinstance(offsets, list) else str(offsets).split(",")
        if len(values) not in (1, int(n_users)):
            raise ConfigError("sat_user_gain_db_offsets must contain one value or N values")

    if for_run:
        required: list[str] = []
        if entrypoint == "no_ris_evaluate":
            required = ["checkpoint"]
        if entrypoint == "pbt":
            if args.get("resume_run_dir") in (None, ""):
                required = ["warmup_model_dir"]
        elif entrypoint == "stage2":
            required = ["phase1_model_dir"]
        elif entrypoint == "evaluate":
            if bool(args.get("run_batch")):
                required = [
                    "atp_pbt_run_dir",
                    "stp_phase2_model_dir",
                    "rnd_stp_phase2_model_dir",
                    "wmmse_phase2_model_dir",
                    "periodic_phase1_model_dir_batch",
                ]
                baselines = args.get("baseline_list", [])
                if isinstance(baselines, str):
                    baselines = [item.strip() for item in baselines.split(",")]
                if "joint_dqn_ppo_ppo" in baselines or "sac_stp" in baselines:
                    sources = {
                        "atp_rl": "atp_pbt_run_dir",
                        "stp_rl": "stp_phase2_model_dir",
                        "rnd_stp": "rnd_stp_phase2_model_dir",
                        "wmmse_stp": "wmmse_phase2_model_dir",
                        "periodic": "periodic_phase1_model_dir_batch",
                        "joint_dqn_ppo_ppo": "joint_checkpoint",
                        "sac_stp": "sac_checkpoint",
                    }
                    required = [sources[method] for method in baselines if method in sources]
            elif args.get("eval_source") == "pbt_best":
                required = ["pbt_run_dir"]
            elif args.get("eval_source") == "phase2_model_dir":
                required = ["phase2_model_dir"]
            elif args.get("eval_source") == "periodic_phase1_model_dir":
                required = ["periodic_phase1_model_dir"]
            elif args.get("eval_source") == "sac_checkpoint":
                required = ["sac_checkpoint"]
            elif args.get("eval_source") == "joint_checkpoint":
                required = ["joint_checkpoint"]
        for key in required:
            if args.get(key) in (None, ""):
                raise ConfigError(f"args.{key} must be provided before running {entrypoint}")


def cli_args(config: Mapping[str, Any]) -> list[str]:
    if config.get("entrypoint") == "sac_stp":
        return ["--settings-json", json.dumps(config["args"])]
    result: list[str] = []
    boolean_optional = {str(item) for item in config.get("boolean_optional_args", [])}
    boolean_values = {str(item) for item in config.get("boolean_value_args", [])}
    if boolean_optional & boolean_values:
        raise ConfigError("Boolean arguments cannot use both optional flags and explicit values")
    for key, value in config["args"].items():
        if value is None:
            continue
        flag = "--" + str(key)
        if isinstance(value, bool) and str(key) in boolean_values:
            result.extend((flag, str(value).lower()))
            continue
        if value is False:
            if str(key) in boolean_optional:
                result.append("--no-" + str(key))
            continue
        if value is True:
            result.append(flag)
        elif isinstance(value, list):
            result.extend((flag, ",".join(str(item) for item in value)))
        else:
            result.extend((flag, str(value)))
    raw_args = config.get("raw_args", [])
    if not isinstance(raw_args, list):
        raise ConfigError("raw_args must be a list")
    return result + [str(item) for item in raw_args]
