from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional


SIGNATURE_VERSION = 1
SIGNATURE_FILENAME = "kernel_signature.json"

PIPELINE_FIELDS = (
    "n",
    "k",
    "topology",
    "fairness_lambda",
    "csi",
    "ris_dbm",
    "dqn_scheduler_impl",
)
CHECKPOINT_FIELDS = ("n", "k", "topology", "dqn_scheduler_impl")
KERNEL_DIMENSION_FIELDS = (
    "dqn_state_dim",
    "scheduler_output_dim",
    "replay_action_dim",
    "power_state_dim",
    "power_action_dim",
)


def build_kernel_signature(
    *,
    n: int,
    k: int,
    topology: Optional[str],
    dqn_scheduler_impl: str,
    fairness_lambda: Any = None,
    csi: Any = None,
    ris_dbm: Any = None,
    dimensions: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    dims = {
        str(name): (None if value is None else int(value))
        for name, value in dict(dimensions or {}).items()
    }
    return {
        "signature_version": SIGNATURE_VERSION,
        "n": int(n),
        "k": int(k),
        "topology": None if topology in (None, "") else str(topology),
        "fairness_lambda": fairness_lambda,
        "csi": csi,
        "ris_dbm": ris_dbm,
        "dqn_scheduler_impl": str(dqn_scheduler_impl).strip().lower(),
        "dimensions": dims,
    }


def kernel_signature_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    args = config["args"]
    if config["entrypoint"] == "pbt":
        return build_kernel_signature(
            n=args.get("p2_num_ground_stations"),
            k=args.get("p2_num_gs_per_rb"),
            topology=args.get("p2_topology_profile"),
            fairness_lambda=args.get("p2_fairness_lambda"),
            csi=args.get("p2_csi_error_variance"),
            ris_dbm=args.get("p2_ris_reflection_power_limit_dBm"),
            dqn_scheduler_impl=args.get("p2_dqn_scheduler_impl", "enumerative"),
        )
    return build_kernel_signature(
        n=args.get("num_ground_stations"),
        k=args.get("num_gs_per_rb"),
        topology=args.get("topology_profile"),
        fairness_lambda=args.get("fairness_lambda"),
        csi=args.get("csi_error_variance"),
        ris_dbm=args.get("ris_reflection_power_limit_dBm"),
        dqn_scheduler_impl=args.get("dqn_scheduler_impl", "enumerative"),
    )


def kernel_signature_from_runtime(
    *,
    args: Any,
    env: Any,
    scheduler_build: Any,
    dqn_state_dim: int,
    power_state_dim: int,
    power_action_dim: int,
    phase_state_dim: int,
) -> dict[str, Any]:
    return build_kernel_signature(
        n=int(env.num_ground_stations),
        k=int(env.num_gs_per_rb),
        topology=getattr(args, "topology_profile", env.config.get("topology_profile")),
        fairness_lambda=getattr(args, "fairness_lambda", None),
        csi=getattr(args, "csi_error_variance", None),
        ris_dbm=getattr(args, "ris_reflection_power_limit_dBm", None),
        dqn_scheduler_impl=scheduler_build.implementation,
        dimensions={
            "dqn_state_dim": int(dqn_state_dim),
            "scheduler_output_dim": int(scheduler_build.network_output_dim),
            "replay_action_dim": int(scheduler_build.replay_action_dim),
            "power_state_dim": int(power_state_dim),
            "power_action_dim": int(power_action_dim),
            "phase_state_dim": int(phase_state_dim),
        },
    )


def pipeline_compatibility_key(signature: Mapping[str, Any]) -> dict[str, Any]:
    return {name: signature.get(name) for name in PIPELINE_FIELDS}


def checkpoint_compatibility_key(signature: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{name: signature.get(name) for name in CHECKPOINT_FIELDS},
        "dimensions": dict(signature.get("dimensions") or {}),
    }


def validate_checkpoint_signature(
    checkpoint_signature: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
    *,
    context: str,
    dimension_fields: Optional[tuple[str, ...]] = None,
) -> None:
    mismatches: dict[str, tuple[Any, Any]] = {}
    for name in CHECKPOINT_FIELDS:
        checkpoint_value = checkpoint_signature.get(name)
        runtime_value = runtime_signature.get(name)
        if (
            checkpoint_value is not None
            and runtime_value is not None
            and checkpoint_value != runtime_value
        ):
            mismatches[name] = (checkpoint_value, runtime_value)

    checkpoint_dims = dict(checkpoint_signature.get("dimensions") or {})
    runtime_dims = dict(runtime_signature.get("dimensions") or {})
    names = checkpoint_dims.keys() if dimension_fields is None else dimension_fields
    for name in names:
        checkpoint_value = checkpoint_dims.get(name)
        runtime_value = runtime_dims.get(name)
        if (
            checkpoint_value is not None
            and runtime_value is not None
            and checkpoint_value != runtime_value
        ):
            mismatches[f"dimensions.{name}"] = (checkpoint_value, runtime_value)

    if mismatches:
        details = ", ".join(
            f"{name}: checkpoint={left!r}, runtime={right!r}"
            for name, (left, right) in mismatches.items()
        )
        raise ValueError(f"{context} kernel signature mismatch before weight loading: {details}")


def validate_checkpoint_metadata(
    metadata: Optional[Mapping[str, Any]],
    runtime_signature: Mapping[str, Any],
    *,
    context: str,
    dimension_fields: Optional[tuple[str, ...]] = None,
    warn_legacy: bool = True,
) -> bool:
    meta = dict(metadata or {})
    if bool(meta.get("_legacy_warning_emitted", False)):
        warn_legacy = False
    signature = meta.get("kernel_signature")
    if isinstance(signature, Mapping):
        validate_checkpoint_signature(
            signature,
            runtime_signature,
            context=context,
            dimension_fields=dimension_fields,
        )
        return True

    legacy_impl = meta.get("dqn_scheduler_impl")
    runtime_impl = runtime_signature.get("dqn_scheduler_impl")
    if (
        legacy_impl is not None
        and str(legacy_impl).strip().lower() != str(runtime_impl).strip().lower()
    ):
        raise ValueError(
            f"{context} scheduler architecture mismatch before weight loading: "
            f"checkpoint={legacy_impl}, runtime={runtime_impl}"
        )
    if warn_legacy:
        warnings.warn(
            f"{context} is a legacy checkpoint without a complete kernel signature; "
            "its original N/K cannot be verified. Falling back to strict state-dict loading.",
            RuntimeWarning,
            stacklevel=2,
        )
    return False


def write_kernel_signature(model_dir: str | Path, signature: Mapping[str, Any]) -> Path:
    path = Path(model_dir) / SIGNATURE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(signature), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_kernel_signature(model_dir: str | Path) -> Optional[dict[str, Any]]:
    path = Path(model_dir) / SIGNATURE_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Kernel signature sidecar must contain a JSON object: {path}")
    return payload
