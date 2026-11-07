from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from atprl.config.loader import ConfigError, cli_args, load_config, set_dotted, validate_config
from atprl.config.kernel_signature import kernel_signature_from_config, pipeline_compatibility_key
from atprl.provenance import (
    create_run_directory,
    finalize_run_metadata,
    project_root,
    write_run_metadata,
)


MODULES = {
    "sac_stp": "atprl.training.sac_stp",
    "joint": "atprl.training.joint",
    "warmup": "atprl.training.warmup",
    "stage1": "atprl.training.stage1",
    "stage2": "atprl.training.stage2",
    "pbt": "atprl.training.pbt.controller",
    "periodic": "atprl.training.periodic",
    "evaluate": "atprl.evaluation.runner",
    "conventional": "atprl.evaluation.conventional",
    "no_ris_evaluate": "atprl.evaluation.no_ris",
}
METHODS = (
    "atp_rl",
    "stp_rl",
    "rnd_stp",
    "wmmse_stp",
    "periodic",
    "stp_sac",
    "joint_rl",
    "no_ris",
    "conventional",
)


def _build_parser():
    parser = argparse.ArgumentParser(prog="atprl", description="ATP-RL and eight benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], dest="overrides")
        p.add_argument("--for-run" if name == "doctor" else "--dry-run", action="store_true")
    p = sub.add_parser("train", help="Run a complete method from scratch")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--artifacts-root", default="artifacts")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("pipeline", help="Run a configured warmup and dependent training job")
    p.add_argument("--warmup-config", required=True)
    p.add_argument("--train-config", required=True)
    p.add_argument("--set-warmup", action="append", default=[], dest="warmup_overrides")
    p.add_argument("--set-train", action="append", default=[], dest="train_overrides")
    p.add_argument("--dry-run", action="store_true")
    return parser


def _doctor(config_path: str, overrides: list[str], for_run: bool) -> int:
    config = load_config(config_path, overrides)
    validate_config(config, for_run=for_run)
    summary = {
        "source": config["_meta"]["source"],
        "config_sha256": config["_meta"]["sha256"],
        "name": config.get("name"),
        "method": config.get("method"),
        "entrypoint": config.get("entrypoint"),
        "cli_args": cli_args(config),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _execute_job(
    config: dict[str, Any], *, dry_run: bool, suffix: str | None = None
) -> tuple[int, Path | None]:
    validate_config(config, for_run=not dry_run)
    module = MODULES[str(config["entrypoint"])]
    command = [sys.executable, "-m", module, *cli_args(config)]
    if dry_run:
        print(json.dumps({"config": config.get("name"), "command": command}, ensure_ascii=False))
        return 0, None

    run_dir = create_run_directory(config, suffix=suffix)
    write_run_metadata(run_dir, config, command)
    env = os.environ.copy()
    from atprl.runtime import THREAD_ENV

    env.update({name: "1" for name in THREAD_ENV})
    src = str(project_root() / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[ATP-RL] run directory: {run_dir}")
    print(f"[ATP-RL] command: {' '.join(command)}")
    result = subprocess.run(command, cwd=run_dir, env=env, check=False)
    (run_dir / "exit_code.txt").write_text(str(result.returncode), encoding="utf-8")
    finalize_run_metadata(run_dir, int(result.returncode))
    return int(result.returncode), run_dir


def _execute(config: dict[str, Any], *, dry_run: bool, suffix: str | None = None) -> int:
    return _execute_job(config, dry_run=dry_run, suffix=suffix)[0]


def _run(config_path: str, overrides: list[str], dry_run: bool) -> int:
    return _execute(load_config(config_path, overrides), dry_run=dry_run)


def _kernel_signature(config: dict[str, Any]) -> dict[str, Any]:
    return pipeline_compatibility_key(kernel_signature_from_config(config))


def _pipeline(
    warmup_path: str,
    train_path: str,
    warmup_overrides: list[str],
    train_overrides: list[str],
    dry_run: bool,
) -> int:
    warmup = load_config(warmup_path, warmup_overrides)
    train = load_config(train_path, train_overrides)
    if warmup["entrypoint"] != "warmup":
        raise ConfigError("pipeline --warmup-config must target entrypoint: warmup")
    if train["entrypoint"] not in {"pbt", "stage2"}:
        raise ConfigError("pipeline --train-config must target entrypoint: pbt or stage2")
    if _kernel_signature(warmup) != _kernel_signature(train):
        raise ConfigError(
            f"Warmup/train kernel mismatch: warmup={_kernel_signature(warmup)}, train={_kernel_signature(train)}"
        )

    warmup_code, warmup_run = _execute_job(warmup, dry_run=dry_run, suffix="warmup")
    if warmup_code != 0 or dry_run:
        if dry_run:
            dependency = "<warmup-run>/models"
            if train["entrypoint"] == "pbt":
                train["args"]["warmup_model_dir"] = dependency
            else:
                train["args"]["phase1_model_dir"] = dependency
            _execute_job(train, dry_run=True, suffix="train")
            return 0
        return warmup_code

    model_files = sorted(warmup_run.rglob("q_network_phase1.pth"))
    if len(model_files) != 1:
        raise ConfigError(
            f"Expected one warmup q_network_phase1.pth under {warmup_run}, found {len(model_files)}"
        )
    model_dir = str(model_files[0].parent.resolve())
    if train["entrypoint"] == "pbt":
        train["args"]["warmup_model_dir"] = model_dir
    else:
        train["args"]["phase1_model_dir"] = model_dir
    return _execute(train, dry_run=False, suffix="train")


def _train(args):
    root = project_root() / "configs"
    if not root.is_dir():
        root = Path(sys.prefix) / "share/atprl/configs"
    config = root / "methods" / f"{args.method}.yaml"
    common = [f"artifacts_root={args.artifacts_root}", f"args.seed={args.seed}"]
    if args.method in ("atp_rl", "stp_rl", "rnd_stp", "wmmse_stp"):
        warmup = root / (
            "base/warmup.yaml" if args.method == "atp_rl" else f"warmup/{args.method}.yaml"
        )
        train_device = (
            [f"args.p1_device={args.device}", f"args.p2_device={args.device}"]
            if args.method == "atp_rl"
            else [f"args.device={args.device}"]
        )
        return _pipeline(
            str(warmup),
            str(config),
            common + [f"args.device={args.device}"],
            common + train_device,
            args.dry_run,
        )
    overrides = list(common)
    if args.method in ("stp_sac", "joint_rl"):
        overrides.append(f"args.seeds=[{args.seed}]")
    if args.method not in ("stp_sac", "conventional"):
        overrides.append(f"args.device={args.device}")
    if args.method == "conventional":
        overrides = [f"artifacts_root={args.artifacts_root}"]
    code, run_dir = _execute_job(load_config(config, overrides), dry_run=args.dry_run)
    if code == 0 and run_dir is not None and args.method == "no_ris":
        checkpoints = sorted(run_dir.rglob("recovery/latest.ckpt"))
        if len(checkpoints) != 1:
            raise ConfigError(f"Expected one No-RIS checkpoint, found {len(checkpoints)}")
        from atprl.evaluation.no_ris import evaluate

        evaluate(checkpoints[0], run_dir / "final_evaluation")
    return code


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            code = _doctor(args.config, args.overrides, args.for_run)
        elif args.command == "run":
            code = _run(args.config, args.overrides, args.dry_run)
        elif args.command == "train":
            code = _train(args)
        else:
            code = _pipeline(
                args.warmup_config,
                args.train_config,
                args.warmup_overrides,
                args.train_overrides,
                args.dry_run,
            )
    except (ConfigError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code)
