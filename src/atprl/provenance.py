from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy
import torch
import yaml

from atprl.config.loader import stable_config_hash


def project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else Path.cwd()


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root()), *args],
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(status),
        "status": status or "",
    }


def runtime_state() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
    }


def create_run_directory(config: Mapping[str, Any], *, suffix: str | None = None) -> Path:
    root = Path(config.get("artifacts_root", "artifacts"))
    if not root.is_absolute():
        root = project_root() / root
    stamp = time.strftime("%Y%m%d_%H%M%S")
    components = [str(config.get("name", "experiment")), str(config.get("method", "method")), stamp]
    if suffix:
        components.append(suffix)
    run_id = "_".join(item.replace(" ", "-") for item in components)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(run_dir: Path, config: Mapping[str, Any], command: list[str]) -> None:
    clean_config = dict(config)
    clean_config.pop("_meta", None)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(clean_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "config_sha256": stable_config_hash(clean_config),
        "command": command,
        "git": git_state(),
        "runtime": runtime_state(),
        "training_seed": clean_config.get("args", {}).get("seed"),
        "evaluation_seed_base": clean_config.get("args", {}).get("eval_seed_base"),
        "created_unix": time.time(),
    }
    if config.get("entrypoint") == "conventional":
        manifest.pop("training_seed", None)
        manifest.update(method_type="non_learning", has_training_trajectory=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def finalize_run_metadata(run_dir: Path, exit_code: int) -> None:
    checkpoints = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".ckpt", ".pth", ".pt"}:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            checkpoints.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    payload = {"schema_version": 1, "exit_code": int(exit_code), "checkpoints": checkpoints}
    (run_dir / "checkpoint_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
