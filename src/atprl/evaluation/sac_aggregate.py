"""Aggregate training seeds equally, keeping environment repeats nested within each run."""

import csv, json
from pathlib import Path
import numpy as np
from atprl.training.sac_recovery import write_json


def aggregate_campaign(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    missing = []
    for seed in manifest["config"]["seeds"]:
        p = root / f"seed{seed}/evaluation/strict_best/metrics_pack.json"
        if not p.exists():
            missing.append(seed)
            continue
        result = json.loads(p.read_text(encoding="utf-8"))
        if not result["complete"]:
            missing.append(seed)
            continue
        if result["artifact"] != "strict_best" or result["training_seed"] != seed:
            raise ValueError("Aggregation identity mismatch")
        rows.append(
            dict(training_seed=seed, **{k: v["mean"] for k, v in result["metrics"].items()})
        )
    summary = dict(complete=not missing, missing_training_seeds=missing, rows=rows, metrics={})
    if rows:
        for k in rows[0]:
            if k != "training_seed":
                summary["metrics"][k] = dict(
                    mean=float(np.mean([r[k] for r in rows])),
                    std=float(np.std([r[k] for r in rows])),
                    training_replicates=len(rows),
                )
        with (root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write_json(root / "summary.json", summary)
    return summary
