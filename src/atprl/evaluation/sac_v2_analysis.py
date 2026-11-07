from pathlib import Path
import json
import numpy as np
from atprl.training.sac_v2_io import read_json, verify, prefix_sha


class RawHistoryUnavailable(ValueError):
    pass


def manifest(root):
    root = Path(root)
    p = read_json(root / "CURRENT.json")
    return read_json(verify(root, p["current"]))


def training_rows(root, stage):
    root = Path(root)
    m = manifest(root)
    ref = m["journals"][stage]
    if prefix_sha(root / ref["path"], ref["bytes"]) != ref["sha256"]:
        raise ValueError("Scientific journal checksum mismatch")
    with (root / ref["path"]).open("rb") as f:
        return [json.loads(x) for x in f.read(ref["bytes"]).splitlines()]


def evaluations(root, stage):
    root = Path(root)
    m = manifest(root)
    return sorted(
        [read_json(verify(root, r)) for r in m["evaluations"].get(stage, [])],
        key=lambda x: x["episodes_done"],
    )


def raw_diagnostics(root, stage, first_episode, last_episode):
    root = Path(root)
    m = manifest(root)
    ds = m["diagnostics_by_stage"][stage]
    refs = ds["raw_refs"]
    sd = "stage1" if stage == "warmup" else "stage2"
    limit = m["journals"].get(sd, {}).get("rows", 0)
    offset = 0 if stage == "warmup" else m["extra"]["stage1_complete"]["episode"]
    if first_episode < 1 or last_episode > limit or first_episode > last_episode:
        raise ValueError("Invalid committed episode range")
    if first_episode + offset <= m["cumulative_episode"] - 640:
        raise RawHistoryUnavailable(
            "Exact old update history is outside the retained raw window; use archived v1 raw data or fixed summaries, not reconstructed samples."
        )
    values = []
    episodes = []
    for ref in refs:
        with np.load(verify(root, ref), allow_pickle=False) as a:
            mask = (a["episode"] >= first_episode) & (a["episode"] <= last_episode)
            values.append(a["values"][mask])
            episodes.append(a["episode"][mask])
    return {
        "columns": ds["columns"],
        "episode": np.concatenate(episodes) if episodes else np.empty(0, np.int64),
        "values": np.concatenate(values)
        if values
        else np.empty((0, len(ds["columns"])), np.float64),
    }


def summary_blocks(root, stage):
    root = Path(root)
    m = manifest(root)
    for ref in m["diagnostics_by_stage"][stage]["summary_refs"]:
        with np.load(verify(root, ref), allow_pickle=False) as a:
            result = {k: a[k].copy() for k in a.files}
            result["columns"] = np.asarray(m["diagnostics_by_stage"][stage]["columns"])
            yield result
