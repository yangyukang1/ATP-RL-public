import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from atprl.evaluation.conventional import ConventionalExperiment
from atprl.evaluation.conventional_config import make_args
from atprl.evaluation.conventional_recovery import (
    load_checkpoint,
    save_checkpoint,
    pack_env_state,
    compatibility,
    repair_result_stream,
    stream_watermark,
)
from atprl.training.recovery import RecoveryError
from atprl.solvers import sdr_phase

ROOT = Path(__file__).resolve().parents[2]
BASE = "51063445c8cd9ca9252fe1640145e6a52a586bb5"


def options(**kwargs):
    return make_args(
        **(
            dict(
                phase_solver="sdr",
                num_eval_seeds=2,
                eval_seed_base=903001,
                max_steps_per_episode=3,
                ao_max_iters=1,
                wmmse_max_iters=3,
                sdr_max_iters=2,
                checkpoint_every_steps=1,
                show_progress=False,
            )
            | kwargs
        )
    )


def trace(run):
    records = []
    original = run.policy.decide

    def decide(env):
        result = original(env)
        records.append((result.users, result.power.copy(), result.phase.copy()))
        return result

    run.policy.decide = decide
    return records


@pytest.mark.parametrize(
    "event",
    [
        "episode_start",
        "before_decision",
        "after_optimization",
        "before_step",
        "after_step",
        "after_commit",
        "episode_boundary",
        "seed_boundary",
    ],
)
def test_F1_transaction_interruption_exact_metrics_state_and_ledger(event, tmp_path):
    expected = ConventionalExperiment(options(), tmp_path / "continuous")
    reference = trace(expected)
    pack = expected.run()
    stopped = []
    captured = {}

    def stop(name, run):
        if name == event and not stopped:
            stopped.append(1)
            if run.safe_boundary and run.env is not None:
                captured["state"] = pack_env_state(run.env)
            raise KeyboardInterrupt("injected transaction cut")

    interrupted = ConventionalExperiment(options(), tmp_path / "resumed", stop)
    with pytest.raises(KeyboardInterrupt):
        interrupted.run()
    path = tmp_path / "resumed"
    saved = load_checkpoint(path / "recovery/latest.ckpt")
    for stream in ("per_seed_metrics.jsonl", "solver_diagnostics.jsonl"):
        with (path / stream).open("a") as handle:
            handle.write("{partial")
    resumed = ConventionalExperiment(options(resume_run_dir=str(path)), path)
    if captured.get("state") is not None:
        actual = pack_env_state(resumed.env)
        assert pickle.dumps(actual) == pickle.dumps(captured["state"])
    records = trace(resumed)
    actual = resumed.run()
    assert pack["metrics"] == actual["metrics"]
    assert expected.policy.scheduler.state_dict() == resumed.policy.scheduler.state_dict()
    assert len((path / "per_seed_metrics.jsonl").read_text().splitlines()) == 2
    assert len((path / "solver_diagnostics.jsonl").read_text().splitlines()) == 2
    start = saved["seed_index"] * 3 + (
        saved["environment"]["fields"]["current_step"] if saved["environment"] else 0
    )
    assert len(records) == len(reference[start:])
    for a, b in zip(records, reference[start:]):
        assert a[0] == b[0]
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
    with patch.object(resumed.policy, "decide", side_effect=AssertionError("completed run")):
        assert resumed.run()["metrics"] == pack["metrics"]


def test_F2_internal_sdp_interrupt_replays_original_decision(tmp_path):
    reference = ConventionalExperiment(options(), tmp_path / "reference").run()
    run = ConventionalExperiment(options(), tmp_path / "cut")
    with patch.object(sdr_phase, "_solve_cone", side_effect=KeyboardInterrupt("inside SDP")):
        with pytest.raises(KeyboardInterrupt):
            run.run()
    assert run.env.current_step == 0 and run.policy.scheduler.decisions == 0
    actual = ConventionalExperiment(
        options(resume_run_dir=str(tmp_path / "cut")), tmp_path / "cut"
    ).run()
    assert reference["metrics"] == actual["metrics"]


@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.parametrize("cut", [5, 6, 7])
def test_F4_six_combination_boundary_and_reset(cut, reset, tmp_path):
    args = options(
        num_eval_seeds=1,
        max_steps_per_episode=8,
        rr_initial_offset=5,
        rr_reset_each_episode=reset,
        sdr_max_iters=1,
    )
    expected = ConventionalExperiment(args, tmp_path / "reference").run()

    def stop(event, run):
        if event == "after_commit" and run.env.current_step == cut:
            raise KeyboardInterrupt

    run = ConventionalExperiment(args, tmp_path / "cut", stop)
    with pytest.raises(KeyboardInterrupt):
        run.run()
    resumed = ConventionalExperiment(
        make_args(**(vars(args) | dict(resume_run_dir=str(tmp_path / "cut")))), tmp_path / "cut"
    )
    assert resumed.policy.scheduler.pointer == (5 + cut) % 6
    assert resumed.run()["metrics"] == expected["metrics"]


@pytest.mark.parametrize(
    "delta",
    [
        dict(sdr_rank_tol=1e-4),
        dict(sdr_scs_eps_abs=1e-7),
        dict(sdr_max_iters=3),
        dict(phase_solver="sca", baseline_name="rr_ao_wmmse_sca"),
        dict(phase_solver="mrt", baseline_name="rr_wmmse_mrt"),
        dict(fairness_lambda=1),
        dict(csi_error_variance=0.02),
    ],
)
def test_F6_incompatible_identity_rejected_before_stream_changes(delta, tmp_path):
    args = options(num_eval_seeds=1)
    ConventionalExperiment(args, tmp_path).run()
    streams = {
        name: (tmp_path / name).read_bytes()
        for name in ("per_seed_metrics.jsonl", "solver_diagnostics.jsonl")
    }
    with pytest.raises((RecoveryError, ValueError)):
        changed = make_args(**(vars(args) | delta | dict(resume_run_dir=str(tmp_path))))
        ConventionalExperiment(changed, tmp_path)
    for name, data in streams.items():
        assert (tmp_path / name).read_bytes() == data


def test_F5_F7_source_and_backend_versions_are_signed(tmp_path):
    args = options(num_eval_seeds=1)
    ConventionalExperiment(args, tmp_path).run()
    signature = compatibility(args)
    assert "solvers/sdr_phase.py" in signature["fields"]["source_sha256"]
    assert signature["fields"]["sdr_runtime"]["cvxpy"] == "1.6.7"
    resumed = options(num_eval_seeds=1, resume_run_dir=str(tmp_path))
    original = sdr_phase.backend_identity
    with patch.object(
        sdr_phase, "backend_identity", side_effect=lambda: original() | dict(scs="different")
    ):
        with pytest.raises(RecoveryError, match="mismatch"):
            ConventionalExperiment(resumed, tmp_path)
    payload = load_checkpoint(tmp_path / "recovery/latest.ckpt")
    payload["compatibility_signature"]["fields"]["source_sha256"]["solvers/sdr_phase.py"] = "bad"
    save_checkpoint(tmp_path / "recovery/latest.ckpt", payload)
    with pytest.raises(RecoveryError, match="mismatch"):
        ConventionalExperiment(resumed, tmp_path)


@pytest.mark.parametrize("stream", ["per_seed_metrics.jsonl", "solver_diagnostics.jsonl"])
def test_F8_committed_prefix_corruption_and_uncommitted_tail(stream, tmp_path):
    args = options(num_eval_seeds=1)
    ConventionalExperiment(args, tmp_path).run()
    path = tmp_path / stream
    valid = path.read_bytes()
    watermark = stream_watermark(path)
    with path.open("ab") as handle:
        handle.write(b"{unfinished")
    repair_result_stream(path, watermark)
    assert path.read_bytes() == valid
    path.write_bytes(valid[:-1])
    with pytest.raises(RecoveryError):
        ConventionalExperiment(options(num_eval_seeds=1, resume_run_dir=str(tmp_path)), tmp_path)


@pytest.mark.parametrize(
    "stage",
    ["after_optimization", "after_step", "atomic_before", "atomic_after", "seed_checkpoint"],
)
def test_F3_actual_child_kill_and_durable_recovery(stage, tmp_path):
    args = options(num_eval_seeds=2, max_steps_per_episode=2)
    expected = ConventionalExperiment(args, tmp_path / "continuous")
    pack = expected.run()
    path = tmp_path / "killed"
    marker = tmp_path / "marker.json"
    worker = tmp_path / "worker.py"
    worker.write_text(
        f"""
import json,threading
from pathlib import Path
from atprl.evaluation.conventional import ConventionalExperiment
from atprl.evaluation.conventional_config import make_args
from atprl.training import recovery
stage={stage!r};marker=Path({str(marker)!r});counts={{'sdr_sdp_solves':0}}
def pause(run):
 marker.write_text(json.dumps(dict(stage=stage,step=run.env.current_step if run.env else None,seed_index=run.seed_index,**counts)))
 threading.Event().wait()
def event(name,run):
 if name==stage:pause(run)
run=ConventionalExperiment(make_args(**{vars(args)!r}),{str(path)!r},event)
decide=run.policy.decide
def counted(env):
 result=decide(env);counts['sdr_sdp_solves']+=result.diagnostics.get('sdr_sdp_solves',0);return result
run.policy.decide=counted
replace=recovery.os.replace
def atomic(source,target):
 active=run.env is not None and run.env.current_step==1
 seed=run.seed_index==1 and run.env is None
 checkpoint=str(target).endswith('latest.ckpt')
 if checkpoint and ((active and stage=='atomic_before') or (seed and stage=='seed_checkpoint')):pause(run)
 result=replace(source,target)
 if checkpoint and active and stage=='atomic_after':pause(run)
 return result
recovery.os.replace=atomic
run.run()
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-B", str(worker)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        limit = time.monotonic() + 60
        while not marker.exists() and process.poll() is None and time.monotonic() < limit:
            time.sleep(0.02)
        assert marker.exists(), (
            process.stderr.read().decode()
            if process.poll() is not None
            else "child marker timed out"
        )
        observation = json.loads(marker.read_text())
        assert observation["sdr_sdp_solves"] > 0
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stderr.close()
    payload = load_checkpoint(path / "recovery/latest.ckpt")
    assert payload["environment"]["fields"]["current_step"] == (
        1 if stage == "atomic_after" else (2 if stage == "seed_checkpoint" else 0)
    )
    actual = ConventionalExperiment(
        options(num_eval_seeds=2, max_steps_per_episode=2, resume_run_dir=str(path)), path
    )
    assert actual.run()["metrics"] == pack["metrics"]
    assert actual.policy.scheduler.state_dict() == expected.policy.scheduler.state_dict()
    rows = [json.loads(line) for line in (path / "per_seed_metrics.jsonl").read_text().splitlines()]
    assert (
        len(rows) == 2
        and len({(r["eval_seed"], r["seed_index"], r["repeat_index"]) for r in rows}) == 2
    )
