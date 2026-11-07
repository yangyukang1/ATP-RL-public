import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from atprl.baselines import rr_ao
from atprl.solvers import sdr_phase
from atprl.solvers.conventional_options import SolverOptions
from atprl.solvers.optimization_context import candidate_terms, snapshot_from_env
from atprl.evaluation.conventional_config import make_args, build_env_config, solver_options
from atprl.envs.actuator import physical_actuator_state
from atprl.envs.leo_sat import LeoSatEnv
from tests.unit.test_conventional_solvers import synthetic
from tests.unit.test_conventional_context import environment

ROOT = Path(__file__).resolve().parents[2]
BASE = "51063445c8cd9ca9252fe1640145e6a52a586bb5"


def test_A7_true_sdr_alternation_and_terminal():
    snapshot = synthetic()
    options = SolverOptions(ao_max_iters=2, wmmse_max_iters=5, sdr_max_iters=3)
    calls = []
    power = rr_ao.solve_rsma_power
    phase = sdr_phase.solve_phase_sdr

    def solve_power(*args, **kwargs):
        calls.append("power")
        return power(*args, **kwargs)

    def solve_phase(*args, **kwargs):
        calls.append("sdr")
        return phase(*args, **kwargs)

    with (
        patch.object(rr_ao, "solve_rsma_power", solve_power),
        patch.object(sdr_phase, "solve_phase_sdr", solve_phase),
    ):
        result = rr_ao.optimize_decision(snapshot, options, "sdr")
    assert calls[0:2] == ["power", "sdr"] and calls[-1] == "power"
    assert len(calls) == 2 * result.diagnostics["ao_iterations"] + 1
    assert (
        result.diagnostics["terminal_power_solves"] == 1 and result.diagnostics["sca_blocks"] == 0
    )
    assert result.diagnostics["sdr_blocks"] > 0 and result.diagnostics["sdr_sdp_solves"] > 0
    assert candidate_terms(snapshot, result.power, result.phase)["is_feasible"]
    json.dumps(result.diagnostics, allow_nan=False)


def test_A8_shared_phase1_and_no_silent_normal_sca():
    snapshot = synthetic(np.array([1.2, 1.2]), np.full((2, 4), 0.3))
    options = SolverOptions(ao_max_iters=1, wmmse_max_iters=3, phase1_max_iters=30, sdr_max_iters=2)
    original = rr_ao.solve_phase_sca
    flags = []

    def restoration(*args, **kwargs):
        flags.append(kwargs.get("phase1", False))
        assert flags[-1] is True
        return original(*args, **kwargs)

    with (
        patch.object(rr_ao, "initialize_grouped_mrt", return_value=np.full(4, np.pi - 0.01)),
        patch.object(rr_ao, "solve_phase_sca", restoration),
    ):
        result = rr_ao.optimize_decision(snapshot, options, "sdr")
    assert flags == [True] and result.diagnostics["phase1_used"] == 1
    assert (
        result.diagnostics["phase1_solver"] == "sca"
        and result.diagnostics["sdr_shared_phase1_used"] == 1
    )
    assert result.diagnostics["sca_blocks"] == 0


def test_A9_sdr_policy_does_not_step_commit_or_mutate_environment():
    import pickle

    env = environment()
    before = pickle.dumps(env.__dict__)
    policy = rr_ao.RRAOPolicy(
        4, 2, SolverOptions(ao_max_iters=1, wmmse_max_iters=3, sdr_max_iters=2), "sdr"
    )
    with patch.object(env, "step", side_effect=AssertionError("internal step")):
        result = policy.decide(env)
    assert policy.scheduler.decisions == 0 and policy.scheduler.peek() == (0, 1)
    assert pickle.dumps(env.__dict__) == before
    assert candidate_terms(
        snapshot_from_env(env, result.users, physical_actuator_state(env)),
        result.power,
        result.phase,
    )["is_feasible"]


@pytest.mark.parametrize("phase", ["sdr"])
def test_L1_lambda_does_not_enter_decision(phase):
    outputs = []
    for lam in (0, 1, 3, 5):
        args = make_args(
            phase_solver=phase,
            fairness_lambda=lam,
            max_steps_per_episode=2,
            ao_max_iters=1,
            wmmse_max_iters=3,
            sca_max_iters=3,
            sdr_max_iters=2,
        )
        env = LeoSatEnv(build_env_config(args, 903001))
        env.reset(seed=903001)
        policy = rr_ao.RRAOPolicy(4, 2, solver_options(args), phase)
        outputs.append(policy.decide(env))
    for actual in outputs[1:]:
        assert actual.users == outputs[0].users and actual.objective == outputs[0].objective
        np.testing.assert_array_equal(actual.power, outputs[0].power)
        np.testing.assert_array_equal(actual.phase, outputs[0].phase)


@pytest.mark.parametrize("phase", ["sdr"])
def test_C1_identity_and_default(phase):
    args = make_args(phase_solver=phase)
    assert make_args(baseline_name=args.baseline_name).phase_solver == phase
    assert make_args().phase_solver == "sdr"
    wrong = "mrt" if phase != "mrt" else "sca"
    with pytest.raises(ValueError, match="mismatch"):
        make_args(baseline_name=args.baseline_name, phase_solver=wrong)
