from dataclasses import replace
from unittest.mock import patch
import builtins
import json
import pickle
import time
import numpy as np
import pytest

from atprl.envs.actuator import RISActuatorState, physical_actuator_state
from atprl.solvers.optimization_context import (
    OptimizationSnapshot,
    snapshot_from_env,
    candidate_terms,
)
from atprl.solvers.conventional_options import SolverOptions, SolverDeadline
from atprl.solvers.mrt_initialization import initialize_grouped_mrt
from atprl.solvers.wmmse_constrained import initial_feasible_power
from atprl.solvers import sdr_phase as sdr
from tests.unit.test_conventional_solvers import synthetic
from tests.unit.test_conventional_context import environment


@pytest.mark.parametrize("g,k", [(1, 1), (2, 2), (4, 2), (8, 5)])
def test_M1_M3_generic_complex_lift(g, k):
    rng = np.random.default_rng(812 + k + g)
    for scale in (1e-12, 1.0, 1e6):
        d = (rng.normal(size=k) + 1j * rng.normal(size=k)) * scale
        c = (rng.normal(size=(k, g)) + 1j * rng.normal(size=(k, g))) * scale
        state = RISActuatorState(np.ones(g), np.full(g, 1 / g), 0, 1, 2)
        snapshot = OptimizationSnapshot(tuple(range(k)), d, c, np.zeros(k), np.ones(k), 1, 0, state)
        theta = rng.uniform(-np.pi, np.pi, g)
        matrix = sdr.phase_matrix(theta)
        channels = sdr.lift_channel_matrices(snapshot)
        actual = abs(d + c @ np.exp(1j * theta)) ** 2
        lifted = np.real(np.einsum("kij,ji->k", channels, matrix))
        np.testing.assert_allclose(lifted, actual, rtol=1e-11, atol=1e-14 * scale**2)
        np.testing.assert_allclose(
            channels, channels.conj().transpose(0, 2, 1), rtol=1e-12, atol=1e-15 * scale**2
        )
        assert np.linalg.eigvalsh(channels).min() >= -1e-12 * scale**2
        np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-15)
        assert np.linalg.eigvalsh(matrix).min() > -1e-12


def test_M2_wrong_conjugate_counterexample():
    a = np.array([1j, 1])
    v = np.array([1j, 1])
    matrix = np.outer(v, v.conj())
    assert abs(a @ v) ** 2 == 0
    assert abs(np.trace(np.outer(a.conj(), a) @ matrix)) < 1e-14
    assert np.trace(np.outer(a, a.conj()) @ matrix).real == 4


@pytest.mark.parametrize("noise", [1e-14, 1e-12, 0.1])
def test_M4_M7_normalized_rate_and_sic_including_epsilon(noise):
    snapshot = replace(synthetic(), base_noise=np.full(2, noise))
    theta = initialize_grouped_mrt(snapshot)
    power = initial_feasible_power(snapshot, theta, SolverOptions())
    model = sdr.build_phase_problem(snapshot, power, theta, SolverOptions())
    y, zc, zp, rc, j = sdr.lifted_terms(model, sdr.phase_matrix(theta))
    original = candidate_terms(snapshot, power, theta)
    np.testing.assert_allclose(
        [rc, j], [original["common_rate"], original["objective"]], rtol=1e-12
    )
    lhs = (model.power[0] - model.power[1:].sum()) * y - model.sic_rhs + model.guard
    np.testing.assert_allclose(lhs, original["sic_margin_normalized"], rtol=1e-12)


@pytest.mark.parametrize("a,b", [(0.0, 0.2), (0.8, 0.0), (0.8, 0.2), (0.8, 1e-6)])
def test_M5_real_soc_hypograph(a, b):
    cp = sdr.require_sdr_backend()
    z = cp.Variable(nonneg=True)
    y = 2.0
    noise = 1.3
    problem = cp.Problem(cp.Maximize(z), sdr.sinr_cone(cp, y, z, a, b, noise))
    assert problem.is_dcp()
    problem.solve(solver="SCS", eps_abs=1e-9, eps_rel=1e-9, max_iters=50000, verbose=False)
    assert problem.status in ("optimal", "optimal_inaccurate")
    np.testing.assert_allclose(z.value, a * y / (b * y + noise), rtol=2e-5, atol=2e-8)


def test_M6_dcp_and_actual_psd_soc_exp_canonicalization():
    snapshot = synthetic()
    theta = initialize_grouped_mrt(snapshot)
    power = initial_feasible_power(snapshot, theta, SolverOptions())
    model = sdr.build_phase_problem(snapshot, power, theta, SolverOptions())
    data, _, _ = model.problem.get_problem_data("SCS")
    assert model.problem.is_dcp()
    assert data["dims"].psd and data["dims"].soc and data["dims"].exp > 0


def test_M8_M9_rank_minorant_gauge_and_deterministic_ties():
    theta = np.array([0.7, -2.0, np.pi, -np.pi])
    matrix = sdr.phase_matrix(theta)
    u, gap, relative, _ = sdr.rank_decomposition(matrix)
    assert abs(gap) < 1e-12 and abs(relative) < 1e-12
    np.testing.assert_allclose(
        np.exp(1j * sdr.recover_phase(matrix, 1e-5)), np.exp(1j * theta), atol=1e-12
    )
    assert abs(np.trace(matrix).real - np.vdot(u, matrix @ u).real) < 1e-12
    high = np.eye(5)
    u1 = sdr.rank_decomposition(high)[0]
    u2 = sdr.rank_decomposition(high)[0]
    np.testing.assert_array_equal(u1, u2)
    assert sdr.rank_decomposition(high)[2] > 0.5
    with pytest.raises(ValueError, match="Rank"):
        sdr.recover_phase(high, 1e-5)
    with pytest.raises(ValueError, match="reference"):
        sdr.recover_phase(np.diag([2.0, 0.0]), 1e-5)
    with pytest.raises(ValueError):
        sdr.recover_phase(np.full((2, 2), np.nan), 1e-5)
    other = 0.5 * matrix + 0.5 * np.eye(5)
    _, actual, _, _ = sdr.rank_decomposition(other)
    assert actual <= np.trace(other).real - np.vdot(u, other @ u).real + 1e-12


def test_A1_real_sdr_original_action_and_no_rng_or_power_mutation():
    snapshot = synthetic()
    theta = initialize_grouped_mrt(snapshot) + 0.15
    options = SolverOptions(sdr_max_iters=15, debug_inner_history=True)
    power = initial_feasible_power(snapshot, theta, options)
    saved = power.copy()
    rng = pickle.dumps(np.random.get_state())
    result = sdr.solve_phase_sdr(snapshot, power, theta, options)
    assert result.diagnostics["sdr_sdp_solves"] > 0
    assert result.diagnostics["sdr_rank_successes"] > 0, result.diagnostics
    assert result.diagnostics["sdr_accepted_candidates"] > 0, result.diagnostics
    assert candidate_terms(snapshot, power, result.value)["is_feasible"]
    assert (
        result.objective
        >= candidate_terms(snapshot, power, theta)["objective"] - options.feasibility_tol
    )
    np.testing.assert_array_equal(power, saved)
    assert pickle.dumps(np.random.get_state()) == rng
    assert np.min(np.diff(result.history)) >= -options.feasibility_tol * max(1, result.objective)
    json.dumps(result.diagnostics, allow_nan=False)


def test_A1_real_environment_snapshot():
    env = environment()
    snapshot = snapshot_from_env(env, (0, 1), physical_actuator_state(env))
    theta = initialize_grouped_mrt(snapshot)
    options = SolverOptions(sdr_max_iters=5)
    power = initial_feasible_power(snapshot, theta, options)
    assert power is not None
    before = pickle.dumps(env.__dict__)
    result = sdr.solve_phase_sdr(snapshot, power, theta, options)
    assert result.diagnostics["sdr_sdp_solves"] > 0
    assert candidate_terms(snapshot, power, result.value)["is_feasible"]
    assert before == pickle.dumps(env.__dict__)


def test_A2_real_infeasible_and_backend_exception_safely_retains_pair():
    cp = sdr.require_sdr_backend()
    v = cp.Variable((2, 2), hermitian=True)
    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(v))), [v >> 0, cp.real(cp.diag(v)) == -1])
    problem.solve(solver="SCS")
    assert problem.status == "infeasible"
    snapshot = synthetic()
    theta = initialize_grouped_mrt(snapshot)
    o = SolverOptions()
    power = initial_feasible_power(snapshot, theta, o)
    with patch.object(sdr, "_solve_cone", side_effect=cp.error.SolverError("injected")):
        result = sdr.solve_phase_sdr(snapshot, power, theta, o)
    assert result.status == "backend_failed" and result.failures == 1
    np.testing.assert_allclose(result.value, theta, atol=1e-15)
    assert result.diagnostics["sdr_safe_fallbacks"] == 1


def test_A3_real_tiny_backend_budget_is_not_converged():
    snapshot = synthetic()
    theta = initialize_grouped_mrt(snapshot)
    o = SolverOptions(sdr_max_iters=1, sdr_scs_max_iters=1)
    power = initial_feasible_power(snapshot, theta, o)
    result = sdr.solve_phase_sdr(snapshot, power, theta, o)
    assert result.status != "converged"
    assert result.diagnostics["sdr_sdp_solves"] == 1
    assert result.diagnostics["sdr_sdp_cap_hits"] == 1
    assert candidate_terms(snapshot, power, result.value)["is_feasible"]


def test_A4_deadline_and_interrupt():
    snapshot = synthetic()
    theta = initialize_grouped_mrt(snapshot)
    o = SolverOptions()
    power = initial_feasible_power(snapshot, theta, o)
    with patch.object(sdr, "_solve_cone", side_effect=AssertionError("late solve")):
        result = sdr.solve_phase_sdr(snapshot, power, theta, o, time.perf_counter() - 1)
    assert result.status == "soft_deadline" and result.diagnostics["sdr_sdp_solves"] == 0
    with patch.object(sdr, "_solve_cone", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            sdr.solve_phase_sdr(snapshot, power, theta, o)
    with patch.object(sdr, "_solve_cone", side_effect=SolverDeadline):
        assert sdr.solve_phase_sdr(snapshot, power, theta, o).status == "soft_deadline"


def test_A5_invalid_backend_and_rank_failure_are_not_success():
    snapshot = synthetic()
    theta = initialize_grouped_mrt(snapshot)
    o = SolverOptions(sdr_max_iters=3)
    power = initial_feasible_power(snapshot, theta, o)
    with patch.object(sdr, "solution_residual", return_value=None):
        result = sdr.solve_phase_sdr(snapshot, power, theta, o)
    assert result.status == "backend_inaccurate" and result.failures == 1
    with patch.object(
        sdr,
        "solution_residual",
        return_value=dict(
            residual=0.0,
            lifted_objective=2.0,
            vector=np.ones(5) / np.sqrt(5),
            gap_abs=4.0,
            gap_rel=0.8,
            matrix=np.eye(5),
        ),
    ):
        result = sdr.solve_phase_sdr(snapshot, power, theta, o)
    assert result.status == "rank_recovery_failed" and result.failures == 1
    assert result.diagnostics["sdr_rank_failures"] == 3
    assert result.diagnostics["sdr_accepted_candidates"] == 0
    np.testing.assert_allclose(result.value, theta, atol=1e-15)


@pytest.mark.parametrize("rejection", ["sic", "objective"])
def test_A6_recovered_original_candidate_must_pass(rejection):
    snapshot = synthetic()
    theta = initialize_grouped_mrt(snapshot)
    o = SolverOptions(sdr_max_iters=2)
    power = initial_feasible_power(snapshot, theta, o)
    original = sdr.candidate_terms
    calls = []

    def reject(s, p, t):
        value = original(s, p, t)
        calls.append(1)
        if len(calls) > 1:
            if rejection == "sic":
                value["is_feasible"] = False
            else:
                value["objective"] = -100
        return value

    with patch.object(sdr, "candidate_terms", side_effect=reject):
        result = sdr.solve_phase_sdr(snapshot, power, theta, o)
    assert result.diagnostics["sdr_accepted_candidates"] == 0
    assert result.failures > 0
    np.testing.assert_allclose(result.value, theta, atol=1e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sdr_max_iters": True},
        {"sdr_solver": "MOSEK"},
        {"sdr_warm_start": 1},
        {"sdr_rank_tol": 1},
        {"sdr_scs_eps_abs": float("nan")},
        {"sdr_tol": "bad"},
        {"sdr_rank_penalty_growth": 1},
        {"sdr_rank_penalty_max": 0.5},
    ],
)
def test_sdr_option_validation(kwargs):
    with pytest.raises(ValueError):
        SolverOptions(**kwargs)


def test_missing_optional_dependencies_are_an_explicit_configuration_error():
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("cvxpy", "scs"):
            raise ImportError("intentionally absent")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocked):
        with pytest.raises(RuntimeError, match="optional CVXPY/SCS"):
            sdr.require_sdr_backend()


@pytest.mark.parametrize("seed", [903001, 903019])
def test_normalized_cones_accept_real_wmmse_power_without_relaxing_constraints(seed):
    from atprl.evaluation.conventional_config import make_args, solver_options
    from atprl.baselines.rr_ao import optimize_decision

    args = make_args(phase_solver="sdr", ao_max_iters=2, wmmse_max_iters=5, sdr_max_iters=3)
    samples = _collect_snapshots(args, dict(scenarios=[[4, 2]], seeds=[seed], decisions_per_seed=2))
    for sample in samples:
        result = optimize_decision(sample["snapshot"], solver_options(args), "sdr")
        assert result.diagnostics["sdr_accepted_candidates"] > 0
        assert result.diagnostics["sdr_effective_updates"] > 0
        assert candidate_terms(sample["snapshot"], result.power, result.phase)["is_feasible"]
        assert result.diagnostics["sdr_lifted_residual_max"] <= args.feasibility_tol


def test_scs_raw_sigint_cannot_be_swallowed_as_a_numerical_fallback():
    from cvxpy.reductions.solvers.solving_chain import SolvingChain

    snapshot = synthetic()
    theta = initialize_grouped_mrt(snapshot)
    options = SolverOptions()
    power = initial_feasible_power(snapshot, theta, options)
    with patch.object(SolvingChain, "solve_via_data", return_value={"info": {"status_val": -5}}):
        with pytest.raises(KeyboardInterrupt, match="interrupted"):
            sdr.solve_phase_sdr(snapshot, power, theta, options)


def _collect_snapshots(base_args, spec):
    from atprl.evaluation.conventional_config import make_args, build_env_config
    from atprl.evaluation.conventional import SicCountingEnv
    from atprl.baselines.round_robin import RoundRobinScheduler
    from atprl.envs.actuator import physical_actuator_state
    from atprl.solvers.optimization_context import snapshot_from_env

    samples = []
    formal = set(base_args.eval_seed_base + base_args.eval_seed_stride * i for i in range(500))
    if formal & set(spec["seeds"]):
        raise ValueError("Calibration seeds overlap formal seeds")
    for n, k in spec["scenarios"]:
        horizon = spec.get("decisions_per_scenario", {}).get(
            f"N{n}K{k}", spec["decisions_per_seed"]
        )
        for seed in spec["seeds"]:
            args = make_args(
                **(
                    vars(base_args)
                    | dict(
                        num_ground_stations=n,
                        num_gs_per_rb=k,
                        topology_profile=f"paper_n{n}",
                        max_steps_per_episode=horizon,
                        num_eval_seeds=1,
                        show_progress=False,
                    )
                )
            )
            env = SicCountingEnv(build_env_config(args, seed))
            env.reset(seed=seed)
            rr = RoundRobinScheduler(n, k)
            for step in range(horizon):
                actuator = physical_actuator_state(env)
                samples.append(
                    dict(
                        scenario=f"N{n}K{k}",
                        seed=seed,
                        step=step,
                        snapshot=snapshot_from_env(env, rr.peek(), actuator),
                    )
                )
                env.step(
                    {
                        "rb_0": (
                            rr.peek(),
                            np.r_[env.satellite_max_power_watt, np.zeros(k)],
                            np.zeros(env.ris_num_groups),
                            actuator.group_weights,
                        )
                    }
                )
                rr.commit()
    return samples
