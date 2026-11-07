from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from atprl.envs.actuator import RISActuatorState, physical_actuator_state
from atprl.solvers.optimization_context import (
    OptimizationSnapshot,
    snapshot_from_env,
    candidate_terms,
)
from atprl.solvers.conventional_options import SolverOptions
from atprl.solvers.mrt_initialization import initialize_grouped_mrt
from atprl.solvers.sca_phase import (
    gain_minorant,
    phase_subproblem,
    solve_phase_sca,
    certified_sic_infeasible,
)
from atprl.solvers.wmmse_constrained import (
    initial_feasible_power,
    solve_rsma_power,
    awmmse_subproblem,
    normalized_channel,
)
from atprl.solvers.wmmse import (
    _compute_mmse_receivers,
    _compute_mmse_mse,
    _compute_mmse_weights,
    _awmmse_constants,
    _common_xi_from_v,
    _private_xi_from_v,
)
from tests.unit.test_conventional_context import environment


def synthetic(direct=None, cascade=None):
    actuator = RISActuatorState(np.ones(4), np.full(4, 0.25), 0, 1, 2)
    return OptimizationSnapshot(
        (0, 1),
        np.array([1 + 0.2j, 1.2 - 0.1j]) if direct is None else direct,
        np.array(
            [[0.1 + 0.2j, 0.2 - 0.1j, 0.3 + 0.1j, 0.2], [0.2 - 0.1j, 0.1 + 0.2j, 0.1 - 0.1j, 0.3j]]
        )
        if cascade is None
        else cascade,
        np.full(2, 0.01),
        np.full(2, 0.1),
        1,
        0.01,
        actuator,
    )


def test_1000_minorants_and_gradient_tightness():
    rng = np.random.default_rng(20261006)
    for i in range(1000):
        scale = 10 ** rng.uniform(-12, 6)
        k = 1 + i % 5
        g = 1 + i % 4
        d = (rng.normal(size=k) + 1j * rng.normal(size=k)) * scale
        c = (rng.normal(size=(k, g)) + 1j * rng.normal(size=(k, g))) * scale
        th = rng.uniform(-np.pi, np.pi, g)
        delta = rng.uniform(-np.pi, np.pi, g)
        q, grad, L = gain_minorant(d, c, th)
        true = abs(d + c @ np.exp(1j * (th + delta))) ** 2
        lower = q + grad @ delta - L * np.dot(delta, delta) / 2
        assert np.all(lower <= true + 1e-10 * max(1, scale**2))
        if i < 25:
            numerical = approx_derivative(lambda x: abs(d + c @ np.exp(1j * x)) ** 2, th)
            np.testing.assert_allclose(
                grad, numerical.reshape(k, g), rtol=2e-5, atol=1e-8 * max(1, scale**2)
            )


def test_rate_wmmse_identity_and_constraint_jacobian():
    s = synthetic()
    th = initialize_grouped_mrt(s)
    o = SolverOptions()
    p = initial_feasible_power(s, th, o)
    h, e, b, t = normalized_channel(s, th)
    x = p / s.pmax
    uc, up = _compute_mmse_receivers(h, e, np.ones(2), x)
    ec, ep = _compute_mmse_mse(h, e, np.ones(2), x, uc, up)
    wc, wp = _compute_mmse_weights(ec, ep)
    co = _awmmse_constants(h, e, np.ones(2), uc, up, wc, wp)
    v = np.sqrt(x)
    value = np.sum(_private_xi_from_v(v, co)) + np.max(_common_xi_from_v(v, co))
    np.testing.assert_allclose(
        value, 3 - np.log(2) * candidate_terms(s, p, th)["objective"], rtol=1e-10
    )
    fun, jac, con, cjac = awmmse_subproblem(
        co, abs(h) ** 2, e, b, t, o.sic_guard * np.maximum(1, b + t)
    )
    z = np.r_[v, np.max(_common_xi_from_v(v, co)) + 0.1]
    np.testing.assert_allclose(jac(z), approx_derivative(fun, z).ravel(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(cjac(z), approx_derivative(con, z), rtol=1e-5, atol=1e-7)
    assert np.min(con(z)) >= -1e-10


@pytest.mark.parametrize("phase1", [False, True])
def test_sca_analytic_jacobian(phase1):
    s = synthetic()
    th = initialize_grouped_mrt(s)
    o = SolverOptions()
    p = np.array([1, 0, 0]) if phase1 else initial_feasible_power(s, th, o)
    fun, jac, con, cjac, z, bounds = phase_subproblem(s, p, th, o, phase1=phase1)
    np.testing.assert_allclose(jac(z), approx_derivative(fun, z).ravel(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(cjac(z), approx_derivative(con, z), rtol=1e-5, atol=1e-7)
    assert np.min(con(z)) >= -1e-10


def test_power_phase_monotonic_deterministic_and_failures():
    s = synthetic()
    th = initialize_grouped_mrt(s)
    o = SolverOptions(wmmse_max_iters=40, sca_max_iters=20, debug_inner_history=True)
    first = solve_rsma_power(s, th, o)
    second = solve_rsma_power(s, th, o)
    np.testing.assert_array_equal(first.value, second.value)
    assert candidate_terms(s, first.value, th)["is_feasible"]
    assert np.min(np.diff(first.history)) >= -o.feasibility_tol
    phase = solve_phase_sca(s, first.value, th, o)
    assert candidate_terms(s, first.value, phase.value)["is_feasible"]
    assert phase.objective >= first.objective - o.feasibility_tol
    with patch("atprl.solvers.wmmse_constrained.minimize") as mock:
        mock.return_value.success = False
        mock.return_value.nit = 1
        failed = solve_rsma_power(s, th, o, initial_power=first.value)
        assert failed.status == "solver_failed"
        np.testing.assert_array_equal(failed.value, first.value)
    with pytest.raises(ValueError):
        solve_rsma_power(s, [np.nan] * 4, o)


def test_phase1_recovery_and_infeasible_certificate():
    s = synthetic(np.array([1.2, 1.2]), np.full((2, 4), 0.3))
    o = SolverOptions(phase1_max_iters=30)
    bad = np.full(4, np.pi / 2)
    bad = np.full(4, np.pi)
    assert not candidate_terms(s, [1, 0, 0], bad + 0.01)["is_feasible"]
    recovered = solve_phase_sca(s, np.array([1, 0, 0]), bad + 0.01, o, phase1=True)
    assert recovered.status == "recovered_feasible"
    impossible = replace(s, direct_hat=np.zeros(2), cascade_hat=np.zeros((2, 4)))
    assert certified_sic_infeasible(impossible)
    result = solve_phase_sca(impossible, np.array([1, 0, 0]), np.zeros(4), o, phase1=True)
    assert result.status == "feasibility_not_found"
    assert np.all(np.isfinite(result.value))


def test_real_n4_n20_no_environment_mutation_and_early_stop():
    import pickle

    for n in (4, 20):
        env = environment(n=n)
        s = snapshot_from_env(env, (0, 1), physical_actuator_state(env))
        before = pickle.dumps(env.__dict__)
        th = initialize_grouped_mrt(s)
        power = solve_rsma_power(s, th, SolverOptions(wmmse_max_iters=5))
        if candidate_terms(s, power.value, th)["is_feasible"]:
            phase = solve_phase_sca(s, power.value, th, SolverOptions(sca_max_iters=5))
            assert np.isfinite(phase.objective)
        assert before == pickle.dumps(env.__dict__)
    s = synthetic(cascade=np.zeros((2, 4)))
    p = solve_rsma_power(s, np.zeros(4), SolverOptions(wmmse_max_iters=100, wmmse_tol=0.01))
    phase = solve_phase_sca(s, p.value, np.zeros(4), SolverOptions(sca_max_iters=50))
    assert phase.status == "converged" and phase.iterations < 50
