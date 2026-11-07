from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from atprl.solvers.wmmse import (
    _compute_mmse_receivers,
    _compute_mmse_mse,
    _compute_mmse_weights,
    _awmmse_constants,
    _common_xi_from_v,
    _private_xi_from_v,
)
from atprl.solvers.optimization_context import OptimizationSnapshot, candidate_terms
from atprl.solvers.conventional_options import (
    SolverOptions,
    BlockResult,
    SolverDeadline,
    check_deadline,
    append_history,
)


def normalized_channel(snapshot, phase):
    d, c, error, base, threshold = snapshot.normalized()
    return d + c @ np.exp(1j * np.asarray(phase)), error, base, threshold


def fraction_to_watt(snapshot, fraction):
    """Preserve the total power budget when converting normalized powers to watts."""
    power = np.asarray(fraction, dtype=float) * snapshot.pmax
    total = float(power.sum())
    target = snapshot.pmax * (1 - (snapshot.k + 2) * np.finfo(float).eps)
    if total > target:
        power *= target / total
    return power


def initial_feasible_power(snapshot: OptimizationSnapshot, phase, options: SolverOptions):
    h, error, base, threshold = normalized_channel(snapshot, phase)
    gain = np.abs(h) ** 2
    guard = options.sic_guard * np.maximum(1, base + threshold)
    margin = gain - error - base - threshold - guard
    if np.any(margin < 0):
        return None
    total_private = min(
        0.25, float(np.min(margin / np.maximum(2 * gain, np.finfo(float).tiny))) / 2
    )
    fractions = np.r_[1 - total_private, np.full(snapshot.k, total_private / snapshot.k)]
    power = fraction_to_watt(snapshot, fractions)
    return power if candidate_terms(snapshot, power, phase)["is_feasible"] else None


def awmmse_subproblem(consts, gain, error, base, threshold, guard):
    k = len(gain)
    if np.any(gain <= error):
        raise ValueError("Fixed-phase SIC is infeasible")
    aa = (gain + error) / (gain - error)
    bb = (base + threshold + guard) / (gain - error)

    def objective(z):
        return float(np.sum(_private_xi_from_v(z[:-1], consts)) + z[-1])

    def objective_jac(z):
        return np.r_[
            2 * np.sum(consts["A_pc"]) * z[0],
            2 * np.sum(consts["A_pp"]) * z[1:-1] - consts["B_p"],
            1.0,
        ]

    def constraints(z):
        v = z[:-1]
        root = np.sqrt(aa * np.sum(v[1:] ** 2) + bb)
        return np.r_[1 - np.sum(v**2), z[-1] - _common_xi_from_v(v, consts), v[0] - root]

    def constraint_jac(z):
        v = z[:-1]
        jac = np.zeros((1 + 2 * k, k + 2))
        jac[0, :-1] = -2 * v
        jac[1 : 1 + k, 0] = -2 * consts["A_c"] * v[0] + consts["B_c"]
        jac[1 : 1 + k, 1:-1] = -2 * consts["A_c"][:, None] * v[None, 1:]
        jac[1 : 1 + k, -1] = 1
        root = np.sqrt(aa * np.sum(v[1:] ** 2) + bb)
        jac[1 + k :, 0] = 1
        jac[1 + k :, 1:-1] = -aa[:, None] * v[None, 1:] / root[:, None]
        return jac

    return objective, objective_jac, constraints, constraint_jac


def solve_rsma_power(
    snapshot: OptimizationSnapshot, phase, options: SolverOptions, initial_power=None, deadline=None
) -> BlockResult:
    phase = np.asarray(phase, dtype=float)
    if phase.shape != (snapshot.g,) or not np.all(np.isfinite(phase)):
        raise ValueError("Invalid fixed phase")
    power = initial_feasible_power(snapshot, phase, options)
    if initial_power is not None:
        terms = candidate_terms(snapshot, initial_power, phase)
        if terms["is_feasible"]:
            power = np.asarray(initial_power, dtype=float).copy()
    if power is None:
        fallback = np.r_[snapshot.pmax, np.zeros(snapshot.k)]
        terms = candidate_terms(snapshot, fallback, phase)
        return BlockResult(
            fallback,
            terms["objective"],
            "fixed_phase_infeasible",
            minimum_sic=float(np.min(terms["sic_margin_normalized"])),
        )
    h, error, base, threshold = normalized_channel(snapshot, phase)
    gain = np.abs(h) ** 2
    guard = options.sic_guard * np.maximum(1, base + threshold)
    terms = candidate_terms(snapshot, power, phase)
    result = BlockResult(
        power.copy(),
        terms["objective"],
        "max_iter_feasible",
        history=[] if options.debug_inner_history else None,
    )
    append_history(result.history, result.objective, options)
    stable = 0
    for iteration in range(options.wmmse_max_iters):
        try:
            check_deadline(deadline)
            previous = power / snapshot.pmax
            uc, up = _compute_mmse_receivers(h, error, np.ones(snapshot.k), previous)
            ec, ep = _compute_mmse_mse(h, error, np.ones(snapshot.k), previous, uc, up)
            wc, wp = _compute_mmse_weights(ec, ep)
            co = _awmmse_constants(h, error, np.ones(snapshot.k), uc, up, wc, wp)
            v = np.sqrt(previous)
            z0 = np.r_[v, np.max(_common_xi_from_v(v, co))]
            fun, jac, con, cjac = awmmse_subproblem(co, gain, error, base, threshold, guard)
            opt = minimize(
                fun,
                z0,
                jac=jac,
                method="SLSQP",
                bounds=[(0, None)] * (snapshot.k + 1) + [(None, None)],
                constraints=[dict(type="ineq", fun=con, jac=cjac)],
                callback=lambda _: check_deadline(deadline),
                options=dict(
                    maxiter=options.wmmse_scipy_maxiter, ftol=options.wmmse_scipy_ftol, disp=False
                ),
            )
            result.iterations = iteration + 1
            result.subproblem_iterations += int(opt.nit)
            if (
                not opt.success
                or not np.all(np.isfinite(opt.x))
                or np.min(con(opt.x)) < -options.feasibility_tol
            ):
                result.failures += 1
                result.status = "solver_failed"
                break
            fraction = np.maximum(opt.x[:-1], 0) ** 2
            if fraction.sum() >= 1:
                fraction *= np.nextafter(1.0, 0.0) / fraction.sum()
            candidate = fraction_to_watt(snapshot, fraction)
            fraction = candidate / snapshot.pmax
            current = candidate_terms(snapshot, candidate, phase)
            if not current["is_feasible"] or current[
                "objective"
            ] < result.objective - options.feasibility_tol * max(1, abs(result.objective)):
                result.failures += 1
                result.status = "stalled_after_failure"
                break
            relative = abs(current["objective"] - result.objective) / max(1, abs(result.objective))
            change = np.linalg.norm(fraction - previous) / max(1, np.linalg.norm(previous))
            power = candidate
            result.value, result.objective = power.copy(), current["objective"]
            append_history(result.history, result.objective, options)
            stable = (
                stable + 1
                if relative <= options.wmmse_tol and change <= np.sqrt(options.wmmse_tol)
                else 0
            )
            if stable >= options.stopping_patience:
                result.status = "converged"
                break
        except SolverDeadline:
            result.status = "soft_deadline"
            break
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            result.failures += 1
            result.status = "solver_failed"
            break
    result.cap_hit = result.status == "max_iter_feasible"
    result.minimum_sic = float(
        np.min(candidate_terms(snapshot, result.value, phase)["sic_margin_normalized"])
    )
    return result
