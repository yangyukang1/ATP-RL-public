from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from atprl.solvers.optimization_context import OptimizationSnapshot, candidate_terms, wrap_phase
from atprl.solvers.conventional_options import (
    SolverOptions,
    BlockResult,
    SolverDeadline,
    check_deadline,
    append_history,
)


def gain_minorant(direct, cascade, phase):
    terms = cascade * np.exp(1j * phase)[None, :]
    h = direct + terms.sum(axis=1)
    gain = np.abs(h) ** 2
    gradient = 2 * np.real(h.conj()[:, None] * 1j * terms)
    curvature = 2 * np.sum(abs(cascade) ** 2, axis=1) + 2 * (
        abs(direct) + np.sum(abs(cascade), axis=1)
    ) * np.max(abs(cascade), axis=1)
    return gain, gradient, curvature


def rate_gain_function(y, signal, interference, noise):
    den = interference * y + noise
    rate = np.log1p(signal * y / den) / np.log(2)
    gradient = signal * noise / (np.log(2) * ((interference + signal) * y + noise) * den)
    return rate, gradient


def phase_subproblem(snapshot, power, phase, options, *, phase1=False):
    d, c, error, base, threshold = snapshot.normalized()
    gain, grad, curvature = gain_minorant(d, c, phase)
    scale = np.maximum(1, gain)
    g, k = snapshot.g, snapshot.k
    x = np.asarray(power) / snapshot.pmax
    private, total = x[1:].sum(), x.sum()
    coefficient = x[0] - private
    noise = 1 + error * total
    guard = options.sic_guard * np.maximum(1, base + threshold)
    rhs = error * total + base + threshold + (0 if phase1 else guard)
    cscale = np.maximum(1, gain)

    def lower(delta):
        return gain + grad @ delta - curvature * np.dot(delta, delta) / 2

    def objective(z):
        y = z[g : g + k] * scale
        return -float(
            z[-1]
            if phase1
            else z[-1]
            + rate_gain_function(y, x[1:], np.maximum(private - x[1:], 0), noise)[0].sum()
        )

    def objective_jac(z):
        out = np.zeros_like(z)
        if not phase1:
            out[g : g + k] = (
                -rate_gain_function(
                    z[g : g + k] * scale, x[1:], np.maximum(private - x[1:], 0), noise
                )[1]
                * scale
            )
        out[-1] = -1
        return out

    def constraints(z):
        y = z[g : g + k] * scale
        channel = (lower(z[:g]) - y) / scale
        if phase1:
            return np.r_[channel, (coefficient * y - rhs - z[-1]) / cscale]
        common = rate_gain_function(y, x[0], private, noise)[0] - z[-1]
        return np.r_[channel, common, (coefficient * y - rhs) / cscale]

    def constraint_jac(z):
        out = np.zeros((k * (2 if phase1 else 3), g + k + 1))
        out[:k, :g] = (grad - curvature[:, None] * z[None, :g]) / scale[:, None]
        out[:k, g : g + k] = -np.eye(k)
        if phase1:
            out[k:, g : g + k] = np.diag(coefficient * scale / cscale)
            out[k:, -1] = -1 / cscale
        else:
            out[k : 2 * k, g : g + k] = np.diag(
                rate_gain_function(z[g : g + k] * scale, x[0], private, noise)[1] * scale
            )
            out[k : 2 * k, -1] = -1
            out[2 * k :, g : g + k] = np.diag(coefficient * scale / cscale)
        return out

    t0 = (
        float(np.min(coefficient * gain - rhs))
        if phase1
        else float(np.min(rate_gain_function(gain, x[0], private, noise)[0]))
    )
    z0 = np.r_[np.zeros(g), gain / scale, t0]
    bounds = (
        [(-options.sca_trust_radius, options.sca_trust_radius)] * g
        + [(0, None)] * k
        + [(None, None) if phase1 else (0, None)]
    )
    return objective, objective_jac, constraints, constraint_jac, z0, bounds


def solve_phase_sca(
    snapshot: OptimizationSnapshot,
    power,
    phase,
    options: SolverOptions,
    deadline=None,
    *,
    phase1=False,
) -> BlockResult:
    th = wrap_phase(phase)
    if not np.all(np.isfinite(th)) or th.shape != (snapshot.g,):
        raise ValueError("Invalid phase initializer")
    initial = candidate_terms(snapshot, power, th)
    if not phase1 and not initial["is_feasible"]:
        raise ValueError("SCA requires a feasible initial power/phase pair")
    merit = float(np.min(initial["sic_margin_normalized"])) if phase1 else initial["objective"]
    result = BlockResult(
        th.copy(),
        merit,
        "feasibility_not_found" if phase1 else "max_iter_feasible",
        history=[] if options.debug_inner_history else None,
    )
    append_history(result.history, merit, options)
    stable = 0
    cap = options.phase1_max_iters if phase1 else options.sca_max_iters
    for iteration in range(cap):
        try:
            check_deadline(deadline)
            fun, jac, con, cjac, z0, bounds = phase_subproblem(
                snapshot, power, th, options, phase1=phase1
            )
            opt = minimize(
                fun,
                z0,
                jac=jac,
                method="SLSQP",
                bounds=bounds,
                constraints=[dict(type="ineq", fun=con, jac=cjac)],
                callback=lambda _: check_deadline(deadline),
                options=dict(
                    maxiter=options.sca_scipy_maxiter, ftol=options.sca_scipy_ftol, disp=False
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
            delta = opt.x[: snapshot.g]
            candidate = wrap_phase(th + delta)
            actual = candidate_terms(snapshot, power, candidate)
            value = (
                float(np.min(actual["sic_margin_normalized"])) if phase1 else actual["objective"]
            )
            if (
                not phase1 and not actual["is_feasible"]
            ) or value < merit - options.feasibility_tol * max(1, abs(merit)):
                result.failures += 1
                result.status = "stalled_after_failure"
                break
            relative = abs(value - merit) / max(1, abs(merit))
            th, merit = candidate, value
            result.value, result.objective = th.copy(), merit
            append_history(result.history, merit, options)
            if (
                phase1
                and actual["is_feasible"]
                and value
                >= options.sic_guard
                * max(1, np.max(snapshot.normalized()[3] + snapshot.normalized()[4]))
            ):
                result.status = "recovered_feasible"
                break
            stable = (
                stable + 1
                if relative <= options.sca_tol and np.linalg.norm(delta) <= options.sca_phase_tol
                else 0
            )
            if stable >= options.stopping_patience:
                result.status = "feasibility_not_found" if phase1 else "converged"
                break
        except SolverDeadline:
            result.status = "soft_deadline"
            break
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            result.failures += 1
            result.status = "solver_failed"
            break
    result.cap_hit = result.iterations == cap and result.status in (
        "max_iter_feasible",
        "feasibility_not_found",
    )
    result.minimum_sic = float(
        np.min(candidate_terms(snapshot, power, result.value)["sic_margin_normalized"])
    )
    return result


def certified_sic_infeasible(snapshot):
    maximum_gain = (abs(snapshot.direct_hat) + np.sum(abs(snapshot.cascade_hat), axis=1)) ** 2
    return bool(
        np.any(
            snapshot.pmax * (maximum_gain - snapshot.error_variance)
            < snapshot.base_noise + snapshot.sic_threshold
        )
    )
