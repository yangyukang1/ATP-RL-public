"""Grouped RIS phase optimization using SDR and deterministic rank penalization."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import io
import os
import platform
import time
import warnings
from contextlib import redirect_stdout
import numpy as np

from atprl.solvers.conventional_options import (
    BlockResult,
    SolverDeadline,
    check_deadline,
    append_history,
)
from atprl.solvers.optimization_context import candidate_terms, wrap_phase

SDR_PROTOCOL = "rsma_active_grouped_sdr_rankmm_soc_v1"
SCS_SETTINGS = dict(
    use_indirect=False,
    acceleration_lookback=0,
    normalize=True,
    adaptive_scale=True,
    scale=0.1,
    rho_x=1e-6,
    alpha=1.5,
    eps_infeas=1e-7,
)


def require_sdr_backend():
    try:
        import cvxpy as cp
        import scs  # noqa: F401 -- check the actual selected optional backend
    except ImportError as exc:
        raise RuntimeError(
            "SDR requires optional CVXPY/SCS dependencies; install atprl[sdr] in an isolated environment"
        ) from exc
    if "SCS" not in cp.installed_solvers():
        raise RuntimeError("SDR requires an installed SCS backend")
    return cp


def backend_identity():
    require_sdr_backend()
    with redirect_stdout(io.StringIO()) as output:
        np.show_config()
    return dict(
        sdr_algorithm_protocol=SDR_PROTOCOL,
        actual_sdp_backend="SCS",
        cvxpy=importlib.metadata.version("cvxpy"),
        scs=importlib.metadata.version("scs"),
        python=platform.python_version(),
        platform=platform.platform(),
        scs_settings=SCS_SETTINGS.copy(),
        phase1_solver="sca",
        blas_configuration=output.getvalue(),
        thread_environment={
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        package_versions={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "cvxpy",
                "scs",
                "osqp",
                "clarabel",
                "cffi",
                "torch",
                "sympy",
                "fsspec",
            )
        },
    )


def lift_channel_matrices(snapshot, *, normalized=False):
    d, c = snapshot.normalized()[:2] if normalized else (snapshot.direct_hat, snapshot.cascade_hat)
    a = np.column_stack((c, d))
    return np.einsum("ki,kj->kij", a.conj(), a)


def phase_matrix(phase):
    v = np.r_[np.exp(1j * np.asarray(phase)), 1.0]
    return np.outer(v, v.conj())


def rank_decomposition(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("Invalid lifted phase matrix")
    sym = (matrix + matrix.conj().T) / 2
    values, vectors = np.linalg.eigh(sym)
    trace = float(np.trace(sym).real)
    if trace <= 0:
        raise ValueError("Lifted matrix has nonpositive trace")
    tied = vectors[:, values >= values[-1] - 1e-10 * max(1, abs(values[-1]))]
    projector = tied @ tied.conj().T
    for column in projector.T:
        norm = np.linalg.norm(column)
        if norm > 1e-10:
            vector = column / norm
            break
    else:
        raise ValueError("Invalid maximum-eigenvalue subspace")
    gap = trace - float(values[-1])
    return vector, gap, gap / trace, float(values[0])


def recover_phase(matrix, rank_tol, feasibility_tol=1e-7):
    matrix = np.asarray(matrix, dtype=complex)
    if np.max(abs(matrix - matrix.conj().T)) > feasibility_tol * max(1, np.linalg.norm(matrix)):
        raise ValueError("Lifted phase matrix is not Hermitian")
    vector, gap, relative, minimum = rank_decomposition(matrix)
    if minimum < -feasibility_tol * max(1, float(np.trace(matrix).real)):
        raise ValueError("Lifted phase matrix is not PSD")
    if relative > rank_tol:
        raise ValueError("Rank-one recovery failed")
    if abs(vector[-1]) < 1e-8:
        raise ValueError("Near-zero phase reference")
    return wrap_phase(np.angle(vector[:-1] / vector[-1]))


def sinr_cone(cp, y, z, a, b, noise, gain_bound=0.0):
    """Construct the exact convex hypograph of a*y/(b*y + noise)."""
    if a == 0:
        return [z == 0]
    if b == 0:
        return [noise * z <= a * y]
    if b * max(1, gain_bound) / noise < 1e-3:
        scale = max(1, gain_bound)
        quadratic = cp.Variable(nonneg=True)
        denominator = 1 + b * y / noise
        constraints = [
            cp.SOC(denominator + quadratic, cp.hstack([2 * y / scale, denominator - quadratic])),
            z + (a * b * scale**2 / noise**2) * quadratic <= a * y / noise,
        ]
        if gain_bound > 0:
            constraints.append(quadratic <= (gain_bound / scale) ** 2)
        return constraints
    upper, denominator = a - b * z, b * y + noise
    scale = (b * gain_bound + noise) / np.sqrt(a * noise)
    upper, denominator = scale * upper, denominator / scale
    return [
        upper >= 0,
        noise * z <= a * y,
        cp.SOC(denominator + upper, cp.hstack([2 * np.sqrt(a * noise), denominator - upper])),
    ]


@dataclass
class PhaseProblem:
    problem: object
    matrix: object
    gain: object
    common_sinr: object
    private_sinr: object
    common_rate: object
    rank_weight: object
    channels: np.ndarray
    xmax: np.ndarray
    power: np.ndarray
    noise: np.ndarray
    sic_rhs: np.ndarray
    guard: np.ndarray
    objective_floor: float
    gain_variable: object
    gain_scale: np.ndarray
    common_variable: object
    private_variable: object
    common_scale: np.ndarray
    private_scale: np.ndarray


def build_phase_problem(snapshot, power, phase, options):
    cp = require_sdr_backend()
    x = np.asarray(power, dtype=float) / snapshot.pmax
    _, _, error, base, threshold = snapshot.normalized()
    private, total = float(x[1:].sum()), float(x.sum())
    channels = lift_channel_matrices(snapshot, normalized=True)
    d, c = snapshot.normalized()[:2]
    xmax = (abs(d) + np.sum(abs(c), axis=1)) ** 2
    noise = 1 + total * error
    guard = options.sic_guard * np.maximum(1, base + threshold)
    rhs = base + total * error + threshold + guard
    k, m = snapshot.k, snapshot.g + 1
    matrix = cp.Variable((m, m), hermitian=True)
    gain_scale = np.maximum(1, xmax)
    gain_variable = cp.Variable(k, nonneg=True)
    gain = cp.multiply(gain_scale, gain_variable)
    common_variable, private_variable = cp.Variable(k, nonneg=True), cp.Variable(k, nonneg=True)
    common_scale = np.where(x[0] > 0, x[0] * gain_scale / noise, 1.0)
    private_scale = np.where(x[1:] > 0, x[1:] * gain_scale / noise, 1.0)
    zc, zp = (
        cp.multiply(common_scale, common_variable),
        cp.multiply(private_scale, private_variable),
    )
    rc = cp.Variable(nonneg=True)
    weight = cp.Parameter((m, m), hermitian=True)
    constraints = [matrix >> 0, cp.real(cp.diag(matrix)) == 1, gain_variable <= xmax / gain_scale]
    for i in range(k):
        constraints.append(
            gain_variable[i] == cp.real(cp.trace(channels[i] @ matrix)) / gain_scale[i]
        )
        for z, a, b in (
            (common_variable[i], x[0], private),
            (private_variable[i], x[i + 1], max(private - x[i + 1], 0)),
        ):
            constraints.extend(
                sinr_cone(
                    cp,
                    gain_variable[i],
                    z,
                    1.0 if a > 0 else 0.0,
                    float(b * gain_scale[i] / noise[i]),
                    1.0,
                    float(xmax[i] / gain_scale[i]),
                )
            )
            constraints.append(
                z <= (xmax[i] / gain_scale[i]) / (1 + b * xmax[i] / noise[i]) if a > 0 else z == 0
            )
        constraints.append(rc <= cp.log1p(zc[i]) / np.log(2))
    sic_scale = np.maximum.reduce([np.ones(k), abs(x[0] - private) * xmax, rhs])
    constraints.append(
        cp.multiply((x[0] - private) * gain_scale / sic_scale, gain_variable) >= rhs / sic_scale
    )
    objective = rc + cp.sum(cp.log1p(zp)) / np.log(2)
    initial = candidate_terms(snapshot, power, phase)["objective"]
    floor = initial - options.feasibility_tol * max(1, abs(initial))
    constraints.append(objective >= floor)
    problem = cp.Problem(cp.Maximize(objective - cp.real(cp.trace(weight @ matrix))), constraints)
    if not problem.is_dcp():
        raise ValueError("SDR model must satisfy DCP")
    model = PhaseProblem(
        problem,
        matrix,
        gain,
        zc,
        zp,
        rc,
        weight,
        channels,
        xmax,
        x.copy(),
        noise,
        rhs,
        guard,
        floor,
        gain_variable,
        gain_scale,
        common_variable,
        private_variable,
        common_scale,
        private_scale,
    )
    initialize_problem(model, phase)
    vector = rank_decomposition(phase_matrix(phase))[0]
    weight.value = options.sdr_rank_penalty_init * (np.eye(m) - np.outer(vector, vector.conj())) / m
    return model


def lifted_terms(model, matrix):
    y = np.real(np.einsum("kij,ji->k", model.channels, matrix))
    x, private = model.power, float(model.power[1:].sum())
    zc = x[0] * y / (private * y + model.noise)
    zp = x[1:] * y / (np.maximum(private - x[1:], 0) * y + model.noise)
    common = float(np.min(np.log1p(zc) / np.log(2)))
    return y, zc, zp, common, float(common + np.log1p(zp).sum() / np.log(2))


def initialize_problem(model, phase):
    matrix = phase_matrix(phase)
    y, zc, zp, rc, _ = lifted_terms(model, matrix)
    model.matrix.value = matrix
    model.gain_variable.value = np.maximum(y, 0) / model.gain_scale
    model.common_variable.value = np.maximum(zc, 0) / model.common_scale
    model.private_variable.value = np.maximum(zp, 0) / model.private_scale
    model.common_rate.value = max(rc, 0)


def solution_residual(model):
    """Check original SINR constraints as well as cone residuals."""
    values = [
        model.matrix.value,
        model.gain.value,
        model.common_sinr.value,
        model.private_sinr.value,
        model.common_rate.value,
    ]
    if any(value is None or not np.all(np.isfinite(value)) for value in values):
        return None
    matrix, y, zc, zp, rc = values
    actual_y, actual_c, actual_p, actual_rc, actual_j = lifted_terms(model, matrix)
    try:
        vector, gap, relative, minimum = rank_decomposition(matrix)
    except ValueError:
        return None
    scale = max(1, float(np.trace(matrix).real))
    violations = [
        np.max(abs(matrix - matrix.conj().T)) / scale,
        max(0, -minimum) / scale,
        np.max(abs(np.diag(matrix) - 1)),
        np.max(abs(y - actual_y) / np.maximum(1, model.xmax)),
        np.max(-y / np.maximum(1, model.xmax)),
        np.max((y - model.xmax) / np.maximum(1, model.xmax)),
        np.max(-zc),
        np.max(-zp),
        -float(rc),
        np.max((zc - actual_c) / np.maximum(1, abs(actual_c))),
        np.max((zp - actual_p) / np.maximum(1, abs(actual_p))),
        (float(rc) - actual_rc) / max(1, abs(actual_rc)),
        (model.objective_floor - actual_j) / max(1, abs(model.objective_floor)),
        np.max(
            (model.sic_rhs - (model.power[0] - model.power[1:].sum()) * actual_y)
            / np.maximum(1, model.sic_rhs)
        ),
    ]
    residual = float(max(0, *violations))
    if not np.isfinite(residual) or not np.isfinite(actual_j):
        return None
    return dict(
        residual=residual,
        lifted_objective=actual_j,
        vector=vector,
        gap_abs=gap,
        gap_rel=relative,
        matrix=(matrix + matrix.conj().T) / 2,
    )


def _solve_cone(model, options, deadline, warm_start, diagnostics):
    check_deadline(deadline)
    data, chain, inverse = model.problem.get_problem_data("SCS")
    check_deadline(deadline)
    settings = dict(
        SCS_SETTINGS,
        max_iters=options.sdr_scs_max_iters,
        eps_abs=options.sdr_scs_eps_abs,
        eps_rel=options.sdr_scs_eps_rel,
    )
    if deadline is not None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise SolverDeadline("SDR soft deadline reached")
        settings["time_limit_secs"] = remaining
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Solution may be inaccurate.*", category=UserWarning
        )
        diagnostics["sdr_sdp_solves"] += 1
        solution = chain.solve_via_data(
            model.problem, data, warm_start=warm_start, verbose=False, solver_opts=settings
        )
        if solution.get("info", {}).get("status_val") == -5:
            raise KeyboardInterrupt("SCS solve interrupted")
        model.problem.unpack_results(solution, chain, inverse)


def _diagnostics():
    counters = (
        "sdr_sdp_solves",
        "sdr_scs_iterations",
        "sdr_sdp_cap_hits",
        "sdr_sdp_inaccurate_count",
        "sdr_rank_successes",
        "sdr_rank_failures",
        "sdr_recovery_failures",
        "sdr_candidate_rejections",
        "sdr_original_sic_rejections",
        "sdr_objective_rejections",
        "sdr_constraint_rejections",
        "sdr_accepted_candidates",
        "sdr_effective_updates",
        "sdr_penalty_updates",
        "sdr_safe_fallbacks",
    )
    result = dict.fromkeys(counters, 0)
    result.update(
        sdr_sdp_status_counts={},
        sdr_sdp_status_val_counts={},
        sdr_block_status_counts={},
        sdr_rank_gap_abs=None,
        sdr_rank_gap_rel=None,
        sdr_lifted_objective=None,
        sdr_penalized_objective=None,
        sdr_lifted_residual_max=None,
        sdr_backend_status=None,
        sdr_modeling_time_s=0.0,
        sdr_solve_time_s=0.0,
        sdr_evd_time_s=0.0,
        sdr_backend_setup_time_s=0.0,
        sdr_backend_solve_time_s=0.0,
    )
    return result


def solve_phase_sdr(snapshot, power, phase, options, deadline=None):
    cp = require_sdr_backend()
    theta = wrap_phase(phase)
    initial = candidate_terms(snapshot, power, theta)
    if not initial["is_feasible"]:
        raise ValueError("SDR requires a feasible original power/phase pair")
    result = BlockResult(
        theta.copy(),
        initial["objective"],
        "max_iter_feasible",
        history=[] if options.debug_inner_history else None,
        diagnostics=_diagnostics(),
    )
    diag = result.diagnostics
    append_history(result.history, result.objective, options)
    _, _, _, base, threshold = snapshot.normalized()
    guard = options.sic_guard * np.maximum(1, base + threshold)
    model = None
    rho, stable = options.sdr_rank_penalty_init, 0
    restart_points = {max(1, options.sdr_max_iters // 3), max(1, 2 * options.sdr_max_iters // 3)}
    try:
        check_deadline(deadline)
        if np.any(initial["sic_margin_normalized"] < guard):
            result.status = "guarded_initializer_rejected"
            result.failures = 1
        else:
            begin = time.perf_counter()
            model = build_phase_problem(snapshot, power, theta, options)
            diag["sdr_modeling_time_s"] += time.perf_counter() - begin
            matrix = phase_matrix(theta)
            for iteration in range(options.sdr_max_iters):
                check_deadline(deadline)
                restart = iteration in restart_points and diag["sdr_accepted_candidates"] == 0
                if restart:
                    next_rho = min(
                        rho * options.sdr_rank_penalty_growth, options.sdr_rank_penalty_max
                    )
                    diag["sdr_penalty_updates"] += int(next_rho != rho)
                    rho = next_rho
                    initialize_problem(model, theta)
                    matrix = phase_matrix(theta)
                begin = time.perf_counter()
                vector = rank_decomposition(matrix)[0]
                diag["sdr_evd_time_s"] += time.perf_counter() - begin
                model.rank_weight.value = (
                    rho
                    * (np.eye(snapshot.g + 1) - np.outer(vector, vector.conj()))
                    / (snapshot.g + 1)
                )
                begin = time.perf_counter()
                result.iterations = iteration + 1
                try:
                    _solve_cone(
                        model,
                        options,
                        deadline,
                        options.sdr_warm_start and iteration > 0 and not restart,
                        diag,
                    )
                finally:
                    diag["sdr_solve_time_s"] += time.perf_counter() - begin
                status = model.problem.status
                diag["sdr_backend_status"] = status
                counts = diag["sdr_sdp_status_counts"]
                counts[status] = counts.get(status, 0) + 1
                stats = model.problem.solver_stats
                info = (stats.extra_stats or {}).get("info", {})
                code = info.get("status_val")
                if code is not None:
                    codes = diag["sdr_sdp_status_val_counts"]
                    key = str(code)
                    codes[key] = codes.get(key, 0) + 1
                diag["sdr_backend_message"] = info.get("status")
                for source, target in (
                    ("res_pri", "sdr_backend_primal_residual"),
                    ("res_dual", "sdr_backend_dual_residual"),
                    ("gap", "sdr_backend_gap"),
                ):
                    value = info.get(source)
                    diag[target] = (
                        float(value) if value is not None and np.isfinite(value) else None
                    )
                for source, target in (
                    ("setup_time", "sdr_backend_setup_time_s"),
                    ("solve_time", "sdr_backend_solve_time_s"),
                ):
                    value = info.get(source)
                    if value is not None and np.isfinite(value):
                        diag[target] += float(value) / 1000
                count = int(stats.num_iters or 0)
                diag["sdr_scs_iterations"] += count
                result.subproblem_iterations += count
                diag["sdr_sdp_cap_hits"] += int(count >= options.sdr_scs_max_iters)
                diag["sdr_sdp_inaccurate_count"] += int(status == cp.OPTIMAL_INACCURATE)
                check_deadline(deadline)
                if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    result.status, result.failures = "backend_failed", 1
                    break
                begin = time.perf_counter()
                checked = solution_residual(model)
                diag["sdr_evd_time_s"] += time.perf_counter() - begin
                if checked is not None:
                    diag["sdr_lifted_residual_max"] = max(
                        diag["sdr_lifted_residual_max"] or 0, checked["residual"]
                    )
                if (
                    checked is None
                    or checked["residual"] > options.feasibility_tol
                    or model.problem.value is None
                    or not np.isfinite(model.problem.value)
                ):
                    diag["sdr_constraint_rejections"] += 1
                    result.status, result.failures = "backend_inaccurate", 1
                    break
                diag.update(
                    sdr_rank_gap_abs=checked["gap_abs"],
                    sdr_rank_gap_rel=checked["gap_rel"],
                    sdr_lifted_objective=checked["lifted_objective"],
                    sdr_penalized_objective=float(model.problem.value),
                )
                matrix = checked["matrix"]
                if checked["gap_rel"] > options.sdr_rank_tol:
                    diag["sdr_rank_failures"] += 1
                    stable = 0
                    continue
                begin = time.perf_counter()
                try:
                    candidate = recover_phase(matrix, options.sdr_rank_tol, options.feasibility_tol)
                except ValueError:
                    diag["sdr_recovery_failures"] += 1
                    continue
                finally:
                    diag["sdr_evd_time_s"] += time.perf_counter() - begin
                check_deadline(deadline)
                diag["sdr_rank_successes"] += 1
                actual = candidate_terms(snapshot, power, candidate)
                previous, change = result.objective, np.linalg.norm(wrap_phase(candidate - theta))
                tolerance = options.feasibility_tol * max(1, abs(previous))
                if not actual["is_feasible"] or np.any(actual["sic_margin_normalized"] < guard):
                    diag["sdr_original_sic_rejections"] += 1
                elif actual["objective"] < previous - tolerance:
                    diag["sdr_objective_rejections"] += 1
                else:
                    diag["sdr_accepted_candidates"] += 1
                    relative = abs(actual["objective"] - previous) / max(1, abs(previous))
                    diag["sdr_effective_updates"] += int(
                        relative > options.sdr_tol or change > options.sdr_phase_tol
                    )
                    theta = candidate
                    result.value, result.objective = theta.copy(), actual["objective"]
                    append_history(result.history, result.objective, options)
                    stable = (
                        stable + 1
                        if (
                            relative <= options.sdr_tol
                            and change <= options.sdr_phase_tol
                            and status == cp.OPTIMAL
                        )
                        else 0
                    )
                    if stable >= options.stopping_patience:
                        result.status = "converged"
                        break
                    continue
                diag["sdr_candidate_rejections"] += 1
                stable = 0
            if diag["sdr_accepted_candidates"] == 0 and result.failures == 0:
                result.status = (
                    "rank_recovery_failed" if diag["sdr_rank_failures"] else "candidate_rejected"
                )
                result.failures = 1
    except SolverDeadline:
        result.status = "soft_deadline"
    except (cp.error.SolverError, FloatingPointError, np.linalg.LinAlgError):
        result.status, result.failures = "backend_failed", 1
    result.cap_hit = result.iterations == options.sdr_max_iters and result.status != "converged"
    diag["sdr_penalty_last"] = float(rho)
    diag["sdr_safe_fallbacks"] = int(result.failures > 0 or result.status == "soft_deadline")
    diag["sdr_block_status_counts"] = {result.status: 1}
    result.minimum_sic = float(
        np.min(candidate_terms(snapshot, power, result.value)["sic_margin_normalized"])
    )
    return result
