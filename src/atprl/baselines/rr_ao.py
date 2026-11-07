from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from atprl.baselines.round_robin import RoundRobinScheduler
from atprl.envs.actuator import physical_actuator_state
from atprl.solvers.optimization_context import snapshot_from_env, candidate_terms, wrap_phase
from atprl.solvers.mrt_initialization import initialize_grouped_mrt
from atprl.solvers.conventional_options import SolverOptions
from atprl.solvers.wmmse_constrained import initial_feasible_power, solve_rsma_power
from atprl.solvers.sca_phase import solve_phase_sca, certified_sic_infeasible


@dataclass
class DecisionResult:
    users: tuple[int, ...]
    power: np.ndarray
    phase: np.ndarray
    weights: np.ndarray
    status: str
    objective: float
    diagnostics: dict

    @property
    def action(self):
        return self.users, self.power.copy(), self.phase.copy(), self.weights.copy()


def optimize_decision(snapshot, options: SolverOptions, phase_solver="sca") -> DecisionResult:
    if phase_solver not in ("sca", "sdr", "mrt"):
        raise ValueError("Unknown phase solver")
    if phase_solver == "sdr":
        from atprl.solvers.sdr_phase import solve_phase_sdr, require_sdr_backend, _diagnostics

        require_sdr_backend()
        phase_function = solve_phase_sdr
    else:
        phase_function = solve_phase_sca
    begin = time.perf_counter()
    deadline = (
        None if options.decision_time_limit_s is None else begin + options.decision_time_limit_s
    )
    phase = initialize_grouped_mrt(snapshot)
    diagnostics = dict(
        ao_iterations=0,
        wmmse_iterations=0,
        sca_iterations=0,
        phase1_iterations=0,
        wmmse_blocks=0,
        sca_blocks=0,
        wmmse_cap_hits=0,
        sca_cap_hits=0,
        ao_cap_hits=0,
        wmmse_failures=0,
        sca_failures=0,
        phase1_failures=0,
        fallback_count=0,
        phase1_used=0,
        terminal_power_solves=0,
        phase1_cap_hits=0,
        outage_count=0,
    )
    trajectory = [] if options.debug_inner_history else None
    if phase_solver == "sdr":
        diagnostics.update(_diagnostics())
        diagnostics.update(
            sdr_iterations=0,
            sdr_blocks=0,
            sdr_cap_hits=0,
            sdr_failures=0,
            phase1_solver="sca",
            sdr_shared_phase1_used=0,
        )

    def absorb(block, kind):
        diagnostics[kind + "_iterations"] += block.iterations
        if kind in ("wmmse", "sca", "sdr"):
            diagnostics[kind + "_blocks"] += 1
            diagnostics[kind + "_cap_hits"] += int(block.cap_hit)
            diagnostics[kind + "_failures"] += block.failures
            if kind == "sdr" and block.diagnostics is not None:
                for key, value in block.diagnostics.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        diagnostics[key] = diagnostics.get(key, 0) + value
                    elif isinstance(value, dict):
                        counts = diagnostics.setdefault(key, {})
                        for status, count in value.items():
                            counts[status] = counts.get(status, 0) + count
                    elif key.endswith("_time_s"):
                        diagnostics[key] = diagnostics.get(key, 0.0) + value
                    elif key == "sdr_lifted_residual_max" and value is not None:
                        diagnostics[key] = max(diagnostics.get(key) or 0, value)
                    elif value is not None:
                        diagnostics[key] = value
        elif kind == "phase1":
            diagnostics["phase1_failures"] += block.failures
            diagnostics["phase1_cap_hits"] += int(block.cap_hit)
        diagnostics["fallback_count"] += int(block.failures > 0 or block.status == "soft_deadline")
        if block.history is not None:
            histories = diagnostics.setdefault(kind + "_objective_histories", [])
            if len(histories) < options.debug_history_limit:
                histories.append(block.history)

    def accept(power, theta, old_objective):
        terms = candidate_terms(snapshot, power, theta)
        return terms["is_feasible"] and terms[
            "objective"
        ] >= old_objective - options.feasibility_tol * max(1, abs(old_objective))

    power = initial_feasible_power(snapshot, phase, options)
    if power is None and phase_solver in ("sca", "sdr") and not certified_sic_infeasible(snapshot):
        diagnostics["phase1_used"] = 1
        if phase_solver == "sdr":
            diagnostics["sdr_shared_phase1_used"] = 1
        restoration = solve_phase_sca(
            snapshot,
            np.r_[snapshot.pmax, np.zeros(snapshot.k)],
            phase,
            options,
            deadline,
            phase1=True,
        )
        absorb(restoration, "phase1")
        phase = restoration.value
        power = initial_feasible_power(snapshot, phase, options)
    status = "max_iter_feasible"
    if power is None:
        power = np.r_[snapshot.pmax, np.zeros(snapshot.k)]
        status = (
            "certified_sic_infeasible"
            if certified_sic_infeasible(snapshot)
            else "feasibility_not_found"
        )
        diagnostics["outage_count"] = 1
        diagnostics["fallback_count"] += 1
        diagnostics["action_kind"] = "sic_outage_action"
    else:
        objective = candidate_terms(snapshot, power, phase)["objective"]
        stable = 0
        if phase_solver in ("sca", "sdr"):
            for iteration in range(options.ao_max_iters):
                previous_p, previous_th, previous_j = power.copy(), phase.copy(), objective
                block = solve_rsma_power(
                    snapshot, phase, options, initial_power=power, deadline=deadline
                )
                absorb(block, "wmmse")
                if accept(block.value, phase, objective):
                    power = block.value
                    objective = candidate_terms(snapshot, power, phase)["objective"]
                phase_block = phase_function(snapshot, power, phase, options, deadline)
                absorb(phase_block, phase_solver)
                if accept(power, phase_block.value, objective):
                    phase = phase_block.value
                    objective = candidate_terms(snapshot, power, phase)["objective"]
                diagnostics["ao_iterations"] = iteration + 1
                if trajectory is not None and len(trajectory) < options.debug_history_limit:
                    trajectory.append(objective)
                if block.status == "soft_deadline" or phase_block.status == "soft_deadline":
                    status = "soft_deadline"
                    break
                if block.failures or phase_block.failures:
                    status = "stalled_after_failure"
                    break
                improvement = abs(objective - previous_j) / max(1, abs(previous_j))
                power_change = np.linalg.norm(power - previous_p) / snapshot.pmax
                phase_change = np.linalg.norm(wrap_phase(phase - previous_th))
                stable = (
                    stable + 1
                    if (
                        improvement <= options.ao_tol
                        and power_change <= np.sqrt(options.ao_tol)
                        and phase_change
                        <= (
                            options.sdr_phase_tol
                            if phase_solver == "sdr"
                            else options.sca_phase_tol
                        )
                    )
                    else 0
                )
                if stable >= options.stopping_patience:
                    status = "converged"
                    break
            diagnostics["ao_cap_hits"] = int(status == "max_iter_feasible")
        terminal = solve_rsma_power(
            snapshot, phase, options, initial_power=power, deadline=deadline
        )
        diagnostics["terminal_power_solves"] = 1
        absorb(terminal, "wmmse")
        if accept(terminal.value, phase, objective):
            power = terminal.value
        if phase_solver == "mrt":
            status = terminal.status
        elif terminal.failures:
            status = "stalled_after_failure"
        elif terminal.status == "soft_deadline":
            status = "soft_deadline"
        diagnostics["action_kind"] = "verified_joint_pair"
    terms = candidate_terms(snapshot, power, phase)
    if diagnostics["outage_count"] == 0 and not terms["is_feasible"]:
        raise RuntimeError("Final joint pair lost feasibility")
    diagnostics.update(
        solver_status=status,
        predicted_objective=terms["objective"],
        minimum_sic_margin_watt=float(np.min(terms["sic_margin_watt"])),
        minimum_sic_margin_normalized=float(np.min(terms["sic_margin_normalized"])),
        power_excess_watt=float(terms["power_excess_watt"]),
        phase_excess=float(terms["phase_excess"]),
        ris_power_excess_watt=float(
            snapshot.actuator.output_power_watt - snapshot.actuator.output_limit_watt
        ),
        amplitude_excess=float(
            np.max(snapshot.actuator.group_amplitudes) - snapshot.actuator.max_amplitude
        ),
        is_predicted_sic_feasible=bool(np.all(terms["sic_margin_watt"] >= 0)),
        optimization_wall_time_s=time.perf_counter() - begin,
    )
    if trajectory is not None:
        diagnostics["ao_objective_history"] = trajectory
    return DecisionResult(
        snapshot.users,
        power.copy(),
        phase.copy(),
        snapshot.actuator.group_weights.copy(),
        status,
        terms["objective"],
        diagnostics,
    )


class RRAOPolicy:
    def __init__(self, n, k, options, phase_solver="sca", offset=0):
        self.scheduler = RoundRobinScheduler(n, k, offset)
        self.options, self.phase_solver = options, phase_solver

    def decide(self, env):
        actuator = physical_actuator_state(env)
        snapshot = snapshot_from_env(env, self.scheduler.peek(), actuator)
        return optimize_decision(snapshot, self.options, self.phase_solver)

    def commit(self):
        self.scheduler.commit()
