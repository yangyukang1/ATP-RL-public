from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import time
import numpy as np


@dataclass(frozen=True)
class SolverOptions:
    ao_max_iters: int = 3
    ao_tol: float = 1e-5
    wmmse_max_iters: int = 10
    wmmse_tol: float = 1e-6
    wmmse_scipy_maxiter: int = 500
    wmmse_scipy_ftol: float = 1e-8
    sca_max_iters: int = 10
    sca_tol: float = 1e-6
    sca_phase_tol: float = 1e-3
    sca_trust_radius: float = 0.5
    sca_scipy_maxiter: int = 300
    sca_scipy_ftol: float = 1e-8
    phase1_max_iters: int = 10
    sic_guard: float = 1e-6
    feasibility_tol: float = 1e-7
    stopping_patience: int = 2
    decision_time_limit_s: Optional[float] = None
    debug_inner_history: bool = False
    debug_history_limit: int = 64
    sdr_solver: str = "SCS"
    sdr_max_iters: int = 15
    sdr_tol: float = 1e-6
    sdr_phase_tol: float = 1e-3
    sdr_rank_tol: float = 1e-5
    sdr_rank_penalty_init: float = 1.0
    sdr_rank_penalty_growth: float = 10.0
    sdr_rank_penalty_max: float = 100.0
    sdr_scs_max_iters: int = 20000
    sdr_scs_eps_abs: float = 1e-8
    sdr_scs_eps_rel: float = 1e-8
    sdr_warm_start: bool = True

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in (
                "debug_inner_history",
                "decision_time_limit_s",
                "sdr_solver",
                "sdr_warm_start",
            ):
                continue
            if key.startswith("sdr_") and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise ValueError(f"{key} must be numeric")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
            if key.endswith(("iters", "maxiter", "patience", "limit")) and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"{key} must be an integer")
        if self.sca_trust_radius > np.pi:
            raise ValueError("Trust radius cannot exceed pi")
        if self.sdr_solver != "SCS" or type(self.sdr_warm_start) is not bool:
            raise ValueError("SDR requires SCS and a boolean warm-start option")
        if self.sdr_phase_tol > np.pi or self.sdr_rank_tol >= 1:
            raise ValueError("Invalid SDR phase/rank tolerance")
        if (
            self.sdr_rank_penalty_growth <= 1
            or self.sdr_rank_penalty_max < self.sdr_rank_penalty_init
        ):
            raise ValueError("Invalid SDR rank penalty schedule")
        if self.decision_time_limit_s is not None and (
            not np.isfinite(self.decision_time_limit_s) or self.decision_time_limit_s <= 0
        ):
            raise ValueError("Soft deadline must be positive or None")


class SolverDeadline(RuntimeError):
    pass


def check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise SolverDeadline("Decision soft deadline reached")


@dataclass
class BlockResult:
    value: np.ndarray
    objective: float
    status: str
    iterations: int = 0
    subproblem_iterations: int = 0
    failures: int = 0
    cap_hit: bool = False
    minimum_sic: float = 0.0
    history: Optional[list[float]] = None
    diagnostics: Optional[dict] = None


def append_history(history, value, options):
    if history is not None and len(history) < options.debug_history_limit:
        history.append(float(value))
