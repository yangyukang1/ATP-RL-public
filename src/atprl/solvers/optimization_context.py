from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from atprl.envs.actuator import RISActuatorState, readonly

ENV_RATE_EPS = 1e-12
FIELD_SOURCES = {
    "direct_hat": "ESTIMATED_CSI: h_sg_hat",
    "cascade_hat": "ESTIMATED_CSI: grouped products h_sr_hat * h_rg_hat, fixed actuator amplitude",
    "error_variance": "DERIVED_FROM_ALLOWED_INFORMATION: estimated-hop norms, public NMSE and path gains",
    "base_noise": "PUBLIC_CONFIG/ESTIMATE: thermal + estimated active noise + public mean BS interference",
    "actuator": "PUBLIC_LEGAL_ACTUATOR_STATE: fixed group amplitudes and weights",
    "pmax": "PUBLIC_CONFIG: satellite_max_power_watt",
    "sic_threshold": "PUBLIC_CONFIG: sic_detection_threshold_watt",
}


@dataclass(frozen=True)
class OptimizationSnapshot:
    users: tuple[int, ...]
    direct_hat: np.ndarray
    cascade_hat: np.ndarray
    error_variance: np.ndarray
    base_noise: np.ndarray
    pmax: float
    sic_threshold: float
    actuator: RISActuatorState

    def __post_init__(self):
        object.__setattr__(self, "users", tuple(self.users))
        for name in ("direct_hat", "cascade_hat", "error_variance", "base_noise"):
            dtype = np.complex128 if name.endswith("hat") else np.float64
            array = readonly(getattr(self, name), dtype)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Non-finite snapshot field {name}")
            object.__setattr__(self, name, array)
        k, g = len(self.users), len(self.actuator.group_amplitudes)
        if (
            k < 1
            or len(set(self.users)) != k
            or self.direct_hat.shape != (k,)
            or self.cascade_hat.shape != (k, g)
            or self.error_variance.shape != (k,)
            or self.base_noise.shape != (k,)
        ):
            raise ValueError("Snapshot dimensions or selected users are invalid")
        if (
            not np.isfinite(self.pmax)
            or self.pmax <= 0
            or not np.isfinite(self.sic_threshold)
            or self.sic_threshold < 0
            or np.any(self.base_noise <= 0)
            or np.any(self.error_variance < 0)
        ):
            raise ValueError("Invalid snapshot budgets/noise")

    @property
    def k(self):
        return len(self.users)

    @property
    def g(self):
        return self.cascade_hat.shape[1]

    def normalized(self):
        scale = self.base_noise + ENV_RATE_EPS
        factor = np.sqrt(self.pmax / scale)
        return (
            self.direct_hat * factor,
            self.cascade_hat * factor[:, None],
            self.pmax * self.error_variance / scale,
            self.base_noise / scale,
            self.sic_threshold / scale,
        )


def public_bs_interference(env, users) -> np.ndarray:
    """Use public geometry and mean interference, without instantaneous BS-channel access."""
    distance_km = np.maximum(
        np.linalg.norm(env.gs_positions[list(users)] - env.bs_positions[0], axis=1) / 1000, 1e-3
    )
    forced = env.config.get("force_bs_gain_db")
    if forced is not None:
        gain = np.full(len(users), 10 ** (float(forced) / 10))
    else:
        gain = 10 ** (-(120.9 + 37.6 * np.log10(distance_km)) / 10)
        gain *= np.exp(0.5 * (np.log(10) / 10) ** 2 * env.log_normal_shadowing_variance_dB)
    return env.terrestrial_cell_power_watt * gain


def snapshot_from_env(env, users, actuator: RISActuatorState) -> OptimizationSnapshot:
    """Read estimated CSI and public data after separate actuator preparation."""
    users = tuple(int(u) for u in users)
    if len(users) != env.num_gs_per_rb or any(u < 0 or u >= env.num_ground_stations for u in users):
        raise ValueError("Invalid selected users")
    if len(actuator.group_amplitudes) != len(env.ris_groups):
        raise ValueError("Actuator/environment group mismatch")
    direct = np.asarray(env.h_sg_hat)[list(users)]
    sr = np.asarray(env.h_sr_hat)
    rg = np.asarray(env.h_rg_hat)[list(users)]
    amplitude = np.zeros(sr.size)
    for i, group in enumerate(env.ris_groups):
        amplitude[group] = actuator.group_amplitudes[i]
    a2 = amplitude**2
    cascade = np.column_stack(
        [np.sum(rg[:, group] * sr[group] * amplitude[group], axis=1) for group in env.ris_groups]
    )
    nmse = float(env.csi_error_variance)
    vrg = nmse * np.asarray(env.large_scale_gain_ris_to_gs)[list(users)]
    vsr = nmse * env.large_scale_gain_sat_to_ris
    norm_rg = np.sum(np.abs(rg) ** 2 * a2, axis=1)
    norm_sr = float(np.sum(np.abs(sr) ** 2 * a2))
    error = nmse * np.asarray(env.large_scale_gain_sat_to_gs_per_user)[list(users)]
    error = error + vsr * norm_rg + vrg * norm_sr + vrg * vsr * np.sum(a2)
    active_noise = (
        env.active_ris_noise_power_watt * (norm_rg + vrg * np.sum(a2))
        if env.active_ris_enabled and not env.disable_ris_path
        else np.zeros(len(users))
    )
    if env.disable_ris_path:
        cascade = np.zeros_like(cascade)
        error = nmse * np.asarray(env.large_scale_gain_sat_to_gs_per_user)[list(users)]
    base = env.noise_power_watt + public_bs_interference(env, users) + active_noise
    return OptimizationSnapshot(
        users,
        direct,
        cascade,
        error,
        base,
        env.satellite_max_power_watt,
        env.sic_detection_threshold_watt,
        actuator,
    )


def candidate_terms(snapshot: OptimizationSnapshot, power, phase) -> dict:
    p, th = np.asarray(power, dtype=float), np.asarray(phase, dtype=float)
    if (
        p.shape != (snapshot.k + 1,)
        or th.shape != (snapshot.g,)
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(th))
    ):
        raise ValueError("Non-finite or incorrectly shaped action")
    if np.any(p < 0):
        raise ValueError("Negative power")
    gain = np.abs(snapshot.direct_hat + snapshot.cascade_hat @ np.exp(1j * th)) ** 2
    total, private = float(p.sum()), float(p[1:].sum())
    interference = snapshot.base_noise + total * snapshot.error_variance
    common = np.log1p(p[0] * gain / (private * gain + interference + ENV_RATE_EPS)) / np.log(2)
    rates = np.log1p(
        p[1:] * gain / ((private - p[1:]) * gain + interference + ENV_RATE_EPS)
    ) / np.log(2)
    sic = (p[0] - private) * gain - interference - snapshot.sic_threshold
    return dict(
        objective=float(common.min() + rates.sum()),
        common_rate=float(common.min()),
        private_rates=rates,
        sic_margin_watt=sic,
        sic_margin_normalized=sic / (snapshot.base_noise + ENV_RATE_EPS),
        power_excess_watt=total - snapshot.pmax,
        phase_excess=float(np.max(np.abs(th)) - np.pi),
        is_feasible=bool(
            np.all(sic >= 0) and total <= snapshot.pmax and np.all(np.abs(th) <= np.pi)
        ),
    )


def wrap_phase(phase):
    return (np.asarray(phase, dtype=float) + np.pi) % (2 * np.pi) - np.pi
