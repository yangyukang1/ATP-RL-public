from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.optimize import minimize


_EPS = 1e-12


_REQUIRED_CTX_KEYS = (
    "served_users",
    "rb_index",
    "h_sg_hat_sel",
    "h_sr_hat",
    "h_rg_hat_sel",
    "ris_phase_response",
    "ls_sg_sel",
    "ls_sr",
    "ls_rg_sel",
    "g_bs_perfect_sel",
    "noise_power_watt",
    "active_ris_enabled",
    "active_ris_noise_power_watt",
    "satellite_max_power_watt",
    "terrestrial_cell_power_watt",
    "sic_detection_threshold_watt",
    "csi_error_variance",
)


def _as_float_array(x: Any, *, ndim: Optional[int] = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape={arr.shape}.")
    return arr


def _as_complex_array(x: Any, *, ndim: Optional[int] = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape={arr.shape}.")
    return arr


def _validate_wmmse_context(wmmse_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(wmmse_ctx, dict):
        raise TypeError(
            "wmmse_ctx must be a dict exported by env.export_wmmse_context_for_rb(...)."
        )

    missing = [k for k in _REQUIRED_CTX_KEYS if k not in wmmse_ctx]
    if missing:
        raise KeyError("wmmse_ctx missing required keys: " + ", ".join(missing))

    ctx: Dict[str, Any] = dict(wmmse_ctx)
    ctx["served_users"] = np.asarray(ctx["served_users"], dtype=np.int64).reshape(-1)
    ctx["h_sg_hat_sel"] = _as_complex_array(ctx["h_sg_hat_sel"], ndim=1, name="h_sg_hat_sel")
    ctx["h_sr_hat"] = _as_complex_array(ctx["h_sr_hat"], ndim=1, name="h_sr_hat")
    ctx["h_rg_hat_sel"] = _as_complex_array(ctx["h_rg_hat_sel"], ndim=2, name="h_rg_hat_sel")
    ctx["ris_phase_response"] = _as_complex_array(
        ctx["ris_phase_response"], ndim=1, name="ris_phase_response"
    )
    ctx["ls_sg_sel"] = _as_float_array(ctx["ls_sg_sel"], ndim=1, name="ls_sg_sel")
    ctx["ls_rg_sel"] = _as_float_array(ctx["ls_rg_sel"], ndim=1, name="ls_rg_sel")
    ctx["g_bs_perfect_sel"] = _as_float_array(
        ctx["g_bs_perfect_sel"], ndim=1, name="g_bs_perfect_sel"
    )

    ctx["ls_sr"] = float(ctx["ls_sr"])
    ctx["noise_power_watt"] = float(ctx["noise_power_watt"])
    ctx["satellite_max_power_watt"] = float(ctx["satellite_max_power_watt"])
    ctx["terrestrial_cell_power_watt"] = float(ctx["terrestrial_cell_power_watt"])
    ctx["sic_detection_threshold_watt"] = float(ctx["sic_detection_threshold_watt"])
    ctx["csi_error_variance"] = float(ctx["csi_error_variance"])
    ctx["active_ris_enabled"] = bool(ctx["active_ris_enabled"])
    ctx["active_ris_noise_power_watt"] = float(ctx["active_ris_noise_power_watt"])

    k = ctx["served_users"].size
    m = ctx["ris_phase_response"].size
    if ctx["h_sg_hat_sel"].shape != (k,):
        raise ValueError(
            f"h_sg_hat_sel shape mismatch: expected ({k},), got {ctx['h_sg_hat_sel'].shape}"
        )
    if ctx["h_sr_hat"].shape != (m,):
        raise ValueError(f"h_sr_hat shape mismatch: expected ({m},), got {ctx['h_sr_hat'].shape}")
    if ctx["h_rg_hat_sel"].shape != (k, m):
        raise ValueError(
            f"h_rg_hat_sel shape mismatch: expected ({k}, {m}), got {ctx['h_rg_hat_sel'].shape}"
        )
    if ctx["ls_sg_sel"].shape != (k,):
        raise ValueError(f"ls_sg_sel shape mismatch: expected ({k},), got {ctx['ls_sg_sel'].shape}")
    if ctx["ls_rg_sel"].shape != (k,):
        raise ValueError(f"ls_rg_sel shape mismatch: expected ({k},), got {ctx['ls_rg_sel'].shape}")
    if ctx["g_bs_perfect_sel"].shape != (k,):
        raise ValueError(
            f"g_bs_perfect_sel shape mismatch: expected ({k},), got {ctx['g_bs_perfect_sel'].shape}"
        )
    return ctx


def build_wmmse_context_from_env(
    env: Any, user_indices: Sequence[int], rb_index: int = 0
) -> Dict[str, Any]:
    if hasattr(env, "export_wmmse_context_for_rb"):
        ctx = env.export_wmmse_context_for_rb(list(user_indices), int(rb_index), copy_arrays=True)
        return _validate_wmmse_context(ctx)

    served_arr = np.asarray(list(user_indices), dtype=np.int64)
    if served_arr.size == 0:
        raise ValueError("user_indices must be non-empty.")
    rb_index = int(rb_index)

    required_attrs = [
        "ris_phase_response",
        "ris_phase_vector",
        "ris_group_phase_vector",
        "h_sg_hat",
        "h_sr_hat",
        "h_rg_hat",
        "large_scale_gain_sat_to_gs_per_user",
        "large_scale_gain_sat_to_ris",
        "large_scale_gain_ris_to_gs",
        "large_scale_gain_bs_to_gs",
        "small_scale_coeffs_bs_to_gs",
        "noise_power_watt",
        "satellite_max_power_watt",
        "terrestrial_cell_power_watt",
        "sic_detection_threshold_watt",
        "csi_error_variance",
        "active_ris_enabled",
        "active_ris_noise_power_watt",
    ]
    for name in required_attrs:
        if not hasattr(env, name):
            raise AttributeError(
                f"env is missing required attribute '{name}', and no export_wmmse_context_for_rb() method was found."
            )

    g_bs_full = np.asarray(env.large_scale_gain_bs_to_gs, dtype=np.float64) * (
        np.abs(np.asarray(env.small_scale_coeffs_bs_to_gs, dtype=np.complex128)) ** 2
    )
    h_eff_hat_all = np.asarray(
        env._compute_effective_satellite_channel_hat(), dtype=np.complex128
    ).reshape(-1)
    active_ris_noise_all = np.asarray(
        env._compute_active_ris_noise_per_user(), dtype=np.float64
    ).reshape(-1)

    ctx = {
        "served_users": served_arr.copy(),
        "rb_index": int(rb_index),
        "num_served_users": int(served_arr.size),
        "num_total_users": int(getattr(env, "num_ground_stations")),
        "num_resource_blocks": int(getattr(env, "num_resource_blocks")),
        "num_ris_elems": int(getattr(env, "ris_num_elems")),
        "num_ris_groups": int(getattr(env, "ris_num_groups")),
        "ris_phase_response": np.asarray(env.ris_phase_response, dtype=np.complex128).copy(),
        "ris_phase_vector": np.asarray(env.ris_phase_vector, dtype=np.float64).copy(),
        "ris_group_phase_vector": np.asarray(env.ris_group_phase_vector, dtype=np.float64).copy(),
        "h_sg_hat_sel": np.asarray(env.h_sg_hat, dtype=np.complex128)
        .reshape(-1)[served_arr]
        .copy(),
        "h_sr_hat": np.asarray(env.h_sr_hat, dtype=np.complex128).reshape(-1).copy(),
        "h_rg_hat_sel": np.asarray(env.h_rg_hat, dtype=np.complex128)[served_arr, :].copy(),
        "h_eff_hat_sel": h_eff_hat_all[served_arr].copy(),
        "g_eff_hat_sel": (np.abs(h_eff_hat_all[served_arr]) ** 2).astype(np.float64),
        "ls_sg_sel": np.asarray(env.large_scale_gain_sat_to_gs_per_user, dtype=np.float64)
        .reshape(-1)[served_arr]
        .copy(),
        "ls_sr": float(env.large_scale_gain_sat_to_ris),
        "ls_rg_sel": np.asarray(env.large_scale_gain_ris_to_gs, dtype=np.float64)
        .reshape(-1)[served_arr]
        .copy(),
        "g_bs_perfect_sel": np.asarray(g_bs_full[served_arr, rb_index], dtype=np.float64).copy(),
        "noise_power_watt": float(env.noise_power_watt),
        "satellite_max_power_watt": float(env.satellite_max_power_watt),
        "terrestrial_cell_power_watt": float(env.terrestrial_cell_power_watt),
        "sic_detection_threshold_watt": float(env.sic_detection_threshold_watt),
        "csi_error_variance": float(env.csi_error_variance),
        "active_ris_enabled": bool(env.active_ris_enabled),
        "active_ris_noise_power_watt": float(env.active_ris_noise_power_watt),
        "disable_ris_path": bool(getattr(env, "disable_ris_path", False)),
        "active_ris_noise_sel": active_ris_noise_all[served_arr].copy(),
        "current_step": int(getattr(env, "current_step", 0)),
        "rho": float(getattr(env, "rho", 0.0)),
    }
    return _validate_wmmse_context(ctx)


def _build_effective_channel_from_ctx(ctx: Dict[str, Any]) -> np.ndarray:
    """Combine direct and cascaded estimated channels without conjugating the latter."""
    h_sg_hat = ctx["h_sg_hat_sel"]
    h_sr_hat = ctx["h_sr_hat"]
    h_rg_hat = ctx["h_rg_hat_sel"]
    psi = ctx["ris_phase_response"]

    if bool(ctx.get("disable_ris_path", False)):
        return h_sg_hat.copy()

    ris_term = np.sum((psi[None, :] * h_sr_hat[None, :]) * h_rg_hat, axis=1)
    return h_sg_hat + ris_term


def _compute_active_ris_noise_from_ctx(ctx: Dict[str, Any]) -> np.ndarray:
    if bool(ctx.get("disable_ris_path", False)):
        return np.zeros(ctx["served_users"].shape[0], dtype=np.float64)
    if (not bool(ctx["active_ris_enabled"])) or (float(ctx["active_ris_noise_power_watt"]) <= 0.0):
        return np.zeros(ctx["served_users"].shape[0], dtype=np.float64)

    psi = ctx["ris_phase_response"]
    abs_psi_sq = np.abs(psi) ** 2
    phi_fro_sq = float(np.sum(abs_psi_sq))

    h_rg_hat = ctx["h_rg_hat_sel"]
    nmse = float(ctx["csi_error_variance"])
    sigma_e_rg = nmse * ctx["ls_rg_sel"]

    proj_sq = np.sum((np.abs(h_rg_hat) ** 2) * abs_psi_sq[None, :], axis=1)
    return float(ctx["active_ris_noise_power_watt"]) * (proj_sq + sigma_e_rg * phi_fro_sq)


def _compute_eeq2_from_ctx(ctx: Dict[str, Any]) -> np.ndarray:
    psi = ctx["ris_phase_response"]
    abs_psi_sq = np.abs(psi) ** 2
    phi_fro_sq = float(np.sum(abs_psi_sq))

    nmse = float(ctx["csi_error_variance"])
    var_sg = nmse * ctx["ls_sg_sel"]
    var_sr = nmse * float(ctx["ls_sr"])
    var_rg = nmse * ctx["ls_rg_sel"]

    h_sr_hat = ctx["h_sr_hat"]
    h_rg_hat = ctx["h_rg_hat_sel"]

    norm_hrg_phi_sq = np.sum((np.abs(h_rg_hat) ** 2) * abs_psi_sq[None, :], axis=1)
    norm_phi_hsr_sq = float(np.sum(abs_psi_sq * (np.abs(h_sr_hat) ** 2)))

    eeq2 = (
        var_sg
        + (var_sr * norm_hrg_phi_sq)
        + (var_rg * norm_phi_hsr_sq)
        + (var_rg * var_sr * phi_fro_sq)
    )
    return np.asarray(eeq2, dtype=np.float64)


def _compute_env_consistent_terms(
    ctx: Dict[str, Any], power_vec: np.ndarray
) -> Dict[str, np.ndarray | float]:
    power_vec = np.asarray(power_vec, dtype=np.float64).reshape(-1)
    k = ctx["served_users"].size
    if power_vec.shape != (k + 1,):
        raise ValueError(f"power_vec must have shape ({k + 1},), got {power_vec.shape}")

    p_common = float(power_vec[0])
    p_priv = np.maximum(power_vec[1:], 0.0)
    p_common = max(p_common, 0.0)

    h_eff_hat = _build_effective_channel_from_ctx(ctx)
    g_sat = np.abs(h_eff_hat) ** 2

    active_ris_noise = _compute_active_ris_noise_from_ctx(ctx)
    eeq2 = _compute_eeq2_from_ctx(ctx)

    external_interference = float(ctx["terrestrial_cell_power_watt"]) * ctx["g_bs_perfect_sel"]

    total_private_tx_power = float(np.sum(p_priv))
    total_stream_power = float(p_common + total_private_tx_power)

    I_csi = total_stream_power * eeq2
    I_tot = external_interference + float(ctx["noise_power_watt"]) + active_ris_noise + I_csi

    common_rx_power = p_common * g_sat
    private_rx_powers = p_priv * g_sat
    total_private_rx_power_at_user = total_private_tx_power * g_sat
    internal_interference = (total_private_tx_power - p_priv) * g_sat

    sinr_common = common_rx_power / np.maximum(total_private_rx_power_at_user + I_tot, _EPS)
    sinr_private = private_rx_powers / np.maximum(internal_interference + I_tot, _EPS)

    common_rate = float(np.min(np.log2(1.0 + sinr_common))) if k > 0 else 0.0
    private_rates = np.log2(1.0 + sinr_private)
    sum_rate_objective = float(np.sum(private_rates) + common_rate)

    sic_margin = (
        common_rx_power
        - total_private_rx_power_at_user
        - I_tot
        - float(ctx["sic_detection_threshold_watt"])
    )
    is_feasible = bool(np.all(sic_margin >= 0.0))

    env_gated_common_rate = common_rate if is_feasible else 0.0
    env_gated_private_rates = private_rates.copy() if is_feasible else np.zeros_like(private_rates)
    env_gated_total_rates = (env_gated_common_rate / max(k, 1)) + env_gated_private_rates

    return {
        "h_eff_hat": h_eff_hat,
        "g_sat": g_sat,
        "eeq2": eeq2,
        "active_ris_noise": active_ris_noise,
        "external_interference": external_interference,
        "I_csi": I_csi,
        "I_tot": I_tot,
        "common_rx_power": common_rx_power,
        "private_rx_powers": private_rx_powers,
        "total_private_rx_power_at_user": total_private_rx_power_at_user,
        "internal_interference": internal_interference,
        "sinr_common": sinr_common,
        "sinr_private": sinr_private,
        "common_rate": common_rate,
        "private_rates": private_rates,
        "sum_rate_objective": sum_rate_objective,
        "sic_margin": sic_margin,
        "is_feasible": is_feasible,
        "env_gated_common_rate": env_gated_common_rate,
        "env_gated_private_rates": env_gated_private_rates,
        "env_gated_total_rates": env_gated_total_rates,
        "p_common": p_common,
        "p_priv": p_priv,
        "total_stream_power": total_stream_power,
        "total_private_tx_power": total_private_tx_power,
    }


def _compute_mmse_receivers(
    h_eff_hat: np.ndarray, eeq2: np.ndarray, base_noise: np.ndarray, power_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p_common = float(power_vec[0])
    p_priv = np.asarray(power_vec[1:], dtype=np.float64)
    v_common = np.sqrt(max(p_common, 0.0))
    v_priv = np.sqrt(np.maximum(p_priv, 0.0))

    g_sat = np.abs(h_eff_hat) ** 2
    total_priv = float(np.sum(p_priv))
    total_power = float(p_common + total_priv)

    T_c = (g_sat + eeq2) * total_power + base_noise
    T_p = g_sat * total_priv + eeq2 * total_power + base_noise

    u_c = np.conjugate(h_eff_hat) * v_common / np.maximum(T_c, _EPS)
    u_p = np.conjugate(h_eff_hat) * v_priv / np.maximum(T_p, _EPS)
    return u_c, u_p


def _compute_mmse_mse(
    h_eff_hat: np.ndarray,
    eeq2: np.ndarray,
    base_noise: np.ndarray,
    power_vec: np.ndarray,
    u_c: np.ndarray,
    u_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_common = float(power_vec[0])
    p_priv = np.asarray(power_vec[1:], dtype=np.float64)
    v_common = np.sqrt(max(p_common, 0.0))
    v_priv = np.sqrt(np.maximum(p_priv, 0.0))

    g_sat = np.abs(h_eff_hat) ** 2
    total_priv = float(np.sum(p_priv))
    total_power = float(p_common + total_priv)

    T_c = (g_sat + eeq2) * total_power + base_noise
    T_p = g_sat * total_priv + eeq2 * total_power + base_noise

    a_c = np.real(u_c * h_eff_hat)
    a_p = np.real(u_p * h_eff_hat)

    e_c = 1.0 - 2.0 * a_c * v_common + (np.abs(u_c) ** 2) * T_c
    e_p = 1.0 - 2.0 * a_p * v_priv + (np.abs(u_p) ** 2) * T_p

    return np.maximum(e_c, _EPS), np.maximum(e_p, _EPS)


def _compute_mmse_weights(e_c: np.ndarray, e_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 1.0 / np.maximum(e_c, _EPS), 1.0 / np.maximum(e_p, _EPS)


def _awmmse_constants(
    h_eff_hat: np.ndarray,
    eeq2: np.ndarray,
    base_noise: np.ndarray,
    u_c: np.ndarray,
    u_p: np.ndarray,
    w_c: np.ndarray,
    w_p: np.ndarray,
) -> Dict[str, np.ndarray]:
    g_sat = np.abs(h_eff_hat) ** 2

    a_c = np.real(u_c * h_eff_hat)
    a_p = np.real(u_p * h_eff_hat)

    abs_uc_sq = np.abs(u_c) ** 2
    abs_up_sq = np.abs(u_p) ** 2

    const_c = w_c * (1.0 + abs_uc_sq * base_noise) - np.log(np.maximum(w_c, _EPS))
    const_p = w_p * (1.0 + abs_up_sq * base_noise) - np.log(np.maximum(w_p, _EPS))

    A_c = w_c * abs_uc_sq * (g_sat + eeq2)
    B_c = 2.0 * w_c * a_c

    A_pc = w_p * abs_up_sq * eeq2
    A_pp = w_p * abs_up_sq * (g_sat + eeq2)
    B_p = 2.0 * w_p * a_p

    return {
        "const_c": const_c,
        "const_p": const_p,
        "A_c": A_c,
        "B_c": B_c,
        "A_pc": A_pc,
        "A_pp": A_pp,
        "B_p": B_p,
    }


def _common_xi_from_v(v: np.ndarray, consts: Dict[str, np.ndarray]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    vc = float(v[0])
    sum_all_sq = float(np.sum(v**2))
    return consts["const_c"] + consts["A_c"] * sum_all_sq - consts["B_c"] * vc


def _private_xi_from_v(v: np.ndarray, consts: Dict[str, np.ndarray]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    vc = float(v[0])
    vp = v[1:]
    sum_priv_sq = float(np.sum(vp**2))

    xi_p = (
        consts["const_p"]
        + consts["A_pc"] * (vc**2)
        + consts["A_pp"] * sum_priv_sq
        - consts["B_p"] * vp
    )
    return xi_p


def _solve_awmmse_tx_subproblem_slsqp(
    consts: Dict[str, np.ndarray],
    v0: np.ndarray,
    pmax: float,
    scipy_method: str = "SLSQP",
    scipy_options: Optional[Dict[str, Any]] = None,
) -> tuple[np.ndarray, float, bool, str]:
    v0 = np.asarray(v0, dtype=np.float64).reshape(-1)
    k = v0.size - 1
    if k <= 0:
        raise ValueError("v0 must contain [v_common, v_privates...].")

    xi_c0 = _common_xi_from_v(v0, consts)
    t0 = float(np.max(xi_c0))
    x0 = np.concatenate([v0, np.array([t0], dtype=np.float64)])

    sum_A_pc = float(np.sum(consts["A_pc"]))
    sum_A_pp = float(np.sum(consts["A_pp"]))
    B_p = consts["B_p"]
    const_sum_p = float(np.sum(consts["const_p"]))

    def objective(x: np.ndarray) -> float:
        v = x[:-1]
        t = float(x[-1])
        vc = float(v[0])
        vp = v[1:]
        sum_priv_sq = float(np.sum(vp**2))
        val = const_sum_p + sum_A_pc * (vc**2) + sum_A_pp * sum_priv_sq - float(np.dot(B_p, vp)) + t
        return float(val)

    def objective_jac(x: np.ndarray) -> np.ndarray:
        v = x[:-1]
        vc = float(v[0])
        vp = v[1:]
        grad = np.zeros_like(x)
        grad[0] = 2.0 * sum_A_pc * vc
        grad[1:-1] = 2.0 * sum_A_pp * vp - B_p
        grad[-1] = 1.0
        return grad

    constraints = []

    def power_constr(x: np.ndarray) -> float:
        v = x[:-1]
        return float(pmax - np.sum(v**2))

    def power_constr_jac(x: np.ndarray) -> np.ndarray:
        v = x[:-1]
        grad = np.zeros_like(x)
        grad[:-1] = -2.0 * v
        return grad

    constraints.append({"type": "ineq", "fun": power_constr, "jac": power_constr_jac})

    for kk in range(k):

        def c_fun(x: np.ndarray, kk: int = kk) -> float:
            v = x[:-1]
            t = float(x[-1])
            xi_c = _common_xi_from_v(v, consts)
            return float(t - xi_c[kk])

        def c_jac(x: np.ndarray, kk: int = kk) -> np.ndarray:
            v = x[:-1]
            vc = float(v[0])
            grad = np.zeros_like(x)
            grad[0] = -(2.0 * consts["A_c"][kk] * vc - consts["B_c"][kk])
            grad[1:-1] = -(2.0 * consts["A_c"][kk] * v[1:])
            grad[-1] = 1.0
            return grad

        constraints.append({"type": "ineq", "fun": c_fun, "jac": c_jac})

    bounds = [(0.0, None)] * (k + 1) + [(None, None)]
    options = {"maxiter": 200, "ftol": 1e-10, "disp": False}
    if scipy_options:
        options.update(dict(scipy_options))

    result = minimize(
        objective,
        x0,
        jac=objective_jac,
        method=scipy_method,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )

    x_star = np.asarray(result.x if result.x is not None else x0, dtype=np.float64)
    v_star = np.maximum(x_star[:-1], 0.0)
    obj_star = float(objective(np.concatenate([v_star, np.array([x_star[-1]], dtype=np.float64)])))
    return v_star, obj_star, bool(result.success), str(result.message)


def _check_sic_feasibility(
    ctx: Dict[str, Any], power_vec: np.ndarray
) -> tuple[bool, np.ndarray, Dict[str, Any]]:
    terms = _compute_env_consistent_terms(ctx, power_vec)
    sic_margin = np.asarray(terms["sic_margin"], dtype=np.float64)
    return bool(np.all(sic_margin >= 0.0)), sic_margin, terms


def _apply_conservative_sic_repair(
    ctx: Dict[str, Any],
    power_vec: np.ndarray,
    max_iters: int = 30,
    step_ratio: float = 0.05,
) -> tuple[np.ndarray, bool, Dict[str, Any], Dict[str, Any]]:
    power_vec = np.asarray(power_vec, dtype=np.float64).reshape(-1)
    candidate = np.maximum(power_vec, 0.0)
    pmax = float(ctx["satellite_max_power_watt"])
    if np.sum(candidate) > pmax + 1e-9:
        candidate = candidate * (pmax / max(np.sum(candidate), _EPS))

    history = {
        "repair_power_history": [candidate.copy()],
        "repair_sic_margin_history": [],
        "repair_feasibility_history": [],
    }

    last_feasible = None
    last_terms = None
    for _ in range(int(max_iters)):
        feasible, sic_margin, terms = _check_sic_feasibility(ctx, candidate)
        history["repair_sic_margin_history"].append(sic_margin.copy())
        history["repair_feasibility_history"].append(bool(feasible))
        if feasible:
            last_feasible = candidate.copy()
            last_terms = terms
            break

        p_common = float(candidate[0])
        p_priv = np.asarray(candidate[1:], dtype=np.float64)
        priv_total = float(np.sum(p_priv))
        if priv_total <= _EPS:
            break

        shift = min(float(step_ratio) * priv_total, priv_total)
        scale = max((priv_total - shift) / max(priv_total, _EPS), 0.0)

        candidate[0] = p_common + shift
        candidate[1:] = p_priv * scale
        history["repair_power_history"].append(candidate.copy())

    if last_feasible is not None and last_terms is not None:
        return last_feasible, True, last_terms, history

    feasible, _, terms = _check_sic_feasibility(ctx, candidate)
    return candidate, bool(feasible), terms, history


def solve_wmmse_power(
    wmmse_ctx: Optional[Dict[str, Any]] = None,
    *,
    env: Optional[Any] = None,
    user_indices: Optional[Sequence[int]] = None,
    rb_index: int = 0,
    max_wmmse_iters: int = 50,
    tol: float = 1e-6,
    scipy_method: str = "SLSQP",
    scipy_options: Optional[Dict[str, Any]] = None,
    sic_repair_max_iters: int = 30,
    sic_repair_step_ratio: float = 0.05,
) -> Dict[str, Any]:
    """Solve RSMA power allocation using the current exported channel context."""
    if wmmse_ctx is None:
        if env is None or user_indices is None:
            raise ValueError(
                "Either provide wmmse_ctx directly, or provide env + user_indices so the solver can call "
                "env.export_wmmse_context_for_rb(...)."
            )
        ctx = build_wmmse_context_from_env(env, user_indices, rb_index)
    else:
        ctx = _validate_wmmse_context(wmmse_ctx)

    k = int(ctx["served_users"].size)
    if k <= 0:
        raise ValueError("WMMSE requires at least one served user.")

    pmax = float(ctx["satellite_max_power_watt"])
    base_noise = (
        float(ctx["noise_power_watt"])
        + float(ctx["terrestrial_cell_power_watt"]) * ctx["g_bs_perfect_sel"]
        + _compute_active_ris_noise_from_ctx(ctx)
    )
    h_eff_hat = _build_effective_channel_from_ctx(ctx)
    eeq2 = _compute_eeq2_from_ctx(ctx)

    p_init = np.full((k + 1,), pmax / float(k + 1), dtype=np.float64)
    v = np.sqrt(np.maximum(p_init, 0.0))
    power_vec = p_init.copy()

    objective_history = []
    awmmse_history = []
    power_history = [power_vec.copy()]
    feasibility_history = []
    sic_margin_history = []
    subproblem_success_history = []
    subproblem_message_history = []

    last_feasible_power = None
    last_feasible_terms = None
    solver_status = "max_iter_infeasible"

    for it in range(int(max_wmmse_iters)):
        prev_power = power_vec.copy()

        terms = _compute_env_consistent_terms(ctx, power_vec)
        objective_history.append(float(terms["sum_rate_objective"]))
        feasibility_history.append(bool(terms["is_feasible"]))
        sic_margin_history.append(np.asarray(terms["sic_margin"], dtype=np.float64).copy())
        if bool(terms["is_feasible"]):
            last_feasible_power = power_vec.copy()
            last_feasible_terms = terms

        u_c, u_p = _compute_mmse_receivers(h_eff_hat, eeq2, base_noise, power_vec)
        e_c, e_p = _compute_mmse_mse(h_eff_hat, eeq2, base_noise, power_vec, u_c, u_p)
        w_c, w_p = _compute_mmse_weights(e_c, e_p)
        consts = _awmmse_constants(h_eff_hat, eeq2, base_noise, u_c, u_p, w_c, w_p)

        xi_c_cur = _common_xi_from_v(v, consts)
        xi_p_cur = _private_xi_from_v(v, consts)
        awmmse_history.append(float(np.sum(xi_p_cur) + np.max(xi_c_cur)))

        v_new, _, sub_ok, sub_msg = _solve_awmmse_tx_subproblem_slsqp(
            consts=consts,
            v0=v,
            pmax=pmax,
            scipy_method=scipy_method,
            scipy_options=scipy_options,
        )
        subproblem_success_history.append(bool(sub_ok))
        subproblem_message_history.append(str(sub_msg))

        if not sub_ok:
            solver_status = "scipy_subproblem_failed"
            break

        power_vec = np.maximum(v_new, 0.0) ** 2
        total_power = float(np.sum(power_vec))
        if total_power > pmax + 1e-9:
            power_vec *= pmax / max(total_power, _EPS)
            v_new = np.sqrt(np.maximum(power_vec, 0.0))

        v = v_new
        power_history.append(power_vec.copy())

        if np.linalg.norm(power_vec - prev_power) < float(tol):
            solver_status = "converged"
            break

    final_terms = _compute_env_consistent_terms(ctx, power_vec)
    if bool(final_terms["is_feasible"]):
        solver_status = (
            "converged_feasible"
            if solver_status == "converged"
            else "max_iter_feasible"
            if solver_status.startswith("max_iter")
            else solver_status
        )
        final_power = power_vec.copy()
        final_is_feasible = True
    else:
        repaired_power, repaired_ok, repaired_terms, repair_debug = _apply_conservative_sic_repair(
            ctx,
            power_vec,
            max_iters=sic_repair_max_iters,
            step_ratio=sic_repair_step_ratio,
        )
        power_vec = repaired_power
        final_terms = repaired_terms
        final_is_feasible = bool(repaired_ok)
        if repaired_ok:
            solver_status = (
                "converged_infeasible_repaired"
                if solver_status == "converged"
                else "max_iter_infeasible_repaired"
            )
        else:
            solver_status = "repair_failed_infeasible"
        final_power = power_vec.copy()

    objective_history.append(float(final_terms["sum_rate_objective"]))
    feasibility_history.append(bool(final_terms["is_feasible"]))
    sic_margin_history.append(np.asarray(final_terms["sic_margin"], dtype=np.float64).copy())
    power_history.append(final_power.copy())

    result = {
        "power_vec": final_power.astype(np.float64),
        "is_feasible": bool(final_is_feasible),
        "iters_used": int(len(power_history) - 1),
        "solver_status": solver_status,
        "served_users": ctx["served_users"].copy(),
        "rb_index": int(ctx["rb_index"]),
        "common_sinr_all": np.asarray(final_terms["sinr_common"], dtype=np.float64).copy(),
        "private_sinr_all": np.asarray(final_terms["sinr_private"], dtype=np.float64).copy(),
        "common_rate": float(final_terms["common_rate"]),
        "private_rates": np.asarray(final_terms["private_rates"], dtype=np.float64).copy(),
        "sum_rate_objective": float(final_terms["sum_rate_objective"]),
        "env_sic_success": bool(final_terms["is_feasible"]),
        "env_gated_common_rate": float(final_terms["env_gated_common_rate"]),
        "env_gated_private_rates": np.asarray(
            final_terms["env_gated_private_rates"], dtype=np.float64
        ).copy(),
        "env_gated_total_rates": np.asarray(
            final_terms["env_gated_total_rates"], dtype=np.float64
        ).copy(),
        "sic_margin_all": np.asarray(final_terms["sic_margin"], dtype=np.float64).copy(),
        "debug": {
            "objective_history": [float(x) for x in objective_history],
            "awmmse_history": [float(x) for x in awmmse_history],
            "power_history": [np.asarray(x, dtype=np.float64).copy() for x in power_history],
            "feasibility_history": [bool(x) for x in feasibility_history],
            "sic_margin_history": [
                np.asarray(x, dtype=np.float64).copy() for x in sic_margin_history
            ],
            "subproblem_success_history": [bool(x) for x in subproblem_success_history],
            "subproblem_message_history": list(subproblem_message_history),
            "h_eff_hat": np.asarray(final_terms["h_eff_hat"], dtype=np.complex128).copy(),
            "g_sat": np.asarray(final_terms["g_sat"], dtype=np.float64).copy(),
            "eeq2": np.asarray(final_terms["eeq2"], dtype=np.float64).copy(),
            "active_ris_noise": np.asarray(
                final_terms["active_ris_noise"], dtype=np.float64
            ).copy(),
            "external_interference": np.asarray(
                final_terms["external_interference"], dtype=np.float64
            ).copy(),
            "I_csi": np.asarray(final_terms["I_csi"], dtype=np.float64).copy(),
            "I_tot": np.asarray(final_terms["I_tot"], dtype=np.float64).copy(),
            "last_feasible_power": None
            if last_feasible_power is None
            else last_feasible_power.copy(),
            "last_feasible_sum_rate": None
            if last_feasible_terms is None
            else float(last_feasible_terms["sum_rate_objective"]),
        },
    }
    if "repair_debug" in locals():
        result["debug"]["repair_debug"] = repair_debug
    return result


def solve_wmmse_power_from_env(
    env: Any,
    user_indices: Sequence[int],
    rb_index: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    return solve_wmmse_power(
        wmmse_ctx=None,
        env=env,
        user_indices=user_indices,
        rb_index=rb_index,
        **kwargs,
    )


if __name__ == "__main__":
    try:
        from atprl.envs.leo_sat import LeoSatEnv

        env = LeoSatEnv(
            {
                "num_ground_stations": 3,
                "num_resource_blocks": 1,
                "num_gs_per_rb": 2,
                "random_seed": 123,
                "reseed_each_episode": True,
            }
        )
        env.reset(seed=123)
        out = solve_wmmse_power_from_env(
            env,
            user_indices=[0, 1],
            rb_index=0,
            max_wmmse_iters=10,
            tol=1e-7,
        )
        print("solver_status:", out["solver_status"])
        print("power_vec:", out["power_vec"])
        print("is_feasible:", out["is_feasible"])
        print("sum_rate_objective:", out["sum_rate_objective"])
    except Exception as exc:  # pragma: no cover
        print("Self-test failed:", repr(exc))
