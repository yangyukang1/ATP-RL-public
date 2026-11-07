import numpy as np
from typing import Dict, Any, Optional
from scipy.special import j0
from typing import Dict, Any, Optional, List, Union

from atprl.envs.topologies import resolve_topology


class LeoSatEnv:
    """LEO RSMA environment with active RIS and terrestrial interference."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.random_seed = int(config.get("random_seed", 1234))
        self.reseed_each_episode = bool(config.get("reseed_each_episode", True))
        self._rng = np.random.default_rng(self.random_seed)

        self.include_angle_features = bool(config.get("include_angle_features", False))

        self.include_sr_incident_pow_groups = bool(
            config.get("include_sr_incident_pow_groups", False)
        )

        self.include_mrt_features = bool(config.get("include_mrt_features", False))

        self.mrt_casc_abs_tau = float(config.get("mrt_casc_abs_tau", 1e-12))

        self._base_obs_dim: Optional[int] = None

        self._obs_layout_spec_base: Optional[Dict[str, Any]] = None
        self._obs_layout_spec_full: Optional[Dict[str, Any]] = None

        self.num_ground_stations = int(config.get("num_ground_stations", 4))

        self.num_resource_blocks = int(config.get("num_resource_blocks", 1))

        self.num_gs_per_rb = int(config.get("num_gs_per_rb", 2))
        if not 1 <= self.num_gs_per_rb <= self.num_ground_stations:
            raise ValueError(
                "num_gs_per_rb must satisfy 1 <= K <= N, "
                f"got K={self.num_gs_per_rb}, N={self.num_ground_stations}"
            )

        self.deployment_area_size_m = float(config.get("deployment_area_size_m", 400.0))

        self.agents = [f"rb_{i}" for i in range(self.num_resource_blocks)]

        self.satellite_max_power_dBm = float(config.get("satellite_max_power_dBm", 50.0))
        self.satellite_max_power_watt = self._dbm_to_watt(self.satellite_max_power_dBm)

        self.terrestrial_cell_power_dBm = float(config.get("terrestrial_cell_power_dBm", 35.0))
        self.terrestrial_cell_power_watt = self._dbm_to_watt(self.terrestrial_cell_power_dBm)

        self.noise_power_dBm = float(config.get("noise_power_dBm", -114.0))
        self.noise_power_watt = self._dbm_to_watt(self.noise_power_dBm)

        self.carrier_frequency_GHz = float(config.get("carrier_frequency_GHz", 4.0))
        self.carrier_frequency_Hz = self.carrier_frequency_GHz * 1e9

        self.light_speed = 3.0e8

        self.satellite_altitude_m = float(config.get("satellite_altitude_m", 600e3))

        self.satellite_antenna_gain_dBi = float(config.get("satellite_antenna_gain_dBi", 24.3))
        self.satellite_antenna_gain_linear = self._dbi_to_linear(self.satellite_antenna_gain_dBi)

        self.uav_rx_antenna_gain_dBi = float(config.get("uav_rx_antenna_gain_dBi", 15.0))
        self.uav_rx_antenna_gain_linear = self._dbi_to_linear(self.uav_rx_antenna_gain_dBi)

        self.ground_station_antenna_gain_dBi = float(
            config.get("ground_station_antenna_gain_dBi", 24.3)
        )
        self.ground_station_antenna_gain_linear = self._dbi_to_linear(
            self.ground_station_antenna_gain_dBi
        )

        self.terrestrial_cell_antenna_gain_dBi = float(
            config.get("terrestrial_cell_antenna_gain_dBi", 0.0)
        )
        self.terrestrial_cell_antenna_gain_linear = self._dbi_to_linear(
            self.terrestrial_cell_antenna_gain_dBi
        )

        self.log_normal_shadowing_variance_dB = float(
            config.get("log_normal_shadowing_variance_dB", 8.0)
        )

        self.rician_k_factor = float(config.get("rician_k_factor", 10.0))

        self.num_micro_steps = int(config.get("num_micro_steps", 20))
        if self.num_micro_steps < 1:
            self.num_micro_steps = 1

        self.channel_update_interval_s = float(config.get("channel_update_interval_s", 0.04))
        if self.channel_update_interval_s <= 0.0:
            raise ValueError("channel_update_interval_s must be > 0.")

        self.micro_step_interval_s = self.channel_update_interval_s / float(self.num_micro_steps)

        self.max_doppler_frequency_Hz = float(config.get("max_doppler_frequency_Hz", 50.0))
        if self.max_doppler_frequency_Hz < 0.0:
            raise ValueError("max_doppler_frequency_Hz must be >= 0.")

        self.compute_rho_from_jakes = bool(config.get("compute_rho_from_jakes", True))

        manual_rho = config.get("rho", None)

        if self.compute_rho_from_jakes:
            self.rho = self._compute_jakes_rho(
                fd_hz=self.max_doppler_frequency_Hz,
                ts_s=self.micro_step_interval_s,
            )
            self.rho_source = "jakes"
        else:
            if manual_rho is None:
                self.rho = self._compute_jakes_rho(
                    fd_hz=self.max_doppler_frequency_Hz,
                    ts_s=self.micro_step_interval_s,
                )
                self.rho_source = "jakes_fallback"
            else:
                self.rho = float(manual_rho)
                self.rho_source = "manual"

        self.rho = float(np.clip(self.rho, -0.999999, 0.999999))

        self._macro_err_sg = None
        self._macro_err_sr = None
        self._macro_err_rg = None

        self.max_steps_per_episode = int(config.get("max_steps_per_episode", 1024))
        self.current_step = 0

        self.sic_detection_threshold_dBm = float(config.get("sic_detection_threshold_dBm", -94.0))
        self.sic_detection_threshold_watt = self._dbm_to_watt(self.sic_detection_threshold_dBm)

        self.qos_rate_threshold_bpsHz = float(config.get("qos_rate_threshold_bpsHz", 0.1))

        self.min_rate_requirement_bps_hz = float(
            config.get("min_rate_requirement_bps_hz", self.qos_rate_threshold_bpsHz)
        )

        self.ris_boost_lambda = float(config.get("ris_boost_lambda", 0.0))

        self._last_boost_ratios_all_users = np.ones(self.num_ground_stations, dtype=np.float64)
        self._last_boost_ratios_direct_only = np.ones(self.num_ground_stations, dtype=np.float64)

        self.sic_penalty = float(config.get("sic_penalty", 0.0))

        self.csi_error_variance = float(config.get("csi_error_variance", 0.01))

        self.fairness_alpha = float(config.get("fairness_alpha", 0.0))

        self.fairness_beta = float(config.get("fairness_beta", 0.01))
        self.fairness_beta = min(max(self.fairness_beta, 0.0), 1.0)

        self.fairness_epsilon = float(config.get("fairness_epsilon", 0.01))

        self.historical_avg_rates = np.zeros(self.num_ground_stations, dtype=np.float64)

        self.fairness_lambda = float(config.get("fairness_lambda", 0.0))

        uav_init_raw = config.get("uav_initial_position_xyz", [0.0, 0.0, 80.0])
        self.uav_initial_position_xyz = np.array(uav_init_raw, dtype=np.float64)

        uav_vel_raw = config.get("uav_velocity_per_step_xyz", [0.282842712, 0.282842712, 0.0])
        self.uav_velocity_per_step_xyz = np.array(uav_vel_raw, dtype=np.float64)

        self.uav_position_xyz = np.zeros(3, dtype=np.float64)

        self.ris_num_elem_x = int(config.get("ris_num_elem_x", 16))
        self.ris_num_elem_y = int(config.get("ris_num_elem_y", 16))
        self.ris_num_elems = self.ris_num_elem_x * self.ris_num_elem_y

        self.ris_element_amplitude = config.get("ris_element_amplitude", 1.0)

        self.ris_power_control_mode = config.get("ris_power_control_mode", "power_budget")

        self.ris_reflection_power_limit_dBm = float(
            config.get("ris_reflection_power_limit_dBm", 0.0)
        )
        self.ris_reflection_power_limit_watt = self._dbm_to_watt(
            self.ris_reflection_power_limit_dBm
        )

        self.active_ris_enabled = bool(config.get("active_ris_enabled", True))
        if (self.ris_element_amplitude > 1.0) and (not self.active_ris_enabled):
            self.active_ris_enabled = True

        self.disable_ris_path = bool(config.get("disable_ris_path", False))
        if self.disable_ris_path:
            self.active_ris_enabled = False

        if "active_ris_noise_power_dBm" in config:
            self.active_ris_noise_power_watt = 10 ** (
                (config["active_ris_noise_power_dBm"] - 30.0) / 10.0
            )
        else:
            default_noise_dbm = -100.0
            self.active_ris_noise_power_watt = 10 ** ((default_noise_dbm - 30.0) / 10.0)

        self._last_active_ris_noise_per_user = np.zeros(self.num_ground_stations, dtype=np.float64)

        self.ris_elem_max_amp_gain_dB = float(config.get("ris_elem_max_amp_gain_dB", 40.0))
        self.ris_elem_max_amp_linear = 10.0 ** (self.ris_elem_max_amp_gain_dB / 20.0)

        self.ris_rx_element_gain_linear = 1.0
        self.ris_tx_element_gain_linear = np.ones(self.num_ground_stations, dtype=np.float64)

        self.ris_elem_spacing_ratio = config.get("ris_elem_spacing_ratio", 0.5)

        self.ris_orientation_euler_deg = np.array(
            config.get("ris_orientation_euler_deg", [0.0, 0.0, 0.0]), dtype=np.float64
        )

        self.wavelength_m = self.light_speed / self.carrier_frequency_Hz
        self.ris_elem_spacing_m = self.ris_elem_spacing_ratio * self.wavelength_m

        if (self.ris_num_elem_x % 2 != 0) or (self.ris_num_elem_y % 2 != 0):
            raise ValueError("四象限分组要求 ris_num_elem_x/ris_num_elem_y 为偶数，例如 8x8。")

        gx_split = self.ris_num_elem_x // 2
        gy_split = self.ris_num_elem_y // 2

        UL = [
            iy * self.ris_num_elem_x + ix for iy in range(0, gy_split) for ix in range(0, gx_split)
        ]
        UR = [
            iy * self.ris_num_elem_x + ix
            for iy in range(0, gy_split)
            for ix in range(gx_split, self.ris_num_elem_x)
        ]
        LL = [
            iy * self.ris_num_elem_x + ix
            for iy in range(gy_split, self.ris_num_elem_y)
            for ix in range(0, gx_split)
        ]
        LR = [
            iy * self.ris_num_elem_x + ix
            for iy in range(gy_split, self.ris_num_elem_y)
            for ix in range(gx_split, self.ris_num_elem_x)
        ]

        self.ris_groups = [UL, UR, LL, LR]
        self.ris_num_groups = 4

        self._obs_sr_groups = np.zeros(self.ris_num_groups, dtype=np.complex128)
        self._obs_rg_groups = np.zeros(
            (self.num_ground_stations, self.ris_num_groups), dtype=np.complex128
        )

        self._obs_casc_groups = np.zeros(
            (self.num_ground_stations, self.ris_num_groups),
            dtype=np.complex128,
        )

        self._obs_sr_incident_pow_groups = np.zeros((self.ris_num_groups,), dtype=np.float64)

        self.ris_group_phase_vector = np.zeros(self.ris_num_groups, dtype=np.float64)

        self.ris_phase_vector = np.zeros(self.ris_num_elems, dtype=np.float64)

        self.ris_phase_response = np.ones(self.ris_num_elems, dtype=np.complex128)

        self.ris_phase_min = -np.pi
        self.ris_phase_max = np.pi

        self.small_scale_sat_to_ris = np.zeros(self.ris_num_elems, dtype=np.complex128)

        self.small_scale_ris_to_gs = np.zeros(
            (self.num_ground_stations, self.ris_num_elems), dtype=np.complex128
        )

        self.large_scale_gain_sat_to_ris = 0.0
        self.large_scale_gain_ris_to_gs = np.zeros(self.num_ground_stations, dtype=np.float64)
        self.large_scale_gain_ris_path = np.zeros(self.num_ground_stations, dtype=np.float64)

        self.k_factor_ris_to_gs_linear = config.get(
            "k_factor_ris_to_gs_linear", self.rician_k_factor
        )

        self.k_factor_sat_to_ris_linear = config.get(
            "k_factor_sat_to_ris_linear", self.rician_k_factor
        )

        self._nlos_sat_to_ris = np.zeros(self.ris_num_elems, dtype=np.complex128)
        self._nlos_ris_to_gs = np.zeros(
            (self.num_ground_stations, self.ris_num_elems), dtype=np.complex128
        )

        self.ris_nlos_model = self.config.get("ris_nlos_model", "panel")

        self.historical_sinrs_per_user = {
            agent_id: np.zeros((self.num_ground_stations, 2), dtype=np.float32)
            for agent_id in self.agents
        }

        self.power_action_dim = 1 + self.num_ground_stations
        self.last_power_actions_per_user = {
            agent_id: np.zeros(self.power_action_dim, dtype=np.float32) for agent_id in self.agents
        }

        self.small_scale_coeffs_sat_to_gs = np.zeros(self.num_ground_stations, dtype=np.complex128)

        self.small_scale_coeffs_bs_to_gs = np.zeros(
            (self.num_ground_stations, self.num_resource_blocks), dtype=np.complex128
        )

        self.estimated_small_scale_coeffs_sat_to_gs = np.zeros(
            self.num_ground_stations, dtype=np.complex128
        )

        self.estimated_small_scale_sat_to_ris = np.zeros((self.ris_num_elems,), dtype=np.complex128)
        self.estimated_small_scale_ris_to_gs = np.zeros(
            (self.num_ground_stations, self.ris_num_elems), dtype=np.complex128
        )

        self.csi_err_sg = np.zeros((self.num_ground_stations,), dtype=np.complex128)
        self.csi_err_sr = np.zeros((self.ris_num_elems,), dtype=np.complex128)
        self.csi_err_rg = np.zeros(
            (self.num_ground_stations, self.ris_num_elems), dtype=np.complex128
        )

        self.h_sg_true = np.zeros((self.num_ground_stations,), dtype=np.complex128)
        self.h_sg_hat = np.zeros((self.num_ground_stations,), dtype=np.complex128)

        self.h_sr_true = np.zeros((self.ris_num_elems,), dtype=np.complex128)
        self.h_sr_hat = np.zeros((self.ris_num_elems,), dtype=np.complex128)

        self.h_rg_true = np.zeros(
            (self.num_ground_stations, self.ris_num_elems), dtype=np.complex128
        )
        self.h_rg_hat = np.zeros(
            (self.num_ground_stations, self.ris_num_elems), dtype=np.complex128
        )

        self.large_scale_gain_sat_to_gs = 0.0

        self.large_scale_gain_sat_to_gs_per_user = np.zeros(
            self.num_ground_stations, dtype=np.float64
        )

        self.large_scale_gain_bs_to_gs = np.zeros(
            (self.num_ground_stations, self.num_resource_blocks), dtype=np.float64
        )

        raw_offsets = self.config.get("sat_user_gain_db_offsets", None)
        if raw_offsets is None:
            self._sat_user_gain_offset_linear = np.ones(self.num_ground_stations, dtype=np.float64)
            self._sat_user_gain_offset_db = np.zeros(self.num_ground_stations, dtype=np.float64)
        else:
            if isinstance(raw_offsets, str):
                parts = [p.strip() for p in raw_offsets.split(",") if p.strip() != ""]
                vals = [float(p) for p in parts]
            else:
                vals = list(raw_offsets)

            if len(vals) == 1:
                vals = vals * self.num_ground_stations
            elif len(vals) < self.num_ground_stations:
                vals = vals + [0.0] * (self.num_ground_stations - len(vals))
            else:
                vals = vals[: self.num_ground_stations]

            self._sat_user_gain_offset_db = np.array(vals, dtype=np.float64)
            self._sat_user_gain_offset_linear = 10 ** (self._sat_user_gain_offset_db / 10.0)

        self._fixed_shadowing_db = self._rng.normal(
            0,
            np.sqrt(self.log_normal_shadowing_variance_dB),
            size=(self.num_ground_stations, self.num_resource_blocks),
        )

        fixed_nlos_real = self._rng.standard_normal(
            (self.num_ground_stations, self.num_resource_blocks)
        )
        fixed_nlos_imag = self._rng.standard_normal(
            (self.num_ground_stations, self.num_resource_blocks)
        )
        self._fixed_bs_to_gs_nlos_component = (fixed_nlos_real + 1j * fixed_nlos_imag) / np.sqrt(2)

        self._fixed_initial_sat_to_gs_coeffs = (
            self._rng.standard_normal(self.num_ground_stations)
            + 1j * self._rng.standard_normal(self.num_ground_stations)
        ) / np.sqrt(2)

        self.ris_num_groups = len(self.ris_groups)
        self.power_action_dim = 1 + self.num_ground_stations
        self.ris_phase_action_dim = self.ris_num_groups
        self.ppo_action_dim = self.power_action_dim + self.ris_phase_action_dim

        self._last_mrt_feat: Optional[np.ndarray] = None

    def _dbm_to_watt(self, dbm: float) -> float:
        return 10 ** ((dbm - 30) / 10)

    def _watt_to_dbm(self, watt: float) -> float:
        if watt <= 0.0:
            return -np.inf
        return 10.0 * np.log10(watt) + 30.0

    def _ls_gain_to_db_norm(
        self, x_lin: np.ndarray, eps: float = 1e-30, denom: float = 100.0
    ) -> np.ndarray:
        x_lin = np.asarray(x_lin, dtype=np.float64)
        x_db = 10.0 * np.log10(np.maximum(x_lin, eps))
        return (x_db / denom).astype(np.float32)

    def _dbi_to_linear(self, dbi: float) -> float:
        return 10 ** (dbi / 10)

    def _compute_jakes_rho(self, fd_hz: float, ts_s: float) -> float:
        """Return J0(2*pi*fd*Ts) for the micro-slot fading process."""
        fd_hz = float(fd_hz)
        ts_s = float(ts_s)

        if fd_hz < 0.0:
            raise ValueError("fd_hz must be >= 0.")
        if ts_s <= 0.0:
            raise ValueError("ts_s must be > 0.")

        x = 2.0 * np.pi * fd_hz * ts_s
        rho = float(j0(x))
        return rho

    def get_temporal_correlation_debug_info(self) -> Dict[str, float]:
        return {
            "max_doppler_frequency_Hz": float(self.max_doppler_frequency_Hz),
            "channel_update_interval_s": float(self.channel_update_interval_s),
            "num_micro_steps": int(self.num_micro_steps),
            "micro_step_interval_s": float(self.micro_step_interval_s),
            "rho": float(self.rho),
        }

    def _cn0(self, shape, var):
        """Sample circular complex Gaussian noise with E[abs(w)**2] = var."""
        if var <= 0.0:
            return np.zeros(shape, dtype=np.complex128)
        std = np.sqrt(var / 2.0)
        return (self._rng.normal(0.0, std, shape) + 1j * self._rng.normal(0.0, std, shape)).astype(
            np.complex128
        )

    def _sample_csi_error_realization_for_macro(self) -> None:
        """Hold one CSI-error realization fixed throughout each decision interval."""
        var = float(getattr(self, "csi_error_variance", 0.0) or 0.0)

        K = int(self.num_ground_stations)
        M = int(self.ris_num_elems)

        if var <= 0.0:
            self._macro_err_sg = np.zeros((K,), dtype=np.complex128)
            self._macro_err_sr = np.zeros((M,), dtype=np.complex128)
            self._macro_err_rg = np.zeros((K, M), dtype=np.complex128)
            return

        self._macro_err_sg = self._cn0((K,), var)
        self._macro_err_sr = self._cn0((M,), var)
        self._macro_err_rg = self._cn0((K, M), var)

    def _apply_fixed_error_to_build_estimates(self) -> None:
        if (
            (self._macro_err_sg is None)
            or (self._macro_err_sr is None)
            or (self._macro_err_rg is None)
        ):
            self._sample_csi_error_realization_for_macro()

        self.csi_err_sg = self._macro_err_sg
        self.estimated_small_scale_coeffs_sat_to_gs = (
            self.small_scale_coeffs_sat_to_gs + self._macro_err_sg
        )

        self.csi_err_sr = self._macro_err_sr
        self.estimated_small_scale_sat_to_ris = self.small_scale_sat_to_ris + self._macro_err_sr

        self.csi_err_rg = self._macro_err_rg
        self.estimated_small_scale_ris_to_gs = self.small_scale_ris_to_gs + self._macro_err_rg

        if hasattr(self, "large_scale_gain_sat_to_gs_per_user"):
            ls_sg = np.asarray(self.large_scale_gain_sat_to_gs_per_user, dtype=np.float64)
        else:
            offset = getattr(
                self,
                "_sat_user_gain_offset_linear",
                np.ones(self.num_ground_stations, dtype=np.float64),
            )
            ls_sg = float(getattr(self, "large_scale_gain_sat_to_gs", 1.0)) * np.asarray(
                offset, dtype=np.float64
            )

        ls_sg = ls_sg.reshape(-1)
        self.h_sg_true = np.sqrt(ls_sg) * self.small_scale_coeffs_sat_to_gs
        self.h_sg_hat = np.sqrt(ls_sg) * self.estimated_small_scale_coeffs_sat_to_gs

        ls_sr = float(getattr(self, "large_scale_gain_sat_to_ris", 1.0))
        self.h_sr_true = np.sqrt(ls_sr) * self.small_scale_sat_to_ris
        self.h_sr_hat = np.sqrt(ls_sr) * self.estimated_small_scale_sat_to_ris

        ls_rg = np.asarray(
            getattr(self, "large_scale_gain_ris_to_gs", np.ones(self.num_ground_stations)),
            dtype=np.float64,
        )
        self.h_rg_true = (np.sqrt(ls_rg)[:, None]) * self.small_scale_ris_to_gs
        self.h_rg_hat = (np.sqrt(ls_rg)[:, None]) * self.estimated_small_scale_ris_to_gs

    def _refresh_imperfect_csi_estimates(self) -> None:
        self._sample_csi_error_realization_for_macro()
        self._apply_fixed_error_to_build_estimates()

    def _complex_to_feats(self, z_complex):
        """Encode each complex value as log-magnitude, cosine, and sine."""
        z = np.asarray(z_complex, dtype=np.complex128)
        amp = np.maximum(np.abs(z), 1e-12)
        log_amp = np.log(amp)
        ang = np.angle(z)
        feats = np.stack([log_amp, np.cos(ang), np.sin(ang)], axis=-1)
        return feats.astype(np.float32)

    def _uav_rel_xy_per_gs_feature(self):
        L = float(max(self.deployment_area_size_m, 1e-9))

        dx = (self.uav_position_xyz[0] - self.gs_positions[:, 0]) / L
        dy = (self.uav_position_xyz[1] - self.gs_positions[:, 1]) / L
        uav_rel_xy_feat = np.stack([dx, dy], axis=1).reshape(-1).astype(np.float32)

        z_over_L = np.array([self.uav_position_xyz[2] / L], dtype=np.float32)

        return uav_rel_xy_feat, z_over_L

    def _uav_direction_cosines_to_gs(self) -> np.ndarray:
        uav_pos = self.uav_position_xyz.astype(np.float64)

        gs_xy = self.gs_positions.astype(np.float64)
        if gs_xy.shape[1] == 2:
            gs_pos = np.concatenate(
                [gs_xy, np.zeros((gs_xy.shape[0], 1), dtype=np.float64)], axis=1
            )
        else:
            gs_pos = gs_xy

        diff = gs_pos - uav_pos[None, :]

        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-9)

        dir_cos = diff / dist
        angle_feats = dir_cos.reshape(-1).astype(np.float32)
        return angle_feats

    def _update_twohop_group_csi_observation(self) -> None:
        """Sum elementwise channel products within each group, without conjugation."""
        G = self.ris_num_groups
        K = self.num_ground_stations

        sr_hat = np.zeros((G,), dtype=np.complex128)
        rg_hat = np.zeros((K, G), dtype=np.complex128)

        casc_hat = np.zeros((K, G), dtype=np.complex128)
        inc_pow = np.zeros((G,), dtype=np.float64)

        for g, idxs in enumerate(self.ris_groups):
            sr_elem = self.estimated_small_scale_sat_to_ris[idxs]

            sr_hat[g] = np.sum(sr_elem)

            inc_pow[g] = float(np.sum(np.abs(sr_elem) ** 2))

            for k in range(K):
                rg_elem = self.estimated_small_scale_ris_to_gs[k, idxs]

                rg_hat[k, g] = np.sum(rg_elem)

                casc_hat[k, g] = np.sum(sr_elem * rg_elem)

        self._obs_sr_groups[:] = sr_hat
        self._obs_rg_groups[:] = rg_hat

        self._obs_casc_groups[:] = casc_hat
        self._obs_sr_incident_pow_groups[:] = inc_pow

    def _compute_direct_aware_mrt_feats(self) -> np.ndarray:
        casc_hat = np.asarray(self._obs_casc_groups, dtype=np.complex128)
        h_dir_hat = np.asarray(self.estimated_small_scale_coeffs_sat_to_gs, dtype=np.complex128)

        K = int(self.num_ground_stations)
        G = int(self.ris_num_groups)

        if casc_hat.shape != (K, G):
            raise ValueError(
                f"_obs_casc_groups shape mismatch: expect {(K, G)}, got {casc_hat.shape}"
            )
        if h_dir_hat.shape[0] != K:
            raise ValueError(
                f"estimated_small_scale_coeffs_sat_to_gs shape mismatch: expect ({K},), got {h_dir_hat.shape}"
            )

        ang_dir = np.angle(h_dir_hat).reshape(K, 1)
        ang_casc = np.angle(casc_hat)
        theta = ang_dir - ang_casc

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        tau = float(getattr(self, "mrt_casc_abs_tau", 0.0))
        if tau > 0.0:
            mask = np.abs(casc_hat) < tau
            if np.any(mask):
                cos_th = cos_th.copy()
                sin_th = sin_th.copy()
                cos_th[mask] = 1.0
                sin_th[mask] = 0.0

        feat = np.stack([cos_th, sin_th], axis=-1).reshape(-1).astype(np.float32)
        return feat

    def _compute_ris_steering_vector(
        self, ris_pos_xyz: np.ndarray, target_pos_xyz: np.ndarray
    ) -> np.ndarray:
        dvec = target_pos_xyz.astype(np.float64) - ris_pos_xyz.astype(np.float64)
        dist = np.linalg.norm(dvec)
        if dist < 1e-9:
            dist = 1e-9
        ux, uy, uz = dvec / dist

        k0 = 2.0 * np.pi / self.wavelength_m
        d = self.ris_elem_spacing_m

        ix_idx = np.arange(self.ris_num_elem_x, dtype=np.float64)
        iy_idx = np.arange(self.ris_num_elem_y, dtype=np.float64)

        a_x = np.exp(1j * k0 * d * ix_idx * ux)
        a_y = np.exp(1j * k0 * d * iy_idx * uy)

        steering_vec = np.kron(a_y, a_x).astype(np.complex128)

        return steering_vec

    def _advance_sat_to_gs_small_scale_one_micro(self):
        """Advance direct-link Rayleigh fading by one AR(1) micro-step."""
        rho = float(getattr(self, "rho", 0.0))
        rho = float(np.clip(rho, -0.999999, 0.999999))

        K = int(self.num_ground_stations)
        w = (self._rng.standard_normal(K) + 1j * self._rng.standard_normal(K)) / np.sqrt(2.0)
        self.small_scale_coeffs_sat_to_gs = (
            rho * self.small_scale_coeffs_sat_to_gs + np.sqrt(1.0 - rho * rho) * w
        )

    def _begin_macro_slot_refresh(self) -> None:
        self._refresh_bs_to_gs_small_scale_for_macro()

        self._resample_ris_nlos_for_macro()

        self._refresh_ris_small_scale_channels()

        self._sample_csi_error_realization_for_macro()

        self._apply_fixed_error_to_build_estimates()

    def _resample_ris_nlos_for_macro(self) -> None:
        if getattr(self, "ris_nlos_model", "panel") == "panel":
            for g_indices in self.ris_groups:
                z_group_sr = (
                    self._rng.standard_normal() + 1j * self._rng.standard_normal()
                ) / np.sqrt(2.0)
                self._nlos_sat_to_ris[g_indices] = z_group_sr

            for k in range(self.num_ground_stations):
                for g_indices in self.ris_groups:
                    z_group_rg = (
                        self._rng.standard_normal() + 1j * self._rng.standard_normal()
                    ) / np.sqrt(2.0)
                    self._nlos_ris_to_gs[k, g_indices] = z_group_rg
        else:
            self._nlos_sat_to_ris = (
                self._rng.standard_normal(self.ris_num_elems)
                + 1j * self._rng.standard_normal(self.ris_num_elems)
            ) / np.sqrt(2.0)
            self._nlos_ris_to_gs = (
                self._rng.standard_normal((self.num_ground_stations, self.ris_num_elems))
                + 1j * self._rng.standard_normal((self.num_ground_stations, self.ris_num_elems))
            ) / np.sqrt(2.0)

    def _refresh_ris_small_scale_channels(self) -> None:
        uav_xyz = self.uav_position_xyz.astype(np.float64)
        sat_xyz = np.array([0.0, 0.0, self.satellite_altitude_m], dtype=np.float64)

        los_vec_sr = self._compute_ris_steering_vector(ris_pos_xyz=uav_xyz, target_pos_xyz=sat_xyz)

        K_sr = self.k_factor_sat_to_ris_linear
        self.small_scale_sat_to_ris = (
            np.sqrt(K_sr / (K_sr + 1.0)) * los_vec_sr
            + np.sqrt(1.0 / (K_sr + 1.0)) * self._nlos_sat_to_ris
        )

        K_rg = self.k_factor_ris_to_gs_linear
        new_ris_to_gs = np.zeros(
            (self.num_ground_stations, self.ris_num_elems), dtype=np.complex128
        )
        for n in range(self.num_ground_stations):
            gs_xyz = np.array(
                [self.gs_positions[n, 0], self.gs_positions[n, 1], 0.0], dtype=np.float64
            )
            los_vec_rg = self._compute_ris_steering_vector(
                ris_pos_xyz=uav_xyz, target_pos_xyz=gs_xyz
            )

            new_ris_to_gs[n, :] = (
                np.sqrt(K_rg / (K_rg + 1.0)) * los_vec_rg
                + np.sqrt(1.0 / (K_rg + 1.0)) * self._nlos_ris_to_gs[n, :]
            )

        self.small_scale_ris_to_gs = new_ris_to_gs

    def _initialize_positions(self):
        if self.config.get("use_fixed_positions", False):
            assert self.num_resource_blocks == 1, "固定位置模式要求 num_resource_blocks 为 1"

            self.bs_positions = np.array([[0.0, -600.0]], dtype=np.float64)

            topology_profile = self.config.get("topology_profile")
            if topology_profile not in (None, ""):
                topology = resolve_topology(str(topology_profile), self.num_ground_stations)
                self.gs_positions, self.bs_positions = topology.arrays()
                return

            if self.num_ground_stations == 3:
                self.gs_positions = np.array(
                    [
                        [-100.0, 0.0],
                        [260.0, 60.0],
                        [120.0, 280.0],
                    ],
                    dtype=np.float64,
                )

            elif self.num_ground_stations == 4:
                self.gs_positions = np.array(
                    [
                        [-20.0, -100.0],
                        [50.0, 150.0],
                        [150.0, 20.0],
                        [250.0, 160.0],
                    ],
                    dtype=np.float64,
                )

            elif self.num_ground_stations == 5:
                self.gs_positions = np.array(
                    [
                        [-20.0, -100.0],
                        [50.0, 150.0],
                        [150.0, 20.0],
                        [250.0, 160.0],
                        [200.0, 250.0],
                    ],
                    dtype=np.float64,
                )

            else:
                raise ValueError(
                    f"固定位置模式当前仅支持 num_ground_stations 为 3、4 或 5，"
                    f"当前收到 {self.num_ground_stations}。"
                )

        else:
            self.gs_positions = (
                self._rng.random((self.num_ground_stations, 2)) * self.deployment_area_size_m
            )
            self.bs_positions = (
                self._rng.random((self.num_resource_blocks, 2)) * self.deployment_area_size_m
            )

    def _initialize_large_scale_fading(self):
        d_s = self.satellite_altitude_m
        path_gain_factor = (self.light_speed / (4 * np.pi * d_s * self.carrier_frequency_Hz)) ** 2
        self.large_scale_gain_sat_to_gs = (
            path_gain_factor
            * self.satellite_antenna_gain_linear
            * self.ground_station_antenna_gain_linear
        )
        offset = np.asarray(
            getattr(self, "_sat_user_gain_offset_linear", np.ones(self.num_ground_stations)),
            dtype=np.float64,
        ).reshape(-1)

        if offset.size != self.num_ground_stations:
            offset = np.ones(self.num_ground_stations, dtype=np.float64)

        self.large_scale_gain_sat_to_gs_per_user = float(self.large_scale_gain_sat_to_gs) * offset

        for n in range(self.num_ground_stations):
            for m in range(self.num_resource_blocks):
                distance_m = np.linalg.norm(self.gs_positions[n] - self.bs_positions[m])
                distance_km = distance_m / 1000.0
                if distance_km < 1e-3:
                    distance_km = 1e-3
                path_loss_db = 120.9 + 37.6 * np.log10(distance_km)
                shadowing_db = self._fixed_shadowing_db[n, m]
                total_loss_db = path_loss_db + shadowing_db
                self.large_scale_gain_bs_to_gs[n, m] = 10 ** (-total_loss_db / 10)
        force_db = self.config.get("force_bs_gain_db", None)
        if force_db is not None:
            fixed_gain_linear = 10 ** (force_db / 10.0)
            self.large_scale_gain_bs_to_gs.fill(fixed_gain_linear)

    def _refresh_bs_to_gs_small_scale_for_macro(self) -> None:
        KI = float(getattr(self, "rician_k_factor", 0.0))
        los = 1.0 + 0.0j

        nlos = (
            self._rng.standard_normal((self.num_ground_stations, self.num_resource_blocks))
            + 1j * self._rng.standard_normal((self.num_ground_stations, self.num_resource_blocks))
        ) / np.sqrt(2.0)

        a = np.sqrt(KI / (KI + 1.0)) if KI > 0 else 0.0
        b = np.sqrt(1.0 / (KI + 1.0)) if KI >= 0 else 1.0

        self.small_scale_coeffs_bs_to_gs = a * los + b * nlos

    def _initialize_small_scale_fading(self):
        self.small_scale_coeffs_sat_to_gs = self._fixed_initial_sat_to_gs_coeffs.copy()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        _ = options

        if seed is not None:
            self.random_seed = int(seed)
            self._rng = np.random.default_rng(self.random_seed)
        elif self.reseed_each_episode:
            self._rng = np.random.default_rng(self.random_seed)

        self.current_step = 0
        self.time_limit_truncated = False

        self.current_step = 0

        self._initialize_positions()

        self._initialize_large_scale_fading()

        self._initialize_small_scale_fading()

        self.uav_position_xyz = np.array(self.uav_initial_position_xyz, dtype=np.float64).copy()

        self.ris_group_phase_vector[:] = 0.0

        for g_idx, elem_idx_list in enumerate(self.ris_groups):
            self.ris_phase_vector[elem_idx_list] = self.ris_group_phase_vector[g_idx]

        self.ris_phase_response = self.ris_element_amplitude * np.exp(1j * self.ris_phase_vector)

        self._update_uav_geometry_large_scale()

        self._begin_macro_slot_refresh()
        if self.ris_power_control_mode == "power_budget":
            self._apply_ris_phase_action(ris_group_phase_action=None, ris_power_weights=None)

        for agent_id in self.agents:
            self.historical_sinrs_per_user[agent_id].fill(0.0)
            self.last_power_actions_per_user[agent_id].fill(0.0)

        self.historical_avg_rates.fill(self.fairness_epsilon)

        initial_observations = {}
        for agent_id in self.agents:
            initial_observations[agent_id] = self._get_next_observation_for_rb(agent_id)

        return initial_observations

    def _apply_ris_phase_action(
        self, ris_group_phase_action: np.ndarray | None, ris_power_weights: np.ndarray | None = None
    ) -> None:
        if ris_group_phase_action is not None:
            ris_group_phase_action = np.asarray(ris_group_phase_action, dtype=np.float64)
            if ris_group_phase_action.shape[0] != self.ris_num_groups:
                raise ValueError(
                    f"RIS phase action dim mismatch: expected {self.ris_num_groups}, "
                    f"got {ris_group_phase_action.shape[0]}"
                )

            clipped = np.clip(
                ris_group_phase_action,
                self.ris_phase_min,
                self.ris_phase_max,
            )

            self.ris_group_phase_vector[:] = clipped

            for g_idx, elem_idx_list in enumerate(self.ris_groups):
                self.ris_phase_vector[elem_idx_list] = self.ris_group_phase_vector[g_idx]

        weights = None
        if self.ris_power_control_mode == "power_budget":
            G = self.ris_num_groups

            if ris_power_weights is not None:
                w = np.asarray(ris_power_weights, dtype=np.float64)
            elif hasattr(self, "_last_ris_power_weights") and (
                self._last_ris_power_weights is not None
            ):
                w = np.asarray(self._last_ris_power_weights, dtype=np.float64)
            else:
                w = np.ones(G, dtype=np.float64)

            s = float(np.sum(w))
            if s > 1e-9:
                w = w / s
            else:
                w = np.ones(G, dtype=np.float64) / float(G)

            weights = w
            self._last_ris_power_weights = w.copy()
        else:
            self._last_ris_power_weights = None

        if (self.ris_power_control_mode != "power_budget") or (weights is None):
            current_amplitude_vector = np.full(
                self.ris_num_elems,
                self.ris_element_amplitude,
                dtype=np.float64,
            )
        else:
            sat_p_watt = getattr(
                self,
                "satellite_max_power_watt",
                self._dbm_to_watt(self.satellite_max_power_dBm),
            )

            abs_h_sr_sq = np.abs(self.small_scale_sat_to_ris) ** 2
            p_inc_signal_per_elem = sat_p_watt * self.large_scale_gain_sat_to_ris * abs_h_sr_sq

            p_input_for_budget_per_elem = p_inc_signal_per_elem

            current_amplitude_vector = np.zeros(self.ris_num_elems, dtype=np.float64)

            max_amp_linear = float(getattr(self, "ris_elem_max_amp_linear", 1e30))

            for g_idx, elem_idx_list in enumerate(self.ris_groups):
                p_target_group = float(self.ris_reflection_power_limit_watt * weights[g_idx])

                p_input_group_sum = float(np.sum(p_input_for_budget_per_elem[elem_idx_list]))

                if (p_target_group <= 0.0) or (p_input_group_sum <= 1e-12):
                    A_g = 0.0
                else:
                    A_g = float(np.sqrt(p_target_group / p_input_group_sum))

                    if A_g > max_amp_linear:
                        A_g = max_amp_linear

                current_amplitude_vector[elem_idx_list] = A_g

        self.ris_phase_response = current_amplitude_vector * np.exp(1j * self.ris_phase_vector)

    def _update_uav_geometry_large_scale(self):
        sat_xyz = np.array([0.0, 0.0, self.satellite_altitude_m], dtype=np.float64)
        uav_xyz = self.uav_position_xyz.astype(np.float64)

        d_sr = np.linalg.norm(sat_xyz - uav_xyz)
        path_gain_sr = (self.light_speed / (4.0 * np.pi * d_sr * self.carrier_frequency_Hz)) ** 2

        uav_rx_gain_linear = float(getattr(self, "uav_rx_antenna_gain_linear", 1.0))

        self.large_scale_gain_sat_to_ris = (
            path_gain_sr * self.satellite_antenna_gain_linear * uav_rx_gain_linear
        )

        self.ris_rx_element_gain_linear = 1.0

        new_ris_to_gs = np.zeros(self.num_ground_stations, dtype=np.float64)
        tx_elem_gains = np.ones(self.num_ground_stations, dtype=np.float64)

        for n in range(self.num_ground_stations):
            gs_xyz = np.array(
                [self.gs_positions[n, 0], self.gs_positions[n, 1], 0.0], dtype=np.float64
            )
            d_rg = np.linalg.norm(uav_xyz - gs_xyz)
            path_gain_rg = (
                self.light_speed / (4.0 * np.pi * d_rg * self.carrier_frequency_Hz)
            ) ** 2
            new_ris_to_gs[n] = path_gain_rg * self.ground_station_antenna_gain_linear
            tx_elem_gains[n] = 1.0

        self.large_scale_gain_ris_to_gs = new_ris_to_gs
        self.ris_tx_element_gain_linear = tx_elem_gains

        self.large_scale_gain_ris_path = (
            self.large_scale_gain_sat_to_ris * self.large_scale_gain_ris_to_gs
        )

    def _compute_effective_satellite_channel(self) -> np.ndarray:
        num_users = self.num_ground_stations

        if hasattr(self, "large_scale_gain_sat_to_gs_per_user"):
            g_sg = np.asarray(self.large_scale_gain_sat_to_gs_per_user, dtype=np.float64).reshape(
                -1
            )
        else:
            offset = np.asarray(
                getattr(self, "_sat_user_gain_offset_linear", np.ones(num_users)), dtype=np.float64
            ).reshape(-1)
            if offset.size != num_users:
                offset = np.ones(num_users, dtype=np.float64)
            g_sg = float(self.large_scale_gain_sat_to_gs) * offset

        direct_term = np.sqrt(g_sg) * self.small_scale_coeffs_sat_to_gs

        if getattr(self, "disable_ris_path", False):
            ris_term = np.zeros(num_users, dtype=np.complex128)
        else:
            pre_vec = (
                np.sqrt(self.large_scale_gain_sat_to_ris) * self.small_scale_sat_to_ris
            ) * self.ris_phase_response

            ris_term = np.zeros(num_users, dtype=np.complex128)
            for n in range(num_users):
                ris_term[n] = np.sqrt(self.large_scale_gain_ris_to_gs[n]) * np.sum(
                    pre_vec * self.small_scale_ris_to_gs[n]
                )

        h_eff = direct_term + ris_term

        p_dir = np.abs(direct_term) ** 2
        p_tot = np.abs(h_eff) ** 2
        eps = 1e-12
        boost_ratios = p_tot / (p_dir + eps)

        if getattr(self, "disable_ris_path", False):
            boost_ratios = np.ones_like(boost_ratios, dtype=np.float64)
            p_tot = p_dir

        self._last_boost_ratios_all_users = boost_ratios.astype(np.float64)
        self._last_boost_ratios_direct_only = p_dir.astype(np.float64)

        return h_eff

    def _compute_effective_satellite_channel_hat(self) -> np.ndarray:
        num_users = self.num_ground_stations

        if hasattr(self, "large_scale_gain_sat_to_gs_per_user"):
            g_sg = np.asarray(self.large_scale_gain_sat_to_gs_per_user, dtype=np.float64).reshape(
                -1
            )
        else:
            offset = np.asarray(
                getattr(self, "_sat_user_gain_offset_linear", np.ones(num_users)), dtype=np.float64
            ).reshape(-1)
            if offset.size != num_users:
                offset = np.ones(num_users, dtype=np.float64)
            g_sg = float(self.large_scale_gain_sat_to_gs) * offset

        if not hasattr(self, "estimated_small_scale_coeffs_sat_to_gs"):
            self.estimated_small_scale_coeffs_sat_to_gs = self.small_scale_coeffs_sat_to_gs.copy()
            self.estimated_small_scale_sat_to_ris = self.small_scale_sat_to_ris.copy()
            self.estimated_small_scale_ris_to_gs = self.small_scale_ris_to_gs.copy()

        direct_term_hat = np.sqrt(g_sg) * self.estimated_small_scale_coeffs_sat_to_gs

        if getattr(self, "disable_ris_path", False):
            ris_term_hat = np.zeros(num_users, dtype=np.complex128)
        else:
            pre_vec_hat = (
                np.sqrt(self.large_scale_gain_sat_to_ris) * self.estimated_small_scale_sat_to_ris
            ) * self.ris_phase_response

            ris_term_hat = np.zeros(num_users, dtype=np.complex128)
            for n in range(num_users):
                ris_term_hat[n] = np.sqrt(self.large_scale_gain_ris_to_gs[n]) * np.sum(
                    pre_vec_hat * self.estimated_small_scale_ris_to_gs[n]
                )

        return direct_term_hat + ris_term_hat

    def _compute_active_ris_noise_per_user(self) -> np.ndarray:
        if getattr(self, "disable_ris_path", False):
            self._last_active_ris_noise_per_user.fill(0.0)
            return self._last_active_ris_noise_per_user

        if (not self.active_ris_enabled) or (self.active_ris_noise_power_watt <= 0.0):
            self._last_active_ris_noise_per_user.fill(0.0)
            return self._last_active_ris_noise_per_user

        abs_psi_sq = np.abs(self.ris_phase_response) ** 2
        phi_fro_sq = float(np.sum(abs_psi_sq))

        h_rg_hat = getattr(self, "h_rg_hat", None)
        if h_rg_hat is None:
            h_rg_hat = (
                (np.sqrt(self.large_scale_gain_ris_to_gs)[:, None]) * self.small_scale_ris_to_gs
            )

        noise_per_user = np.zeros(self.num_ground_stations, dtype=np.float64)
        nmse = float(getattr(self, "csi_error_variance", 0.0) or 0.0)

        h_rg_abs_sq = np.abs(h_rg_hat) ** 2
        proj_sq = np.sum(h_rg_abs_sq * abs_psi_sq[None, :], axis=1)

        sigma_e_rg = nmse * np.asarray(self.large_scale_gain_ris_to_gs, dtype=np.float64).reshape(
            -1
        )

        noise_per_user = self.active_ris_noise_power_watt * (proj_sq + sigma_e_rg * phi_fro_sq)

        self._last_active_ris_noise_per_user = noise_per_user
        return noise_per_user

    def _compute_active_ris_noise_per_user_true(self) -> np.ndarray:
        if (not self.active_ris_enabled) or (self.active_ris_noise_power_watt <= 0.0):
            return np.zeros(self.num_ground_stations, dtype=np.float64)

        abs_psi_sq = np.abs(self.ris_phase_response) ** 2

        h_rg_true = (np.sqrt(self.large_scale_gain_ris_to_gs)[:, None]) * self.small_scale_ris_to_gs

        proj_sq = np.sum((np.abs(h_rg_true) ** 2) * abs_psi_sq[None, :], axis=1)
        return (self.active_ris_noise_power_watt * proj_sq).astype(np.float64)

    def get_ris_reflection_power(
        self, sat_tx_power_watt: float = None, include_noise: bool = True, cache: bool = True
    ) -> dict:
        if sat_tx_power_watt is None:
            sat_tx_power_watt = getattr(
                self, "satellite_max_power_watt", self._dbm_to_watt(self.satellite_max_power_dBm)
            )

        abs_h_sr_sq = np.abs(self.small_scale_sat_to_ris) ** 2
        p_inc_per_elem = sat_tx_power_watt * self.large_scale_gain_sat_to_ris * abs_h_sr_sq
        p_inc_total = float(np.sum(p_inc_per_elem))

        abs_psi_sq = np.abs(self.ris_phase_response) ** 2

        p_refl_sig_per_elem = abs_psi_sq * p_inc_per_elem
        p_refl_sig_total = float(np.sum(p_refl_sig_per_elem))

        if include_noise and self.active_ris_enabled and (self.active_ris_noise_power_watt > 0.0):
            p_refl_noise_per_elem = abs_psi_sq * self.active_ris_noise_power_watt
            p_refl_noise_total = float(np.sum(p_refl_noise_per_elem))
        else:
            p_refl_noise_per_elem = np.zeros_like(p_inc_per_elem)
            p_refl_noise_total = 0.0

        p_out_total = p_refl_sig_total + p_refl_noise_total

        result = {
            "p_inc_per_elem_watt": p_inc_per_elem,
            "p_refl_sig_per_elem_watt": p_refl_sig_per_elem,
            "p_refl_noise_per_elem_watt": p_refl_noise_per_elem,
            "p_inc_total_watt": p_inc_total,
            "p_refl_sig_total_watt": p_refl_sig_total,
            "p_refl_noise_total_watt": p_refl_noise_total,
            "p_out_total_watt": p_out_total,
            "p_inc_total_dBm": self._watt_to_dbm(p_inc_total),
            "p_refl_sig_total_dBm": self._watt_to_dbm(p_refl_sig_total),
            "p_refl_noise_total_dBm": self._watt_to_dbm(p_refl_noise_total),
            "p_out_total_dBm": self._watt_to_dbm(p_out_total),
            "per_elem_amplitude_A": np.sqrt(abs_psi_sq).astype(np.float64),
            "num_elements_M": int(self.ris_num_elems),
        }

        if cache:
            self._last_ris_power_dict = result
        return result

    def _get_current_channel_gains(self) -> Dict[str, np.ndarray]:
        _ = self._compute_effective_satellite_channel()

        h_eff_hat = self._compute_effective_satellite_channel_hat()
        sat_gains_hat = np.abs(h_eff_hat) ** 2

        bs_gains = self.large_scale_gain_bs_to_gs * (np.abs(self.small_scale_coeffs_bs_to_gs) ** 2)

        gains_dict = {
            "sat_to_gs": sat_gains_hat,
            "bs_to_gs": bs_gains,
        }

        if self.active_ris_enabled and self.active_ris_noise_power_watt > 0.0:
            gains_dict["active_ris_noise"] = self._compute_active_ris_noise_per_user()

        return gains_dict

    def _get_current_channel_gains_perfect(self) -> Dict[str, np.ndarray]:
        h_eff_true = self._compute_effective_satellite_channel()
        sat_gains_true = np.abs(h_eff_true) ** 2

        bs_gains = self.large_scale_gain_bs_to_gs * (np.abs(self.small_scale_coeffs_bs_to_gs) ** 2)

        gains_dict = {"sat_to_gs": sat_gains_true, "bs_to_gs": bs_gains}

        if self.active_ris_enabled and self.active_ris_noise_power_watt > 0.0:
            gains_dict["active_ris_noise"] = self._compute_active_ris_noise_per_user_true()

        return gains_dict

    def _update_channel_state(self):
        noise_std = np.sqrt(max(0.0, 1.0 - float(self.rho) ** 2))
        z = (
            self._rng.standard_normal(self.num_ground_stations)
            + 1j * self._rng.standard_normal(self.num_ground_stations)
        ) / np.sqrt(2.0)
        self.small_scale_coeffs_sat_to_gs = (
            float(self.rho) * self.small_scale_coeffs_sat_to_gs + noise_std * z
        )

    def _calculate_sinrs_for_rb(
        self,
        user_indices: List[int],
        m: int,
        power_map: Dict[Union[str, int], float],
        gains: Dict[str, np.ndarray],
    ) -> dict:
        served_users = list(user_indices)
        if len(served_users) == 0:
            return {
                "sinr_common": np.array([], dtype=np.float64),
                "sinr_private": np.array([], dtype=np.float64),
                "common_rx_power": np.array([], dtype=np.float64),
                "private_rx_powers": np.array([], dtype=np.float64),
                "total_private_rx_power_at_user": np.array([], dtype=np.float64),
                "internal_interference": np.array([], dtype=np.float64),
                "external_interference": np.array([], dtype=np.float64),
                "I_csi": np.array([], dtype=np.float64),
                "I_tot": np.array([], dtype=np.float64),
                "active_ris_noise": np.array([], dtype=np.float64),
            }

        if "sat_to_gs" not in gains:
            h_eff_hat = self._compute_effective_satellite_channel_hat()
            g_sat_all = np.abs(h_eff_hat) ** 2
        else:
            g_sat_all = gains["sat_to_gs"]

        g_sat = np.asarray(g_sat_all, dtype=np.float64)[served_users]

        g_bs_all = gains.get("bs_to_gs", None)
        if g_bs_all is None:
            g_bs_all = self.large_scale_gain_bs_to_gs * (
                np.abs(self.small_scale_coeffs_bs_to_gs) ** 2
            )
        g_bs_all = gains.get("bs_to_gs", None)
        if g_bs_all is None:
            g_bs_all = self.large_scale_gain_bs_to_gs * (
                np.abs(self.small_scale_coeffs_bs_to_gs) ** 2
            )

        g_bs_all = np.asarray(g_bs_all, dtype=np.float64)
        if g_bs_all.ndim == 2:
            g_bs = g_bs_all[served_users, m]
        else:
            g_bs = g_bs_all[served_users]
        external_interference = self.terrestrial_cell_power_watt * g_bs

        active_ris_noise_all = gains.get("active_ris_noise", None)
        if active_ris_noise_all is None:
            active_ris_noise = 0.0
        else:
            active_ris_noise = np.asarray(active_ris_noise_all, dtype=np.float64)[served_users]

        p_common = float(power_map.get("common", 0.0))
        p_privates = np.array([float(power_map[u]) for u in served_users], dtype=np.float64)
        total_private_tx_power = float(np.sum(p_privates))
        total_stream_power = p_common + total_private_tx_power

        nmse = float(getattr(self, "csi_error_variance", 0.0) or 0.0)

        if hasattr(self, "large_scale_gain_sat_to_gs_per_user"):
            ls_sg_vec = np.asarray(
                self.large_scale_gain_sat_to_gs_per_user, dtype=np.float64
            ).reshape(-1)
        else:
            offset = np.asarray(
                getattr(self, "_sat_user_gain_offset_linear", np.ones(self.num_ground_stations)),
                dtype=np.float64,
            ).reshape(-1)
            if offset.size != self.num_ground_stations:
                offset = np.ones(self.num_ground_stations, dtype=np.float64)
            ls_sg_vec = float(self.large_scale_gain_sat_to_gs) * offset

        var_sg = nmse * ls_sg_vec[served_users]
        var_sr = nmse * float(self.large_scale_gain_sat_to_ris)
        var_rg = (
            nmse
            * np.asarray(self.large_scale_gain_ris_to_gs, dtype=np.float64).reshape(-1)[
                served_users
            ]
        )

        abs_psi_sq = np.abs(self.ris_phase_response) ** 2
        phi_fro_sq = float(np.sum(abs_psi_sq))

        h_sr_hat = getattr(self, "h_sr_hat", None)
        h_rg_hat = getattr(self, "h_rg_hat", None)
        if h_sr_hat is None or h_rg_hat is None:
            h_sr_hat = np.sqrt(self.large_scale_gain_sat_to_ris) * self.small_scale_sat_to_ris
            h_rg_hat = (
                (np.sqrt(self.large_scale_gain_ris_to_gs)[:, None]) * self.small_scale_ris_to_gs
            )

        norm_hrg_phi_sq = np.sum(
            (np.abs(h_rg_hat[served_users, :]) ** 2) * abs_psi_sq[None, :], axis=1
        )

        norm_phi_hsr_sq = float(np.sum(abs_psi_sq * (np.abs(h_sr_hat) ** 2)))

        eeq2 = (
            var_sg
            + (var_sr * norm_hrg_phi_sq)
            + (var_rg * norm_phi_hsr_sq)
            + (var_rg * var_sr * phi_fro_sq)
        )

        I_csi = total_stream_power * eeq2

        I_tot = external_interference + self.noise_power_watt + active_ris_noise + I_csi

        common_rx_power = p_common * g_sat
        private_rx_powers = p_privates * g_sat
        total_private_rx_power_at_user = total_private_tx_power * g_sat
        internal_interference = (total_private_tx_power - p_privates) * g_sat

        eps = 1e-12
        denom_common = total_private_rx_power_at_user + I_tot
        sinr_common = common_rx_power / (denom_common + eps)

        denom_private = internal_interference + I_tot
        sinr_private = private_rx_powers / (denom_private + eps)

        return {
            "sinr_common": sinr_common,
            "sinr_private": sinr_private,
            "common_rx_power": common_rx_power,
            "private_rx_powers": private_rx_powers,
            "total_private_rx_power_at_user": total_private_rx_power_at_user,
            "internal_interference": internal_interference,
            "external_interference": external_interference,
            "I_csi": I_csi,
            "I_tot": I_tot,
            "active_ris_noise": active_ris_noise,
        }

    def _calculate_rates_for_rb(self, sinr_results: dict) -> dict:
        active_ris_noise = sinr_results.get("active_ris_noise", 0.0)
        I_csi = sinr_results.get("I_csi", 0.0)

        if "I_tot" in sinr_results:
            I_tot = sinr_results["I_tot"]
        else:
            I_tot = (
                sinr_results.get("external_interference", 0.0)
                + self.noise_power_watt
                + active_ris_noise
                + I_csi
            )

        power_difference = (
            sinr_results["common_rx_power"] - sinr_results["total_private_rx_power_at_user"] - I_tot
        )
        sic_success = np.all(power_difference >= self.sic_detection_threshold_watt)

        if sic_success:
            if len(sinr_results.get("sinr_common", [])) > 0:
                common_rate = np.min(np.log2(1.0 + sinr_results["sinr_common"]))
            else:
                common_rate = 0.0

            private_rates = np.log2(1.0 + sinr_results.get("sinr_private", 0.0))

            total_rates = (common_rate / self.num_gs_per_rb) + private_rates
        else:
            common_rate = 0.0
            _p = np.asarray(sinr_results.get("sinr_private", 0.0))
            private_rates = np.zeros_like(_p, dtype=np.float64)
            total_rates = np.zeros_like(_p, dtype=np.float64)

        return {
            "common_rate": common_rate,
            "private_rates": private_rates,
            "total_rates": total_rates,
            "sic_success": sic_success,
        }

    def _build_wmmse_context_bundle(
        self,
        served_users: List[int],
        rb_index: int,
    ) -> Dict[str, Any]:
        served_users = [int(u) for u in served_users]
        rb_index = int(rb_index)

        if len(served_users) == 0:
            raise ValueError("served_users must be non-empty for WMMSE context export.")

        if rb_index < 0 or rb_index >= int(self.num_resource_blocks):
            raise ValueError(
                f"rb_index={rb_index} out of range [0, {self.num_resource_blocks - 1}]."
            )

        required_attrs = [
            "ris_phase_response",
            "h_sg_hat",
            "h_sr_hat",
            "h_rg_hat",
            "large_scale_gain_sat_to_gs_per_user",
            "large_scale_gain_sat_to_ris",
            "large_scale_gain_ris_to_gs",
            "small_scale_coeffs_bs_to_gs",
            "large_scale_gain_bs_to_gs",
        ]
        for name in required_attrs:
            if not hasattr(self, name):
                raise RuntimeError(
                    f"WMMSE export failed: missing required env attribute '{name}'. "
                    f"Please ensure env.reset() has been called and channel caches are initialized."
                )

        served_arr = np.asarray(served_users, dtype=np.int64)

        ris_phase_response = np.asarray(self.ris_phase_response, dtype=np.complex128)

        h_sg_hat_all = np.asarray(self.h_sg_hat, dtype=np.complex128).reshape(-1)
        h_sr_hat = np.asarray(self.h_sr_hat, dtype=np.complex128).reshape(-1)
        h_rg_hat_all = np.asarray(self.h_rg_hat, dtype=np.complex128)

        h_sg_hat_sel = h_sg_hat_all[served_arr]
        h_rg_hat_sel = h_rg_hat_all[served_arr, :]

        ls_sg_all = np.asarray(self.large_scale_gain_sat_to_gs_per_user, dtype=np.float64).reshape(
            -1
        )
        ls_sr = float(self.large_scale_gain_sat_to_ris)
        ls_rg_all = np.asarray(self.large_scale_gain_ris_to_gs, dtype=np.float64).reshape(-1)

        ls_sg_sel = ls_sg_all[served_arr]
        ls_rg_sel = ls_rg_all[served_arr]

        g_bs_full = np.asarray(self.large_scale_gain_bs_to_gs, dtype=np.float64) * (
            np.abs(np.asarray(self.small_scale_coeffs_bs_to_gs, dtype=np.complex128)) ** 2
        )

        if g_bs_full.ndim != 2:
            raise RuntimeError(
                f"Expected BS->GS gain tensor shape [N_users, N_RB], got {g_bs_full.shape}."
            )

        g_bs_perfect_sel = g_bs_full[served_arr, rb_index]

        h_eff_hat_all = np.asarray(
            self._compute_effective_satellite_channel_hat(), dtype=np.complex128
        ).reshape(-1)
        h_eff_hat_sel = h_eff_hat_all[served_arr]
        g_eff_hat_sel = np.abs(h_eff_hat_sel) ** 2

        active_ris_noise_all = np.asarray(
            self._compute_active_ris_noise_per_user(), dtype=np.float64
        ).reshape(-1)
        active_ris_noise_sel = active_ris_noise_all[served_arr]

        obs_casc_groups_sel = None
        if hasattr(self, "_obs_casc_groups"):
            obs_casc_groups_sel = np.asarray(self._obs_casc_groups, dtype=np.complex128)[served_arr]

        obs_sr_incident_pow_groups = None
        if hasattr(self, "_obs_sr_incident_pow_groups"):
            obs_sr_incident_pow_groups = np.asarray(
                self._obs_sr_incident_pow_groups, dtype=np.float64
            ).copy()

        ctx = {
            "served_users": served_arr.copy(),
            "rb_index": int(rb_index),
            "num_served_users": int(len(served_users)),
            "num_total_users": int(self.num_ground_stations),
            "num_resource_blocks": int(self.num_resource_blocks),
            "num_ris_elems": int(self.ris_num_elems),
            "num_ris_groups": int(self.ris_num_groups),
            "ris_phase_response": ris_phase_response.copy(),
            "ris_phase_vector": np.asarray(self.ris_phase_vector, dtype=np.float64).copy(),
            "ris_group_phase_vector": np.asarray(
                self.ris_group_phase_vector, dtype=np.float64
            ).copy(),
            "h_sg_hat_sel": h_sg_hat_sel.copy(),
            "h_sr_hat": h_sr_hat.copy(),
            "h_rg_hat_sel": h_rg_hat_sel.copy(),
            "h_eff_hat_sel": h_eff_hat_sel.copy(),
            "g_eff_hat_sel": g_eff_hat_sel.copy(),
            "ls_sg_sel": ls_sg_sel.copy(),
            "ls_sr": float(ls_sr),
            "ls_rg_sel": ls_rg_sel.copy(),
            "g_bs_perfect_sel": np.asarray(g_bs_perfect_sel, dtype=np.float64).copy(),
            "noise_power_watt": float(self.noise_power_watt),
            "satellite_max_power_watt": float(self.satellite_max_power_watt),
            "terrestrial_cell_power_watt": float(self.terrestrial_cell_power_watt),
            "sic_detection_threshold_watt": float(self.sic_detection_threshold_watt),
            "csi_error_variance": float(self.csi_error_variance),
            "active_ris_enabled": bool(self.active_ris_enabled),
            "active_ris_noise_power_watt": float(self.active_ris_noise_power_watt),
            "disable_ris_path": bool(self.disable_ris_path),
            "active_ris_noise_sel": active_ris_noise_sel.copy(),
            "current_step": int(self.current_step),
            "rho": float(self.rho),
            "obs_casc_groups_sel": None
            if obs_casc_groups_sel is None
            else obs_casc_groups_sel.copy(),
            "obs_sr_incident_pow_groups": (
                None if obs_sr_incident_pow_groups is None else obs_sr_incident_pow_groups.copy()
            ),
        }

        return ctx

    def export_wmmse_context_for_rb(
        self,
        user_indices: List[int],
        rb_index: int,
        *,
        copy_arrays: bool = True,
    ) -> Dict[str, Any]:
        """Export the current solver inputs without advancing or modifying the environment."""
        ctx = self._build_wmmse_context_bundle(
            served_users=list(user_indices),
            rb_index=int(rb_index),
        )

        if copy_arrays:
            safe_ctx: Dict[str, Any] = {}
            for k, v in ctx.items():
                if isinstance(v, np.ndarray):
                    safe_ctx[k] = v.copy()
                else:
                    safe_ctx[k] = v
            return safe_ctx

        return ctx

    def preview_perfect_csi_infos(self, actions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        saved_nmse = float(getattr(self, "csi_error_variance", 0.0) or 0.0)

        saved_group_phase = self.ris_group_phase_vector.copy()
        saved_phase_vec = self.ris_phase_vector.copy()
        saved_phase_resp = self.ris_phase_response.copy()
        had_last_weights = hasattr(self, "_last_ris_power_weights")
        saved_last_weights = getattr(self, "_last_ris_power_weights", None)

        try:
            for _, act in actions.items():
                if isinstance(act, (list, tuple)) and len(act) >= 3 and act[2] is not None:
                    theta = act[2]
                    w = act[3] if len(act) >= 4 else None
                    self._apply_ris_phase_action(theta, w)
                    break

            gains_true = self._get_current_channel_gains_perfect()

            self.csi_error_variance = 0.0

            infos = {}
            for agent_id, act in actions.items():
                served_users = list(map(int, act[0]))
                power_vector = np.asarray(act[1], dtype=np.float64)

                m = int(agent_id.split("_")[-1]) if agent_id.startswith("rb_") else 0

                power_map = {"common": float(power_vector[0])}
                for i, u in enumerate(served_users):
                    power_map[int(u)] = float(power_vector[1 + i])

                sinr_results = self._calculate_sinrs_for_rb(served_users, m, power_map, gains_true)
                rates_results = self._calculate_rates_for_rb(sinr_results)

                full_rates = np.zeros(self.num_ground_stations, dtype=np.float64)
                if len(served_users) > 0:
                    full_rates[np.asarray(served_users, dtype=int)] = np.asarray(
                        rates_results["total_rates"], dtype=np.float64
                    )

                infos[agent_id] = {
                    "users_served": served_users,
                    "total_rates_per_user": full_rates,
                    "common_rate": float(rates_results["common_rate"]),
                    "private_rates": np.asarray(rates_results["private_rates"], dtype=np.float64),
                    "sic_success": bool(rates_results.get("sic_success", True)),
                }
            return infos

        finally:
            self.csi_error_variance = saved_nmse
            self.ris_group_phase_vector = saved_group_phase
            self.ris_phase_vector = saved_phase_vec
            self.ris_phase_response = saved_phase_resp
            if had_last_weights:
                self._last_ris_power_weights = saved_last_weights
            else:
                if hasattr(self, "_last_ris_power_weights"):
                    delattr(self, "_last_ris_power_weights")

    def _calculate_reward_for_rb(
        self,
        rates_results: dict,
        num_served_users: int,
        served_user_indices: np.ndarray,
        fairness_weights: np.ndarray = None,
    ) -> float:
        num_served = int(num_served_users)

        sic_success = bool(rates_results.get("sic_success", True))
        if not sic_success:
            return 0.0

        total_rates_all = np.asarray(
            rates_results.get("total_rates", np.zeros(num_served, dtype=np.float64)),
            dtype=np.float64,
        )

        served_idx = np.asarray(served_user_indices, dtype=int)

        if total_rates_all.size == num_served:
            served_rates = total_rates_all
        else:
            try:
                served_rates = total_rates_all[served_idx]
            except Exception:
                served_rates = total_rates_all[:num_served]

        if num_served > 0:
            rsma_reward = float(np.sum(served_rates))
        else:
            rsma_reward = 0.0

        total_reward = rsma_reward

        lambda_boost = float(getattr(self, "ris_boost_lambda", 0.0))
        if lambda_boost > 0.0 and num_served > 0:
            boost_all = getattr(self, "_last_boost_ratios_all_users", None)

            if boost_all is not None:
                boost_all = np.asarray(boost_all, dtype=np.float64)

                try:
                    boost_served = boost_all[served_idx]
                except Exception:
                    boost_served = boost_all[:num_served]

                boost_clip = 50.0
                boost_served_clipped = np.minimum(boost_served, boost_clip)

                delta_boost = np.maximum(boost_served_clipped - 1.0, 0.0)

                r_boost_per_user = np.log1p(delta_boost)

                if r_boost_per_user.size > 0:
                    r_boost = float(np.mean(r_boost_per_user))
                else:
                    r_boost = 0.0

                total_reward += lambda_boost * r_boost

        return float(total_reward)

    def _get_next_observation_for_rb(self, agent_id: str) -> np.ndarray:
        _ = self.historical_sinrs_per_user[agent_id]
        _ = self.last_power_actions_per_user[agent_id]

        self._update_twohop_group_csi_observation()

        csi_feat_blocks = []

        for ui in range(self.num_ground_stations):
            casc_feats_i = self._complex_to_feats(self._obs_casc_groups[ui])
            csi_feat_blocks.append(casc_feats_i.reshape(-1))

        h_dir_obs = self.estimated_small_scale_coeffs_sat_to_gs.astype(np.complex128)
        dir_feats = self._complex_to_feats(h_dir_obs)
        csi_feat_blocks.append(dir_feats.reshape(-1))

        est_csi_feats = np.concatenate(csi_feat_blocks, axis=0).astype(np.float32)

        if getattr(self, "include_sr_incident_pow_groups", False):
            inc_pow = self._obs_sr_incident_pow_groups.astype(np.float32)
            inc_pow_feat = np.log(inc_pow + 1e-12).astype(np.float32)
        else:
            inc_pow_feat = None

        uav_rel_xy_feat, uav_z_feat = self._uav_rel_xy_per_gs_feature()

        hist_rate_feat = np.log(self.historical_avg_rates + self.fairness_epsilon).astype(
            np.float32
        )

        obs_parts = [
            est_csi_feats.astype(np.float32),
            uav_rel_xy_feat.astype(np.float32),
            uav_z_feat.astype(np.float32),
            hist_rate_feat,
        ]

        if inc_pow_feat is not None:
            obs_parts.insert(1, inc_pow_feat)

        if self.include_angle_features:
            angle_feats = self._uav_direction_cosines_to_gs()
            obs_parts.append(angle_feats.astype(np.float32))

        ls_sr = np.array([float(self.large_scale_gain_sat_to_ris)], dtype=np.float64)
        ls_rg = np.asarray(self.large_scale_gain_ris_to_gs, dtype=np.float64).reshape(-1)

        if hasattr(self, "large_scale_gain_sat_to_gs_per_user"):
            ls_sg = np.asarray(self.large_scale_gain_sat_to_gs_per_user, dtype=np.float64).reshape(
                -1
            )
        else:
            offset = np.asarray(
                getattr(self, "_sat_user_gain_offset_linear", np.ones(self.num_ground_stations)),
                dtype=np.float64,
            ).reshape(-1)
            if offset.size != self.num_ground_stations:
                offset = np.ones(self.num_ground_stations, dtype=np.float64)
            ls_sg = float(self.large_scale_gain_sat_to_gs) * offset

        ls_feat = np.concatenate(
            [
                self._ls_gain_to_db_norm(ls_sr),
                self._ls_gain_to_db_norm(ls_rg),
                self._ls_gain_to_db_norm(ls_sg),
            ],
            axis=0,
        ).astype(np.float32)

        obs_parts.insert(1, ls_feat)

        obs_base = np.concatenate(obs_parts, axis=-1).astype(np.float32)
        base_obs_dim = int(obs_base.shape[0])

        if self._base_obs_dim is None:
            self._base_obs_dim = base_obs_dim

        n_users = int(self.num_ground_stations)
        G = int(self.ris_num_groups)

        casc_feats_per_user = int(3 * G)
        direct_feats_per_user = 3
        mrt_feats_per_user = int(2 * G)

        if n_users <= 0:
            raise ValueError(f"Invalid num_ground_stations: {n_users}")
        if G <= 0:
            raise ValueError(f"Invalid ris_num_groups: {G}")

        if uav_rel_xy_feat.shape[0] % n_users != 0:
            raise ValueError(
                f"uav_rel_xy_feat dim {uav_rel_xy_feat.shape[0]} cannot be evenly split by n_users={n_users}"
            )
        if hist_rate_feat.shape[0] % n_users != 0:
            raise ValueError(
                f"hist_rate_feat dim {hist_rate_feat.shape[0]} cannot be evenly split by n_users={n_users}"
            )

        uav_xy_feats_per_user = int(uav_rel_xy_feat.shape[0] // n_users)
        hist_rate_feats_per_user = int(hist_rate_feat.shape[0] // n_users)

        offset = 0

        casc_slice = (offset, offset + n_users * casc_feats_per_user)
        offset = casc_slice[1]

        direct_slice = (offset, offset + n_users * direct_feats_per_user)
        offset = direct_slice[1]

        csi_slice = (0, offset)

        ls_slice = (offset, offset + int(ls_feat.shape[0]))
        offset = ls_slice[1]

        inc_pow_slice = None
        if inc_pow_feat is not None:
            inc_pow_slice = (offset, offset + int(inc_pow_feat.shape[0]))
            offset = inc_pow_slice[1]

        uav_xy_slice = (offset, offset + int(uav_rel_xy_feat.shape[0]))
        offset = uav_xy_slice[1]

        uav_z_slice = (offset, offset + int(uav_z_feat.shape[0]))
        offset = uav_z_slice[1]

        hist_rate_slice = (offset, offset + int(hist_rate_feat.shape[0]))
        offset = hist_rate_slice[1]

        angle_slice = None
        if self.include_angle_features:
            angle_dim = int(angle_feats.shape[0])
            angle_slice = (offset, offset + angle_dim)
            offset = angle_slice[1]

        if offset != base_obs_dim:
            raise ValueError(
                f"Base obs layout mismatch: offset={offset}, base_obs_dim={base_obs_dim}"
            )

        layout_base = {
            "obs_dim": int(base_obs_dim),
            "base_obs_dim": int(base_obs_dim),
            "n_users": int(n_users),
            "k_users_per_rb": int(self.num_gs_per_rb),
            "ris_num_groups": int(G),
            "csi_slice": tuple(csi_slice),
            "casc_slice": tuple(casc_slice),
            "direct_slice": tuple(direct_slice),
            "ls_slice": tuple(ls_slice),
            "uav_xy_slice": tuple(uav_xy_slice),
            "uav_z_slice": tuple(uav_z_slice),
            "hist_rate_slice": tuple(hist_rate_slice),
            "casc_feats_per_user": int(casc_feats_per_user),
            "direct_feats_per_user": int(direct_feats_per_user),
            "uav_xy_feats_per_user": int(uav_xy_feats_per_user),
            "hist_rate_feats_per_user": int(hist_rate_feats_per_user),
            "mrt_feats_per_user": int(mrt_feats_per_user),
            "include_sr_incident_pow_groups": bool(inc_pow_feat is not None),
            "include_angle_features": bool(self.include_angle_features),
            "include_mrt_features": False,
        }

        if inc_pow_slice is not None:
            layout_base["inc_pow_slice"] = tuple(inc_pow_slice)
        if angle_slice is not None:
            layout_base["angle_slice"] = tuple(angle_slice)

        self._obs_layout_spec_base = dict(layout_base)

        mrt_total_dim = int(n_users * mrt_feats_per_user)
        layout_full = dict(layout_base)
        layout_full["mrt_slice"] = (base_obs_dim, base_obs_dim + mrt_total_dim)
        layout_full["obs_dim"] = int(base_obs_dim + mrt_total_dim)
        layout_full["include_mrt_features"] = True
        self._obs_layout_spec_full = layout_full

        if getattr(self, "include_mrt_features", False):
            mrt_feat = self._compute_direct_aware_mrt_feats()

            expect_dim = int(self.num_ground_stations) * int(self.ris_num_groups) * 2
            if mrt_feat.shape[0] != expect_dim:
                raise ValueError(
                    f"MRT feat dim mismatch: expect {expect_dim}, got {mrt_feat.shape[0]}"
                )

            self._last_mrt_feat = mrt_feat.astype(np.float32).copy()
            obs = np.concatenate([obs_base, self._last_mrt_feat], axis=-1).astype(np.float32)
        else:
            self._last_mrt_feat = None
            obs = obs_base

        return obs

    def get_base_obs_dim(self) -> Optional[int]:
        return self._base_obs_dim

    def get_obs_layout_spec(
        self, include_mrt_features: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """Return slices for the base observation and optional MRT features."""
        base_spec = getattr(self, "_obs_layout_spec_base", None)
        full_spec = getattr(self, "_obs_layout_spec_full", None)

        if base_spec is None:
            return None

        if include_mrt_features is None:
            use_full = bool(getattr(self, "include_mrt_features", False))
        else:
            use_full = bool(include_mrt_features)

        spec = full_spec if use_full else base_spec
        if spec is None:
            spec = base_spec

        return dict(spec)

    def get_mrt_features(self) -> np.ndarray:
        if getattr(self, "_last_mrt_feat", None) is not None:
            return self._last_mrt_feat.copy()
        return self._compute_direct_aware_mrt_feats().astype(np.float32)

    def step(self, actions: Dict[str, Any]) -> tuple:
        """Apply selected users, common/private powers in watts, and optional grouped RIS phases."""

        rewards = {}
        infos = {}
        next_observations = {}

        ris_phase_action_to_apply = None
        ris_weights_to_apply = None
        parsed_actions = {}

        for agent_id, agent_action in actions.items():
            if len(agent_action) == 4:
                user_indices, power_vec, ris_phase_vec, ris_power_weights = agent_action
            elif len(agent_action) == 3:
                user_indices, power_vec, ris_phase_vec = agent_action
                ris_power_weights = None
            elif len(agent_action) == 2:
                user_indices, power_vec = agent_action
                ris_phase_vec = None
                ris_power_weights = None
            else:
                raise ValueError(
                    f"Action for {agent_id} must be tuple of len 2, 3 or 4, got len={len(agent_action)}."
                )

            users = tuple(user_indices)
            if len(users) != self.num_gs_per_rb:
                raise ValueError(
                    f"Action for {agent_id} must select exactly K={self.num_gs_per_rb} users, "
                    f"got {len(users)}"
                )
            if any(not isinstance(user, (int, np.integer)) for user in users):
                raise ValueError(
                    f"Action for {agent_id} contains a non-integer user index: {users}"
                )
            users = tuple(int(user) for user in users)
            if len(set(users)) != len(users):
                raise ValueError(f"Action for {agent_id} contains duplicate users: {users}")
            if any(user < 0 or user >= self.num_ground_stations for user in users):
                raise ValueError(
                    f"Action for {agent_id} contains a user outside [0, {self.num_ground_stations}): {users}"
                )

            local_power = np.asarray(power_vec)
            expected_power_dim = 1 + len(users)
            if local_power.ndim != 1 or local_power.shape[0] != expected_power_dim:
                raise ValueError(
                    f"Action for {agent_id} must provide a local power vector of shape "
                    f"({expected_power_dim},), got {local_power.shape}"
                )
            if not np.all(np.isfinite(local_power)):
                raise ValueError(f"Action for {agent_id} contains non-finite power values")

            parsed_actions[agent_id] = (users, local_power)

            if ris_phase_action_to_apply is None and ris_phase_vec is not None:
                ris_phase_action_to_apply = ris_phase_vec

            if ris_weights_to_apply is None and ris_power_weights is not None:
                ris_weights_to_apply = ris_power_weights

        self._apply_ris_phase_action(ris_phase_action_to_apply, ris_weights_to_apply)

        K = int(self.num_ground_stations)
        N_micro = int(getattr(self, "num_micro_steps", 20))
        if N_micro < 1:
            N_micro = 1

        if self.num_resource_blocks != 1:
            raise RuntimeError("Implementation A (current patch) assumes num_resource_blocks == 1.")

        base_reward_sum = 0.0
        rates_vec_sum = np.zeros((K,), dtype=np.float64)

        _last_sinr_results = None
        _last_rates_results = None
        _last_power_map = None
        _last_user_indices = None

        for mi in range(N_micro):
            self._advance_sat_to_gs_small_scale_one_micro()
            self._apply_fixed_error_to_build_estimates()

            current_gains = self._get_current_channel_gains()

            for agent_id, (user_indices, local_power_vector) in parsed_actions.items():
                m = int(agent_id.split("_")[1])

                num_served = len(user_indices)
                if num_served == 0:
                    base_reward_micro = 0.0
                    rates_vec_micro = np.zeros((K,), dtype=np.float64)
                    sinr_results = None
                    rates_results = {
                        "sic_success": True,
                        "total_rates": rates_vec_micro,
                        "common_rate": 0.0,
                        "private_rates": np.zeros((0,), dtype=np.float64),
                    }
                    power_map = {"common": 0.0}
                else:
                    power_map = {"common": float(local_power_vector[0])}
                    for i_user, u_idx in enumerate(user_indices):
                        power_map[u_idx] = float(local_power_vector[i_user + 1])

                    sinr_results = self._calculate_sinrs_for_rb(
                        user_indices=user_indices, m=m, power_map=power_map, gains=current_gains
                    )
                    rates_results = self._calculate_rates_for_rb(sinr_results)

                    if not bool(rates_results.get("sic_success", True)):
                        base_reward_micro = 0.0
                        rates_vec_micro = np.zeros((K,), dtype=np.float64)
                        self.historical_sinrs_per_user[agent_id].fill(0.0)
                    else:
                        full_rates_vector = np.zeros((K,), dtype=np.float64)
                        if len(user_indices) > 0:
                            full_rates_vector[list(user_indices)] = rates_results["total_rates"]
                        rates_results["total_rates"] = full_rates_vector
                        rates_vec_micro = full_rates_vector

                        if self.fairness_alpha > 0.0:
                            current_weights = np.power(
                                self.historical_avg_rates + self.fairness_epsilon,
                                -self.fairness_alpha,
                            )
                            current_weights = np.clip(current_weights, 0.0, 30.0)
                        else:
                            current_weights = np.ones((K,), dtype=np.float64)

                        base_reward_micro = self._calculate_reward_for_rb(
                            rates_results,
                            num_served_users=num_served,
                            served_user_indices=np.array(user_indices, dtype=int),
                            fairness_weights=current_weights,
                        )

                        current_sinrs = np.zeros((K, 2), dtype=np.float32)
                        sc = np.atleast_1d(
                            np.asarray(sinr_results["sinr_common"], dtype=np.float32)
                        )
                        sp = np.atleast_1d(
                            np.asarray(sinr_results["sinr_private"], dtype=np.float32)
                        )
                        sinrs_of_served = np.stack([sc, sp], axis=-1)
                        idx_arr = np.array(user_indices, dtype=int)
                        current_sinrs[idx_arr] = sinrs_of_served
                        self.historical_sinrs_per_user[agent_id] = current_sinrs

                base_reward_sum += float(base_reward_micro)
                rates_vec_sum += np.asarray(rates_vec_micro, dtype=np.float64)

                _last_sinr_results = sinr_results
                _last_rates_results = rates_results
                _last_power_map = power_map
                _last_user_indices = user_indices

        base_reward_macro = base_reward_sum / float(N_micro)
        rates_macro_vec = rates_vec_sum / float(N_micro)

        beta = float(self.fairness_beta)
        self.historical_avg_rates = (
            1.0 - beta
        ) * self.historical_avg_rates + beta * rates_macro_vec

        if self.fairness_lambda > 0.0:
            avg_rates = self.historical_avg_rates
            mean_rate = float(np.mean(avg_rates))
            G_dev = float(np.mean(np.abs(avg_rates - mean_rate)))
        else:
            G_dev = 0.0

        final_reward = float(base_reward_macro - self.fairness_lambda * G_dev)

        for agent_id in self.agents:
            rewards[agent_id] = final_reward
            infos[agent_id] = infos.get(agent_id, {})
            infos[agent_id].update(
                {
                    "base_reward_macro": float(base_reward_macro),
                    "G_dev": float(G_dev),
                    "rates_macro_vec": rates_macro_vec.copy(),
                    "users_served": _last_user_indices,
                    "sic_success_last_micro": bool(_last_rates_results.get("sic_success", True))
                    if isinstance(_last_rates_results, dict)
                    else True,
                    "power_alloc_watt_last_micro": _last_power_map,
                }
            )

        for agent_id, (user_indices, local_power_vector) in parsed_actions.items():
            current_powers = np.zeros(self.power_action_dim, dtype=np.float32)
            if local_power_vector is not None and len(user_indices) > 0:
                current_powers[0] = float(local_power_vector[0])
                private_slot_indices = np.array(user_indices, dtype=int) + 1
                current_powers[private_slot_indices] = local_power_vector[1:].astype(np.float32)
            self.last_power_actions_per_user[agent_id] = current_powers

        self.uav_position_xyz += self.uav_velocity_per_step_xyz
        self._update_uav_geometry_large_scale()
        self._begin_macro_slot_refresh()

        for agent_id in self.agents:
            next_observations[agent_id] = self._get_next_observation_for_rb(agent_id)

        self.current_step += 1
        done_flag = self.current_step >= self.max_steps_per_episode
        time_limit_truncated = bool(done_flag)
        terminated = False

        for agent_id in self.agents:
            if agent_id not in infos or infos[agent_id] is None:
                infos[agent_id] = {}
            infos[agent_id]["time_limit_truncated"] = time_limit_truncated
            infos[agent_id]["terminated"] = terminated

        dones = {"__all__": done_flag}
        return next_observations, rewards, dones, infos


if __name__ == "__main__":
    test_config = {
        "num_ground_stations": 4,
        "num_resource_blocks": 1,
        "num_gs_per_rb": 2,
        "deployment_area_size_m": 400.0,
        "random_seed": 42,
        "reseed_each_episode": True,
        "max_steps_per_episode": 1024,
        "satellite_max_power_dBm": 50.0,
        "terrestrial_cell_power_dBm": 35.0,
        "noise_power_dBm": -114.0,
        "sic_detection_threshold_dBm": -94.0,
        "carrier_frequency_GHz": 4.0,
        "satellite_altitude_m": 600e3,
        "satellite_antenna_gain_dBi": 24.3,
        "ground_station_antenna_gain_dBi": 24.3,
        "terrestrial_cell_antenna_gain_dBi": 0,
        "log_normal_shadowing_variance_dB": 8.0,
        "rho": 0.95,
        "csi_error_variance": 0.01,
        "min_rate_requirement_bps_hz": 0.1,
        "use_fixed_positions": True,
        "ris_num_elem_x": 8,
        "ris_num_elem_y": 8,
        "ris_element_amplitude": 500.0,
        "ris_elem_spacing_ratio": 0.5,
        "active_ris_enabled": True,
        "active_ris_noise_power_dBm": -100,
        "rician_k_factor": 10.0,
        "k_factor_ris_to_gs_linear": 10.0,
        "k_factor_sat_to_ris_linear": 30.0,
        "uav_initial_position_xyz": [0.0, 0.0, 80.0],
        "uav_velocity_per_step_xyz": [0.282842712, 0.282842712, 0.0],
        "focus_user_idx": 2,
        "track_ris_every_step": True,
        "focus_plot_percentile_hi": 99.0,
        "focus_plot_percentile_lo": 1.0,
        "focus_plot_ylim_pad": 0.05,
    }
