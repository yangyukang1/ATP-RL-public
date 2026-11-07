from __future__ import annotations

from dataclasses import asdict, fields
from argparse import Namespace
import math
from atprl.solvers.conventional_options import SolverOptions
from atprl.envs.topologies import resolve_topology

EVALUATION_DEFAULTS = dict(
    baseline_name="rr_ao_wmmse_sdr",
    phase_solver="sdr",
    ris_init="mrt",
    budget_profile="smoke",
    num_ground_stations=4,
    num_gs_per_rb=2,
    topology_profile="paper_n4",
    rr_initial_offset=0,
    rr_reset_each_episode=True,
    output_dir=".",
    resume_run_dir=None,
    checkpoint_every_steps=64,
    num_eval_seeds=500,
    eval_seed_base=10001,
    eval_seed_stride=37,
    evaluation_repeats=1,
    max_steps_per_episode=1024,
    diagnostic_sample_limit=256,
    debug_decision_limit=4,
    device="cpu",
    num_threads=1,
    show_progress=True,
    fairness_lambda=3.0,
    fairness_beta=0.05,
    fairness_alpha=0.0,
    fairness_epsilon=0.01,
    csi_error_variance=0.01,
    sic_detection_threshold_dBm=-94.0,
    sic_penalty=0.0,
    ris_reflection_power_limit_dBm=-5.0,
    active_ris_noise_power_dBm=-100.0,
    uav_rx_antenna_gain_dBi=15.0,
    ris_elem_max_amp_gain_dB=40.0,
    ris_num_elem_x=16,
    ris_num_elem_y=16,
    min_rate_req=0.1,
    rho=0.95,
    force_bs_gain_db=None,
    sat_user_gain_db_offsets=None,
    lambda_boost=0.0,
    satellite_max_power_dBm=50.0,
    terrestrial_cell_power_dBm=35.0,
    noise_power_dBm=-114.0,
    num_micro_steps=20,
    decision_interval_s=0.04,
    residual_doppler_hz=50.0,
)
DEFAULTS = EVALUATION_DEFAULTS | asdict(SolverOptions())
METHOD_TO_PHASE = {"rr_ao_wmmse_sdr": "sdr"}


def resolve_method_phase(baseline_name=None, phase_solver=None):
    if baseline_name is None and phase_solver is None:
        return "rr_ao_wmmse_sdr", "sdr"
    if baseline_name is None:
        matches = [name for name, phase in METHOD_TO_PHASE.items() if phase == phase_solver]
        if not matches:
            raise ValueError("Unknown conventional phase solver")
        baseline_name = matches[0]
    if baseline_name not in METHOD_TO_PHASE:
        raise ValueError("Unknown conventional baseline")
    expected = METHOD_TO_PHASE[baseline_name]
    if phase_solver is not None and phase_solver != expected:
        raise ValueError("Method/phase solver/initialization mismatch")
    return baseline_name, expected


def make_args(**overrides):
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown conventional options: {sorted(unknown)}")
    method, phase = resolve_method_phase(
        overrides.get("baseline_name"), overrides.get("phase_solver")
    )
    overrides.update(baseline_name=method, phase_solver=phase)
    args = Namespace(**(DEFAULTS | overrides))
    validate_args(args)
    return args


def solver_options(args):
    return SolverOptions(**{f.name: getattr(args, f.name) for f in fields(SolverOptions)})


def validate_args(args):
    solver_options(args)
    resolve_method_phase(args.baseline_name, args.phase_solver)
    if args.ris_init != "mrt":
        raise ValueError("Method/phase solver/initialization mismatch")
    if (
        args.budget_profile not in ("smoke", "calibration", "formal_candidate")
        or args.device != "cpu"
    ):
        raise ValueError("Conventional solver requires CPU and a known budget profile")
    for key in (
        "num_ground_stations",
        "num_gs_per_rb",
        "num_eval_seeds",
        "eval_seed_stride",
        "evaluation_repeats",
        "max_steps_per_episode",
        "diagnostic_sample_limit",
        "debug_decision_limit",
        "num_micro_steps",
        "num_threads",
    ):
        value = getattr(args, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 1 <= args.num_gs_per_rb <= args.num_ground_stations:
        raise ValueError("Invalid RR dimensions/offset")
    if type(args.rr_initial_offset) is not int or not 0 <= args.rr_initial_offset < math.comb(
        args.num_ground_stations, args.num_gs_per_rb
    ):
        raise ValueError("Invalid RR combination offset; require 0 <= offset < C(N,K)")
    if not isinstance(args.checkpoint_every_steps, int) or args.checkpoint_every_steps < 0:
        raise ValueError("Invalid checkpoint interval")
    for key in (
        "fairness_lambda",
        "fairness_beta",
        "fairness_alpha",
        "fairness_epsilon",
        "csi_error_variance",
        "decision_interval_s",
        "residual_doppler_hz",
        "rho",
        "satellite_max_power_dBm",
        "terrestrial_cell_power_dBm",
        "noise_power_dBm",
        "sic_detection_threshold_dBm",
        "ris_reflection_power_limit_dBm",
        "active_ris_noise_power_dBm",
    ):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"{key} must be finite")
    if (
        args.csi_error_variance < 0
        or not 0 <= args.fairness_beta <= 1
        or args.fairness_epsilon <= 0
    ):
        raise ValueError("Invalid CSI/fairness configuration")
    resolve_topology(args.topology_profile, args.num_ground_stations)


def seed_matrix(args):
    return [
        [
            args.eval_seed_base + args.eval_seed_stride * (i + r * args.num_eval_seeds)
            for i in range(args.num_eval_seeds)
        ]
        for r in range(args.evaluation_repeats)
    ]


def build_env_config(args, seed):
    return dict(
        num_ground_stations=args.num_ground_stations,
        num_resource_blocks=1,
        num_gs_per_rb=args.num_gs_per_rb,
        deployment_area_size_m=400.0,
        satellite_max_power_dBm=args.satellite_max_power_dBm,
        terrestrial_cell_power_dBm=args.terrestrial_cell_power_dBm,
        noise_power_dBm=args.noise_power_dBm,
        sic_detection_threshold_dBm=args.sic_detection_threshold_dBm,
        carrier_frequency_GHz=4.0,
        satellite_antenna_gain_dBi=24.3,
        ground_station_antenna_gain_dBi=24.3,
        satellite_altitude_m=600e3,
        log_normal_shadowing_variance_dB=8.0,
        rician_k_factor=10.0,
        k_factor_ris_to_gs_linear=10.0,
        k_factor_sat_to_ris_linear=30.0,
        max_doppler_frequency_Hz=args.residual_doppler_hz,
        channel_update_interval_s=args.decision_interval_s,
        num_micro_steps=args.num_micro_steps,
        min_rate_requirement_bps_hz=args.min_rate_req,
        max_steps_per_episode=args.max_steps_per_episode,
        ris_boost_lambda=args.lambda_boost,
        random_seed=int(seed),
        use_fixed_positions=True,
        topology_profile=args.topology_profile,
        rho=args.rho,
        csi_error_variance=args.csi_error_variance,
        force_bs_gain_db=args.force_bs_gain_db,
        reseed_each_episode=True,
        sat_user_gain_db_offsets=args.sat_user_gain_db_offsets,
        include_angle_features=False,
        ris_num_elem_x=args.ris_num_elem_x,
        ris_num_elem_y=args.ris_num_elem_y,
        ris_reflection_power_limit_dBm=args.ris_reflection_power_limit_dBm,
        ris_power_control_mode="power_budget",
        ris_elem_spacing_ratio=0.5,
        active_ris_enabled=True,
        active_ris_noise_power_dBm=args.active_ris_noise_power_dBm,
        disable_ris_path=False,
        uav_initial_position_xyz=[0.0, 0.0, 80.0],
        uav_velocity_per_step_xyz=[0.282842712, 0.282842712, 0.0],
        uav_rx_antenna_gain_dBi=args.uav_rx_antenna_gain_dBi,
        ris_elem_max_amp_gain_dB=args.ris_elem_max_amp_gain_dB,
        fairness_alpha=args.fairness_alpha,
        fairness_beta=args.fairness_beta,
        fairness_epsilon=args.fairness_epsilon,
        fairness_lambda=args.fairness_lambda,
        sic_penalty=args.sic_penalty,
        include_sr_incident_pow_groups=False,
        include_mrt_features=False,
        mrt_casc_abs_tau=1e-12,
    )
