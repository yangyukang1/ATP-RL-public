#!/usr/bin/env python3


from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
import time
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from atprl.envs.leo_sat import LeoSatEnv
from atprl.envs.topologies import resolve_topology
from atprl.config.kernel_signature import (
    read_kernel_signature,
    validate_checkpoint_metadata,
)
from atprl.training.stage2 import Phase2Trainer
from atprl.evaluation.utils import (
    DEFAULT_METRIC_KEYS,
    summarize_metrics_by_seed,
    build_eval_curve_record,
    run_episode_validation_metrics,
    extract_step_rate_vector_from_infos,
)


@contextmanager
def preserve_rng_state():
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _set_agents_mode(agents: Dict[str, Any], mode: str = "eval") -> None:
    network_names = ["q_network", "power_actor", "power_critic", "phase_actor", "phase_critic"]
    for agent in agents.values():
        for net_name in network_names:
            if hasattr(agent, net_name):
                net = getattr(agent, net_name)
                if net is not None and hasattr(net, mode):
                    getattr(net, mode)()


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {v}")


def parse_sat_user_gain_offsets(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None
    return [float(x.strip()) for x in txt.split(",") if x.strip() != ""]


VALID_BASELINES = (
    "atp_rl",
    "stp_rl",
    "rnd_stp",
    "wmmse_stp",
    "periodic",
    "joint_dqn_ppo_ppo",
    "sac_stp",
)


def _periodic_unsupported_reason(
    num_ground_stations: int,
    num_gs_per_rb: int,
    dqn_scheduler_impl: str,
) -> Optional[str]:
    n = int(num_ground_stations)
    k = int(num_gs_per_rb)
    impl = str(dqn_scheduler_impl).strip().lower()
    if n not in (3, 4, 5):
        return f"periodic baseline has no predefined plan for N={n}; supported N values are 3, 4, and 5"
    if k != 2:
        return f"periodic baseline is pair-based and supports only K=2, got K={k}"
    if impl != "enumerative":
        return (
            f"periodic baseline requires an enumerative action map, got dqn_scheduler_impl={impl}"
        )
    return None


def parse_csv_str_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    txt = str(raw).strip()
    if txt == "":
        return []
    return [x.strip() for x in txt.split(",") if x.strip() != ""]


def parse_float_list(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None
    return [float(x.strip()) for x in txt.split(",") if x.strip() != ""]


def _dbm_tag(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified train-aligned evaluator for PBT / STP-RL / periodic baseline."
    )

    g = parser.add_argument_group("Evaluation source")
    g.add_argument(
        "--eval_source",
        type=str,
        default="pbt_best",
        choices=[
            "pbt_best",
            "phase2_model_dir",
            "periodic_phase1_model_dir",
            "joint_checkpoint",
            "sac_checkpoint",
        ],
        help=(
            "Which saved model source to evaluate. "
            "pbt_best = auto/manual best worker ckpt; "
            "phase2_model_dir = STP-RL / Phase-II saved model directory; "
            "periodic_phase1_model_dir = periodic baseline Phase-I saved model directory."
        ),
    )

    g.add_argument(
        "--pbt_run_dir",
        type=str,
        default=None,
        help="PBT run root containing worker_xx/checkpoints/best.ckpt.",
    )
    g.add_argument("--sac_checkpoint", type=str, default=None)
    g.add_argument(
        "--joint_checkpoint",
        type=str,
        default=None,
        help="Complete committed Joint best policy artifact.",
    )
    g.add_argument("--select_worker", type=str, default="auto", choices=["auto", "manual"])
    g.add_argument("--worker_id", type=int, default=None)
    g.add_argument("--best_ckpt_name", type=str, default="best.ckpt")

    g.add_argument(
        "--phase2_model_dir",
        type=str,
        default=None,
        help="Phase-II / STP-RL run root or models directory.",
    )
    g.add_argument(
        "--phase2_phase_ckpt",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Which phase checkpoint to load for Phase-II evaluation.",
    )
    g.add_argument(
        "--phase2_phase_ckpt_strict",
        action="store_true",
        help="If set and --phase2_phase_ckpt=best, require phase_actor_phase2_best.pth / phase_critic_phase2_best.pth.",
    )

    g.add_argument(
        "--periodic_phase1_model_dir",
        type=str,
        default=None,
        help="Periodic baseline run root or models directory.",
    )
    g.add_argument(
        "--phase1_ckpt",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Which Phase-I kernel checkpoint to load for periodic baseline.",
    )
    g.add_argument(
        "--phase1_ckpt_strict",
        action="store_true",
        help="If set and --phase1_ckpt=best, require *_best.pth for periodic baseline.",
    )

    g = parser.add_argument_group("Output / seeds")
    g.add_argument("--output_dir", type=str, default=None)
    g.add_argument("--num_eval_seeds", type=int, default=8)
    g.add_argument("--eval_seed_base", type=int, default=10001)
    g.add_argument("--eval_seed_stride", type=int, default=37)
    g.add_argument("--save_per_seed_metrics", action="store_true")
    g.add_argument("--csv_name", type=str, default="eval_summary.csv")
    g.add_argument("--show_progress", type=str2bool, default=True)
    g.add_argument("--stream_write", type=str2bool, default=True)
    g.add_argument("--stream_flush_every", type=int, default=1)

    g = parser.add_argument_group("Experiment tags")
    g.add_argument("--eval_tag", type=str, default="train_aligned_eval")
    g.add_argument("--lambda_tag", type=str, default=None)
    g.add_argument("--csi_tag", type=str, default=None)
    g.add_argument("--mrt_tag", type=str, default=None)

    g = parser.add_argument_group("Train-aligned rebuild args")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--exp_name", type=str, default="eval_train_aligned")
    g.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    g.add_argument(
        "--save_model", action="store_true", help="Unused during eval; kept for compatibility."
    )

    g.add_argument("--scheduler_mode", type=str, default="dqn", choices=["dqn", "random"])
    g.add_argument(
        "--dqn_scheduler_impl",
        type=str,
        default="enumerative",
        choices=["enumerative"],
    )
    g.add_argument("--power_mode", type=str, default="ppo", choices=["ppo", "wmmse"])

    g.add_argument("--wmmse_rb_index", type=int, default=0)
    g.add_argument("--wmmse_max_iters", type=int, default=50)
    g.add_argument("--wmmse_tol", type=float, default=1e-6)
    g.add_argument("--wmmse_scipy_method", type=str, default="SLSQP")
    g.add_argument("--wmmse_scipy_options", type=str, default=None)
    g.add_argument("--wmmse_sic_repair_max_iters", type=int, default=30)
    g.add_argument("--wmmse_sic_repair_step_ratio", type=float, default=0.05)

    g.add_argument("--num_ground_stations", type=int, default=4)
    g.add_argument("--num_gs_per_rb", type=int, default=2)
    g.add_argument("--topology_profile", type=str, default="paper_n4")
    g.add_argument("--min_rate_req", type=float, default=0.1)
    g.add_argument("--csi_error_variance", type=float, default=0.01)
    g.add_argument("--rho", type=float, default=0.95)
    g.add_argument("--lambda_boost", type=float, default=0.0)
    g.add_argument("--active_ris_noise_power_dBm", type=float, default=-100.0)
    g.add_argument("--disable_ris_path", type=str2bool, default=False)
    g.add_argument("--uav_rx_antenna_gain_dBi", type=float, default=15.0)
    g.add_argument("--ris_elem_max_amp_gain_dB", type=float, default=40.0)
    g.add_argument("--ris_reflection_power_limit_dBm", type=float, default=-5.0)
    g.add_argument("--fairness_lambda", type=float, default=0.0)
    g.add_argument("--fairness_alpha", type=float, default=0.0)
    g.add_argument("--fairness_beta", type=float, default=0.01)
    g.add_argument("--fairness_epsilon", type=float, default=0.01)
    g.add_argument("--sic_penalty", type=float, default=0.0)
    g.add_argument("--sic_detection_threshold_dBm", type=float, default=-94.0)
    g.add_argument("--force_bs_gain_db", type=float, default=None)
    g.add_argument("--train_reseed_each_episode", action="store_true")
    g.add_argument("--sat_user_gain_db_offsets", type=str, default=None)
    g.add_argument("--ris_num_elem_x", type=int, default=16)
    g.add_argument("--ris_num_elem_y", type=int, default=16)

    g.add_argument("--cycle_dqn_before_power", type=int, default=2)
    g.add_argument("--cycle_power", type=int, default=1)
    g.add_argument("--cycle_dqn_before_phase", type=int, default=2)
    g.add_argument("--cycle_phase", type=int, default=1)
    g.add_argument("--phase_delay_cycles", type=int, default=0)
    g.add_argument("--phase_delay_episodes", type=int, default=0)

    g.add_argument("--ppo_power_opt_episodes", type=int, default=8)
    g.add_argument("--ppo_phase_opt_episodes", type=int, default=16)
    g.add_argument("--ppo_power_epochs", type=int, default=12)
    g.add_argument("--ppo_phase_epochs", type=int, default=6)
    g.add_argument("--ppo_power_buffer_size", type=int, default=1024)
    g.add_argument("--ppo_phase_buffer_size", type=int, default=1024)
    g.add_argument("--entropy_coef_power", type=float, default=0.012)
    g.add_argument("--entropy_coef_phase", type=float, default=0.006)
    g.add_argument("--lr_power_ppo", type=float, default=3e-4)
    g.add_argument("--lr_phase_ppo", type=float, default=6e-5)
    g.add_argument("--lr_dqn", type=float, default=5e-4)
    g.add_argument("--gamma", type=float, default=0.9)
    g.add_argument("--gae_lambda", type=float, default=0.95)

    g.add_argument("--max_train_episodes", type=int, default=5000)
    g.add_argument("--eval_every_episodes", type=int, default=32)
    g.add_argument("--curve_k", type=int, default=0)
    g.add_argument("--epsilon_start", type=float, default=0.05)
    g.add_argument("--epsilon_end", type=float, default=0.05)
    g.add_argument("--epsilon_decay_episodes", type=int, default=1)
    g.add_argument("--eps_floor_start", type=float, default=0.0)
    g.add_argument("--eps_floor_end", type=float, default=0.0)
    g.add_argument("--eps_floor_decay_steps", type=int, default=500000)

    g.add_argument("--phase2_power_deterministic_mean", type=str2bool, default=True)
    g.add_argument("--phase_wrap_mode", type=str, default="pi", choices=["pi", "2pi"])
    g.add_argument("--include_sr_incident_pow_groups", type=str2bool, default=False)
    g.add_argument(
        "--learn_amp_weights", type=str, default="auto", choices=["auto", "true", "false"]
    )
    g.add_argument("--force_phase2_uniform_amp", action="store_true")
    g.add_argument("--phase2_alt_enable", action="store_true")
    g.add_argument("--phase2_alt_theta_episodes", type=int, default=12)
    g.add_argument("--phase2_alt_amp_episodes", type=int, default=4)
    g.add_argument("--phase2_alt_warmup_theta_only_episodes", type=int, default=0)
    g.add_argument("--include_mrt_features", type=str2bool, default=False)
    g.add_argument("--mrt_casc_abs_tau", type=float, default=1e-12)
    g.add_argument(
        "--phase_state_mode",
        type=str,
        default="legacy_full",
        choices=["legacy_full", "compact_selected"],
    )
    g.add_argument("--phase_compact_include_mrt_features", action="store_true")
    g.add_argument(
        "--phase_policy_dist",
        type=str,
        default="gaussian_latent",
        choices=["gaussian_latent", "von_mises"],
    )

    g = parser.add_argument_group("Batch sweep")
    g.add_argument(
        "--run_batch",
        type=str2bool,
        default=False,
        help="If true, sweep multiple baselines and multiple RIS reflection power values in one run.",
    )
    g.add_argument(
        "--baseline_list",
        type=str,
        default="atp_rl,stp_rl,rnd_stp,wmmse_stp,periodic",
        help="Comma-separated baseline list. Supported: atp_rl, stp_rl, rnd_stp, wmmse_stp, periodic",
    )
    g.add_argument(
        "--sweep_ris_reflection_power_limit_dBm",
        type=str,
        default=None,
        help='Comma-separated RIS reflection power list, e.g. "-25,-20,-15,-10,-5,0".',
    )
    g.add_argument(
        "--batch_root_dir", type=str, default=None, help="Root directory for batch outputs."
    )

    g.add_argument("--atp_pbt_run_dir", type=str, default=None)
    g.add_argument("--stp_phase2_model_dir", type=str, default=None)
    g.add_argument("--rnd_stp_phase2_model_dir", type=str, default=None)
    g.add_argument("--wmmse_phase2_model_dir", type=str, default=None)
    g.add_argument("--periodic_phase1_model_dir_batch", type=str, default=None)
    args = parser.parse_args()

    args.baseline_list = parse_csv_str_list(args.baseline_list)
    args.sweep_ris_reflection_power_limit_dBm = parse_float_list(
        args.sweep_ris_reflection_power_limit_dBm
    )

    args.phase2_only = True
    args.freeze_dqn_power = True
    args.kernel_frozen = True
    args.skip_phase1_kernel_load = True
    args.phase1_model_dir = None
    args.early_stop_patience_eval = 0
    args.min_improvement = 0.0
    args.eval_every_steps = 0
    args.eval_env_seed = int(args.eval_seed_base)
    args.eval_tag_steps = "Eval"

    args.sat_user_gain_db_offsets = parse_sat_user_gain_offsets(args.sat_user_gain_db_offsets)

    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    n_users = int(args.num_ground_stations)
    k_users = int(args.num_gs_per_rb)
    if n_users < 1:
        raise ValueError("--num_ground_stations must satisfy N >= 1.")
    if k_users <= 0 or k_users > n_users:
        raise ValueError("--num_gs_per_rb must satisfy 1 <= K <= N.")
    resolve_topology(args.topology_profile, n_users)

    if bool(getattr(args, "run_batch", False)):
        if not args.baseline_list:
            raise ValueError("When --run_batch true, --baseline_list cannot be empty.")

        invalid = [x for x in args.baseline_list if x not in VALID_BASELINES]
        if invalid:
            raise ValueError(
                f"Unsupported baseline(s) in --baseline_list: {invalid}. "
                f"Supported = {list(VALID_BASELINES)}"
            )
        if "sac_stp" in args.baseline_list and not args.sac_checkpoint:
            raise ValueError("Batch SAC evaluation requires --sac_checkpoint")
        if "joint_dqn_ppo_ppo" in args.baseline_list and not args.joint_checkpoint:
            raise ValueError("Batch Joint evaluation requires --joint_checkpoint.")

        if args.sweep_ris_reflection_power_limit_dBm is None:
            args.sweep_ris_reflection_power_limit_dBm = [float(args.ris_reflection_power_limit_dBm)]

        if "atp_rl" in args.baseline_list:
            if not (args.atp_pbt_run_dir or args.pbt_run_dir):
                raise ValueError(
                    "Batch mode baseline 'atp_rl' requires --atp_pbt_run_dir or --pbt_run_dir."
                )

        if any(x in args.baseline_list for x in ("stp_rl", "rnd_stp", "wmmse_stp")):
            if not (
                args.stp_phase2_model_dir
                or args.rnd_stp_phase2_model_dir
                or args.wmmse_phase2_model_dir
                or args.phase2_model_dir
            ):
                raise ValueError(
                    "Batch mode for STP/RND/WMMSE baselines requires at least one Phase-II model dir "
                    "(--phase2_model_dir or dedicated per-baseline dirs)."
                )

        periodic_reason = _periodic_unsupported_reason(
            n_users,
            k_users,
            getattr(args, "dqn_scheduler_impl", "enumerative"),
        )
        if "periodic" in args.baseline_list and periodic_reason is None:
            if not (args.periodic_phase1_model_dir_batch or args.periodic_phase1_model_dir):
                raise ValueError(
                    "Batch mode baseline 'periodic' requires "
                    "--periodic_phase1_model_dir_batch or --periodic_phase1_model_dir."
                )
        return

    if args.eval_source == "sac_checkpoint" and not args.sac_checkpoint:
        raise ValueError("SAC evaluation requires --sac_checkpoint")
    if args.eval_source == "pbt_best":
        if not args.pbt_run_dir:
            raise ValueError("--pbt_run_dir is required when --eval_source pbt_best.")
    elif args.eval_source == "phase2_model_dir":
        if not args.phase2_model_dir:
            raise ValueError("--phase2_model_dir is required when --eval_source phase2_model_dir.")
    elif args.eval_source == "periodic_phase1_model_dir":
        periodic_reason = _periodic_unsupported_reason(
            n_users,
            k_users,
            getattr(args, "dqn_scheduler_impl", "enumerative"),
        )
        if periodic_reason is not None:
            raise ValueError(f"Periodic baseline is not applicable: {periodic_reason}.")
        if not args.periodic_phase1_model_dir:
            raise ValueError(
                "--periodic_phase1_model_dir is required when --eval_source periodic_phase1_model_dir."
            )
    elif args.eval_source == "joint_checkpoint" and not args.joint_checkpoint:
        raise ValueError("--joint_checkpoint is required when --eval_source joint_checkpoint.")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_external_models_dir(user_path: str) -> Path:
    p = Path(user_path)
    if p.is_dir() and (p / "models").is_dir():
        return p / "models"
    return p


def _load_required_state_dict(path: Path, device) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint file not found: {path}")
    obj = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a state_dict-like dict: {path}")
    return obj


def _pick_ckpt(model_dir: Path, base_name: str, prefer: str, strict_best: bool) -> Path:
    final_path = model_dir / f"{base_name}.pth"
    best_path = model_dir / f"{base_name}_best.pth"
    if prefer == "best":
        if best_path.is_file():
            return best_path
        if strict_best:
            raise FileNotFoundError(f"Requested best checkpoint but not found: {best_path}")
        return final_path
    return final_path


def _pick_phase2_phase_ckpt(
    model_dir: Path, base_name: str, prefer: str, strict_best: bool
) -> Path:
    final_path = model_dir / f"{base_name}_final.pth"
    best_path = model_dir / f"{base_name}_phase2_best.pth"
    if prefer == "best":
        if best_path.is_file():
            return best_path
        if strict_best:
            raise FileNotFoundError(f"Requested best phase checkpoint but not found: {best_path}")
        return final_path
    return final_path


def _find_worker_dirs(pbt_run_dir: Path) -> List[Path]:
    return sorted(
        [p for p in pbt_run_dir.iterdir() if p.is_dir() and p.name.lower().startswith("worker_")]
    )


def select_worker_and_ckpt(
    pbt_run_dir: Path,
    select_worker: str,
    worker_id: Optional[int],
    best_ckpt_name: str,
) -> Dict[str, Any]:
    pbt_run_dir = Path(pbt_run_dir)
    if not pbt_run_dir.is_dir():
        raise FileNotFoundError(f"PBT run directory not found: {pbt_run_dir}")

    worker_dirs = _find_worker_dirs(pbt_run_dir)
    if not worker_dirs:
        raise FileNotFoundError(f"No worker_xx directories found under: {pbt_run_dir}")

    if select_worker == "manual":
        if worker_id is None:
            raise ValueError("--worker_id is required when --select_worker manual")
        target = pbt_run_dir / f"worker_{int(worker_id):02d}"
        if not target.is_dir():
            raise FileNotFoundError(f"Requested worker directory not found: {target}")
        meta_path = target / "checkpoints" / "best_meta.json"
        ckpt_path = target / "checkpoints" / best_ckpt_name
        if not meta_path.is_file():
            raise FileNotFoundError(f"best_meta.json not found: {meta_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")
        meta = _load_json(meta_path)
        return {
            "selected_worker_id": int(worker_id),
            "worker_dir": target,
            "best_meta": meta,
            "best_meta_path": meta_path,
            "best_ckpt_path": ckpt_path,
        }

    candidates: List[Tuple[float, int, Path, Dict[str, Any], Path, Path]] = []
    for wd in worker_dirs:
        ckpt_dir = wd / "checkpoints"
        meta_path = ckpt_dir / "best_meta.json"
        ckpt_path = ckpt_dir / best_ckpt_name
        if (not meta_path.is_file()) or (not ckpt_path.is_file()):
            continue
        meta = _load_json(meta_path)
        score = float(meta.get("best_ready_score", float("-inf")))
        wid = int(wd.name.split("_")[-1])
        candidates.append((score, wid, wd, meta, meta_path, ckpt_path))

    if not candidates:
        raise FileNotFoundError(
            f"No valid worker checkpoints found under {pbt_run_dir}. "
            f"Expected worker_xx/checkpoints/best_meta.json and {best_ckpt_name}."
        )

    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    _, wid, wd, meta, meta_path, ckpt_path = candidates[0]
    return {
        "selected_worker_id": wid,
        "worker_dir": wd,
        "best_meta": meta,
        "best_meta_path": meta_path,
        "best_ckpt_path": ckpt_path,
    }


class _NullWriter:
    def __init__(self, log_dir=None):
        self._enabled = False
        self.log_dir = None if log_dir is None else str(log_dir)

    @property
    def enabled(self) -> bool:
        return False

    def get_logdir(self):
        return self.log_dir

    def add_scalar(self, *args, **kwargs):
        return

    def add_histogram(self, *args, **kwargs):
        return

    def add_image(self, *args, **kwargs):
        return

    def add_text(self, *args, **kwargs):
        return

    def add_figure(self, *args, **kwargs):
        return

    def add_graph(self, *args, **kwargs):
        return

    def add_embedding(self, *args, **kwargs):
        return

    def flush(self):
        return

    def close(self):
        return

    def __getattr__(self, item):
        def _noop(*args, **kwargs):
            return

        return _noop


def build_phase2_trainer_for_eval(args: argparse.Namespace, output_dir: Path) -> Phase2Trainer:
    tmp_log_dir = output_dir / "_tmp_phase2_eval_build"
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    return Phase2Trainer(args, log_dir=tmp_log_dir, writer=_NullWriter(log_dir=tmp_log_dir))


def _ckpt_get(obj: Any, key: str):
    if isinstance(obj, dict):
        return obj[key]
    if hasattr(obj, key):
        return getattr(obj, key)
    raise KeyError(f"Checkpoint does not contain field '{key}'")


def apply_unified_ckpt_to_phase2_trainer(
    trainer: Phase2Trainer, ckpt_obj: Any, *, strict: bool = True
) -> None:
    try:
        checkpoint_meta = _ckpt_get(ckpt_obj, "meta")
    except Exception:
        checkpoint_meta = None
    validate_checkpoint_metadata(
        checkpoint_meta,
        trainer.kernel_signature,
        context="PBT unified evaluation checkpoint",
    )
    if checkpoint_meta is None:
        checkpoint_meta = {"_legacy_warning_emitted": True}
    elif "kernel_signature" not in checkpoint_meta:
        checkpoint_meta = {**dict(checkpoint_meta), "_legacy_warning_emitted": True}
    kernel_state = {
        "meta": checkpoint_meta,
        "models": {
            "q_network": _ckpt_get(ckpt_obj, "q_network"),
            "target_q_network": _ckpt_get(ckpt_obj, "target_q_network"),
            "power_actor": _ckpt_get(ckpt_obj, "power_actor"),
            "power_critic": _ckpt_get(ckpt_obj, "power_critic"),
        },
    }
    trainer.load_kernel_state(kernel_state, strict=strict, load_optim=False)
    trainer.phase_actor.load_state_dict(_ckpt_get(ckpt_obj, "phase_actor"), strict=strict)
    trainer.phase_critic.load_state_dict(_ckpt_get(ckpt_obj, "phase_critic"), strict=strict)

    try:
        counters = _ckpt_get(ckpt_obj, "counters")
    except Exception:
        counters = None

    if isinstance(counters, dict):
        if hasattr(trainer, "total_steps") and counters.get("total_steps") is not None:
            trainer.total_steps = int(counters["total_steps"])
        if hasattr(trainer, "episodes_done"):
            v = counters.get("episodes_done", counters.get("episode_count", None))
            if v is not None:
                trainer.episodes_done = int(v)
        if hasattr(trainer, "episode_count"):
            v = counters.get("episode_count", counters.get("episodes_done", None))
            if v is not None:
                trainer.episode_count = int(v)
        if hasattr(trainer, "current_epsilon") and counters.get("current_epsilon") is not None:
            trainer.current_epsilon = float(counters["current_epsilon"])
        if hasattr(trainer, "segment_idx") and counters.get("segment_idx") is not None:
            trainer.segment_idx = int(counters["segment_idx"])
        if (
            hasattr(trainer, "segment_ep_remaining")
            and counters.get("segment_ep_remaining") is not None
        ):
            trainer.segment_ep_remaining = int(counters["segment_ep_remaining"])
        if hasattr(trainer, "cycles_done") and counters.get("cycles_done") is not None:
            trainer.cycles_done = int(counters["cycles_done"])


def apply_phase2_saved_models_to_trainer(
    trainer: Phase2Trainer,
    model_dir: Path,
    *,
    phase_prefer: str,
    phase_strict_best: bool,
    device: str,
) -> Dict[str, str]:
    model_dir = Path(model_dir)
    signature = read_kernel_signature(model_dir)
    metadata = None if signature is None else {"kernel_signature": signature}
    validate_checkpoint_metadata(
        metadata,
        trainer.kernel_signature,
        context=f"Phase-II model directory {model_dir}",
    )
    if metadata is None:
        metadata = {"_legacy_warning_emitted": True}

    q_state = _load_required_state_dict(model_dir / "q_network_final.pth", device)
    pa_state = _load_required_state_dict(model_dir / "power_actor_final.pth", device)
    pc_state = _load_required_state_dict(model_dir / "power_critic_final.pth", device)

    phase_actor_path = _pick_phase2_phase_ckpt(
        model_dir, "phase_actor", phase_prefer, phase_strict_best
    )
    phase_critic_path = _pick_phase2_phase_ckpt(
        model_dir, "phase_critic", phase_prefer, phase_strict_best
    )
    pha_state = _load_required_state_dict(phase_actor_path, device)
    phc_state = _load_required_state_dict(phase_critic_path, device)

    kernel_state = {
        "meta": metadata,
        "models": {
            "q_network": q_state,
            "target_q_network": q_state,
            "power_actor": pa_state,
            "power_critic": pc_state,
        },
    }
    trainer.load_kernel_state(kernel_state, strict=True, load_optim=False)
    trainer.phase_actor.load_state_dict(pha_state, strict=True)
    trainer.phase_critic.load_state_dict(phc_state, strict=True)

    return {
        "model_dir": str(model_dir.resolve()),
        "q_network": str((model_dir / "q_network_final.pth").resolve()),
        "power_actor": str((model_dir / "power_actor_final.pth").resolve()),
        "power_critic": str((model_dir / "power_critic_final.pth").resolve()),
        "phase_actor": str(phase_actor_path.resolve()),
        "phase_critic": str(phase_critic_path.resolve()),
    }


def apply_phase1_kernel_to_trainer(
    trainer: Phase2Trainer,
    model_dir: Path,
    *,
    prefer: str,
    strict_best: bool,
    device: str,
) -> Dict[str, str]:
    model_dir = Path(model_dir)
    signature = read_kernel_signature(model_dir)
    metadata = None if signature is None else {"kernel_signature": signature}
    q_path = _pick_ckpt(model_dir, "q_network_phase1", prefer, strict_best)
    pa_path = _pick_ckpt(model_dir, "power_actor_phase1", prefer, strict_best)
    pc_path = _pick_ckpt(model_dir, "power_critic_phase1", prefer, strict_best)

    q_state = _load_required_state_dict(q_path, device)
    pa_state = _load_required_state_dict(pa_path, device)
    pc_state = _load_required_state_dict(pc_path, device)

    kernel_state = {
        "meta": metadata,
        "models": {
            "q_network": q_state,
            "target_q_network": q_state,
            "power_actor": pa_state,
            "power_critic": pc_state,
        },
    }
    trainer.load_kernel_state(kernel_state, strict=True, load_optim=False)
    return {
        "model_dir": str(model_dir.resolve()),
        "q_network": str(q_path.resolve()),
        "power_actor": str(pa_path.resolve()),
        "power_critic": str(pc_path.resolve()),
    }


def make_train_aligned_eval_env(base_env_config: Dict[str, Any], seed: int) -> LeoSatEnv:
    cfg = copy.deepcopy(base_env_config)
    cfg["random_seed"] = int(seed)
    cfg["reseed_each_episode"] = True
    return LeoSatEnv(config=cfg)


def build_eval_seed_list(num_eval_seeds: int, base: int, stride: int) -> List[int]:
    if num_eval_seeds <= 0:
        raise ValueError("--num_eval_seeds must be positive.")
    if stride <= 0:
        raise ValueError("--eval_seed_stride must be positive.")
    return [int(base + i * stride) for i in range(int(num_eval_seeds))]


def run_multi_seed_validation_metrics_train_aligned(
    base_env_config: Dict[str, Any],
    agents: Dict[str, Any],
    args,
    seed_list: Sequence[int],
    show_progress: bool = True,
    on_seed_done=None,
) -> Dict[str, Any]:
    seed_list = list(map(int, list(seed_list) if seed_list is not None else []))
    metrics_all: List[Dict[str, float]] = []

    _set_agents_mode(agents, mode="eval")
    try:
        with preserve_rng_state(), torch.no_grad():
            iterator = seed_list
            if bool(show_progress):
                iterator = tqdm(
                    seed_list,
                    total=len(seed_list),
                    desc="Eval seeds",
                    dynamic_ncols=True,
                    leave=True,
                )
            for seed in iterator:
                eval_env = make_train_aligned_eval_env(base_env_config, seed=int(seed))
                try:
                    m = run_episode_validation_metrics(
                        eval_env=eval_env, agents=agents, args=args, seed=int(seed)
                    )
                finally:
                    try:
                        eval_env.close()
                    except Exception:
                        pass
                metrics_all.append(m)
                if callable(on_seed_done):
                    on_seed_done(seed=int(seed), metrics=m, metrics_all=metrics_all)
                if bool(show_progress) and hasattr(iterator, "set_postfix"):
                    postfix = {}
                    if "avg_sum_rate" in m:
                        postfix["sum_rate"] = f"{float(m['avg_sum_rate']):.4f}"
                    if "min_user_avg_rate" in m:
                        postfix["min_user"] = f"{float(m['min_user_avg_rate']):.4f}"
                    if postfix:
                        iterator.set_postfix(postfix)
    finally:
        _set_agents_mode(agents, mode="train")

    summary = summarize_metrics_by_seed(metrics_all=metrics_all, metric_keys=DEFAULT_METRIC_KEYS)
    return {
        "seed_list": seed_list,
        "metrics": summary,
        "per_seed_metrics": metrics_all,
    }


PERIODIC_PAIR_CYCLE_12_3U: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 2),
]
PERIODIC_TARGET_CYCLE_12_3U: List[int] = [
    0,
    2,
    1,
    1,
    0,
    2,
    0,
    2,
    1,
    1,
    0,
    2,
]

PERIODIC_PAIR_CYCLE_12_4U: List[Tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (0, 3),
    (1, 2),
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (0, 3),
    (1, 2),
]
PERIODIC_TARGET_CYCLE_12_4U: List[int] = [
    0,
    2,
    0,
    1,
    3,
    1,
    1,
    3,
    2,
    3,
    0,
    2,
]

PERIODIC_PAIR_CYCLE_20_5U: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
]
PERIODIC_TARGET_CYCLE_20_5U: List[int] = [
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    3,
    1,
    2,
    3,
    4,
    2,
    3,
    4,
    3,
    4,
    4,
]


def _get_periodic_cycles_for_topology(
    num_ground_stations: int, num_gs_per_rb: int
) -> Tuple[List[Tuple[int, int]], List[int]]:
    n = int(num_ground_stations)
    k = int(num_gs_per_rb)
    if k != 2:
        raise ValueError("Periodic baseline only supports K=2.")
    if n == 3:
        return list(PERIODIC_PAIR_CYCLE_12_3U), list(PERIODIC_TARGET_CYCLE_12_3U)
    if n == 4:
        return list(PERIODIC_PAIR_CYCLE_12_4U), list(PERIODIC_TARGET_CYCLE_12_4U)
    if n == 5:
        return list(PERIODIC_PAIR_CYCLE_20_5U), list(PERIODIC_TARGET_CYCLE_20_5U)
    raise ValueError(f"Unsupported periodic topology N={n}, K={k}.")


def _build_pair_to_action_idx(discrete_actions_map) -> Dict[Tuple[int, ...], int]:
    out: Dict[Tuple[int, ...], int] = {}

    if isinstance(discrete_actions_map, dict):
        iterator = discrete_actions_map.items()
    elif isinstance(discrete_actions_map, (list, tuple)):
        iterator = enumerate(discrete_actions_map)
    else:
        raise TypeError(
            "discrete_actions_map must be dict / list / tuple, "
            f"got type={type(discrete_actions_map)}"
        )

    for idx, combo in iterator:
        if not isinstance(combo, (list, tuple)):
            raise TypeError(
                f"Each action entry must be list/tuple of user ids, got type={type(combo)}"
            )
        key = tuple(sorted(int(x) for x in combo))
        out[key] = int(idx)

    return out


def _wrap_phase_to_pi(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float32)
    return ((phi + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _get_periodic_pair_and_target(
    step_in_episode: int, num_ground_stations: int, num_gs_per_rb: int
) -> Tuple[Tuple[int, int], int]:
    pair_cycle, target_cycle = _get_periodic_cycles_for_topology(num_ground_stations, num_gs_per_rb)
    idx = int(step_in_episode) % len(pair_cycle)
    pair = tuple(sorted(int(u) for u in pair_cycle[idx]))
    target_user = int(target_cycle[idx])
    if target_user not in pair:
        raise ValueError(
            f"Periodic target mismatch at step={step_in_episode}: pair={pair}, target={target_user}"
        )
    return pair, target_user


def _build_periodic_power_state_tensor(
    agent, local_observation: np.ndarray, selected_users: Tuple[int, ...]
) -> torch.Tensor:
    obs_full = np.asarray(local_observation, dtype=np.float32).reshape(-1)
    dqn_state = obs_full[: agent.dqn_state_dim].astype(np.float32, copy=True)
    alpha = np.zeros(agent.n_users, dtype=np.float32)
    alpha[list(selected_users)] = 1.0
    power_state = np.concatenate([alpha, dqn_state], axis=0).astype(np.float32)
    return torch.from_numpy(power_state).float().to(agent.device)


def _build_periodic_single_user_mrt_phase(env: LeoSatEnv, target_user: int) -> np.ndarray:
    G = int(getattr(env, "ris_num_groups", 0))
    K = int(getattr(env, "num_ground_stations", env.config.get("num_ground_stations", 0)))

    if G <= 0:
        return np.zeros((0,), dtype=np.float32)
    if bool(getattr(env, "disable_ris_path", False)) or bool(
        getattr(env, "config", {}).get("disable_ris_path", False)
    ):
        return np.zeros((G,), dtype=np.float32)

    target_user = int(target_user)
    if target_user < 0 or target_user >= K:
        raise IndexError(f"target_user out of range: {target_user}, K={K}")

    mrt_feat = np.asarray(env.get_mrt_features(), dtype=np.float32).reshape(K, G, 2)
    cos_th = mrt_feat[target_user, :, 0]
    sin_th = mrt_feat[target_user, :, 1]
    phi = np.arctan2(sin_th, cos_th).astype(np.float32)
    return _wrap_phase_to_pi(phi)


def _attach_periodic_phase_to_env_action(
    env: LeoSatEnv, selected_users: Tuple[int, ...], power_vec: np.ndarray, target_user: int
):
    phi = _build_periodic_single_user_mrt_phase(env, target_user)
    return (list(selected_users), np.asarray(power_vec, dtype=np.float32).copy(), phi)


def _build_periodic_action_info(
    agent,
    env: LeoSatEnv,
    local_observation: np.ndarray,
    pair_to_action_idx: Dict[Tuple[int, ...], int],
    step_in_episode: int,
    *,
    deterministic_power: bool,
) -> Dict[str, Any]:
    selected_users, target_user = _get_periodic_pair_and_target(
        step_in_episode=step_in_episode,
        num_ground_stations=int(
            getattr(env, "num_ground_stations", env.config["num_ground_stations"])
        ),
        num_gs_per_rb=int(getattr(env, "num_gs_per_rb", env.config["num_gs_per_rb"])),
    )
    discrete_action_idx = int(pair_to_action_idx[tuple(sorted(selected_users))])

    power_state_tensor = _build_periodic_power_state_tensor(
        agent, local_observation, selected_users
    )
    power_state_batch = power_state_tensor.unsqueeze(0)

    with torch.no_grad():
        dist_power = agent.power_actor(power_state_batch)
        value_power = agent.power_critic(power_state_batch)
        power_action_raw = dist_power.mean if bool(deterministic_power) else dist_power.sample()
        log_prob_power = dist_power.log_prob(power_action_raw)

    power_shares = torch.softmax(power_action_raw, dim=-1)
    final_power_watt = power_shares * float(agent.max_power_watt)
    final_power_watt_np = final_power_watt.detach().cpu().numpy().squeeze(0).astype(np.float32)

    env_action = _attach_periodic_phase_to_env_action(
        env, selected_users, final_power_watt_np, target_user
    )

    return {
        "discrete_action_idx": int(discrete_action_idx),
        "selected_users": tuple(sorted(selected_users)),
        "scheduler_mode": "periodic",
        "power_state": power_state_tensor.detach().cpu(),
        "power_action": final_power_watt_np,
        "power_log_prob": log_prob_power.squeeze(0).detach().cpu(),
        "power_value": value_power.squeeze(0).detach().cpu(),
        "target_user": int(target_user),
        "env_action": env_action,
    }


def _jain_index_local(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    denom = float(x.size) * float(np.sum(np.square(x)))
    if denom <= 0.0:
        return 0.0
    return float((np.sum(x) ** 2) / denom)


def _run_single_seed_validation_periodic(
    base_env_config: Dict[str, Any],
    agents: Dict[str, Any],
    seed: int,
    pair_to_action_idx: Dict[Tuple[int, ...], int],
) -> Dict[str, float]:
    eval_env = make_train_aligned_eval_env(base_env_config, seed=int(seed))
    try:
        reset_ret = eval_env.reset(seed=int(seed))
        if isinstance(reset_ret, tuple) and len(reset_ret) == 2 and isinstance(reset_ret[1], dict):
            obs = reset_ret[0]
        else:
            obs = reset_ret

        episode_return = 0.0
        n_steps = 0
        sum_rate_per_user_over_steps = None
        sum_rate_total_over_steps = 0.0

        while True:
            actions_to_env = {}
            for agent_id in list(getattr(eval_env, "agents", list(agents.keys()))):
                if agent_id not in agents:
                    raise KeyError(f"[PeriodicEval] Missing agent for agent_id={agent_id}")
                act_info = _build_periodic_action_info(
                    agent=agents[agent_id],
                    env=eval_env,
                    local_observation=obs[agent_id],
                    pair_to_action_idx=pair_to_action_idx,
                    step_in_episode=n_steps,
                    deterministic_power=True,
                )
                actions_to_env[agent_id] = act_info["env_action"]

            next_obs, rewards, dones, infos = eval_env.step(actions_to_env)
            if isinstance(rewards, dict):
                episode_return += float(sum(float(v) for v in rewards.values()))
            else:
                episode_return += float(rewards)

            step_rate_per_user = extract_step_rate_vector_from_infos(infos, eval_env)
            step_rate_per_user = np.asarray(step_rate_per_user, dtype=np.float64).reshape(-1)
            if sum_rate_per_user_over_steps is None:
                sum_rate_per_user_over_steps = np.zeros_like(step_rate_per_user, dtype=np.float64)
            sum_rate_per_user_over_steps += step_rate_per_user
            sum_rate_total_over_steps += float(np.sum(step_rate_per_user))

            n_steps += 1
            obs = next_obs
            done_flag = bool(dones["__all__"]) if isinstance(dones, dict) else bool(dones)
            if done_flag:
                break

        T = max(1, int(n_steps))
        if sum_rate_per_user_over_steps is None:
            avg_rates = np.zeros(
                int(eval_env.config.get("num_ground_stations", 1)), dtype=np.float64
            )
            avg_sum_rate = 0.0
        else:
            avg_rates = (sum_rate_per_user_over_steps / float(T)).reshape(-1)
            avg_sum_rate = float(sum_rate_total_over_steps / float(T))

        min_user_avg_rate = float(np.min(avg_rates)) if avg_rates.size > 0 else 0.0
        jfi = _jain_index_local(avg_rates)
        return {
            "undiscounted_return": float(episode_return),
            "avg_sum_rate": float(avg_sum_rate),
            "min_user_avg_rate": float(min_user_avg_rate),
            "jain_index": float(jfi),
            "n_steps": int(n_steps),
        }
    finally:
        try:
            eval_env.close()
        except Exception:
            pass


def run_multi_seed_periodic_validation_metrics(
    base_env_config: Dict[str, Any],
    agents: Dict[str, Any],
    seed_list: Sequence[int],
    pair_to_action_idx: Dict[Tuple[int, ...], int],
    show_progress: bool = True,
    on_seed_done=None,
) -> Dict[str, Any]:
    seed_list = list(map(int, list(seed_list) if seed_list is not None else []))
    metrics_all: List[Dict[str, float]] = []

    _set_agents_mode(agents, mode="eval")
    try:
        with preserve_rng_state(), torch.no_grad():
            iterator = seed_list
            if bool(show_progress):
                iterator = tqdm(
                    seed_list,
                    total=len(seed_list),
                    desc="Eval seeds",
                    dynamic_ncols=True,
                    leave=True,
                )
            for seed in iterator:
                m = _run_single_seed_validation_periodic(
                    base_env_config, agents, int(seed), pair_to_action_idx
                )
                metrics_all.append(m)
                if callable(on_seed_done):
                    on_seed_done(seed=int(seed), metrics=m, metrics_all=metrics_all)
                if bool(show_progress) and hasattr(iterator, "set_postfix"):
                    postfix = {}
                    if "avg_sum_rate" in m:
                        postfix["sum_rate"] = f"{float(m['avg_sum_rate']):.4f}"
                    if "min_user_avg_rate" in m:
                        postfix["min_user"] = f"{float(m['min_user_avg_rate']):.4f}"
                    if postfix:
                        iterator.set_postfix(postfix)
    finally:
        _set_agents_mode(agents, mode="train")

    summary = summarize_metrics_by_seed(metrics_all=metrics_all, metric_keys=DEFAULT_METRIC_KEYS)
    return {
        "seed_list": seed_list,
        "metrics": summary,
        "per_seed_metrics": metrics_all,
    }


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def jsonl_append(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def save_summary_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _infer_baseline_name_from_args(args: argparse.Namespace) -> str:
    explicit = getattr(args, "baseline_name", None)
    if explicit not in (None, "", "None"):
        return str(explicit)

    if args.eval_source == "sac_checkpoint":
        return "sac_stp"

    if args.eval_source == "pbt_best":
        return "atp_rl"

    if args.eval_source == "periodic_phase1_model_dir":
        return "periodic"

    scheduler_mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()
    power_mode = str(getattr(args, "power_mode", "ppo")).strip().lower()

    if scheduler_mode == "random" and power_mode == "ppo":
        return "rnd_stp"
    if scheduler_mode == "dqn" and power_mode == "wmmse":
        return "wmmse_stp"
    if scheduler_mode == "dqn" and power_mode == "ppo":
        return "stp_rl"

    return "unknown"


def _resolve_batch_run_args(
    base_args: argparse.Namespace,
    baseline_name: str,
    ris_dbm: float,
    batch_root_dir: Path,
) -> argparse.Namespace:
    args = copy.deepcopy(base_args)
    args.run_batch = False
    args.baseline_name = str(baseline_name)
    args.ris_reflection_power_limit_dBm = float(ris_dbm)

    args.output_dir = str(
        Path(batch_root_dir)
        / f"N{int(args.num_ground_stations)}_K{int(args.num_gs_per_rb)}"
        / f"dbm_{_dbm_tag(ris_dbm)}"
        / str(baseline_name)
    )

    if baseline_name == "sac_stp":
        args.eval_source = "sac_checkpoint"
        return args

    if baseline_name == "atp_rl":
        args.eval_source = "pbt_best"
        args.pbt_run_dir = args.atp_pbt_run_dir or args.pbt_run_dir
        if not args.pbt_run_dir:
            raise ValueError("Missing PBT run dir for baseline 'atp_rl'.")
        args.scheduler_mode = "dqn"
        args.power_mode = "ppo"
        args.disable_ris_path = False
        return args

    if baseline_name == "joint_dqn_ppo_ppo":
        if not args.joint_checkpoint:
            raise ValueError("Missing Joint policy artifact for baseline 'joint_dqn_ppo_ppo'.")
        args.eval_source = "joint_checkpoint"
        return args

    if baseline_name == "stp_rl":
        args.eval_source = "phase2_model_dir"
        args.phase2_model_dir = args.stp_phase2_model_dir or args.phase2_model_dir
        if not args.phase2_model_dir:
            raise ValueError("Missing Phase-II model dir for baseline 'stp_rl'.")
        args.scheduler_mode = "dqn"
        args.power_mode = "ppo"
        args.disable_ris_path = False
        return args

    if baseline_name == "rnd_stp":
        args.eval_source = "phase2_model_dir"
        args.phase2_model_dir = args.rnd_stp_phase2_model_dir or args.phase2_model_dir
        if not args.phase2_model_dir:
            raise ValueError("Missing Phase-II model dir for baseline 'rnd_stp'.")
        args.scheduler_mode = "random"
        args.power_mode = "ppo"
        args.disable_ris_path = False
        return args

    if baseline_name == "wmmse_stp":
        args.eval_source = "phase2_model_dir"
        args.phase2_model_dir = args.wmmse_phase2_model_dir or args.phase2_model_dir
        if not args.phase2_model_dir:
            raise ValueError("Missing Phase-II model dir for baseline 'wmmse_stp'.")
        args.scheduler_mode = "dqn"
        args.power_mode = "wmmse"
        args.disable_ris_path = False
        return args

    if baseline_name == "periodic":
        args.eval_source = "periodic_phase1_model_dir"
        args.periodic_phase1_model_dir = (
            args.periodic_phase1_model_dir_batch or args.periodic_phase1_model_dir
        )
        if not args.periodic_phase1_model_dir:
            raise ValueError("Missing periodic Phase-I model dir for baseline 'periodic'.")
        args.scheduler_mode = "dqn"
        args.power_mode = "ppo"
        args.disable_ris_path = False
        return args

    raise ValueError(f"Unsupported baseline_name: {baseline_name}")


def make_stream_writer(
    *,
    output_dir: Path,
    selection_info: Dict[str, Any],
    eval_meta: Dict[str, Any],
    seed_list: Sequence[int],
    flush_every: int = 1,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    flush_every = max(1, int(flush_every))
    progress_path = output_dir / "live_progress.json"
    partial_pack_path = output_dir / "live_metrics_pack.json"
    jsonl_path = output_dir / "per_seed_metrics.jsonl"

    if jsonl_path.exists():
        jsonl_path.unlink()

    start_time = time.time()
    total = int(len(list(seed_list)))

    json_dump(selection_info, output_dir / "selection_info.json")
    json_dump(eval_meta, output_dir / "eval_meta.json")
    json_dump(
        {
            "status": "running",
            "completed": 0,
            "total": total,
            "seed_list": list(map(int, seed_list)),
            "completed_seed_list": [],
            "last_seed": None,
            "start_time": start_time,
            "last_update_time": start_time,
        },
        progress_path,
    )

    state = {
        "completed_seed_list": [],
        "start_time": start_time,
        "last_partial_metrics_pack": None,
    }

    def _writer(*, seed: int, metrics: Dict[str, Any], metrics_all: Sequence[Dict[str, Any]]):
        state["completed_seed_list"].append(int(seed))
        completed = len(state["completed_seed_list"])
        elapsed = float(time.time() - start_time)

        jsonl_append({"seed": int(seed), "metrics": metrics}, jsonl_path)

        should_flush = (completed % flush_every == 0) or (completed == total)
        if should_flush:
            partial_summary = summarize_metrics_by_seed(
                metrics_all=list(metrics_all), metric_keys=DEFAULT_METRIC_KEYS
            )
            partial_pack = {
                "seed_list": list(state["completed_seed_list"]),
                "metrics": partial_summary,
            }
            state["last_partial_metrics_pack"] = partial_pack
            json_dump(partial_pack, partial_pack_path)

        json_dump(
            {
                "status": "running" if completed < total else "finished",
                "completed": completed,
                "total": total,
                "seed_list": list(map(int, seed_list)),
                "completed_seed_list": list(state["completed_seed_list"]),
                "last_seed": int(seed),
                "start_time": start_time,
                "elapsed_seconds": elapsed,
                "last_update_time": time.time(),
                "latest_metrics": metrics,
            },
            progress_path,
        )

    def _finalize(final_metrics_pack: Dict[str, Any]):
        elapsed = float(time.time() - start_time)
        json_dump(
            {
                "status": "finished",
                "completed": total,
                "total": total,
                "seed_list": list(map(int, seed_list)),
                "completed_seed_list": list(state["completed_seed_list"]),
                "last_seed": int(state["completed_seed_list"][-1])
                if state["completed_seed_list"]
                else None,
                "start_time": start_time,
                "elapsed_seconds": elapsed,
                "last_update_time": time.time(),
                "final_metrics": final_metrics_pack.get("metrics", {}),
            },
            progress_path,
        )
        json_dump(
            {
                "seed_list": list(final_metrics_pack.get("seed_list", [])),
                "metrics": copy.deepcopy(final_metrics_pack.get("metrics", {})),
            },
            partial_pack_path,
        )

    return _writer, _finalize


def _default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.eval_source == "pbt_best":
        return Path(args.pbt_run_dir) / "evaluation_train_aligned"
    if args.eval_source == "phase2_model_dir":
        return Path(args.phase2_model_dir) / "evaluation_train_aligned"
    if args.eval_source == "periodic_phase1_model_dir":
        return Path(args.periodic_phase1_model_dir) / "evaluation_train_aligned"
    raise ValueError(f"Unsupported eval_source: {args.eval_source}")


def _build_train_alignment(
    args: argparse.Namespace, *, scheduler_mode_for_meta: str
) -> Dict[str, Any]:
    return {
        "eval_source": str(args.eval_source),
        "baseline_name": str(getattr(args, "baseline_name", None)),
        "num_ground_stations": int(args.num_ground_stations),
        "num_gs_per_rb": int(args.num_gs_per_rb),
        "fairness_lambda": float(args.fairness_lambda),
        "fairness_beta": float(args.fairness_beta),
        "csi_error_variance": float(args.csi_error_variance),
        "disable_ris_path": bool(args.disable_ris_path),
        "include_mrt_features": bool(args.include_mrt_features),
        "scheduler_mode": str(scheduler_mode_for_meta),
        "dqn_scheduler_impl": str(getattr(args, "dqn_scheduler_impl", "enumerative")),
        "power_mode": str(args.power_mode),
        "phase_policy_dist": str(args.phase_policy_dist),
        "phase_state_mode": str(args.phase_state_mode),
        "phase_compact_include_mrt_features": bool(args.phase_compact_include_mrt_features),
        "include_sr_incident_pow_groups": bool(args.include_sr_incident_pow_groups),
        "learn_amp_weights": str(args.learn_amp_weights),
        "force_phase2_uniform_amp": bool(args.force_phase2_uniform_amp),
        "mrt_casc_abs_tau": float(args.mrt_casc_abs_tau),
        "ris_reflection_power_limit_dBm": float(
            getattr(args, "ris_reflection_power_limit_dBm", 5.0)
        ),
        "active_ris_noise_power_dBm": float(args.active_ris_noise_power_dBm),
    }


def _selection_info_for_pbt(args: argparse.Namespace) -> Dict[str, Any]:
    selection = select_worker_and_ckpt(
        pbt_run_dir=Path(args.pbt_run_dir),
        select_worker=args.select_worker,
        worker_id=args.worker_id,
        best_ckpt_name=args.best_ckpt_name,
    )
    return {
        "eval_source": "pbt_best",
        "pbt_run_dir": str(Path(args.pbt_run_dir).resolve()),
        "selected_worker_id": int(selection["selected_worker_id"]),
        "worker_dir": str(Path(selection["worker_dir"]).resolve()),
        "best_meta_path": str(Path(selection["best_meta_path"]).resolve()),
        "best_ckpt_path": str(Path(selection["best_ckpt_path"]).resolve()),
        "best_meta": dict(selection["best_meta"]),
        "_selection_obj": selection,
    }


def _selection_info_for_phase2(
    args: argparse.Namespace, loaded_paths: Dict[str, str], source_path: str
) -> Dict[str, Any]:
    return {
        "eval_source": "phase2_model_dir",
        "phase2_model_dir_user": str(Path(source_path).resolve()),
        "phase2_model_dir_resolved": loaded_paths["model_dir"],
        "loaded_model_paths": dict(loaded_paths),
    }


def _selection_info_for_periodic(
    args: argparse.Namespace, loaded_paths: Dict[str, str], source_path: str
) -> Dict[str, Any]:
    pair_cycle, target_cycle = _get_periodic_cycles_for_topology(
        args.num_ground_stations, args.num_gs_per_rb
    )
    return {
        "eval_source": "periodic_phase1_model_dir",
        "periodic_phase1_model_dir_user": str(Path(source_path).resolve()),
        "periodic_phase1_model_dir_resolved": loaded_paths["model_dir"],
        "loaded_model_paths": dict(loaded_paths),
        "periodic_cycle_len": int(len(pair_cycle)),
        "periodic_pair_cycle": [list(map(int, p)) for p in pair_cycle],
        "periodic_target_cycle": [int(x) for x in target_cycle],
    }


def _single_run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    if args.eval_source == "sac_checkpoint":
        from atprl.evaluation.sac_stp import evaluate_from_runner

        return evaluate_from_runner(args)
    if args.eval_source == "joint_checkpoint":
        from atprl.evaluation.joint import evaluate_from_runner

        return evaluate_from_runner(args)
    output_dir = _default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_name = _infer_baseline_name_from_args(args)

    trainer = build_phase2_trainer_for_eval(args, output_dir=output_dir)
    try:
        if args.eval_source == "pbt_best":
            selection_info = _selection_info_for_pbt(args)
            selection = selection_info.pop("_selection_obj")
            ckpt_obj = torch.load(
                Path(selection["best_ckpt_path"]), map_location="cpu", weights_only=False
            )
            apply_unified_ckpt_to_phase2_trainer(trainer, ckpt_obj, strict=True)
            scheduler_mode_for_meta = str(args.scheduler_mode)
            row_source_path = str(Path(args.pbt_run_dir).resolve())

        elif args.eval_source == "phase2_model_dir":
            model_dir = _resolve_external_models_dir(args.phase2_model_dir)
            loaded_paths = apply_phase2_saved_models_to_trainer(
                trainer,
                model_dir=model_dir,
                phase_prefer=str(args.phase2_phase_ckpt),
                phase_strict_best=bool(args.phase2_phase_ckpt_strict),
                device=args.device,
            )
            selection_info = _selection_info_for_phase2(args, loaded_paths, args.phase2_model_dir)
            scheduler_mode_for_meta = str(args.scheduler_mode)
            row_source_path = str(Path(args.phase2_model_dir).resolve())

        elif args.eval_source == "periodic_phase1_model_dir":
            model_dir = _resolve_external_models_dir(args.periodic_phase1_model_dir)
            loaded_paths = apply_phase1_kernel_to_trainer(
                trainer,
                model_dir=model_dir,
                prefer=str(args.phase1_ckpt),
                strict_best=bool(args.phase1_ckpt_strict),
                device=args.device,
            )
            selection_info = _selection_info_for_periodic(
                args, loaded_paths, args.periodic_phase1_model_dir
            )
            scheduler_mode_for_meta = "periodic"
            row_source_path = str(Path(args.periodic_phase1_model_dir).resolve())
        else:
            raise ValueError(f"Unsupported eval_source: {args.eval_source}")

        selection_info["baseline_name"] = baseline_name

        base_env_config = copy.deepcopy(trainer.env_config)
        seed_list = build_eval_seed_list(
            num_eval_seeds=int(args.num_eval_seeds),
            base=int(args.eval_seed_base),
            stride=int(args.eval_seed_stride),
        )

        train_alignment = _build_train_alignment(
            args, scheduler_mode_for_meta=scheduler_mode_for_meta
        )
        eval_meta = {
            "eval_tag": args.eval_tag,
            "baseline_name": baseline_name,
            "lambda_tag": args.lambda_tag,
            "csi_tag": args.csi_tag,
            "mrt_tag": args.mrt_tag,
            "generated_seed_list": seed_list,
            "train_alignment": train_alignment,
            "trainer_env_config": base_env_config,
            "args": vars(args),
            "time": float(time.time()),
        }

        seed_done_cb = None
        finalize_stream = None
        if bool(args.stream_write):
            seed_done_cb, finalize_stream = make_stream_writer(
                output_dir=output_dir,
                selection_info=selection_info,
                eval_meta=eval_meta,
                seed_list=seed_list,
                flush_every=int(args.stream_flush_every),
            )

        if args.eval_source == "periodic_phase1_model_dir":
            first_agent = next(iter(trainer.agents.values()))
            pair_to_action_idx = _build_pair_to_action_idx(first_agent.discrete_actions_map)
            metrics_pack = run_multi_seed_periodic_validation_metrics(
                base_env_config=base_env_config,
                agents=trainer.agents,
                seed_list=seed_list,
                pair_to_action_idx=pair_to_action_idx,
                show_progress=bool(args.show_progress),
                on_seed_done=seed_done_cb,
            )
        else:
            metrics_pack = run_multi_seed_validation_metrics_train_aligned(
                base_env_config=base_env_config,
                agents=trainer.agents,
                args=args,
                seed_list=seed_list,
                show_progress=bool(args.show_progress),
                on_seed_done=seed_done_cb,
            )

        if callable(finalize_stream):
            finalize_stream(metrics_pack)

        metrics_only_pack = {
            "seed_list": list(metrics_pack["seed_list"]),
            "metrics": copy.deepcopy(metrics_pack["metrics"]),
        }
        record = build_eval_curve_record(episodes_done=0, metrics_pack=metrics_only_pack)

        json_dump(selection_info, output_dir / "selection_info.json")
        json_dump(eval_meta, output_dir / "eval_meta.json")
        json_dump(metrics_only_pack, output_dir / "eval_metrics_pack.json")
        json_dump(record, output_dir / "eval_record.json")
        if args.save_per_seed_metrics:
            json_dump(
                {"per_seed_metrics": metrics_pack.get("per_seed_metrics", [])},
                output_dir / "per_seed_metrics.json",
            )

        row = {
            "baseline_name": baseline_name,
            "eval_source": str(args.eval_source),
            "source_path": row_source_path,
            "num_ground_stations": int(
                base_env_config.get("num_ground_stations", args.num_ground_stations)
            ),
            "num_gs_per_rb": int(base_env_config.get("num_gs_per_rb", args.num_gs_per_rb)),
            "fairness_lambda": float(args.fairness_lambda),
            "csi_error_variance": float(args.csi_error_variance),
            "ris_reflection_power_limit_dBm": float(
                getattr(args, "ris_reflection_power_limit_dBm", 5.0)
            ),
            "disable_ris_path": int(bool(args.disable_ris_path)),
            "include_mrt_features": int(bool(args.include_mrt_features)),
            "scheduler_mode": str(scheduler_mode_for_meta),
            "power_mode": (
                "ppo" if args.eval_source == "periodic_phase1_model_dir" else str(args.power_mode)
            ),
            "phase_policy_dist": str(args.phase_policy_dist),
            "phase_state_mode": str(args.phase_state_mode),
            "lambda_tag": args.lambda_tag,
            "csi_tag": args.csi_tag,
            "mrt_tag": args.mrt_tag,
            "selected_worker_id": None,
            "best_ready_score": None,
            "best_ready_idx": None,
            "periodic_cycle_len": None,
        }

        if args.eval_source == "pbt_best":
            row["selected_worker_id"] = int(selection_info["selected_worker_id"])
            row["best_ready_score"] = float(
                selection_info["best_meta"].get("best_ready_score", float("nan"))
            )
            row["best_ready_idx"] = selection_info["best_meta"].get("best_ready_idx", None)

        if args.eval_source == "periodic_phase1_model_dir":
            pair_cycle, _ = _get_periodic_cycles_for_topology(
                args.num_ground_stations, args.num_gs_per_rb
            )
            row["periodic_cycle_len"] = int(len(pair_cycle))

        for metric_name in DEFAULT_METRIC_KEYS:
            metric_obj = metrics_only_pack["metrics"].get(metric_name, {})
            row[f"{metric_name}_mean"] = float(metric_obj.get("mean", 0.0))
            row[f"{metric_name}_std"] = float(metric_obj.get("std", 0.0))

        save_summary_csv(output_dir / args.csv_name, row)

        print(
            f"[Eval] Finished: baseline={baseline_name}, RIS dBm={float(getattr(args, 'ris_reflection_power_limit_dBm', 5.0))}"
        )
        for metric_name in DEFAULT_METRIC_KEYS:
            metric_obj = metrics_only_pack["metrics"].get(metric_name, {})
            print(
                f"[Eval] {metric_name}: mean={float(metric_obj.get('mean', 0.0)):.6f}, std={float(metric_obj.get('std', 0.0)):.6f}"
            )
        print(f"[Eval] Outputs written to: {output_dir}")

        return row

    finally:
        try:
            trainer.close()
        except Exception:
            pass
        tmp_dir = output_dir / "_tmp_phase2_eval_build"
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
        except Exception:
            pass


def main() -> None:
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    args = parse_args()

    if not bool(getattr(args, "run_batch", False)):
        _single_run_eval(args)
        return

    batch_root_dir = (
        Path(args.batch_root_dir)
        if args.batch_root_dir
        else (Path(args.output_dir) if args.output_dir else Path("evaluation_train_aligned_batch"))
    )
    batch_root_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    skipped_baselines: List[Dict[str, Any]] = []

    for ris_dbm in args.sweep_ris_reflection_power_limit_dBm:
        for baseline_name in args.baseline_list:
            if baseline_name == "periodic":
                reason = _periodic_unsupported_reason(
                    args.num_ground_stations,
                    args.num_gs_per_rb,
                    getattr(args, "dqn_scheduler_impl", "enumerative"),
                )
                if reason is not None:
                    skipped = {
                        "baseline_name": "periodic",
                        "num_ground_stations": int(args.num_ground_stations),
                        "num_gs_per_rb": int(args.num_gs_per_rb),
                        "ris_reflection_power_limit_dBm": float(ris_dbm),
                        "reason": reason,
                    }
                    skipped_baselines.append(skipped)
                    print(f"[BatchEval] SKIP periodic: {reason}")
                    continue
            run_args = _resolve_batch_run_args(
                base_args=args,
                baseline_name=baseline_name,
                ris_dbm=float(ris_dbm),
                batch_root_dir=batch_root_dir,
            )
            row = _single_run_eval(run_args)
            all_rows.append(row)

    save_rows_csv(batch_root_dir / args.csv_name, all_rows)
    json_dump(
        {"rows": all_rows, "skipped_baselines": skipped_baselines},
        batch_root_dir / "batch_summary.json",
    )
    if skipped_baselines:
        json_dump(
            {"skipped_baselines": skipped_baselines},
            batch_root_dir / "skipped_baselines.json",
        )

    print(f"[BatchEval] Finished {len(all_rows)} runs.")
    print(f"[BatchEval] Skipped {len(skipped_baselines)} inapplicable baseline runs.")
    print(f"[BatchEval] Aggregate CSV: {batch_root_dir / args.csv_name}")
    print(f"[BatchEval] Aggregate JSON: {batch_root_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
