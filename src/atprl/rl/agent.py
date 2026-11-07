import math
import random
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from atprl.nn.models import (
    QNetwork,
    ActorNetwork,
    CriticNetwork,
    PhaseActorGaussianLatent,
    PhaseActorVonMises,
    PhaseCritic,
)

from atprl.solvers.wmmse import solve_wmmse_power


class DPRA_Agent:
    def __init__(
        self,
        q_network,
        power_actor,
        power_critic,
        phase_actor,
        phase_critic,
        discrete_actions_map,
        n_users,
        dqn_state_dim,
        max_power_watt,
        device="cpu",
        ris_num_groups: Optional[int] = None,
        phase_wrap_mode: str = "pi",
        force_phase2_uniform_amp: bool = False,
        include_sr_incident_pow_groups: bool = False,
        learn_amp_weights: bool = True,
        phase_state_mode: str = "legacy_full",
        phase_obs_layout: Optional[Dict[str, Any]] = None,
        phase_compact_include_mrt_features: bool = False,
        phase_policy_dist: str = "gaussian_latent",
        dqn_scheduler_impl: str = "enumerative",
        k_users_per_rb: Optional[int] = None,
        continuous_backend=None,
    ):
        self.continuous_backend = continuous_backend
        self.q_network = q_network
        self.power_actor = power_actor
        self.power_critic = power_critic
        self.phase_actor = phase_actor
        self.phase_critic = phase_critic
        self.discrete_actions_map = discrete_actions_map
        self.n_users = n_users
        self.dqn_state_dim = dqn_state_dim
        self.device = device
        self.max_power_watt = max_power_watt
        self.dqn_scheduler_impl = str(dqn_scheduler_impl).strip().lower()
        if self.dqn_scheduler_impl not in ("enumerative",):
            raise ValueError(
                f"dqn_scheduler_impl must be one of ['enumerative'], got {dqn_scheduler_impl!r}"
            )
        self.k_users_per_rb = int(k_users_per_rb or len(next(iter(discrete_actions_map.values()))))

        if phase_wrap_mode not in ("pi", "2pi"):
            raise ValueError(f"phase_wrap_mode must be 'pi' or '2pi', got {phase_wrap_mode}")
        self.phase_wrap_mode = phase_wrap_mode
        self.ris_num_groups = ris_num_groups
        self.force_phase2_uniform_amp = bool(force_phase2_uniform_amp)

        self.include_sr_incident_pow_groups = bool(include_sr_incident_pow_groups)

        self.learn_amp_weights = bool(learn_amp_weights)

        if phase_state_mode not in ("legacy_full", "compact_selected", "all_users_no_action"):
            raise ValueError(
                f"phase_state_mode must be one of ['legacy_full', 'compact_selected'], got {phase_state_mode}"
            )
        self.phase_state_mode = str(phase_state_mode)
        self.phase_obs_layout = None if phase_obs_layout is None else dict(phase_obs_layout)
        self.phase_compact_include_mrt_features = bool(phase_compact_include_mrt_features)

        if phase_policy_dist not in ("gaussian_latent", "von_mises"):
            raise ValueError(
                f"phase_policy_dist must be one of ['gaussian_latent', 'von_mises'], got {phase_policy_dist}"
            )
        self.phase_policy_dist = str(phase_policy_dist)

        if self.phase_policy_dist == "von_mises" and bool(self.learn_amp_weights):
            raise ValueError(
                "phase_policy_dist='von_mises' requires learn_amp_weights=False, "
                "because the Von Mises path is theta-only (1G) in this codebase."
            )

        actor_policy_name = getattr(self.phase_actor, "policy_dist_name", None)
        if actor_policy_name is not None:
            actor_policy_name = str(actor_policy_name).strip().lower()
            if actor_policy_name != self.phase_policy_dist:
                raise ValueError(
                    f"phase_policy_dist mismatch: agent asked for '{self.phase_policy_dist}', "
                    f"but phase_actor.policy_dist_name='{actor_policy_name}'."
                )

    @staticmethod
    def _slice_1d(vec: np.ndarray, start: int, end: int, name: str) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        s = int(start)
        e = int(end)
        if s < 0 or e < s or e > arr.shape[0]:
            raise ValueError(
                f"[PhaseState] invalid slice for {name}: start={s}, end={e}, obs_dim={arr.shape[0]}"
            )
        return arr[s:e].astype(np.float32)

    @staticmethod
    def _pick_user_block(
        user_flat: np.ndarray, n_users: int, user_idx: int, per_user_dim: int, name: str
    ) -> np.ndarray:
        arr = np.asarray(user_flat, dtype=np.float32).reshape(-1)
        n = int(n_users)
        u = int(user_idx)
        d = int(per_user_dim)
        if n <= 0 or d <= 0:
            raise ValueError(f"[PhaseState] invalid n_users/per_user_dim for {name}: n={n}, d={d}")
        if arr.shape[0] != n * d:
            raise ValueError(
                f"[PhaseState] {name} total dim mismatch: got {arr.shape[0]}, expect {n * d} (= {n} * {d})"
            )
        if u < 0 or u >= n:
            raise IndexError(
                f"[PhaseState] user_idx out of range for {name}: user={u}, n_users={n}"
            )
        st = u * d
        ed = st + d
        return arr[st:ed].astype(np.float32)

    def _build_phase_state_legacy(
        self,
        local_observation: np.ndarray,
        user_indices_sorted: Tuple[int, ...],
        power_shares_np: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(local_observation, dtype=np.float32).reshape(-1)

        one_hot_users = np.zeros(self.n_users, dtype=np.float32)
        one_hot_users[list(user_indices_sorted)] = 1.0

        power_shares_np = np.asarray(power_shares_np, dtype=np.float32).reshape(-1)

        phase_state = np.concatenate([obs, one_hot_users, power_shares_np], axis=0).astype(
            np.float32
        )
        return phase_state

    def _build_phase_state_compact(
        self,
        local_observation: np.ndarray,
        user_indices_sorted: Tuple[int, ...],
        power_shares_np: np.ndarray,
    ) -> np.ndarray:
        if self.phase_obs_layout is None:
            raise ValueError(
                "[PhaseState] phase_obs_layout is required when phase_state_mode='compact_selected'."
            )

        obs = np.asarray(local_observation, dtype=np.float32).reshape(-1)
        layout = self.phase_obs_layout
        users = tuple(int(u) for u in user_indices_sorted)

        if len(users) <= 0:
            raise ValueError("[PhaseState] compact_selected requires non-empty selected users.")

        n_users_layout = int(layout.get("n_users", self.n_users))
        if n_users_layout != int(self.n_users):
            raise ValueError(
                f"[PhaseState] layout n_users mismatch: layout={n_users_layout}, agent={self.n_users}"
            )

        def _get_slice(key: str) -> Tuple[int, int]:
            if key not in layout:
                raise KeyError(f"[PhaseState] phase_obs_layout missing key: '{key}'")
            v = layout[key]
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(
                    f"[PhaseState] layout['{key}'] must be a 2-item list/tuple, got {v}"
                )
            return int(v[0]), int(v[1])

        casc_s, casc_e = _get_slice("casc_slice")
        direct_s, direct_e = _get_slice("direct_slice")
        ls_s, ls_e = _get_slice("ls_slice")
        uxy_s, uxy_e = _get_slice("uav_xy_slice")
        uz_s, uz_e = _get_slice("uav_z_slice")
        hist_s, hist_e = _get_slice("hist_rate_slice")

        G = int(
            layout.get(
                "ris_num_groups", self.ris_num_groups if self.ris_num_groups is not None else 0
            )
        )
        if G <= 0:
            raise ValueError(f"[PhaseState] invalid ris_num_groups in layout: {G}")

        casc_all = self._slice_1d(obs, casc_s, casc_e, "casc_slice")
        direct_all = self._slice_1d(obs, direct_s, direct_e, "direct_slice")
        ls_all = self._slice_1d(obs, ls_s, ls_e, "ls_slice")
        uxy_all = self._slice_1d(obs, uxy_s, uxy_e, "uav_xy_slice")
        uz_all = self._slice_1d(obs, uz_s, uz_e, "uav_z_slice")
        hist_all = self._slice_1d(obs, hist_s, hist_e, "hist_rate_slice")

        casc_dim_per_user = int(layout.get("casc_feats_per_user", 3 * G))
        direct_dim_per_user = int(layout.get("direct_feats_per_user", 3))
        mrt_dim_per_user = int(layout.get("mrt_feats_per_user", 2 * G))

        if casc_all.shape[0] != n_users_layout * casc_dim_per_user:
            raise ValueError(
                f"[PhaseState] casc block dim mismatch: got {casc_all.shape[0]}, "
                f"expect {n_users_layout * casc_dim_per_user}"
            )
        if direct_all.shape[0] != n_users_layout * direct_dim_per_user:
            raise ValueError(
                f"[PhaseState] direct block dim mismatch: got {direct_all.shape[0]}, "
                f"expect {n_users_layout * direct_dim_per_user}"
            )

        uxy_dim_per_user = int((uxy_e - uxy_s) // n_users_layout)
        hist_dim_per_user = int((hist_e - hist_s) // n_users_layout)

        if uxy_dim_per_user <= 0 or (uxy_e - uxy_s) != n_users_layout * uxy_dim_per_user:
            raise ValueError(
                f"[PhaseState] uav_xy block cannot be evenly split by n_users: "
                f"dim={uxy_e - uxy_s}, n_users={n_users_layout}"
            )
        if hist_dim_per_user <= 0 or (hist_e - hist_s) != n_users_layout * hist_dim_per_user:
            raise ValueError(
                f"[PhaseState] hist_rate block cannot be evenly split by n_users: "
                f"dim={hist_e - hist_s}, n_users={n_users_layout}"
            )

        expected_ls_dim = 1 + n_users_layout + n_users_layout
        if ls_all.shape[0] != expected_ls_dim:
            raise ValueError(
                f"[PhaseState] ls block dim mismatch: got {ls_all.shape[0]}, expect {expected_ls_dim}"
            )

        ls_sr = ls_all[:1].astype(np.float32)
        ls_rg_all = ls_all[1 : 1 + n_users_layout].astype(np.float32)
        ls_sg_all = ls_all[1 + n_users_layout : 1 + 2 * n_users_layout].astype(np.float32)

        parts: List[np.ndarray] = []

        for u in users:
            parts.append(
                self._pick_user_block(casc_all, n_users_layout, u, casc_dim_per_user, "casc")
            )

        for u in users:
            parts.append(
                self._pick_user_block(direct_all, n_users_layout, u, direct_dim_per_user, "direct")
            )

        parts.append(ls_sr)
        parts.append(ls_rg_all[list(users)].astype(np.float32))
        parts.append(ls_sg_all[list(users)].astype(np.float32))

        for u in users:
            parts.append(
                self._pick_user_block(uxy_all, n_users_layout, u, uxy_dim_per_user, "uav_xy")
            )
        parts.append(uz_all.astype(np.float32))

        for u in users:
            parts.append(
                self._pick_user_block(hist_all, n_users_layout, u, hist_dim_per_user, "hist_rate")
            )

        if self.phase_compact_include_mrt_features:
            mrt_s, mrt_e = _get_slice("mrt_slice")
            mrt_all = self._slice_1d(obs, mrt_s, mrt_e, "mrt_slice")
            if mrt_all.shape[0] != n_users_layout * mrt_dim_per_user:
                raise ValueError(
                    f"[PhaseState] mrt block dim mismatch: got {mrt_all.shape[0]}, "
                    f"expect {n_users_layout * mrt_dim_per_user}"
                )
            for u in users:
                parts.append(
                    self._pick_user_block(mrt_all, n_users_layout, u, mrt_dim_per_user, "mrt")
                )

        parts.append(np.asarray(power_shares_np, dtype=np.float32).reshape(-1))

        phase_state = np.concatenate(parts, axis=0).astype(np.float32)
        return phase_state

    def _build_phase_state_all_users_no_action(self, local_observation: np.ndarray) -> np.ndarray:
        """Construct the phase observation from pre-action environment features only."""
        layout = self.phase_obs_layout
        if layout is None:
            raise ValueError("[PhaseState] all_users_no_action requires phase_obs_layout")
        obs = np.asarray(local_observation, dtype=np.float32).reshape(-1)
        if obs.size != int(layout["obs_dim"]):
            raise ValueError("[PhaseState] observation length does not match layout obs_dim")
        n = int(layout["n_users"])
        if n <= 0 or n != int(self.n_users):
            raise ValueError(
                f"[PhaseState] layout n_users mismatch: layout={n}, agent={self.n_users}"
            )
        blocks = [
            ("casc", n * int(layout["casc_feats_per_user"])),
            ("direct", n * int(layout["direct_feats_per_user"])),
            ("ls", 1 + 2 * n),
            ("uav_xy", n * int(layout["uav_xy_feats_per_user"])),
            ("uav_z", int(layout["uav_z_slice"][1] - layout["uav_z_slice"][0])),
            ("hist_rate", n * int(layout["hist_rate_feats_per_user"])),
        ]
        if self.phase_compact_include_mrt_features:
            if not layout.get("include_mrt_features", False):
                raise ValueError("[PhaseState] all-users MRT state requires MRT observations")
            blocks.append(("mrt", n * int(layout["mrt_feats_per_user"])))
        parts = []
        for name, expected in blocks:
            block = self._slice_1d(obs, *layout[f"{name}_slice"], f"{name}_slice")
            if expected <= 0 or block.size != expected:
                raise ValueError(
                    f"[PhaseState] {name} block dim mismatch: got {block.size}, expect {expected}"
                )
            parts.append(block)
        state = np.concatenate(parts).astype(np.float32)
        for name, model in (("actor", self.phase_actor), ("critic", self.phase_critic)):
            expected = next(
                layer for layer in model.modules() if isinstance(layer, nn.Linear)
            ).in_features
            if state.size != expected:
                raise ValueError(
                    f"[PhaseState] {name} input dim mismatch: got {state.size}, expect {expected}"
                )
        if not np.isfinite(state).all():
            raise ValueError("[PhaseState] all_users_no_action contains NaN/Inf")
        return state

    def _build_phase_state(
        self,
        local_observation: np.ndarray,
        user_indices_sorted: Tuple[int, ...],
        power_shares_np: np.ndarray,
    ) -> np.ndarray:
        if self.phase_state_mode == "legacy_full":
            return self._build_phase_state_legacy(
                local_observation=local_observation,
                user_indices_sorted=user_indices_sorted,
                power_shares_np=power_shares_np,
            )
        if self.phase_state_mode == "compact_selected":
            return self._build_phase_state_compact(
                local_observation=local_observation,
                user_indices_sorted=user_indices_sorted,
                power_shares_np=power_shares_np,
            )
        if self.phase_state_mode == "all_users_no_action":
            return self._build_phase_state_all_users_no_action(local_observation)
        raise ValueError(f"Unsupported phase_state_mode: {self.phase_state_mode}")

    def _run_phase_policy_gaussian(
        self,
        phase_state_tensor: torch.Tensor,
        *,
        deterministic_phase: bool,
        deterministic_amp: bool,
        phase_ppo_mode: str,
        force_uniform_amp: bool,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            phase_dist = self.phase_actor(phase_state_tensor)
            phase_value = self.phase_critic(phase_state_tensor)

            u_mean = phase_dist.mean
            need_sample = (not bool(deterministic_phase)) or (
                self.learn_amp_weights and (not bool(deterministic_amp))
            )
            u_samp = phase_dist.sample() if need_sample else None

        action_dim = int(u_mean.shape[-1])

        if self.ris_num_groups is not None:
            G = int(self.ris_num_groups)
        else:
            if self.learn_amp_weights:
                if action_dim % 3 != 0:
                    raise ValueError(
                        f"Phase latent dim must be 3G when learn_amp_weights=True, got {action_dim}"
                    )
                G = action_dim // 3
            else:
                if action_dim % 2 != 0:
                    raise ValueError(
                        f"Phase latent dim must be 2G when learn_amp_weights=False, got {action_dim}"
                    )
                G = action_dim // 2

        learn_amp = bool(self.learn_amp_weights)

        if learn_amp:
            expected = 3 * G
            if action_dim != expected:
                raise ValueError(
                    f"Expected phase latent dim=3G={expected} (learn_amp), got {action_dim}"
                )
        else:
            expected = 2 * G
            if action_dim != expected:
                raise ValueError(
                    f"Expected phase latent dim=2G={expected} (phase-only), got {action_dim}"
                )

        if deterministic_phase:
            u_phase_xy = u_mean[:, : 2 * G]
        else:
            if u_samp is None:
                u_samp = phase_dist.sample()
            u_phase_xy = u_samp[:, : 2 * G]

        if learn_amp:
            if deterministic_amp:
                u_amp = u_mean[:, 2 * G :]
            else:
                if u_samp is None:
                    u_samp = phase_dist.sample()
                u_amp = u_samp[:, 2 * G :]
        else:
            u_amp = None

        should_force_uniform = (not learn_amp) or force_uniform_amp or self.force_phase2_uniform_amp
        if learn_amp and should_force_uniform:
            u_amp = torch.zeros_like(u_amp)

        u = torch.cat([u_phase_xy, u_amp], dim=-1) if learn_amp else u_phase_xy

        base = getattr(phase_dist, "base_dist", None)
        if base is None or (not hasattr(base, "log_prob")):
            raise RuntimeError(
                "phase_dist must expose base_dist.log_prob for masked log_prob computation."
            )
        per_dim_lp = base.log_prob(u)

        log_prob_phase = per_dim_lp[..., : 2 * G].sum(dim=-1)
        log_prob_amp = (
            per_dim_lp[..., 2 * G :].sum(dim=-1) if learn_amp else torch.zeros_like(log_prob_phase)
        )

        if learn_amp and should_force_uniform:
            joint_log_prob = log_prob_phase
            effective_mode = "theta"
        else:
            effective_mode = phase_ppo_mode
            if phase_ppo_mode == "theta":
                joint_log_prob = log_prob_phase
            elif phase_ppo_mode == "amp":
                joint_log_prob = log_prob_amp if learn_amp else log_prob_phase
            else:
                joint_log_prob = log_prob_phase + log_prob_amp

        xy = u_phase_xy.view(-1, G, 2)
        x = xy[..., 0]
        y = xy[..., 1]
        eps = 1e-8
        norm = torch.sqrt(x * x + y * y + eps)
        x_n = x / norm
        y_n = y / norm
        phase_action_theta = torch.atan2(y_n, x_n)

        if learn_amp:
            ris_power_weights_t = torch.softmax(u_amp, dim=-1)
        else:
            ris_power_weights_t = torch.full((1, G), 1.0 / float(G), device=u.device, dtype=u.dtype)

        return {
            "G": int(G),
            "phase_value": phase_value,
            "u": u,
            "u_amp": u_amp,
            "phase_action_theta": phase_action_theta,
            "ris_power_weights_t": ris_power_weights_t,
            "log_prob_phase": log_prob_phase,
            "log_prob_amp": log_prob_amp,
            "joint_log_prob": joint_log_prob,
            "effective_mode": effective_mode,
        }

    def _run_phase_policy_von_mises(
        self,
        phase_state_tensor: torch.Tensor,
        *,
        deterministic_phase: bool,
        deterministic_amp: bool,
        phase_ppo_mode: str,
        force_uniform_amp: bool,
    ) -> Dict[str, Any]:
        _ = deterministic_amp
        _ = phase_ppo_mode
        _ = force_uniform_amp

        if bool(self.learn_amp_weights):
            raise ValueError(
                "Von Mises phase policy is theta-only (1G) and requires learn_amp_weights=False."
            )

        with torch.no_grad():
            phase_dist = self.phase_actor(phase_state_tensor)
            phase_value = self.phase_critic(phase_state_tensor)

            base = getattr(phase_dist, "base_dist", None)
            if base is None or (not hasattr(base, "loc")):
                raise RuntimeError(
                    "Von Mises phase_dist must expose base_dist.loc. "
                    "Please ensure phase_actor returns Independent(VonMises(...), 1)."
                )

            theta_mean = base.loc
            if theta_mean.ndim != 2:
                raise ValueError(
                    f"Von Mises mean direction must be 2D [B,G], got shape={theta_mean.shape}"
                )

            action_dim = int(theta_mean.shape[-1])

            if self.ris_num_groups is not None:
                G = int(self.ris_num_groups)
                if action_dim != G:
                    raise ValueError(
                        f"Von Mises theta dim mismatch: actor outputs G={action_dim}, "
                        f"but ris_num_groups={G}."
                    )
            else:
                G = action_dim

            if deterministic_phase:
                theta = theta_mean
            else:
                theta = phase_dist.sample()

            if theta.shape[-1] != G:
                raise ValueError(
                    f"Von Mises sampled theta shape mismatch: got {theta.shape}, expected last dim={G}"
                )

            log_prob_phase = phase_dist.log_prob(theta)
            log_prob_amp = torch.zeros_like(log_prob_phase)
            joint_log_prob = log_prob_phase

            ris_power_weights_t = torch.full(
                (theta.shape[0], G),
                1.0 / float(G),
                device=theta.device,
                dtype=theta.dtype,
            )

            phase_action_theta = theta
            u = theta
            u_amp = None
            effective_mode = "theta"

        return {
            "G": int(G),
            "phase_value": phase_value,
            "u": u,
            "u_amp": u_amp,
            "phase_action_theta": phase_action_theta,
            "ris_power_weights_t": ris_power_weights_t,
            "log_prob_phase": log_prob_phase,
            "log_prob_amp": log_prob_amp,
            "joint_log_prob": joint_log_prob,
            "effective_mode": effective_mode,
        }

    def select_scheduler_context(
        self, local_observation, current_epsilon=0.0, deterministic_dqn=True, scheduler_mode="dqn"
    ):
        obs_full = local_observation.astype(np.float32)
        dqn_state = obs_full[: self.dqn_state_dim]

        if dqn_state.shape[0] != self.dqn_state_dim:
            dqn_state = dqn_state[: self.dqn_state_dim]

        dqn_state_tensor = torch.from_numpy(dqn_state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(dqn_state_tensor)
        if scheduler_mode == "random":
            discrete_action_idx = random.randint(0, len(self.discrete_actions_map) - 1)
        elif deterministic_dqn:
            discrete_action_idx = q_values.argmax().item()
        elif random.random() < current_epsilon:
            discrete_action_idx = random.randint(0, len(self.discrete_actions_map) - 1)
        else:
            discrete_action_idx = q_values.argmax().item()
        scheduler_sequence = tuple(self.discrete_actions_map[discrete_action_idx])
        user_indices_sorted = tuple(sorted(scheduler_sequence))
        scheduler_action = int(discrete_action_idx)

        return dict(
            dqn_state=dqn_state,
            selected_users=user_indices_sorted,
            scheduler_sequence=scheduler_sequence,
            scheduler_action=scheduler_action,
            discrete_action_idx=discrete_action_idx,
        )

    def select_action(
        self,
        local_observation: np.ndarray,
        current_epsilon: float = 0.0,
        current_min_private_power_fraction: float = 0.0,
        deterministic: bool = False,
        deterministic_dqn: Optional[bool] = None,
        deterministic_power: Optional[bool] = None,
        deterministic_phase: Optional[bool] = None,
        deterministic_amp: Optional[bool] = None,
        phase_ppo_mode: str = "joint",
        force_uniform_amp: bool = False,
        scheduler_mode: str = "dqn",
        power_mode: str = "ppo",
        wmmse_ctx: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
        rb_index: int = 0,
        max_wmmse_iters: int = 50,
        wmmse_tol: float = 1e-6,
        wmmse_scipy_method: str = "SLSQP",
        wmmse_scipy_options: Optional[Dict[str, Any]] = None,
        wmmse_sic_repair_max_iters: int = 30,
        wmmse_sic_repair_step_ratio: float = 0.05,
    ) -> Dict[str, Any]:
        if deterministic_dqn is None:
            deterministic_dqn = deterministic
        if deterministic_power is None:
            deterministic_power = deterministic
        if deterministic_phase is None:
            deterministic_phase = deterministic
        if deterministic_amp is None:
            deterministic_amp = deterministic_phase

        if phase_ppo_mode not in ("theta", "amp", "joint"):
            raise ValueError(
                f"phase_ppo_mode must be one of ['theta','amp','joint'], got {phase_ppo_mode}"
            )

        if scheduler_mode not in ("dqn", "random"):
            raise ValueError(
                f"scheduler_mode must be one of ['dqn', 'random'], got {scheduler_mode}"
            )

        if power_mode not in ("ppo", "wmmse"):
            raise ValueError(f"power_mode must be one of ['ppo', 'wmmse'], got {power_mode}")

        context = self.select_scheduler_context(
            local_observation, current_epsilon, deterministic_dqn, scheduler_mode
        )
        dqn_state = context["dqn_state"]
        user_indices_sorted = context["selected_users"]
        scheduler_sequence = context["scheduler_sequence"]
        scheduler_action = context["scheduler_action"]
        discrete_action_idx = context["discrete_action_idx"]

        alpha = np.zeros(self.n_users, dtype=np.float32)
        alpha[list(user_indices_sorted)] = 1.0
        power_state = np.concatenate([alpha, dqn_state]).astype(np.float32)
        power_state_tensor = torch.from_numpy(power_state).float().unsqueeze(0).to(self.device)

        if self.continuous_backend is not None:
            return self.continuous_backend(
                self,
                local_observation,
                context,
                power_state,
                deterministic_power,
                deterministic_phase,
            )

        wmmse_result: Optional[Dict[str, Any]] = None

        if power_mode == "ppo":
            with torch.no_grad():
                dist_power = self.power_actor(power_state_tensor)
                value_power = self.power_critic(power_state_tensor)

                if deterministic_power:
                    power_action_raw = dist_power.mean
                else:
                    power_action_raw = dist_power.sample()

                log_prob_power = dist_power.log_prob(power_action_raw)

            power_shares = torch.softmax(power_action_raw, dim=-1)
            final_power_watt = power_shares * self.max_power_watt

            final_power_watt_np = final_power_watt.detach().cpu().numpy().squeeze(0)

        else:
            with torch.no_grad():
                wmmse_result = solve_wmmse_power(
                    wmmse_ctx=wmmse_ctx,
                    env=env,
                    user_indices=user_indices_sorted,
                    rb_index=int(rb_index),
                    max_wmmse_iters=int(max_wmmse_iters),
                    tol=float(wmmse_tol),
                    scipy_method=str(wmmse_scipy_method),
                    scipy_options=wmmse_scipy_options,
                    sic_repair_max_iters=int(wmmse_sic_repair_max_iters),
                    sic_repair_step_ratio=float(wmmse_sic_repair_step_ratio),
                )

            if not isinstance(wmmse_result, dict):
                raise RuntimeError("solve_wmmse_power(...) must return a dict.")

            if "power_vec" not in wmmse_result:
                raise KeyError("WMMSE result missing required key: 'power_vec'.")

            final_power_watt_np = np.asarray(wmmse_result["power_vec"], dtype=np.float32).reshape(
                -1
            )

            expected_power_dim = 1 + len(user_indices_sorted)
            if final_power_watt_np.shape != (expected_power_dim,):
                raise ValueError(
                    f"WMMSE power_vec shape mismatch: expected {(expected_power_dim,)}, got {final_power_watt_np.shape}"
                )

            if not np.all(np.isfinite(final_power_watt_np)):
                raise ValueError(
                    f"WMMSE power_vec contains non-finite values: {final_power_watt_np}"
                )

            total_power = float(np.sum(final_power_watt_np))
            if total_power < -1e-8:
                raise ValueError(f"WMMSE power_vec total power is negative: {total_power}")

            if total_power > float(self.max_power_watt) + 1e-8:
                final_power_watt_np = final_power_watt_np * (
                    float(self.max_power_watt) / max(total_power, 1e-12)
                )
                total_power = float(np.sum(final_power_watt_np))

            if np.any(final_power_watt_np < -1e-8):
                raise ValueError(f"WMMSE power_vec has negative entries: {final_power_watt_np}")

            final_power_watt_np = np.maximum(final_power_watt_np, 0.0).astype(np.float32)

            power_shares_np = final_power_watt_np / max(float(self.max_power_watt), 1e-12)
            power_shares_np = np.clip(power_shares_np, 0.0, 1.0).astype(np.float32)

            eps = 1e-12
            raw_proxy_np = np.log(np.maximum(power_shares_np, eps)).astype(np.float32)

            power_action_raw = torch.from_numpy(raw_proxy_np).unsqueeze(0).to(self.device)
            log_prob_power = torch.zeros((1,), dtype=torch.float32, device=self.device)
            value_power = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            power_shares = torch.from_numpy(power_shares_np).unsqueeze(0).to(self.device)

        power_shares_np_for_phase = power_shares.detach().cpu().numpy().flatten().astype(np.float32)

        phase_state = self._build_phase_state(
            local_observation=local_observation,
            user_indices_sorted=user_indices_sorted,
            power_shares_np=power_shares_np_for_phase,
        )
        phase_state_tensor = torch.from_numpy(phase_state).unsqueeze(0).to(self.device)

        if self.phase_policy_dist == "gaussian_latent":
            phase_out = self._run_phase_policy_gaussian(
                phase_state_tensor,
                deterministic_phase=bool(deterministic_phase),
                deterministic_amp=bool(deterministic_amp),
                phase_ppo_mode=str(phase_ppo_mode),
                force_uniform_amp=bool(force_uniform_amp),
            )
        elif self.phase_policy_dist == "von_mises":
            phase_out = self._run_phase_policy_von_mises(
                phase_state_tensor,
                deterministic_phase=bool(deterministic_phase),
                deterministic_amp=bool(deterministic_amp),
                phase_ppo_mode=str(phase_ppo_mode),
                force_uniform_amp=bool(force_uniform_amp),
            )
        else:
            raise ValueError(f"Unsupported phase_policy_dist: {self.phase_policy_dist}")

        G = int(phase_out["G"])
        phase_value = phase_out["phase_value"]
        u = phase_out["u"]
        u_amp = phase_out["u_amp"]
        phase_action_theta = phase_out["phase_action_theta"]
        ris_power_weights_t = phase_out["ris_power_weights_t"]
        log_prob_phase = phase_out["log_prob_phase"]
        log_prob_amp = phase_out["log_prob_amp"]
        joint_log_prob = phase_out["joint_log_prob"]
        effective_mode = phase_out["effective_mode"]

        phi_G_raw = phase_action_theta.detach().cpu().numpy().reshape(-1)

        G_inferred = int(phi_G_raw.shape[0])

        if self.ris_num_groups is not None:
            if G_inferred != int(self.ris_num_groups):
                raise ValueError(
                    f"[PhaseWrap] G mismatch: inferred={G_inferred} vs ris_num_groups={self.ris_num_groups}. "
                    "Check PhaseActor output dim = G and env.ris_num_groups."
                )
        G = int(self.ris_num_groups) if (self.ris_num_groups is not None) else G_inferred

        if self.phase_wrap_mode == "2pi":
            phi_min, phi_max = 0.0, 2.0 * np.pi
            phi_G = np.mod(phi_G_raw, (phi_max - phi_min)) + phi_min
            in_range = (phi_G >= phi_min - 1e-7) & (phi_G < phi_max + 1e-7)
        else:
            phi_min, phi_max = -np.pi, np.pi
            phi_G = (phi_G_raw - phi_min) % (phi_max - phi_min) + phi_min
            in_range = (phi_G >= phi_min - 1e-7) & (phi_G < phi_max + 1e-7)

        if not np.all(np.isfinite(phi_G)):
            raise ValueError(f"[PhaseWrap] phi_G contains non-finite values: {phi_G}")
        if phi_G.shape != (G,):
            raise ValueError(f"[PhaseWrap] phi_G shape mismatch: {phi_G.shape} vs {(G,)}")
        if not np.all(in_range):
            bad = phi_G[~in_range]
            raise ValueError(
                f"[PhaseWrap] phi_G out of expected range mode={self.phase_wrap_mode}, "
                f"range=[{phi_min},{phi_max}) bad_samples={bad[:8]}"
            )

        phase_wrap_delta_abs = float(np.mean(np.abs(phi_G - phi_G_raw)))
        if self.phase_wrap_mode == "2pi":
            phase_wrap_oor_ratio = float(np.mean((phi_G_raw < 0.0) | (phi_G_raw >= 2.0 * np.pi)))
        else:
            phase_wrap_oor_ratio = float(np.mean((phi_G_raw < -np.pi) | (phi_G_raw >= np.pi)))

        ris_power_weights = ris_power_weights_t.detach().cpu().numpy().reshape(-1)
        if ris_power_weights.shape != (G,):
            raise ValueError(
                f"ris_power_weights shape mismatch: {ris_power_weights.shape} vs {(G,)}"
            )
        if not np.all(np.isfinite(ris_power_weights)):
            raise ValueError(f"ris_power_weights contains non-finite values: {ris_power_weights}")
        s = float(np.sum(ris_power_weights))
        if not (0.999 <= s <= 1.001):
            raise ValueError(f"ris_power_weights must sum to 1, got sum={s}")
        if np.any(ris_power_weights < -1e-8):
            raise ValueError(f"ris_power_weights has negative entries: {ris_power_weights}")

        phi_G = phi_G.astype(np.float32)
        ris_power_weights = ris_power_weights.astype(np.float32)

        env_action = (user_indices_sorted, final_power_watt_np, phi_G, ris_power_weights)

        phase_joint_action = u.squeeze(0).detach().cpu()

        phase_action_theta_squeezed = phase_action_theta.squeeze(0).detach().cpu()
        if u_amp is None:
            amp_action_raw_squeezed = torch.zeros(G, dtype=torch.float32)
        else:
            amp_action_raw_squeezed = u_amp.squeeze(0).detach().cpu()

        info = {
            "dqn_state": dqn_state,
            "discrete_action_idx": None
            if discrete_action_idx is None
            else int(discrete_action_idx),
            "scheduler_sequence": scheduler_sequence,
            "scheduler_action": scheduler_action,
            "dqn_scheduler_impl": self.dqn_scheduler_impl,
            "selected_users": user_indices_sorted,
            "scheduler_mode": scheduler_mode,
            "power_mode": power_mode,
            "power_state": power_state_tensor.squeeze(0).detach().cpu(),
            "power_action_raw": power_action_raw.squeeze(0).detach().cpu(),
            "power_action": final_power_watt_np,
            "power_log_prob": log_prob_power.squeeze(0).detach().cpu(),
            "power_value": value_power.squeeze(0).detach().cpu(),
            "wmmse_is_feasible": (
                None if wmmse_result is None else bool(wmmse_result.get("is_feasible", False))
            ),
            "wmmse_solver_status": (
                None if wmmse_result is None else str(wmmse_result.get("solver_status", "unknown"))
            ),
            "wmmse_iters_used": (
                None if wmmse_result is None else int(wmmse_result.get("iters_used", 0))
            ),
            "wmmse_common_sinr_all": (
                None
                if wmmse_result is None
                else np.asarray(wmmse_result.get("common_sinr_all", []), dtype=np.float32)
            ),
            "wmmse_sum_rate_objective": (
                None if wmmse_result is None else float(wmmse_result.get("sum_rate_objective", 0.0))
            ),
            "phase_state": phase_state_tensor.squeeze(0).detach().cpu(),
            "phase_action_theta": phase_action_theta_squeezed.detach().cpu(),
            "phase_amp_raw": amp_action_raw_squeezed.detach().cpu(),
            "phase_joint_action": phase_joint_action.detach().cpu(),
            "phase_log_prob_phase": log_prob_phase.squeeze(0).detach().cpu(),
            "phase_log_prob_amp": log_prob_amp.squeeze(0).detach().cpu(),
            "phase_log_prob": joint_log_prob.squeeze(0).detach().cpu(),
            "phase_joint_log_prob": joint_log_prob.squeeze(0).detach().cpu(),
            "phase_value": phase_value.squeeze(0).detach().cpu(),
            "phase_state_mode": self.phase_state_mode,
            "phase_compact_include_mrt_features": bool(self.phase_compact_include_mrt_features),
            "phase_policy_dist": self.phase_policy_dist,
            "env_action": env_action,
            "phase_wrap_oor_ratio": phase_wrap_oor_ratio,
            "phase_wrap_delta_abs": phase_wrap_delta_abs,
            "phase_effective_mode": effective_mode,
        }

        return info
