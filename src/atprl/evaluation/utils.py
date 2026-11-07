import copy
import json
import random
from typing import Dict, Any, Sequence, Optional, Iterable, List

import numpy as np
import torch

from atprl.envs.leo_sat import LeoSatEnv


FIXED_VALIDATION_SEEDS = (
    10001,
    20002,
    30003,
    40004,
    50005,
    60006,
    70007,
    80008,
    90009,
    100010,
    110011,
    120012,
    130013,
    140014,
    150015,
    160016,
    170017,
    180018,
    190019,
    200002,
    210012,
    220013,
    230004,
    230005,
)

EVAL_FEATURE_PROFILES = {
    "phase1": {
        "include_mrt_features": False,
        "include_sr_incident_pow_groups": False,
        "ris_boost_lambda": 0.0,
    },
    "phase2": {
        "include_mrt_features": True,
        "include_sr_incident_pow_groups": False,
        "ris_boost_lambda": 0.0,
    },
    "phase2_no_mrt": {
        "include_mrt_features": False,
        "include_sr_incident_pow_groups": False,
        "ris_boost_lambda": 0.0,
    },
}

EVAL_MODE_PRESETS = {
    "standard": {},
    "no_ris": {
        "disable_ris_path": True,
        "active_ris_enabled": False,
    },
    "force_ris_on": {
        "disable_ris_path": False,
    },
}


def summarize_values(values):
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "values": []}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "values": [float(x) for x in arr.tolist()],
    }


def summarize_metrics_by_seed(metrics_all, metric_keys):
    out = {}
    for k in metric_keys:
        vals = [float(m.get(k, 0.0)) for m in (metrics_all or [])]
        out[k] = summarize_values(vals)
    return out


def get_eval_config(
    base_config: dict,
    mode: str = "standard",
    profile: str = "phase1",
    *,
    force_fixed_positions: bool = True,
) -> dict:
    cfg = copy.deepcopy(base_config)

    cfg["reseed_each_episode"] = True
    if force_fixed_positions:
        cfg["use_fixed_positions"] = True

    if profile not in EVAL_FEATURE_PROFILES:
        raise ValueError(
            f"Unknown eval profile: {profile}. Available: {list(EVAL_FEATURE_PROFILES.keys())}"
        )
    cfg.update(EVAL_FEATURE_PROFILES[profile])

    if mode not in EVAL_MODE_PRESETS:
        raise ValueError(f"Unknown eval mode: {mode}. Available: {list(EVAL_MODE_PRESETS.keys())}")
    cfg.update(EVAL_MODE_PRESETS[mode])

    cfg["include_sr_incident_pow_groups"] = False
    cfg["ris_boost_lambda"] = 0.0

    if cfg.get("disable_ris_path", False):
        cfg["active_ris_enabled"] = False

    return cfg


def make_eval_env(config: dict, seed: int) -> LeoSatEnv:
    env_cfg = copy.deepcopy(config)
    env_cfg["random_seed"] = int(seed)
    env_cfg["reseed_each_episode"] = True
    env = LeoSatEnv(config=env_cfg)
    return env


DEFAULT_METRIC_KEYS = (
    "undiscounted_return",
    "avg_sum_rate",
    "min_user_avg_rate",
    "jain_index",
)


def _jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    denom = float(x.size) * float(np.sum(np.square(x)))
    if denom <= 0.0:
        return 0.0
    return float((np.sum(x) ** 2) / denom)


def _infer_num_users_from_env(eval_env: LeoSatEnv) -> int:
    try:
        n = int(eval_env.config.get("num_ground_stations", 0))
        if n > 0:
            return n
    except Exception:
        pass

    try:
        n = int(len(np.asarray(eval_env.historical_avg_rates).reshape(-1)))
        if n > 0:
            return n
    except Exception:
        pass

    return 1


def extract_step_rate_vector_from_infos(
    infos: Dict[str, Dict[str, Any]],
    eval_env: LeoSatEnv,
) -> np.ndarray:
    if infos is None:
        infos = {}

    for _, info in infos.items():
        if not isinstance(info, dict):
            continue
        v = info.get("rates_macro_vec", None)
        if v is not None:
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size > 0:
                return arr

    for _, info in infos.items():
        if not isinstance(info, dict):
            continue
        v = info.get("total_rates_per_user", None)
        if v is not None:
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size > 0:
                return arr

    try:
        arr = np.asarray(eval_env.historical_avg_rates, dtype=np.float64).reshape(-1)
        if arr.size > 0:
            return arr
    except Exception:
        pass

    return np.zeros(_infer_num_users_from_env(eval_env), dtype=np.float64)


def _resolve_scheduler_mode(args) -> str:
    mode = str(getattr(args, "scheduler_mode", "dqn")).strip().lower()
    if mode not in ("dqn", "random"):
        return "dqn"
    return mode


def _resolve_power_mode(args) -> str:
    mode = str(getattr(args, "power_mode", "ppo")).strip().lower()
    if mode not in ("ppo", "wmmse"):
        return "ppo"
    return mode


def _build_wmmse_eval_kwargs(args, eval_env: LeoSatEnv) -> Dict[str, Any]:
    power_mode = _resolve_power_mode(args)

    kwargs: Dict[str, Any] = {
        "power_mode": power_mode,
    }

    if power_mode != "wmmse":
        return kwargs

    scipy_options = getattr(args, "wmmse_scipy_options", None)
    if isinstance(scipy_options, str):
        txt = scipy_options.strip()
        if txt == "":
            scipy_options = None
        else:
            try:
                scipy_options = json.loads(txt)
            except Exception:
                scipy_options = None

    kwargs.update(
        {
            "env": eval_env,
            "rb_index": int(getattr(args, "wmmse_rb_index", 0)),
            "max_wmmse_iters": int(getattr(args, "wmmse_max_iters", 50)),
            "wmmse_tol": float(getattr(args, "wmmse_tol", 1e-6)),
            "wmmse_scipy_method": str(getattr(args, "wmmse_scipy_method", "SLSQP")),
            "wmmse_scipy_options": scipy_options,
            "wmmse_sic_repair_max_iters": int(getattr(args, "wmmse_sic_repair_max_iters", 30)),
            "wmmse_sic_repair_step_ratio": float(
                getattr(args, "wmmse_sic_repair_step_ratio", 0.05)
            ),
        }
    )
    return kwargs


def _select_eval_action_for_agent(
    agent,
    obs_i,
    agent_id: str,
    args,
    eval_env: LeoSatEnv,
):
    scheduler_mode = _resolve_scheduler_mode(args)
    power_mode = _resolve_power_mode(args)
    power_kwargs = _build_wmmse_eval_kwargs(args, eval_env)

    try:
        return agent.select_action(
            local_observation=obs_i,
            current_epsilon=0.0,
            current_min_private_power_fraction=0.0,
            deterministic=True,
            deterministic_dqn=True,
            deterministic_power=True,
            deterministic_phase=True,
            deterministic_amp=True,
            scheduler_mode=scheduler_mode,
            **power_kwargs,
        )
    except TypeError:
        pass

    try:
        return agent.select_action(
            local_observation=obs_i,
            current_epsilon=0.0,
            current_min_private_power_fraction=0.0,
            deterministic=True,
            scheduler_mode=scheduler_mode,
            **power_kwargs,
        )
    except TypeError:
        pass

    if scheduler_mode == "random":
        try:
            return agent.select_action(
                obs_i,
                rb_idx=0,
                deterministic=True,
                evaluate=True,
                current_min_private_power_fraction=0.0,
                scheduler_mode="random",
            )
        except TypeError:
            pass

        try:
            return agent.select_action(
                obs_i,
                rb_idx=0,
                deterministic=True,
                current_min_private_power_fraction=0.0,
                scheduler_mode="random",
            )
        except TypeError:
            pass

        try:
            return agent.select_action(
                obs_i,
                rb_idx=0,
                evaluate=True,
                current_min_private_power_fraction=0.0,
                scheduler_mode="random",
            )
        except TypeError:
            pass

        try:
            return agent.select_action(
                obs_i,
                rb_idx=0,
                current_min_private_power_fraction=0.0,
                scheduler_mode="random",
            )
        except TypeError:
            pass

        try:
            return agent.select_action(
                obs_i,
                deterministic=True,
                scheduler_mode="random",
            )
        except TypeError:
            pass

        try:
            return agent.select_action(
                obs_i,
                scheduler_mode="random",
            )
        except TypeError:
            pass

    try:
        return agent.select_action(
            obs_i,
            rb_idx=0,
            deterministic=True,
            evaluate=True,
            current_min_private_power_fraction=0.0,
        )
    except TypeError:
        pass

    try:
        return agent.select_action(
            obs_i,
            rb_idx=0,
            deterministic=True,
            current_min_private_power_fraction=0.0,
        )
    except TypeError:
        pass

    try:
        return agent.select_action(
            obs_i,
            rb_idx=0,
            evaluate=True,
            current_min_private_power_fraction=0.0,
        )
    except TypeError:
        pass

    try:
        return agent.select_action(
            obs_i,
            rb_idx=0,
            current_min_private_power_fraction=0.0,
        )
    except TypeError:
        pass

    try:
        return agent.select_action(
            local_observation=obs_i,
            current_epsilon=0.0,
            current_min_private_power_fraction=0.0,
            deterministic=True,
            deterministic_dqn=True,
            deterministic_power=True,
            deterministic_phase=True,
            deterministic_amp=True,
            scheduler_mode="dqn",
            power_mode="ppo",
        )
    except TypeError:
        pass

    try:
        return agent.select_action(
            local_observation=obs_i,
            current_epsilon=0.0,
            current_min_private_power_fraction=0.0,
            deterministic=True,
            scheduler_mode="dqn",
            power_mode="ppo",
        )
    except TypeError:
        pass

    try:
        return agent.select_action(obs_i, deterministic=True)
    except TypeError:
        pass

    try:
        return agent.select_action(obs_i)
    except TypeError as e:
        raise TypeError(
            f"[eval_utils_PBT_v14] Unsupported agent.select_action signature "
            f"for agent_id={agent_id}, scheduler_mode={scheduler_mode}, power_mode={power_mode}: {e}"
        )


def run_episode_validation_metrics(
    eval_env: LeoSatEnv,
    agents: Dict[str, Any],
    args,
    seed: int,
) -> Dict[str, float]:
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
        actions = {}
        for agent_id in list(getattr(eval_env, "agents", list(agents.keys()))):
            if agent_id not in agents:
                raise KeyError(f"[eval_utils_PBT_v14] Missing agent for agent_id={agent_id}")
            obs_i = obs[agent_id]

            raw_action_info = _select_eval_action_for_agent(
                agent=agents[agent_id],
                obs_i=obs_i,
                agent_id=agent_id,
                args=args,
                eval_env=eval_env,
            )

            if isinstance(raw_action_info, dict) and "env_action" in raw_action_info:
                actions[agent_id] = raw_action_info["env_action"]
            else:
                actions[agent_id] = raw_action_info

        next_obs, rewards, dones, infos = eval_env.step(actions)

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

        obs = next_obs
        n_steps += 1

        if bool(dones.get("__all__", False)):
            break

    T = max(1, int(n_steps))
    if sum_rate_per_user_over_steps is None:
        avg_rates = np.zeros(_infer_num_users_from_env(eval_env), dtype=np.float64)
        avg_sum_rate = 0.0
    else:
        avg_rates = (sum_rate_per_user_over_steps / float(T)).reshape(-1)
        avg_sum_rate = float(sum_rate_total_over_steps / float(T))

    min_user_avg_rate = float(np.min(avg_rates)) if avg_rates.size > 0 else 0.0
    jfi = _jain_index(avg_rates)

    return {
        "undiscounted_return": float(episode_return),
        "avg_sum_rate": float(avg_sum_rate),
        "min_user_avg_rate": float(min_user_avg_rate),
        "jain_index": float(jfi),
        "n_steps": int(T),
    }


def _set_agents_mode(agents: Dict[str, Any], mode: str = "eval"):
    network_names = ["q_network", "power_actor", "power_critic", "phase_actor", "phase_critic"]
    for agent in agents.values():
        for net_name in network_names:
            if hasattr(agent, net_name):
                net = getattr(agent, net_name)
                if net is not None and hasattr(net, mode):
                    getattr(net, mode)()


def run_multi_seed_validation_metrics(
    base_env_config: Dict[str, Any],
    agents: Dict[str, Any],
    args,
    seed_list: Sequence[int],
    *,
    profile: str = "phase1",
    mode: str = "standard",
    force_fixed_positions: bool = True,
) -> Dict[str, Any]:
    seed_list = list(map(int, list(seed_list) if seed_list is not None else []))

    eval_cfg = get_eval_config(
        base_config=base_env_config,
        mode=mode,
        profile=profile,
        force_fixed_positions=force_fixed_positions,
    )

    metrics_all = []

    _set_agents_mode(agents, mode="eval")

    try:
        with preserve_rng_state(), torch.no_grad():
            for seed in seed_list:
                eval_env = make_eval_env(eval_cfg, seed=int(seed))
                m = run_episode_validation_metrics(
                    eval_env=eval_env,
                    agents=agents,
                    args=args,
                    seed=int(seed),
                )
                metrics_all.append(m)
    finally:
        _set_agents_mode(agents, mode="train")

    summary = summarize_metrics_by_seed(
        metrics_all=metrics_all,
        metric_keys=DEFAULT_METRIC_KEYS,
    )

    return {
        "seed_list": seed_list,
        "metrics": summary,
    }


def build_eval_curve_record(
    episodes_done: int,
    metrics_pack: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "episodes_done": int(episodes_done),
        "metrics": copy.deepcopy(metrics_pack.get("metrics", {})),
        "seed_list": list(metrics_pack.get("seed_list", [])),
    }


class preserve_rng_state:
    def __enter__(self):
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)
