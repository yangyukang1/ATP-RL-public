"""Evaluate the single-stage No-RIS policy selected by validation return."""

from pathlib import Path
import argparse
import copy
import hashlib
import itertools
import math

import torch

from atprl.training.recovery import load_warmup_recovery, atomic_json_save
from atprl.training import warmup as w
from atprl.evaluation.utils import summarize_values


def build_evaluation(payload):
    import torch
    from types import SimpleNamespace
    from atprl.training import warmup as w
    from atprl.nn.models import QNetwork

    m = payload["no_ris_runtime"]
    dims = m["dims"]
    a = SimpleNamespace(**m["args"])
    device = torch.device("cpu")
    assert (
        a.disable_ris_path and a.phase1_only and a.scheduler_mode == "dqn" and a.power_mode == "ppo"
    )
    env = w.LeoSatEnv(config=copy.deepcopy(m["env_config"]))
    assert env.disable_ris_path and not env.active_ris_enabled
    q = QNetwork(
        state_dim=dims["dqn_state_dim"], action_dim=math.comb(dims["n_users"], dims["k_users"])
    ).to(device)
    actor = w.ActorNetwork(
        state_dim=dims["power_state_dim"],
        action_dim=dims["power_action_dim"],
        max_power_watt=env.satellite_max_power_watt,
    ).to(device)
    critic = w.CriticNetwork(state_dim=dims["power_state_dim"]).to(device)
    phase = w.PhaseActorGaussianLatent(
        state_dim=dims["phase_state_dim"],
        num_phase_groups=dims["phase_groups"],
        learn_amp=w._resolve_learn_amp_weights(a),
    ).to(device)
    pc = w.PhaseCritic(state_dim=dims["phase_state_dim"]).to(device)
    policy = payload["no_ris_best_policy"]
    assert policy and policy["episode"] > 0
    for name, net in [("q_network", q), ("power_actor", actor), ("power_critic", critic)]:
        net.load_state_dict(policy["models"][name], strict=True)
    phase.load_state_dict(payload["models"]["phase_actor"], strict=True)
    pc.load_state_dict(payload["models"]["phase_critic"], strict=True)
    agents = {
        aid: w.DPRA_Agent(
            q_network=q,
            power_actor=actor,
            power_critic=critic,
            phase_actor=phase,
            phase_critic=pc,
            discrete_actions_map={
                i: c
                for i, c in enumerate(
                    itertools.combinations(range(dims["n_users"]), dims["k_users"])
                )
            },
            n_users=dims["n_users"],
            dqn_state_dim=dims["dqn_state_dim"],
            max_power_watt=env.satellite_max_power_watt,
            device=device,
            include_sr_incident_pow_groups=bool(a.include_sr_incident_pow_groups),
            learn_amp_weights=w._resolve_learn_amp_weights(a),
            ris_num_groups=dims["phase_groups"],
            phase_state_mode=a.phase_model_state_mode,
            phase_obs_layout=m["phase_obs_layout"],
            phase_compact_include_mrt_features=bool(a.phase_model_compact_include_mrt_features),
            phase_policy_dist=a.phase_policy_dist,
            dqn_scheduler_impl="enumerative",
            k_users_per_rb=dims["k_users"],
        )
        for aid in env.agents
    }
    getattr(env, "close", lambda: None)()
    return agents, m["eval_env_config"], a


def evaluate(checkpoint, output, seeds=None):
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    payload = load_warmup_recovery(Path(checkpoint), map_location="cpu")
    if not payload["no_ris_runtime"]["args"]["disable_ris_path"]:
        raise ValueError("Expected a No-RIS checkpoint")
    seeds = list(seeds) if seeds is not None else [10001 + 37 * i for i in range(500)]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be nonempty and unique")
    agents, envcfg, args = build_evaluation(payload)
    w._set_agents_eval_mode(agents)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = payload["no_ris_best_policy"]
    digest = hashlib.sha256()
    for name, state in sorted(policy["models"].items()):
        for key, tensor in sorted(state.items()):
            digest.update((name + "/" + key).encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    identity = dict(
        training_seed=args.seed,
        selected_episode=policy["episode"],
        selected_tensor_sha256=digest.hexdigest(),
        environment=envcfg,
        seed_list=seeds,
    )
    import json

    state_file = output / "evaluation_state.json"
    state = (
        json.loads(state_file.read_text())
        if state_file.exists()
        else dict(identity=identity, records=[])
    )
    if state["identity"] != identity:
        raise ValueError("Evaluation identity changed; use a different output directory")
    records = state["records"]
    if [r["environment_seed"] for r in records] != seeds[: len(records)]:
        raise ValueError("Invalid committed evaluation seed prefix")
    with w.preserve_rng_state_local(), torch.no_grad():
        for seed in seeds[len(records) :]:
            metrics = w._run_single_seed_validation_phase1_random_episode(
                envcfg, agents, args, seed
            )
            records.append(dict(environment_seed=seed, **metrics))
            atomic_json_save(dict(identity=identity, records=records), state_file)
    pack = dict(
        **identity,
        metrics={key: summarize_values([r[key] for r in records]) for key in w.DEFAULT_METRIC_KEYS},
    )
    atomic_json_save(pack, output / "eval_metrics_pack.json")
    atomic_json_save(dict(records=records), output / "per_seed_metrics.json")
    return pack


def main():
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default=".")
    p.add_argument("--num_eval_seeds", type=int, default=500)
    args = p.parse_args()
    evaluate(args.checkpoint, args.output_dir, [10001 + 37 * i for i in range(args.num_eval_seeds)])


if __name__ == "__main__":
    main()
