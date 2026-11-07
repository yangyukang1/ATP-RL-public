from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import numpy as np

from atprl.envs.leo_sat import LeoSatEnv
from atprl.baselines.rr_ao import RRAOPolicy
from atprl.evaluation.conventional_config import (
    DEFAULTS,
    make_args,
    solver_options,
    build_env_config,
    seed_matrix,
    resolve_method_phase,
)
from atprl.evaluation.conventional_metrics import EpisodeMetricAccumulator, DiagnosticAccumulator
from atprl.evaluation.utils import DEFAULT_METRIC_KEYS, summarize_metrics_by_seed
from atprl.training.recovery import _atomic_write_bytes
from atprl.solvers.optimization_context import FIELD_SOURCES


def write_json(path, value):
    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    _atomic_write_bytes(Path(path), lambda handle: handle.write(encoded))


def write_csv(path, rows):
    output = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write_bytes(Path(path), lambda handle: handle.write(output.getvalue().encode("utf-8")))


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class SicCountingEnv(LeoSatEnv):
    def reset(self, *args, **kwargs):
        self.sic_trials = 0
        self.sic_failures = 0
        return super().reset(*args, **kwargs)

    def _calculate_rates_for_rb(self, sinr_results):
        result = super()._calculate_rates_for_rb(sinr_results)
        self.sic_trials += 1
        self.sic_failures += int(not result["sic_success"])
        return result


class ConventionalExperiment:
    def __init__(self, args, output, on_event=None):
        self.sdr_backend = None
        if args.phase_solver == "sdr":
            from atprl.solvers.sdr_phase import backend_identity

            self.sdr_backend = backend_identity()
        self.args = args
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.options = solver_options(args)
        self.seeds = seed_matrix(args)
        self.flat_seeds = [s for repeat in self.seeds for s in repeat]
        self.policy = RRAOPolicy(
            args.num_ground_stations,
            args.num_gs_per_rb,
            self.options,
            args.phase_solver,
            args.rr_initial_offset,
        )
        self.seed_index = 0
        self.env = None
        self.metrics = None
        self.diagnostics = None
        self.completed = []
        self.completed_diagnostics = []
        self.safe_boundary = True
        self.on_event = on_event
        self.execution_state = "in_progress"
        self.debug_saved = 0
        self.stream = self.output / "per_seed_metrics.jsonl"
        self.diagnostic_stream = self.output / "solver_diagnostics.jsonl"
        if args.resume_run_dir:
            from atprl.evaluation.conventional_recovery import restore_experiment

            restore_experiment(self)
            return
        if self.stream.exists() and self.stream.stat().st_size:
            raise ValueError("Existing result stream requires explicit resume")
        self.export_metadata()

    def event(self, name):
        if self.on_event:
            self.on_event(name, self)

    def physical_config(self):
        value = build_env_config(self.args, self.flat_seeds[0])
        value.pop("random_seed")
        return value

    def scheduling_policy(self):
        return dict(
            kind=self.policy.scheduler.SCHEDULE_KIND,
            combination_order="lexicographic",
            cycle_length=self.policy.scheduler.cycle_length,
            initial_combination_offset=self.args.rr_initial_offset,
            reset_each_episode=self.args.rr_reset_each_episode,
        )

    def phase_metadata(self):
        if self.sdr_backend is None:
            return {}
        options = asdict(self.options)
        return dict(
            sdr_backend=self.sdr_backend,
            actual_sdp_backend="SCS",
            sdr_algorithm_protocol=self.sdr_backend["sdr_algorithm_protocol"],
            phase1_solver="sca",
            effective_phase_options={
                key: value for key, value in options.items() if key.startswith("sdr_")
            },
            effective_phase1_options={
                key: options[key]
                for key in (
                    "phase1_max_iters",
                    "sca_trust_radius",
                    "sca_scipy_maxiter",
                    "sca_scipy_ftol",
                    "sic_guard",
                    "feasibility_tol",
                )
            },
        )

    def export_metadata(self):
        physical = self.physical_config()
        self.scenario_id = (
            f"N{self.args.num_ground_stations}K{self.args.num_gs_per_rb}_{identity(physical)[:12]}"
        )
        write_json(
            self.output / "eval_meta.json",
            dict(
                method=self.args.baseline_name,
                method_type="non_learning",
                has_training_trajectory=False,
                phase_solver=self.args.phase_solver,
                ris_init="mrt",
                scheduling_policy=self.scheduling_policy(),
                scenario_id=self.scenario_id,
                physical_protocol_hash=identity(physical),
                seed_matrix_hash=identity(self.seeds),
                generated_seed_list=self.flat_seeds,
                seed_matrix=self.seeds,
                resolved_physical_config=physical,
                solver_options=asdict(self.options),
                resolved_args=vars(self.args),
                rate_unit="bps/Hz",
                statistic_axis="environment_seed",
                std_ddof=0,
                imperfect_csi_audit=FIELD_SOURCES,
                bs_interference_information="public_geometry_conditional_mean_no_instantaneous_true_CSI",
                **self.phase_metadata(),
            ),
        )
        write_json(
            self.output / "selection_info.json",
            dict(
                baseline_name=self.args.baseline_name,
                selection_kind="no_training_no_model_selection",
            ),
        )

    def start_episode(self):
        seed = self.flat_seeds[self.seed_index]
        self.env = SicCountingEnv(build_env_config(self.args, seed))
        self.env.reset(seed=seed)
        if self.args.rr_reset_each_episode:
            self.policy.scheduler.reset()
        self.metrics = EpisodeMetricAccumulator(self.args.num_ground_stations)
        self.diagnostics = DiagnosticAccumulator(self.args.diagnostic_sample_limit)
        self.safe_boundary = True
        self.event("episode_start")
        self.checkpoint()

    def checkpoint(self):
        from atprl.evaluation.conventional_recovery import save_experiment

        save_experiment(self)

    def advance(self):
        self.event("before_decision")
        result = self.policy.decide(self.env)
        self.event("after_optimization")
        self.event("before_step")
        self.safe_boundary = False
        _, rewards, dones, infos = self.env.step({"rb_0": result.action})
        self.event("after_step")
        self.metrics.absorb(rewards, infos, self.env)
        self.diagnostics.absorb(result.diagnostics)
        self.policy.commit()
        self.safe_boundary = True
        if self.args.debug_inner_history and self.debug_saved < self.args.debug_decision_limit:
            write_json(self.output / f"debug/decision_{self.debug_saved}.json", result.diagnostics)
            self.debug_saved += 1
        self.event("after_commit")
        if (
            self.args.checkpoint_every_steps
            and self.metrics.steps % self.args.checkpoint_every_steps == 0
        ):
            self.checkpoint()
        return bool(dones["__all__"])

    def finish_episode(self):
        repeat, index = divmod(self.seed_index, self.args.num_eval_seeds)
        row = dict(
            eval_seed=self.flat_seeds[self.seed_index],
            seed_index=index,
            repeat_index=repeat,
            **self.metrics.finish(),
        )
        diag = dict(
            eval_seed=row["eval_seed"],
            **self.diagnostics.finish(),
            environment_sic_trials=self.env.sic_trials,
            environment_sic_failures=self.env.sic_failures,
            environment_micro_outage_probability=self.env.sic_failures
            / max(1, self.env.sic_trials),
        )
        for path, value in ((self.stream, row), (self.diagnostic_stream, diag)):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
        self.completed.append(row)
        self.completed_diagnostics.append(diag)
        self.seed_index += 1
        self.env = None
        self.metrics = None
        self.diagnostics = None
        self.checkpoint()
        write_json(
            self.output / "live_progress.json",
            dict(
                completed_seeds=self.seed_index,
                total_seeds=len(self.flat_seeds),
                scenario_id=self.scenario_id,
            ),
        )
        self.event("seed_boundary")
        if self.args.show_progress:
            print(
                f"[Conventional] {self.args.baseline_name}: {self.seed_index}/{len(self.flat_seeds)} seeds"
            )

    def export_results(self):
        metrics = summarize_metrics_by_seed(self.completed, DEFAULT_METRIC_KEYS)
        physical = self.physical_config()
        common = dict(
            method=self.args.baseline_name,
            baseline_name=self.args.baseline_name,
            method_type="non_learning",
            has_training_trajectory=False,
            phase_solver=self.args.phase_solver,
            ris_init="mrt",
            scenario_id=self.scenario_id,
            scheduling_policy=self.scheduling_policy(),
            num_ground_stations=self.args.num_ground_stations,
            num_gs_per_rb=self.args.num_gs_per_rb,
            topology_profile=self.args.topology_profile,
            fairness_lambda=self.args.fairness_lambda,
            csi_error_variance=self.args.csi_error_variance,
            ris_reflection_power_limit_dBm=self.args.ris_reflection_power_limit_dBm,
            num_eval_seeds=self.args.num_eval_seeds,
            evaluation_repeats=self.args.evaluation_repeats,
            physical_protocol_hash=identity(physical),
            seed_matrix_hash=identity(self.seeds),
            statistic_axis="environment_seed",
        )
        flat = {
            f"{key}_{stat}": metrics[key][stat]
            for key in DEFAULT_METRIC_KEYS
            for stat in ("mean", "std")
        }
        pack = dict(
            **common,
            **flat,
            metrics=metrics,
            seed_list=self.flat_seeds,
            seed_matrix=self.seeds,
            reference_metrics={f"{key}_mean": metrics[key]["mean"] for key in DEFAULT_METRIC_KEYS},
            resolved_physical_config=physical,
            solver_options=asdict(self.options),
            execution_state="complete",
            **self.phase_metadata(),
        )
        write_json(self.output / "eval_metrics_pack.json", pack)
        write_json(self.output / "eval_record.json", dict(**common, metrics=metrics))
        write_json(self.output / "per_seed_metrics.json", dict(per_seed_metrics=self.completed))
        write_csv(self.output / "eval_summary.csv", [dict(**common, **flat)])
        statuses = Counter()
        for row in self.completed_diagnostics:
            statuses.update(row["solver_status_counts"])
        write_json(
            self.output / "solver_diagnostics_summary.json",
            dict(
                method=self.args.baseline_name,
                decisions=sum(d["decisions"] for d in self.completed_diagnostics),
                solver_status_counts=dict(statuses),
                per_seed_diagnostics=self.completed_diagnostics,
            ),
        )
        from atprl.evaluation.paper_data import publish_result_index

        publish_result_index(self.output)
        return pack

    def run(self):
        try:
            while self.seed_index < len(self.flat_seeds):
                if self.env is None:
                    self.start_episode()
                while self.env.current_step < self.args.max_steps_per_episode:
                    if self.advance():
                        break
                self.event("episode_boundary")
                self.finish_episode()
            self.execution_state = "complete"
            pack = self.export_results()
            self.checkpoint()
            return pack
        except BaseException:
            if self.safe_boundary:
                self.checkpoint()
            raise


def build_parser():
    parser = argparse.ArgumentParser(description="Round-robin AO WMMSE SDR evaluation")
    for name, default in DEFAULTS.items():
        kwargs = dict(default=default)
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        elif name in ("decision_time_limit_s", "force_bs_gain_db"):
            kwargs["type"] = float
        elif default is None:
            kwargs["type"] = str
        else:
            kwargs["type"] = type(default)
        if name in ("baseline_name", "phase_solver"):
            kwargs["default"] = argparse.SUPPRESS
        parser.add_argument("--" + name, **kwargs)
    return parser


def evaluate_from_runner(args):
    overrides = {
        name: getattr(args, name)
        for name in DEFAULTS
        if hasattr(args, name) and getattr(args, name) is not None
    }
    explicit = getattr(args, "_explicit_conventional_identity", None)
    if explicit is not None:
        for key in ("baseline_name", "phase_solver"):
            if key not in explicit:
                overrides.pop(key, None)
    name, phase = resolve_method_phase(
        overrides.get("baseline_name"), overrides.get("phase_solver")
    )
    overrides.update(baseline_name=name, phase_solver=phase, device="cpu")
    options = make_args(**overrides)
    pack = ConventionalExperiment(options, options.resume_run_dir or options.output_dir).run()
    return {key: value for key, value in pack.items() if not isinstance(value, (dict, list))}


def main():
    from atprl.runtime import configure_cpu_threads

    configure_cpu_threads()
    args = make_args(**vars(build_parser().parse_args()))
    ConventionalExperiment(args, args.resume_run_dir or args.output_dir).run()


if __name__ == "__main__":
    main()
