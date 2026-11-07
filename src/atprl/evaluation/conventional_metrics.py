from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np
from atprl.evaluation.utils import extract_step_rate_vector_from_infos, _jain_index


class EpisodeMetricAccumulator:
    def __init__(self, n):
        self.n = n
        self.steps = 0
        self.reward_sum = 0.0
        self.rate_sum = np.zeros(n)
        self.sum_rate_sum = 0.0

    def absorb(self, rewards, infos, env):
        rates = np.asarray(extract_step_rate_vector_from_infos(infos, env), dtype=float)
        reward = float(sum(rewards.values())) if isinstance(rewards, dict) else float(rewards)
        if rates.shape != (self.n,) or not np.all(np.isfinite(rates)) or not np.isfinite(reward):
            raise ValueError("Invalid environment metric sequence")
        self.steps += 1
        self.reward_sum += reward
        self.rate_sum += rates
        self.sum_rate_sum += float(rates.sum())

    def finish(self):
        average = self.rate_sum / max(1, self.steps)
        return dict(
            undiscounted_return=self.reward_sum,
            avg_sum_rate=self.sum_rate_sum / max(1, self.steps),
            min_user_avg_rate=float(average.min()),
            jain_index=_jain_index(average),
            n_steps=self.steps,
        )

    def state_dict(self):
        return dict(
            n=self.n,
            steps=self.steps,
            reward_sum=self.reward_sum,
            rate_sum=self.rate_sum.copy(),
            sum_rate_sum=self.sum_rate_sum,
        )

    def load_state_dict(self, state):
        if state["n"] != self.n or state["steps"] < 0:
            raise ValueError("Invalid metric recovery dimensions")
        self.steps = int(state["steps"])
        self.reward_sum = float(state["reward_sum"])
        self.rate_sum = np.asarray(state["rate_sum"], dtype=float).copy()
        self.sum_rate_sum = float(state["sum_rate_sum"])
        if self.rate_sum.shape != (self.n,) or not np.all(np.isfinite(self.rate_sum)):
            raise ValueError("Invalid metric recovery values")


class DiagnosticAccumulator:
    def __init__(self, sample_limit=256):
        self.sample_limit = sample_limit
        self.count = 0
        self.totals = Counter()
        self.statuses = Counter()
        self.minimum_sic = None
        self.minimum_sic_normalized = None
        self.maximum_power_excess = None
        self.samples = []
        self.runtime_total = 0.0
        self.runtime_max = 0.0
        self.sdr_statuses = {}
        self.sdr_floats = {}

    def absorb(self, row):
        self.count += 1
        for key, value in row.items():
            if isinstance(value, int) and not isinstance(value, bool):
                self.totals[key] += value
            if key.startswith("sdr_") and isinstance(value, dict) and key.endswith("_counts"):
                self.sdr_statuses.setdefault(key, Counter()).update(value)
            elif key.startswith("sdr_") and isinstance(value, float):
                if not np.isfinite(value):
                    raise ValueError("Non-finite SDR diagnostic")
                stats = self.sdr_floats.setdefault(
                    key, dict(count=0, sum=0.0, min=value, max=value, last=value)
                )
                stats["count"] += 1
                stats["sum"] += value
                stats["min"] = min(stats["min"], value)
                stats["max"] = max(stats["max"], value)
                stats["last"] = value
        self.statuses[row["solver_status"]] += 1
        value = float(row["minimum_sic_margin_watt"])
        self.minimum_sic = value if self.minimum_sic is None else min(self.minimum_sic, value)
        value = float(row["minimum_sic_margin_normalized"])
        self.minimum_sic_normalized = (
            value
            if self.minimum_sic_normalized is None
            else min(self.minimum_sic_normalized, value)
        )
        value = float(row["power_excess_watt"])
        self.maximum_power_excess = (
            value if self.maximum_power_excess is None else max(self.maximum_power_excess, value)
        )
        runtime = float(row["optimization_wall_time_s"])
        self.runtime_total += runtime
        self.runtime_max = max(self.runtime_max, runtime)
        if len(self.samples) < self.sample_limit:
            self.samples.append(runtime)
        else:
            j = (
                int.from_bytes(
                    hashlib.blake2b(str(self.count).encode(), digest_size=8).digest(), "little"
                )
                % self.count
            )
            if j < self.sample_limit:
                self.samples[j] = runtime

    def finish(self):
        result = dict(
            decisions=self.count,
            **dict(self.totals),
            solver_status_counts=dict(self.statuses),
            minimum_sic_margin_watt=self.minimum_sic,
            minimum_sic_margin_normalized=self.minimum_sic_normalized,
            maximum_power_excess_watt=self.maximum_power_excess,
            optimization_wall_time_s=self.runtime_total,
            runtime_mean_s=self.runtime_total / max(1, self.count),
            runtime_max_s=self.runtime_max,
            runtime_median_s=float(np.median(self.samples)) if self.samples else None,
            runtime_p95_s=float(np.percentile(self.samples, 95)) if self.samples else None,
            runtime_quantile_kind="bounded_deterministic_sample",
            runtime_sample_count=len(self.samples),
        )
        if self.sdr_statuses or self.sdr_floats:
            result.update({key: dict(value) for key, value in self.sdr_statuses.items()})
            result["sdr_float_statistics"] = {
                key: dict(value, mean=value["sum"] / value["count"])
                for key, value in self.sdr_floats.items()
            }
        return result
