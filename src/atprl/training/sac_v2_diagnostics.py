import copy, io, time
import numpy as np
from atprl.training.sac_v2_io import atomic_bytes, descriptor, StoragePressure


def flatten(row):
    row = dict(row)
    if "q1_mean" in row and "q2_mean" in row:
        row["derived_q_mean_gap"] = abs(row["q1_mean"] - row["q2_mean"])
    for name in ["q1", "q2"]:
        if name + "_gradient_norm" in row:
            row["derived_" + name + "_clip_scale"] = min(
                1.0, 0.5 / (row[name + "_gradient_norm"] + 1e-6)
            )
    result = {}
    for k, v in sorted(row.items()):
        if isinstance(v, str):
            continue
        if isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                result[f"{k}[{i}]"] = float(x)
        elif v is not None:
            result[k] = float(v)
    return result


class Diagnostics:
    def __init__(self, state=None):
        self.state = copy.deepcopy(
            state
            or {
                "columns": [],
                "count": 0,
                "mean": [],
                "m2": [],
                "min": [],
                "max": [],
                "last": [],
                "consumed_update": 0,
                "clip_counts": {},
                "cross": {},
                "summary_refs": [],
                "raw_refs": [],
                "scope_start_update": 1,
            }
        )
        self.pending_raw = []
        self.pending_episodes = []
        self.episode_rows = []
        self.episode = 0

    def begin(self, episode):
        self.episode = int(episode)
        self.episode_rows = []

    def consume(self, row):
        flat = flatten(row)
        s = self.state
        columns = list(flat)
        if not s["columns"]:
            s["columns"] = columns
        if columns != s["columns"]:
            raise ValueError("Diagnostic numeric schema changed")
        v = np.asarray(list(flat.values()), np.float64)
        if len(v) > 64:
            raise StoragePressure("Diagnostic schema exceeds validated 64-column budget")
        if not np.isfinite(v).all():
            raise FloatingPointError("Nonfinite diagnostic record")
        u = int(row["updates"])
        if u != s["consumed_update"] + 1:
            raise ValueError("Repeated or missing diagnostic update")
        s["consumed_update"] = u
        n = s["count"] + 1
        if n == 1:
            mean = v.copy()
            m2 = np.zeros_like(v)
            mn = v.copy()
            mx = v.copy()
        else:
            mean = np.asarray(s["mean"])
            delta = v - mean
            mean = mean + delta / n
            m2 = np.asarray(s["m2"]) + delta * (v - mean)
            mn = np.minimum(s["min"], v)
            mx = np.maximum(s["max"], v)
        s.update(
            count=n,
            mean=mean.tolist(),
            m2=m2.tolist(),
            min=mn.tolist(),
            max=mx.tolist(),
            last=v.tolist(),
        )
        for name in ["q1", "q2", "actor"]:
            k = name + "_gradient_norm"
            limit = row.get(name + "_clip_limit", 0.5)
            if k in row:
                s["clip_counts"][name] = s["clip_counts"].get(name, 0) + int(row[k] > limit)
        if "q1_gradient_norm" in row and "q1_parameter_delta_l2" in row:
            x = float(row["q1_gradient_norm"])
            y = float(row["q1_parameter_delta_l2"])
            a = s["cross"].setdefault("q1_gradient_parameter_delta", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for i, z in enumerate([1, x, y, x * x, y * y, x * y]):
                a[i] += z
        self.pending_raw.append((self.episode, v))
        self.episode_rows.append(v)

    def finish(self):
        d = len(self.state["columns"])
        a = np.asarray(self.episode_rows, np.float64).reshape(-1, d) if d else np.empty((0, 0))
        if len(a):
            stats = np.stack([a.mean(0), ((a - a.mean(0)) ** 2).sum(0), a.min(0), a.max(0), a[-1]])
        else:
            stats = np.zeros((5, d))
        clips = np.zeros(3, np.int64)
        cross = np.zeros(6, np.float64)
        columns = self.state["columns"]
        if len(a):
            for i, name in enumerate(["q1", "q2", "actor"]):
                key = name + "_gradient_norm"
                if key in columns:
                    clips[i] = np.count_nonzero(a[:, columns.index(key)] > 0.5)
            if "q1_gradient_norm" in columns and "q1_parameter_delta_l2" in columns:
                x = a[:, columns.index("q1_gradient_norm")]
                y = a[:, columns.index("q1_parameter_delta_l2")]
                cross = np.array(
                    [len(a), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]
                )
        self.pending_episodes.append((self.episode, len(a), stats, clips, cross))

    def preview(self, root, stage, generation, episode):
        """Write pending diagnostic blocks without consuming them before the commit."""
        root = __import__("pathlib").Path(root)
        s = copy.deepcopy(self.state)
        base = root / "diagnostics" / stage
        if self.pending_episodes:
            d = len(s["columns"])
            stats = np.zeros((len(self.pending_episodes), 5, d))
            for i, (_, _, values, _, _) in enumerate(self.pending_episodes):
                if values.shape[1]:
                    stats[i] = values
            b = io.BytesIO()
            np.savez(
                b,
                episode=np.array([x[0] for x in self.pending_episodes], np.int64),
                count=np.array([x[1] for x in self.pending_episodes], np.int64),
                stats=stats,
                clip_counts=np.asarray([x[3] for x in self.pending_episodes]),
                cross=np.asarray([x[4] for x in self.pending_episodes]),
            )
            p = base / "summary" / f"{generation}.npz"
            atomic_bytes(p, b.getvalue())
            s["summary_refs"].append(descriptor(root, p))
        if self.pending_raw and self.pending_raw[-1][0] > episode - 640:
            b = io.BytesIO()
            np.savez(
                b,
                episode=np.array([x[0] for x in self.pending_raw], np.int64),
                values=np.asarray([x[1] for x in self.pending_raw]),
            )
            p = base / "raw" / f"{generation}.npz"
            atomic_bytes(p, b.getvalue())
            ref = descriptor(root, p)
            ref.update(first_episode=self.pending_raw[0][0], last_episode=self.pending_raw[-1][0])
            s["raw_refs"].append(ref)
        s["raw_refs"] = [r for r in s["raw_refs"] if r["last_episode"] > episode - 640]
        return s

    def published(self, state):
        self.state = state
        self.pending_raw = []
        self.pending_episodes = []


class CompactTrajectory:
    def __init__(self):
        self.rows = []

    def observe(
        self, info, action, next_obs, z, reward, *, epsilon=0.0, dqn_loss=None, center=None
    ):
        self.rows.append(
            {
                "selected_users": np.asarray(info["selected_users"], np.int16),
                "epsilon": float(epsilon),
                "dqn_update": dqn_loss is not None,
                "dqn_loss": 0.0 if dqn_loss is None else float(dqn_loss),
            }
        )


class CompactRecorder:
    def __init__(self, tr):
        self.trainer = tr
        self.active = None

    def start(self, mode, revision, replay_before, cleared):
        tr = self.trainer
        c = tr.config
        ep = tr.episodes_done + 1
        self.started_cpu = time.process_time()
        self.started_wall = time.perf_counter()
        self.segment = dict(
            cycle=(ep - 1) // (c.cycle_dqn_episodes + c.cycle_power_episodes) + 1,
            segment=mode,
            dqn_revision_start=revision,
            power_replay_before=replay_before,
            power_replay_clear_event=cleared,
        )
        self.active = CompactTrajectory()

    def finish(self, row):
        rt = self.trainer.runtime
        counts = {
            ",".join(map(str, pair)): int(count)
            for pair, count in zip(
                *np.unique(
                    np.asarray([x["selected_users"] for x in self.active.rows]),
                    axis=0,
                    return_counts=True,
                )
            )
        }
        losses = [x["dqn_loss"] for x in self.active.rows if x["dqn_update"]]
        row.update(
            self.segment,
            dqn_revision=self.trainer.dqn_revision,
            dqn_target_updates=rt.dqn_target_updates,
            dqn_replay_size=len(rt.dqn_replay),
            dqn_epsilon_first=self.active.rows[0]["epsilon"],
            dqn_epsilon_last=self.active.rows[-1]["epsilon"],
            dqn_q_loss_mean=float(np.mean(losses)) if losses else None,
            dqn_updates_this_episode=len(losses),
            selected_user_combination_counts=counts,
            episode_cpu_seconds=time.process_time() - self.started_cpu,
            episode_wall_seconds=time.perf_counter() - self.started_wall,
        )
        self.active = None

    def evaluation_sink(self, seed):
        return None
