import copy, hashlib, json, os, shutil, time, uuid, contextlib
from pathlib import Path
import torch
from atprl.config.sac import SACRunConfig
from atprl.training.sac_recovery import (
    digest,
    full_state,
    source_signature,
    source_version,
    scientific_config,
    kernel_bundle,
    physics,
)
from atprl.training.recovery import load_replay_buffer_state, restore_rng_state
from atprl.training.sac_v2_io import (
    sha,
    lock,
    atomic_bytes,
    atomic_json,
    torch_bytes,
    read_json,
    verify,
    descriptor,
    prefix_sha,
    append_json,
    bounded_event,
    sync_dir,
    durable_mkdir,
    StoragePressure,
)
from atprl.training.sac_v2_diagnostics import Diagnostics, CompactRecorder

LEGACY_SOURCES = {"f6bdbcc82de0d079ffcc0c852eb869ef5e8d22a98c42d646cfeaf41b7a69fb7e"}
COMPATIBLE_V2_SOURCES = {
    "ab86f1720a22f4adfca6c5f0c8e84a44370c7ec6f960c5d8ac999fcfa5d328c0": "6c1f082122485aee68a159c01f3ada6191499d8e"
}
FROZEN = {
    "q": "q_network",
    "target_q": "target_q_network",
    "q_optimizer": "q_optimizer",
    "dqn_replay": "dqn_replay",
    "power": "power",
}


def validate_identity(s, job=None):
    cfg = SACRunConfig(**s["config"])
    if digest(scientific_config(cfg)) != s["config_identity"]:
        raise ValueError("Scientific configuration identity mismatch")
    if job is not None:
        if (
            cfg.seed != job["seed"]
            or cfg.seeds != [job["seed"]]
            or cfg.fairness_lambda != job["lambda"]
        ):
            raise ValueError("Task seed/lambda mismatch")
        if "config" in job:
            expected = dict(job["config"])
            if s["stage"] == "warmup":
                expected.update(
                    stage="warmup",
                    phase_distribution=None,
                    kernel_checkpoint=None,
                    kernel_checkpoint_sha256=None,
                )
            a = scientific_config(SACRunConfig(**expected))
            b = scientific_config(cfg)
            a.pop("kernel_checkpoint_sha256", None)
            b.pop("kernel_checkpoint_sha256", None)
            if digest(a) != digest(b):
                raise ValueError("Task scientific configuration differs from immutable manifest")
    limit = cfg.stage1_episodes if s["stage"] == "warmup" else cfg.stage2_episodes
    if not 0 <= s["progress"]["episodes_done"] <= limit:
        raise ValueError("Recovery progress exceeds scientific budget")
    if s["stage"] == "stage2":
        k = s["kernel"]
        if k["training_seed"] != cfg.seed or k["physics"]["fairness_lambda"] != cfg.fairness_lambda:
            raise ValueError("Kernel seed/lambda mismatch")
        if cfg.phase_distribution == "wrapped_normal_v1":
            if (
                s["phase"].get("phase_distribution") != "wrapped_normal_v1"
                or s["phase"].get("density_version") != "centered_logsumexp_k5_float64_v1"
            ):
                raise ValueError("WN density identity mismatch")
    return cfg


def rebuild(s, job=None):
    from atprl.training.sac_stp import SACWarmupTrainer, SACPhase2Trainer

    cfg = validate_identity(s, job)
    tr = SACWarmupTrainer(cfg) if s["stage"] == "warmup" else SACPhase2Trainer(cfg, s["kernel"])
    rt = tr.runtime
    rt.q_network.load_state_dict(s["q"])
    rt.target_q_network.load_state_dict(s["target_q"])
    rt.q_optimizer.load_state_dict(s["q_optimizer"])
    load_replay_buffer_state(rt.dqn_replay, s["dqn_replay"])
    rt.power.load_state_dict(s["power"])
    if s["phase"] is not None:
        rt.phase.load_state_dict(s["phase"])
    rt.dqn_updates = s["dqn_updates"]
    rt.dqn_target_updates = s["dqn_target_updates"]
    rt.env = s["environment"]
    for k, v in s["progress"].items():
        setattr(tr, k, copy.deepcopy(v))
    tr.selection = copy.deepcopy(s["selection"])
    restore_rng_state(s["rng"], env=rt.env)
    return tr


def thin_policy(bundle):
    """Export inference networks without replay buffers or resumable optimizer state."""
    is_phase = bundle["kind"] == "sac_stp_policy"
    k = bundle["kernel"] if is_phase else bundle

    def controller(s):
        return {
            key: copy.deepcopy(s[key])
            for key in [
                "version",
                "algorithm",
                "reference",
                "config",
                "state_dim",
                "freedom",
                "phase",
                "models",
                "log_temperature",
                "phase_distribution",
                "density_version",
            ]
            if key in s
        }

    return {
        "kind": "sac_stp_policy_v2",
        "format_version": 2,
        "algorithm": "sac_stp",
        "config": copy.deepcopy(bundle["config"]),
        "training_seed": bundle["training_seed"],
        "episode": bundle["episode"],
        "artifact_identity": bundle["artifact_identity"],
        "stage": "stage2" if is_phase else "warmup",
        "physics": copy.deepcopy(k["physics"]),
        "q_network": copy.deepcopy(k["q_network"]),
        "target_q_network": copy.deepcopy(k["target_q_network"]),
        "power": controller(k["power"]),
        "phase": controller(bundle["phase"]) if is_phase else None,
        "source_signature": bundle["source_signature"],
        "source_version": bundle["source_version"],
        "export_source_signature": source_signature(),
        "export_source_version": source_version(),
        "legacy_model_digest": digest(
            {
                "q": k["q_network"],
                "power": k["power"]["models"],
                "phase": bundle["phase"]["models"] if is_phase else None,
            }
        ),
    }


def policy_runtime(p):
    from atprl.training.sac_runtime import SACSTPRuntime

    if p.get("kind") != "sac_stp_policy_v2" or p.get("format_version") != 2:
        raise ValueError("Not an inference v2 policy")
    cfg = SACRunConfig(**p["config"])
    rt = SACSTPRuntime(cfg, p["stage"])
    if cfg.seed != p["training_seed"]:
        raise ValueError("Policy seed mismatch")
    if p["physics"] != physics(cfg):
        raise ValueError("Policy physics mismatch")
    rt.q_network.load_state_dict(p["q_network"])
    rt.target_q_network.load_state_dict(p["target_q_network"])
    for obj, key in [(rt.power, "power"), (rt.phase, "phase")]:
        if obj is None:
            continue
        s = p[key]
        expected = {
            "version": 2 if obj.phase_distribution == "wrapped_normal_v1" else 1,
            "algorithm": "sac_stp",
            "reference": "uniform_simplex_torus_v1",
            "config": obj.config.to_dict(),
            "state_dim": obj.state_dim,
            "freedom": obj.freedom,
            "phase": obj.phase,
        }
        if any(s.get(k) != v for k, v in expected.items()):
            raise ValueError("Policy controller identity mismatch")
        if s.get("phase_distribution", "tanh_normal_v1") != obj.phase_distribution:
            raise ValueError("Policy distribution mismatch")
        if (
            obj.phase_distribution == "wrapped_normal_v1"
            and s.get("density_version") != obj.state_dict()["density_version"]
        ):
            raise ValueError("Policy density mismatch")
        for name in obj.model_names:
            getattr(obj, name).load_state_dict(s["models"][name], strict=True)
        with torch.no_grad():
            obj.log_temperature.copy_(s["log_temperature"])
    rt.freeze_kernel()
    return rt


def state_identity(s):
    return digest(
        {
            k: s[k]
            for k in [
                "q",
                "target_q",
                "q_optimizer",
                "dqn_replay",
                "dqn_updates",
                "dqn_target_updates",
                "power",
                "phase",
                "rng",
            ]
        }
    )


class Store:
    def __init__(self, root):
        self.root = Path(root)
        durable_mkdir(self.root)
        self.current_path = self.root / "CURRENT.json"
        self.lock_path = self.root / "commit.lock"
        self.manifest = None
        self.diagnostics = None

    def put_object(self, value, role):
        ident = digest(value)
        p = self.root / "objects" / f"{role}_{ident}.ckpt"
        if not p.exists():
            data = torch_bytes(value)
            limit = (160 if role == "kernel" else 2) * 2**20
            if len(data) > limit:
                raise StoragePressure(f"{role} exceeds validated object budget")
            atomic_bytes(p, data)
        elif digest(torch.load(p, map_location="cpu", weights_only=False)) != ident:
            raise ValueError("Corrupt immutable object")
        return descriptor(self.root, p)

    def _decode(self, m):
        cp = verify(self.root, m["checkpoint"])
        s = torch.load(cp, map_location="cpu", weights_only=False)
        if s.get("kind") != "sac_stp_recovery_v2" or s.get("format_version") != 2:
            raise ValueError("Not a full v2 recovery")
        if s["source_signature"] != source_signature():
            expected_commit = COMPATIBLE_V2_SOURCES.get(s["source_signature"])
            if (
                expected_commit is None
                or s.get("source_version", {}).get("commit") != expected_commit
            ):
                raise ValueError("v2 source signature mismatch")
        if s.get("kernel_ref"):
            s["kernel"] = torch.load(
                verify(self.root, s.pop("kernel_ref")), map_location="cpu", weights_only=False
            )
            for k, v in s.pop("frozen_from_kernel", {}).items():
                s[k] = copy.deepcopy(s["kernel"][v])
            bound = m.get("extra", {}).get("stage1_complete", {})
            if (
                bound.get("kernel_state_digest")
                and digest(s["kernel"]) != bound["kernel_state_digest"]
            ):
                raise ValueError("Frozen kernel origin changed")
        else:
            s["kernel"] = None
        stage = "stage1" if s["stage"] == "warmup" else "stage2"
        for ref in m.get("objects", []):
            verify(self.root, ref)
        for refs in m.get("evaluations", {}).values():
            for ref in refs:
                verify(self.root, ref)
        histories = {}
        for name, ref in m["journals"].items():
            p = self.root / ref["path"]
            if prefix_sha(p, ref["bytes"]) != ref["sha256"]:
                raise ValueError("Committed journal checksum mismatch")
            with p.open("rb") as f:
                rows = [json.loads(x) for x in f.read(ref["bytes"]).splitlines()]
            if len(rows) != ref["rows"]:
                raise ValueError("Committed journal row mismatch")
            if [r["episode"] for r in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("Journal episode order")
            histories[name] = rows
        s["progress"]["history"] = histories.get(stage, [])
        if len(s["progress"]["history"]) != s["progress"]["episodes_done"]:
            raise ValueError("Recovery/history progress mismatch")
        for ds in m.get("diagnostics_by_stage", {}).values():
            for ref in ds["summary_refs"] + ds["raw_refs"]:
                verify(self.root, ref)
        validate_identity(s)
        return s

    def load(self, *, truncate=False, job=None):
        guard = (
            lock(self.root / "worker.lock", blocking=False)
            if truncate
            else contextlib.nullcontext()
        )
        with guard, lock(self.lock_path):
            pointer = read_json(self.current_path)
            errors = []
            for ref in [pointer["current"], pointer.get("previous")]:
                if not ref:
                    continue
                try:
                    m = read_json(verify(self.root, ref))
                    s = self._decode(m)
                    validate_identity(s, job)
                except Exception as e:
                    errors.append(str(e))
                    continue
                if truncate:
                    for j in m["journals"].values():
                        with (self.root / j["path"]).open("r+b") as f:
                            f.truncate(j["bytes"])
                            f.flush()
                            os.fsync(f.fileno())
                    if ref != pointer["current"]:
                        atomic_json(
                            self.current_path,
                            {
                                "format_version": 2,
                                "current": ref,
                                "previous": None,
                                "recovered_from": errors,
                            },
                        )
                        bounded_event(
                            self.root / "recovery_events.jsonl",
                            {
                                "time": time.time(),
                                "generation": m["generation"],
                                "rollback_errors": errors,
                            },
                        )
                self.manifest = m
                self.diagnostics = Diagnostics(m["diagnostics_by_stage"].get(s["stage"], None))
                return s, m
            raise ValueError("No complete generation: " + "; ".join(errors))

    def restore(self, job=None):
        with lock(self.root / "worker.lock", blocking=False):
            s, m = self.load(truncate=True, job=job)
            tr = rebuild(s, job)
            return tr, m

    def publish(self, tr, diagnostics, *, extra=None, legacy_state=None):
        with lock(self.root / "worker.lock", blocking=False), lock(self.lock_path):
            old = read_json(self.current_path) if self.current_path.exists() else None
            previous_m = read_json(verify(self.root, old["current"])) if old else None
            generation = f"g{int(time.time_ns()):020d}_{uuid.uuid4().hex[:8]}"
            folder = self.root / "generations" / generation
            durable_mkdir(folder)
            self._gc_unpublished(old, keep=generation)
            s = copy.deepcopy(legacy_state) if legacy_state is not None else full_state(tr)
            stage = s["stage"]
            s["kind"] = "sac_stp_recovery_v2"
            s["format_version"] = 2
            s["source_signature"] = source_signature()
            s["source_version"] = source_version()
            s["progress"].pop("history", None)
            s.pop("journal_bytes", None)
            s.pop("learner_bytes", None)
            s.pop("journal_rows", None)
            objects = []
            if s.get("kernel") is not None:
                k = s.pop("kernel")
                ref = self.put_object(k, "kernel")
                s["kernel_ref"] = ref
                objects.append(ref)
                same = {}
                for field, key in FROZEN.items():
                    if digest(s[field]) == digest(k[key]):
                        same[field] = key
                        s.pop(field)
                s["frozen_from_kernel"] = same
            for key in ["policy_ref", "kernel_ref"]:
                ref = s["selection"].get(key)
                if ref:
                    objects.append(ref)
            ep = s["progress"]["episodes_done"]
            ds = diagnostics.preview(self.root, stage, generation, ep)
            diag = (
                copy.deepcopy(previous_m.get("diagnostics_by_stage", {}))
                if previous_m
                else copy.deepcopy((extra or {}).get("initial_diagnostics_by_stage", {}))
            )
            diag[stage] = ds
            cumulative = ep + (tr.config.stage1_episodes if stage == "stage2" else 0)
            for name, values in diag.items():
                offset = tr.config.stage1_episodes if name == "stage2" else 0
                values["raw_refs"] = [
                    r for r in values["raw_refs"] if r["last_episode"] + offset > cumulative - 640
                ]
            journals = copy.deepcopy(previous_m.get("journals", {})) if previous_m else {}
            prior_journal = self.root / "stage1/training.jsonl"
            if stage == "stage2" and "stage1" not in journals and prior_journal.exists():
                journals["stage1"] = dict(
                    descriptor(self.root, prior_journal), rows=tr.config.stage1_episodes
                )
            stage_dir = "stage1" if stage == "warmup" else "stage2"
            jp = self.root / stage_dir / "training.jsonl"
            if not jp.exists():
                jp.parent.mkdir(exist_ok=True)
                jp.touch()
            with jp.open("rb") as f:
                os.fsync(f.fileno())
            journals[stage_dir] = dict(descriptor(self.root, jp), rows=ep)
            evaluations = {}
            for name, journal in journals.items():
                evaluations[name] = [
                    descriptor(self.root, p)
                    for p in sorted((self.root / name).glob("evaluation_ep*.json"))
                    if int(p.stem.split("ep")[-1]) <= journal["rows"]
                ]
            data = torch_bytes(s)
            max_size = (160 if stage == "warmup" else 96) * 2**20
            if len(data) > max_size:
                raise StoragePressure(
                    f"Recovery exceeds validated stage budget: {len(data)} > {max_size}"
                )
            atomic_bytes(folder / "checkpoint.ckpt", data)
            extra = copy.deepcopy(
                extra if extra is not None else (previous_m or {}).get("extra", {})
            )
            extra.pop("initial_diagnostics_by_stage", None)
            objects.extend(extra.get("persistent_objects", []))
            m = {
                "format_version": 2,
                "generation": generation,
                "stage": stage,
                "episode": ep,
                "cumulative_episode": ep + (tr.config.stage1_episodes if stage == "stage2" else 0),
                "checkpoint": descriptor(self.root, folder / "checkpoint.ckpt"),
                "objects": objects,
                "journals": journals,
                "diagnostics_by_stage": diag,
                "evaluations": evaluations,
                "selection": copy.deepcopy(s["selection"]),
                "extra": extra,
                "source_signature": s["source_signature"],
            }
            atomic_json(folder / "manifest.json", m)
            ref = descriptor(self.root, folder / "manifest.json")
            self._decode(m)
            atomic_json(
                self.current_path,
                {"format_version": 2, "current": ref, "previous": old["current"] if old else None},
            )
            self.manifest = m
            diagnostics.published(ds)
            self.diagnostics = diagnostics
            self.gc_locked()
            return m

    def _gc_unpublished(self, pointer, keep=None):
        retained = {keep}
        for r in [pointer.get("current"), pointer.get("previous")] if pointer else []:
            if r:
                retained.add(Path(r["path"]).parent.name)
        for p in (self.root / "generations").glob("g*"):
            if p.name not in retained:
                shutil.rmtree(p)

    def gc_locked(self):
        pointer = read_json(self.current_path)
        keep = set()
        gens = set()
        for ref in [pointer["current"], pointer.get("previous")]:
            if not ref:
                continue
            m = read_json(verify(self.root, ref))
            gens.add(m["generation"])
            for r in m["objects"]:
                keep.add(r["path"])
            for refs in m.get("evaluations", {}).values():
                keep.update(r["path"] for r in refs)
            for ds in m["diagnostics_by_stage"].values():
                keep.update(r["path"] for r in ds["raw_refs"] + ds["summary_refs"])
        for p in (self.root / "evaluation").rglob("manifest.json"):
            identity = read_json(p)
            wanted = identity.get("policy_sha256")
            if wanted:
                matches = [f for f in (self.root / "objects").glob("*.ckpt") if sha(f) == wanted]
                if not matches:
                    raise ValueError("Evaluation still references a missing model")
                keep.update(str(f.relative_to(self.root)) for f in matches)
        self._gc_unpublished(pointer)
        for parent in ["objects", "diagnostics"]:
            for p in (self.root / parent).rglob("*"):
                if p.is_file() and str(p.relative_to(self.root)) not in keep:
                    p.unlink()
        for p in self.root.glob("stage*/evaluation_ep*.json"):
            if str(p.relative_to(self.root)) not in keep:
                p.unlink()

    def gc(self):
        with lock(self.root / "worker.lock", blocking=False), lock(self.lock_path):
            self.gc_locked()


def make_policy(tr, identity):
    from atprl.training.sac_recovery import policy_bundle

    return thin_policy(policy_bundle(tr, identity))


def enter_phase(store, base_config):
    from dataclasses import replace
    from atprl.training.sac_stp import SACPhase2Trainer

    complete = store.manifest["extra"]["stage1_complete"]
    bound = complete["kernel_ref"]
    kernel = torch.load(verify(store.root, bound), map_location="cpu", weights_only=False)
    cfg = replace(
        base_config,
        stage="stage2",
        phase_distribution="wrapped_normal_v1",
        resume=False,
        kernel_checkpoint=str(store.root / bound["path"]),
        kernel_checkpoint_sha256=bound["sha256"],
    )
    if (
        complete["episode"] != cfg.stage1_episodes
        or kernel["training_seed"] != cfg.seed
        or kernel["physics"]["fairness_lambda"] != cfg.fairness_lambda
    ):
        raise ValueError("Stage transition identity mismatch")
    tr = SACPhase2Trainer(cfg, kernel)
    tr.selection = {"best_score": None, "best_episode": None}
    store.publish(tr, Diagnostics(), extra=store.manifest["extra"])
    return tr


def run_stage_v2(tr, store, diagnostics, *, max_episodes=None):
    from atprl.evaluation.sac_stp import evaluate_curve

    cfg = tr.config
    stage = tr.runtime.stage
    sd = "stage1" if stage == "warmup" else "stage2"
    if not hasattr(tr, "selection"):
        tr.selection = {"best_score": None, "best_episode": None}
    if stage == "warmup":
        tr.stage1_recorder = CompactRecorder(tr)
    (tr.runtime.phase or tr.runtime.power).metrics_hook = diagnostics.consume
    budget = cfg.stage1_episodes if stage == "warmup" else cfg.stage2_episodes
    begin = tr.episodes_done
    extra = copy.deepcopy((store.manifest or {}).get("extra", {}))
    while tr.episodes_done < budget:
        if (store.root / "stop.request").exists():
            if tr.episodes_done != (store.manifest or {}).get("episode"):
                store.publish(tr, diagnostics, extra=extra)
            return tr, False
        diagnostics.begin(tr.episodes_done + 1)
        row = tr.train_episode()
        diagnostics.finish()
        append_json(store.root / sd / "training.jsonl", row, 32768 if stage == "warmup" else 8192)
        if tr.episodes_done % cfg.eval_every_episodes == 0 or tr.episodes_done == budget:
            ev = evaluate_curve(tr)
            atomic_json(store.root / sd / f"evaluation_ep{tr.episodes_done}.json", ev)
            score = ev["metrics"]["undiscounted_return"]["mean"]
            if tr.selection.get("best_score") is None or score > tr.selection["best_score"]:
                pref = store.put_object(make_policy(tr, "strict_best"), "policy")
                selection = {
                    "best_score": score,
                    "best_episode": tr.episodes_done,
                    "policy_ref": pref,
                    "best_path": pref["path"],
                    "best_sha256": pref["sha256"],
                }
                if stage == "warmup":
                    from atprl.training.sac_recovery import policy_bundle

                    selection["kernel_ref"] = store.put_object(
                        policy_bundle(tr, "strict_best"), "kernel"
                    )
                tr.selection = selection
        if tr.episodes_done == budget:
            endpoint = store.put_object(make_policy(tr, "endpoint"), "policy")
            extra.setdefault("endpoints", {})[stage] = endpoint
            extra.setdefault("persistent_objects", []).append(endpoint)
            if stage == "warmup":
                k = torch.load(
                    verify(store.root, tr.selection["kernel_ref"]),
                    map_location="cpu",
                    weights_only=False,
                )
                extra["stage1_complete"] = {
                    "episode": budget,
                    "kernel_ref": tr.selection["kernel_ref"],
                    "best": copy.deepcopy(tr.selection),
                    "kernel_state_digest": digest(k),
                }
                extra["persistent_objects"].append(tr.selection["policy_ref"])
        if tr.episodes_done % cfg.checkpoint_every_episodes == 0 or tr.episodes_done == budget:
            store.publish(tr, diagnostics, extra=extra)
        atomic_json(
            store.root / "progress.json",
            {
                "stage": stage,
                "episode": tr.episodes_done,
                "budget": budget,
                "pid": os.getpid(),
                "committed_generation": (store.manifest or {}).get("generation"),
                "live_may_be_uncommitted": True,
            },
        )
        print(
            f"SAC-V2 seed={cfg.seed} stage={stage} episode={tr.episodes_done}/{budget}", flush=True
        )
        if max_episodes is not None and tr.episodes_done - begin >= max_episodes:
            if (store.manifest or {}).get("episode") != tr.episodes_done:
                store.publish(tr, diagnostics, extra=extra)
            return tr, tr.episodes_done == budget
    return tr, True
