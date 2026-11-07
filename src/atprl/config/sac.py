from dataclasses import dataclass, field, asdict
from atprl.rl.sac import SACConfig
from atprl.config.budgets import WARMUP_EPISODES, POST_WARMUP_EPISODES


@dataclass
class SACRunConfig:
    stage: str = "pipeline"
    seed: int = 42
    seeds: list = field(default_factory=lambda: [3, 6, 8, 9, 42])
    stage1_episodes: int = WARMUP_EPISODES
    stage2_episodes: int = POST_WARMUP_EPISODES
    cycle_dqn_episodes: int = 6
    cycle_power_episodes: int = 10
    macro_steps_per_episode: int = 1024
    num_micro_steps: int = 20
    num_ground_stations: int = 4
    num_gs_per_rb: int = 2
    topology_profile: str = "paper_n4"
    ris_num_elem_x: int = 16
    ris_num_elem_y: int = 16
    fairness_lambda: float = 3.0
    csi_error_variance: float = 0.01
    dqn_scheduler_impl: str = "enumerative"
    num_threads: int = 1
    eval_every_episodes: int = 32
    evaluation_seeds: list = field(
        default_factory=lambda: [10001, 20002, 30003, 40004, 50005, 60006, 70007, 80008]
    )
    num_eval_seeds: int = 500
    eval_seed_base: int = 10001
    eval_seed_stride: int = 37
    checkpoint_every_episodes: int = 16
    power_sac: dict = field(default_factory=dict)
    phase_sac: dict = field(
        default_factory=lambda: dict(actor_lr=6e-5, critic_lr=6e-5, temperature_lr=6e-5)
    )
    phase_distribution: str | None = None
    pbt_enabled: bool = False
    learn_amp_weights: bool = False
    profile: str = "formal"
    kernel_checkpoint: str | None = None
    kernel_checkpoint_sha256: str | None = None
    auto_final_evaluation: bool | None = None
    resume_checkpoint: str | None = None
    checkpoint: str | None = None
    output_dir: str = "."
    resume: bool = False
    dqn_checkpoint: str | None = None
    dqn_checkpoint_sha256: str | None = (
        "24091840c5e0334f3f85670e5696a27df2c5b7ce144c82aafba8ff62b636290f"
    )
    fixed_power_episodes: int = 64
    dqn_replay_capacity: int = 100000
    evaluation_episodes: list = field(default_factory=lambda: [1, 4, 8, 16, 32, 64])
    experiment_group: str | None = None

    def __post_init__(self):
        if self.phase_distribution not in [None, "tanh_normal_v1", "wrapped_normal_v1"]:
            raise ValueError("Unknown Phase distribution")
        if self.phase_distribution == "wrapped_normal_v1" and self.stage not in [
            "pipeline",
            "warmup",
            "stage2",
            "evaluate",
        ]:
            raise ValueError("Wrapped-normal requires the STP training or evaluation pipeline")
        if self.auto_final_evaluation is not None and not isinstance(
            self.auto_final_evaluation, bool
        ):
            raise ValueError("auto_final_evaluation must be bool or None")
        if self.kernel_checkpoint_sha256 is not None and (
            len(self.kernel_checkpoint_sha256) != 64
            or any(x not in "0123456789abcdef" for x in self.kernel_checkpoint_sha256)
        ):
            raise ValueError("kernel_checkpoint_sha256")
        if self.stage not in ["pipeline", "warmup", "stage2", "evaluate", "fixed_power"]:
            raise ValueError("SAC stage")
        if self.pbt_enabled or self.learn_amp_weights:
            raise ValueError("SAC-STP has no PBT or amplitude learning")
        if not 1 <= self.num_gs_per_rb <= self.num_ground_stations:
            raise ValueError("N/K")
        if self.num_threads != 1:
            raise ValueError("SAC-STP CPU workers require one thread")
        if self.dqn_replay_capacity < 129:
            raise ValueError("DQN Replay must allow original >128 update gate")
        for name in [
            "stage1_episodes",
            "stage2_episodes",
            "cycle_dqn_episodes",
            "cycle_power_episodes",
            "macro_steps_per_episode",
            "num_micro_steps",
            "eval_every_episodes",
            "num_eval_seeds",
            "checkpoint_every_episodes",
            "eval_seed_stride",
        ]:
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(name)
        if (
            len(set(self.evaluation_seeds)) != len(self.evaluation_seeds)
            or not self.evaluation_seeds
        ):
            raise ValueError("Evaluation seeds")
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("Training seeds")
        if self.experiment_group is not None and self.experiment_group not in [
            "SAC_BASE",
            "SAC_LOW_ENTROPY",
            "SAC_Q_CLIP5",
            "SAC_ACTOR_LR1",
        ]:
            raise ValueError("SAC experiment group")
        if self.experiment_group is not None and self.stage != "fixed_power":
            raise ValueError("Power ablations require fixed_power scope")
        SACConfig(**self.power_sac)
        SACConfig(**self.phase_sac)
        if self.profile == "formal":
            expected = dict(
                stage1_episodes=WARMUP_EPISODES,
                stage2_episodes=POST_WARMUP_EPISODES,
                cycle_dqn_episodes=6,
                cycle_power_episodes=10,
                macro_steps_per_episode=1024,
                num_micro_steps=20,
                eval_every_episodes=32,
                num_eval_seeds=500,
            )
            if any(getattr(self, k) != v for k, v in expected.items()):
                raise ValueError("Formal SAC budget/protocol changed; use a signed smoke profile")

    def to_dict(self):
        v = asdict(self)
        if self.phase_distribution in [None, "tanh_normal_v1"]:
            v.pop("phase_distribution")
        if self.experiment_group is None:
            v.pop("experiment_group")
        for k in ["kernel_checkpoint_sha256", "auto_final_evaluation"]:
            if v[k] is None:
                v.pop(k)
        return v
