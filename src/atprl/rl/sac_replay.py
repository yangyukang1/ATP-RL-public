import copy
import numpy as np
import torch
from atprl.rl.sac_distributions import validate_action


class SACReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, *, seed=0, phase=False):
        self.capacity = capacity
        self.phase = phase
        self.ptr = self.size = self.store_count = self.sample_count = 0
        self.rng = np.random.default_rng(seed)
        self.context_revision = None
        shapes = dict(
            states=(state_dim,),
            actions=(action_dim,),
            rewards=(1,),
            next_states=(state_dim,),
            terminated=(1,),
            truncated=(1,),
        )
        self.arrays = {k: np.zeros((capacity, *v), np.float32) for k, v in shapes.items()}
        self.episode_ids = np.full(capacity, -1, np.int64)
        self.macro_steps = np.zeros(capacity, np.int64)

    def __len__(self):
        return self.size

    def clear_context(self, revision):
        self.ptr = self.size = 0
        self.context_revision = revision
        self.episode_ids.fill(-1)
        self.macro_steps.fill(0)

    def store(self, s, a, r, ns, terminated, truncated, episode_id=0, macro_step=0):
        values = dict(
            states=s,
            actions=a,
            rewards=[r],
            next_states=ns,
            terminated=[terminated],
            truncated=[truncated],
        )
        validate_action(torch.as_tensor(a), self.phase)
        for k, v in values.items():
            v = np.asarray(v, np.float32)
            if v.shape != self.arrays[k].shape[1:] or not np.isfinite(v).all():
                raise ValueError("Invalid SAC replay " + k)
        for flag in [terminated, truncated]:
            if flag not in [False, True, 0, 1]:
                raise ValueError("Invalid termination flag")
        for k, v in values.items():
            self.arrays[k][self.ptr] = v
        self.episode_ids[self.ptr] = episode_id
        self.macro_steps[self.ptr] = macro_step
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.store_count += 1

    def sample(self, n):
        if self.size < n:
            raise ValueError("Replay not ready")
        ids = self.rng.integers(self.size, size=n)
        self.sample_count += n
        return {k: torch.from_numpy(v[ids].copy()) for k, v in self.arrays.items()}

    def state_dict(self):
        return copy.deepcopy(
            dict(
                version=1,
                capacity=self.capacity,
                phase=self.phase,
                ptr=self.ptr,
                size=self.size,
                arrays=self.arrays,
                episode_ids=self.episode_ids,
                macro_steps=self.macro_steps,
                rng=self.rng.bit_generator.state,
                context_revision=self.context_revision,
                store_count=self.store_count,
                sample_count=self.sample_count,
            )
        )

    def load_state_dict(self, s):
        if s["version"] != 1 or s["capacity"] != self.capacity or s["phase"] != self.phase:
            raise ValueError("Replay signature mismatch")
        if not 0 <= s["size"] <= self.capacity or not 0 <= s["ptr"] < self.capacity:
            raise ValueError("Replay position")
        for k, v in self.arrays.items():
            if s["arrays"][k].shape != v.shape or not np.isfinite(s["arrays"][k]).all():
                raise ValueError("Replay arrays")
            v[:] = s["arrays"][k]
        for k in ["ptr", "size", "context_revision", "store_count", "sample_count"]:
            setattr(self, k, copy.deepcopy(s[k]))
        self.episode_ids[:] = s["episode_ids"]
        self.macro_steps[:] = s["macro_steps"]
        self.rng.bit_generator.state = copy.deepcopy(s["rng"])
