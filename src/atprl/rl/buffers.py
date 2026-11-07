import torch
import numpy as np
from typing import Dict, Tuple


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "states": torch.from_numpy(self.states[indices]),
            "actions": torch.from_numpy(self.actions[indices]),
            "rewards": torch.from_numpy(self.rewards[indices]),
            "next_states": torch.from_numpy(self.next_states[indices]),
            "dones": torch.from_numpy(self.dones[indices]),
        }
        return batch

    def __len__(self) -> int:
        return self.size

    def clear(self):
        self.ptr = 0
        self.size = 0


class TrajectoryBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        batch = {
            "states": torch.stack(self.states, dim=0).detach(),
            "actions": torch.stack(self.actions, dim=0).detach(),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32).unsqueeze(1),
            "dones": torch.tensor(self.dones, dtype=torch.float32).unsqueeze(1),
            "log_probs": torch.stack(self.log_probs, dim=0).detach(),
            "values": torch.stack(self.values, dim=0).detach(),
        }
        return batch

    def clear(self):
        self.__init__()

    def __len__(self) -> int:
        return len(self.states)
