"""Simple experience replay buffer."""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int = 50000, seed: int = 42):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward_vector, next_state, done, n_step=1):
        """
        Args:
            state: np.ndarray of shape (state_dim,)
            action: int
            reward_vector: np.ndarray (n_channels,) — accumulated n-step discounted return
            next_state: np.ndarray (state_dim,) — bootstrap state (after n_step steps)
            done: bool — terminal reached within the n steps
            n_step: int — effective horizon for this transition (γ**n_step in the target)
        """
        self.buffer.append((state, action, reward_vector, next_state, done, n_step))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, n_steps = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(n_steps, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
