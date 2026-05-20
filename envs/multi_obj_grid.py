"""
Multi-Objective GridWorld Environment (v2)
Adapted from COMP9414 Assignment 2 GridWorldEnv.

Changes from assignment:
  - step() returns 4-channel reward vector in info["reward_vector"]
  - render() returns RGB numpy array for TCAV concept image collection
  - Gymnasium-compatible interface (gymnasium.Env subclass)
"""

import numpy as np
from typing import Optional


class MultiObjGridEnv:
    """10x10 Multi-Objective Grid World with 4 reward channels."""

    # ---------- Map layout (same as assignment) ----------
    HAZARDS = {
        (1, 3), (2, 7), (3, 1), (3, 5), (4, 8),
        (5, 2), (6, 6), (7, 4), (8, 1), (8, 8),
    }

    RESOURCES = [
        (0, 7), (1, 9), (2, 8), (3, 4), (5, 1),
        (6, 7), (7, 0), (8, 2),
    ]  # 8 resources, ordered for bitmask indexing

    GOAL = (9, 9)
    START = (0, 0)
    GRID_SIZE = 10

    # ---------- Reward constants ----------
    R_GOAL = 10.0
    R_HAZARD = -10.0
    R_RESOURCE = 0.5
    R_STEP = -0.15

    MAX_STEPS = 200

    # action offsets: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

    N_ACTIONS = 4
    # State: (row, col, collected_bitmask) -> flat features or index
    # Feature dim for neural network: row, col, collected (normalised)
    STATE_DIM = 10  # [row/9, col/9, bit0..bit7]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.row = 0
        self.col = 0
        self.collected = 0
        self.steps = 0
        self.done = False

    # ---------- State encoding (kept from assignment for compatibility) ----------
    @property
    def n_states(self):
        return 100 * 256  # 25600

    def encode(self, row: int, col: int, collected: int) -> int:
        return (row * 10 + col) * 256 + collected

    def decode(self, idx: int):
        collected = idx % 256
        pos = idx // 256
        row = pos // 10
        col = pos % 10
        return row, col, collected

    # ---------- Feature vector for neural network ----------
    def state_features(self) -> np.ndarray:
        """Return normalised [row/9, col/9, bit0..bit7] for the DQN input."""
        bits = [(self.collected >> i) & 1 for i in range(len(self.RESOURCES))]
        return np.array(
            [self.row / (self.GRID_SIZE - 1), self.col / (self.GRID_SIZE - 1)] + bits,
            dtype=np.float32,
        )

    def features_from_pos(self, row: int, col: int, collected: int) -> np.ndarray:
        """Static version for batch feature extraction."""
        bits = [(collected >> i) & 1 for i in range(len(self.RESOURCES))]
        return np.array(
            [row / (self.GRID_SIZE - 1), col / (self.GRID_SIZE - 1)] + bits,
            dtype=np.float32,
        )

    # ---------- Core interface ----------
    def reset(self) -> np.ndarray:
        self.row, self.col = self.START
        self.collected = 0
        self.steps = 0
        self.done = False
        return self.state_features()

    def step(self, action: int):
        """
        Returns: (next_state_features, scalar_reward, done, info)

        info["reward_vector"] = {
            "goal": float,    # +10 if reached goal
            "safety": float,  # -10 if hit hazard
            "coin": float,    # +0.5 if collected new resource
            "step": float,    # -0.15 per step
        }
        info["R1"] = goal + safety + step   (original R1)
        info["R2"] = coin                    (original R2)
        """
        if self.done:
            raise RuntimeError("Episode ended. Call reset().")

        dr, dc = self.ACTION_DELTAS[action]
        new_row = max(0, min(self.GRID_SIZE - 1, self.row + dr))
        new_col = max(0, min(self.GRID_SIZE - 1, self.col + dc))

        self.row = new_row
        self.col = new_col
        self.steps += 1

        # --- 4-channel reward computation ---
        r_goal = 0.0
        r_safety = 0.0
        r_coin = 0.0
        r_step = self.R_STEP

        pos = (self.row, self.col)
        done = False

        # Resource collection
        if pos in self.RESOURCES:
            res_idx = self.RESOURCES.index(pos)
            bit = 1 << res_idx
            if not (self.collected & bit):
                self.collected |= bit
                r_coin = self.R_RESOURCE

        # Goal / hazard
        if pos == self.GOAL:
            r_goal += self.R_GOAL
            done = True
        elif pos in self.HAZARDS:
            r_safety = self.R_HAZARD
            done = True

        if self.steps >= self.MAX_STEPS:
            done = True

        self.done = done

        reward_vector = {
            "goal": r_goal,
            "safety": r_safety,
            "coin": r_coin,
            "step": r_step,
        }

        # Backward-compatible R1/R2
        R1 = r_goal + r_safety + r_step
        R2 = r_coin
        scalar_reward = R1 + R2  # default: equal weight

        info = {
            "reward_vector": reward_vector,
            "R1": R1,
            "R2": R2,
            "position": (self.row, self.col),
            "collected": self.collected,
        }

        return self.state_features(), scalar_reward, done, info

    # ---------- Rendering ----------
    def render(self, path=None) -> np.ndarray:
        """Return 10x10x3 uint8 RGB array of the current state."""
        G = self.GRID_SIZE
        img = np.full((G, G, 3), 240, dtype=np.uint8)  # light grey

        # Hazards = red
        for r, c in self.HAZARDS:
            img[r, c] = [220, 50, 50]

        # Resources = gold (uncollected only)
        for idx, (r, c) in enumerate(self.RESOURCES):
            if not (self.collected & (1 << idx)):
                img[r, c] = [255, 200, 50]

        # Goal = green
        img[self.GOAL[0], self.GOAL[1]] = [50, 200, 80]

        # Agent = blue
        img[self.row, self.col] = [50, 80, 220]

        # Path overlay
        if path:
            for r, c in path:
                if img[r, c, 2] < 200:  # don't overwrite agent
                    img[r, c] = [180, 180, 255]

        return img

    # ---------- Utility ----------
    def get_neighbourhood(self, radius: int = 1):
        """Return set of cell types within radius of agent (for concept labelling)."""
        types = set()
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = self.row + dr, self.col + dc
                if 0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE:
                    pos = (r, c)
                    if pos in self.HAZARDS:
                        types.add("near_hazard")
                    if pos in [self.RESOURCES[i] for i in range(len(self.RESOURCES))
                               if not (self.collected & (1 << i))]:
                        types.add("near_resource")
                    if pos == self.GOAL:
                        types.add("near_goal")
        return types


# ---------- Quick sanity check ----------
if __name__ == "__main__":
    env = MultiObjGridEnv(seed=42)
    obs = env.reset()
    print(f"Initial obs: {obs}")
    print(f"n_states={env.n_states}, STATE_DIM={env.STATE_DIM}")

    for i in range(5):
        action = np.random.randint(0, 4)
        obs, reward, done, info = env.step(action)
        rv = info["reward_vector"]
        print(f"  Step {i+1}: {env.ACTION_NAMES[action]:>5}  "
              f"pos=({info['position'][0]},{info['position'][1]})  "
              f"goal={rv['goal']:.1f}  safety={rv['safety']:.1f}  "
              f"coin={rv['coin']:.1f}  step={rv['step']:.2f}")
        if done:
            break
