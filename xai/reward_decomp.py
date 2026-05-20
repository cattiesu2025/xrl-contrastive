"""
Reward decomposition utilities.

Since DecomposedDQN has per-channel Q-heads, decomposition is intrinsic —
we just read each head's output. This module provides convenience functions
for extracting and formatting decomposed Q-values.
"""

import numpy as np
from agents.decomposed_dqn import DecomposedDQN
from agents.configs import REWARD_CHANNELS
from envs.multi_obj_grid import MultiObjGridEnv


def decompose_state(
    model: DecomposedDQN,
    state: np.ndarray,
    weights: list[float],
) -> dict:
    """
    Decompose Q-values at a single state.

    Returns: {
        "action_names": ["UP", "DOWN", "LEFT", "RIGHT"],
        "greedy_action": int,
        "q_total": np.ndarray (4,),
        "channels": {
            "goal":   np.ndarray (4,),  # Q-values per action
            "safety": np.ndarray (4,),
            "coin":   np.ndarray (4,),
            "step":   np.ndarray (4,),
        },
        "greedy_channel_contributions": {
            "goal":   float,  # Q_goal for the greedy action
            "safety": float,
            "coin":   float,
            "step":   float,
        }
    }
    """
    q_info = model.get_decomposed_q(state, weights)
    greedy = int(q_info["total"].argmax())

    channels = {}
    contributions = {}
    for i, ch_name in enumerate(REWARD_CHANNELS):
        channels[ch_name] = q_info["channels"][i]
        contributions[ch_name] = float(
            weights[i] * q_info["channels"][i][greedy]
        )

    return {
        "action_names": MultiObjGridEnv.ACTION_NAMES,
        "greedy_action": greedy,
        "q_total": q_info["total"],
        "channels": channels,
        "greedy_channel_contributions": contributions,
    }


def trajectory_decomposition(
    model: DecomposedDQN,
    weights: list[float],
    max_steps: int = 200,
) -> list[dict]:
    """
    Run greedy episode and return per-step decomposition.
    Each entry is the output of decompose_state plus position info.
    """
    env = MultiObjGridEnv(seed=0)
    obs = env.reset()
    steps = []
    visited = set()
    visited.add(env.encode(env.row, env.col, env.collected))

    for t in range(max_steps):
        decomp = decompose_state(model, obs, weights)
        decomp["position"] = (env.row, env.col)
        decomp["step_idx"] = t
        steps.append(decomp)

        action = decomp["greedy_action"]
        obs, _, done, info = env.step(action)

        state_idx = env.encode(env.row, env.col, env.collected)
        if done or state_idx in visited:
            break
        visited.add(state_idx)

    return steps


def format_decomposition(decomp: dict) -> str:
    """Pretty-print a single state's Q-value decomposition."""
    lines = []
    pos = decomp.get("position", "?")
    greedy = decomp["greedy_action"]
    action_name = decomp["action_names"][greedy]
    lines.append(f"Position: {pos}  →  Action: {action_name}")
    lines.append(f"{'Channel':<10} {'Q(greedy)':>10} {'Contribution':>13}")
    lines.append("-" * 35)

    for ch_name in REWARD_CHANNELS:
        q_val = decomp["channels"][ch_name][greedy]
        contrib = decomp["greedy_channel_contributions"][ch_name]
        lines.append(f"{ch_name:<10} {q_val:>10.3f} {contrib:>13.3f}")

    lines.append("-" * 35)
    lines.append(f"{'TOTAL':<10} {decomp['q_total'][greedy]:>10.3f}")
    return "\n".join(lines)
