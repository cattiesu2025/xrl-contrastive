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
        "weights": list(weights),
    }


def differentiating_channel(decomp: dict) -> tuple[str, float]:
    """Return (channel, score): the channel that *decides* the chosen action — the
    one without which the agent would have acted differently.

    Largest-magnitude ("dominant") is misleading: the goal channel can be biggest
    everywhere yet nearly equal across actions, so it does not explain the choice.
    Instead we use leave-one-out: for each channel, recompute the greedy action
    with that channel removed; if it flips, the channel was decisive. E.g. a coin
    detour is driven by the coin channel because removing it makes the agent head
    straight for the goal — even though the goal channel is larger in absolute
    terms. Falls back to the largest greedy-vs-runner-up margin if no single
    channel is individually decisive.
    """
    q_total = list(decomp["q_total"])
    greedy = decomp["greedy_action"]
    weights = decomp["weights"]
    channels = decomp["channels"]
    n = len(q_total)

    best_ch, best_score = None, -1e18
    for i, ch in enumerate(REWARD_CHANNELS):
        total_without = [q_total[a] - float(weights[i]) * float(channels[ch][a]) for a in range(n)]
        alt = max(range(n), key=lambda a: total_without[a])
        if alt != greedy:  # removing ch flips the decision → ch is decisive
            score = float(weights[i]) * float(channels[ch][greedy] - channels[ch][alt])
            if score > best_score:
                best_score, best_ch = score, ch
    if best_ch is not None:
        return best_ch, best_score

    # No single channel flips the choice → report the largest greedy-vs-runner-up margin.
    order = sorted(range(n), key=lambda a: q_total[a], reverse=True)
    runner_up = order[1] if n > 1 else greedy
    diffs = {ch: float(weights[i]) * float(channels[ch][greedy] - channels[ch][runner_up])
             for i, ch in enumerate(REWARD_CHANNELS)}
    best = max(diffs, key=lambda c: diffs[c])
    return best, diffs[best]


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
