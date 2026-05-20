"""
Diagnostic script — investigates WHY trained agents get 0% success.

Does NOT modify training. Only reads checkpoints and traces behavior.
"""

import numpy as np
import torch
from agents.train import load_agent
from agents.configs import PROFILES, REWARD_CHANNELS
from envs.multi_obj_grid import MultiObjGridEnv


def trace_greedy_episode(model, weights, profile_name):
    """Run a single greedy episode and print every step."""
    env = MultiObjGridEnv(seed=42)
    obs = env.reset()

    print(f"\n{'=' * 70}")
    print(f"GREEDY TRACE: {profile_name}  weights={weights}")
    print(f"{'=' * 70}")
    print(f"  Start: pos=(0,0)  obs={obs}")

    visited_positions = []
    action_counts = [0, 0, 0, 0]
    same_state_count = 0
    prev_pos = (env.row, env.col)

    for t in range(env.MAX_STEPS):
        q_info = model.get_decomposed_q(obs, weights)
        q_total = q_info["total"]
        action = int(q_total.argmax())
        action_counts[action] += 1

        next_obs, _, done, info = env.step(action)
        pos = (env.row, env.col)
        visited_positions.append(pos)

        if pos == prev_pos:
            same_state_count += 1

        if t < 10 or t in (50, 100, 150, 199):
            ch_names = ["goal", "safety", "coin", "step"]
            ch_q_at_action = [q_info["channels"][c][action] for c in range(4)]
            print(f"  t={t:3d}  pos={prev_pos}->{pos}  "
                  f"a={env.ACTION_NAMES[action]:>5}  "
                  f"q_total={q_total.round(2).tolist()}  "
                  f"q@a=[g:{ch_q_at_action[0]:.2f}, s:{ch_q_at_action[1]:.2f}, "
                  f"c:{ch_q_at_action[2]:.2f}, st:{ch_q_at_action[3]:.2f}]  "
                  f"done={done}")

        prev_pos = pos
        obs = next_obs
        if done:
            print(f"  >> Terminated at t={t}, pos={pos}, reason={'GOAL' if pos == env.GOAL else 'HAZARD' if pos in env.HAZARDS else 'TIMEOUT'}")
            break

    print(f"\n  Summary:")
    print(f"    Total steps: {t+1}")
    print(f"    Action distribution: UP={action_counts[0]} DOWN={action_counts[1]} LEFT={action_counts[2]} RIGHT={action_counts[3]}")
    print(f"    Unique positions visited: {len(set(visited_positions))}")
    print(f"    Steps where pos didn't change (wall-bump): {same_state_count}")
    print(f"    Final pos: {pos}")


def inspect_q_values_at_start(model, weights, profile_name):
    """Look at Q-values at the START state — to see if the agent prefers wall direction."""
    env = MultiObjGridEnv(seed=42)
    obs = env.reset()

    print(f"\n--- Q-values at start state (0,0) for {profile_name} ---")
    q_info = model.get_decomposed_q(obs, weights)
    print(f"  Q_total      = {q_info['total'].round(3).tolist()}  (UP, DOWN, LEFT, RIGHT)")
    for i, ch in enumerate(REWARD_CHANNELS):
        print(f"  Q_{ch:<7} = {q_info['channels'][i].round(3).tolist()}")
    print(f"  argmax       = {env.ACTION_NAMES[int(q_info['total'].argmax())]}")
    print(f"  (UP and LEFT bump into wall at (0,0))")


def check_random_baseline(seed=42, n_eval=50):
    """Sanity check: what does a RANDOM policy achieve?"""
    print(f"\n--- RANDOM POLICY BASELINE (n={n_eval}) ---")
    env = MultiObjGridEnv(seed=seed)
    rng = np.random.RandomState(seed)

    successes = 0
    hazards = 0
    coins_collected_total = 0
    timeouts = 0

    for ep in range(n_eval):
        env.reset()
        for _ in range(env.MAX_STEPS):
            a = rng.randint(0, 4)
            _, _, done, info = env.step(a)
            if done:
                if (env.row, env.col) == env.GOAL:
                    successes += 1
                elif (env.row, env.col) in env.HAZARDS:
                    hazards += 1
                break
        else:
            timeouts += 1
        coins_collected_total += bin(env.collected).count("1")

    print(f"  Random success rate: {successes}/{n_eval} = {100*successes/n_eval:.0f}%")
    print(f"  Random hazards:      {hazards}/{n_eval}")
    print(f"  Random timeouts:     {timeouts}/{n_eval}")
    print(f"  Avg coins per ep:    {coins_collected_total/n_eval:.2f}")


if __name__ == "__main__":
    check_random_baseline()

    for key, profile in PROFILES.items():
        path = f"checkpoints/{key}.pt"
        model, ckpt = load_agent(path)
        inspect_q_values_at_start(model, profile["weights"], profile["name"])
        trace_greedy_episode(model, profile["weights"], profile["name"])
