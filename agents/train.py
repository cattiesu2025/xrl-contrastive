"""
Training loop for DecomposedDQN agents.

Each Q-head is trained with its own reward channel's TD target,
mirroring the assignment's scalarised Q-learning but in neural form.
"""

import os
import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from envs.multi_obj_grid import MultiObjGridEnv
from agents.decomposed_dqn import DecomposedDQN
from agents.replay_buffer import ReplayBuffer
from agents.configs import TRAIN_CONFIG, REWARD_CHANNELS


def train_agent(
    profile_name: str,
    weights: list[float],
    config: dict = None,
    save_dir: str = "checkpoints",
    verbose: bool = True,
) -> tuple[DecomposedDQN, list[float]]:
    """
    Train a DecomposedDQN agent with the given reward channel weights.

    Args:
        profile_name: identifier for saving (e.g. "safety_first")
        weights: list of 4 floats [w_goal, w_safety, w_coin, w_step]
        config: training hyperparameters (defaults to TRAIN_CONFIG)
        save_dir: directory for model checkpoints
        verbose: print progress

    Returns:
        (trained_model, episode_returns)
    """
    cfg = {**TRAIN_CONFIG, **(config or {})}
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.RandomState(cfg["seed"])

    env = MultiObjGridEnv(seed=cfg["seed"])

    # Online and target networks
    online_net = DecomposedDQN(
        state_dim=env.STATE_DIM,
        n_actions=env.N_ACTIONS,
        n_channels=len(REWARD_CHANNELS),
        hidden_dim=cfg["hidden_dim"],
    )
    target_net = copy.deepcopy(online_net)
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(capacity=cfg["buffer_size"], seed=cfg["seed"])
    w_tensor = torch.tensor(weights, dtype=torch.float32)

    eps = cfg["eps_start"]
    episode_returns = []
    global_step = 0

    # --- Diagnostic accumulators (per-window of episodes) ---
    diag_window = 100
    diag_q_max = [-1e18] * len(REWARD_CHANNELS)
    diag_q_min = [1e18] * len(REWARD_CHANNELS)
    diag_td_abs_max = 0.0
    diag_loss_max = 0.0
    diag_ep_outcomes = []  # 'goal' / 'hazard' / 'timeout' per episode in window

    # --- Best-model tracking (catastrophic-forgetting guard) ---
    # Select the snapshot with the highest window-average SCALARISED return
    # (Σ wᵢ·channelᵢ). This is the quantity each agent actually optimises, so it
    # works for any profile — a collector (which ignores the goal) is scored on the
    # coins it gathers, not on a goal-rate that would be ~0 for it.
    best_avg_return = -1e18
    best_state_dict = None
    best_episode = 0

    # --- n-step return setup ---
    n_step = cfg["n_step"]
    n_channels = len(REWARD_CHANNELS)

    def push_n_step_transition(queue):
        """Accumulate the n-step discounted return for the OLDEST transition in
        `queue` (looking ahead up to n_step items) and push it to the buffer."""
        s0, a0 = queue[0][0], queue[0][1]
        R = np.zeros(n_channels, dtype=np.float32)
        discount = 1.0
        bootstrap_state = queue[0][3]
        done_n = False
        steps_used = 0
        for (_, _, r_k, ns_k, d_k) in list(queue)[:n_step]:
            R += discount * r_k
            discount *= cfg["gamma"]
            bootstrap_state = ns_k
            steps_used += 1
            if d_k:
                done_n = True
                break
        buffer.push(s0, a0, R, bootstrap_state, done_n, steps_used)

    iterator = trange(cfg["n_episodes"], desc=profile_name) if verbose else range(cfg["n_episodes"])

    for ep in iterator:
        obs = env.reset()
        ep_return = 0.0
        ep_outcome = "timeout"  # default if neither done
        n_step_queue = deque()  # holds recent transitions for n-step accumulation

        for t in range(env.MAX_STEPS):
            # Epsilon-greedy
            action = online_net.select_action(obs, weights, epsilon=eps, rng=rng)

            next_obs, _, done, info = env.step(action)
            rv = info["reward_vector"]

            # Store per-channel rewards as array
            reward_vec = np.array([
                rv["goal"], rv["safety"], rv["coin"], rv["step"]
            ], dtype=np.float32)

            n_step_queue.append((obs, action, reward_vec, next_obs, done))
            if len(n_step_queue) >= n_step:
                push_n_step_transition(n_step_queue)
                n_step_queue.popleft()

            # Scalarised return for logging (same as assignment)
            ep_return += float(np.dot(weights, reward_vec))

            obs = next_obs
            global_step += 1

            # --- Learn ---
            if len(buffer) >= cfg["batch_size"]:
                states, actions, rewards, next_states, dones, n_steps = buffer.sample(cfg["batch_size"])

                s_t = torch.tensor(states)
                a_t = torch.tensor(actions).long()
                r_t = torch.tensor(rewards)  # (batch, n_channels) — accumulated n-step return
                ns_t = torch.tensor(next_states)
                d_t = torch.tensor(dones)
                n_t = torch.tensor(n_steps)
                gamma_n = cfg["gamma"] ** n_t  # (batch,) — γ^n per transition

                # Forward pass
                _, q_channels_online, _ = online_net(s_t)
                with torch.no_grad():
                    _, q_channels_target, _ = target_net(ns_t)
                    # Use total weighted Q to pick greedy next action (correct NDQL)
                    q_total_next = sum(w * q for w, q in zip(w_tensor, q_channels_target))
                    greedy_next = q_total_next.argmax(dim=1, keepdim=True)  # (batch, 1)

                # Per-channel TD loss — weights only affect action selection, NOT loss
                total_loss = torch.tensor(0.0)
                for ch_idx in range(len(REWARD_CHANNELS)):
                    q_pred = q_channels_online[ch_idx].gather(1, a_t.unsqueeze(1)).squeeze(1)
                    q_next = q_channels_target[ch_idx].gather(1, greedy_next).squeeze(1)
                    td_target = r_t[:, ch_idx] + gamma_n * q_next * (1.0 - d_t.float())
                    channel_loss = nn.functional.mse_loss(q_pred, td_target)
                    total_loss = total_loss + channel_loss

                    # Diagnostics: track per-channel Q range & worst |TD error|
                    with torch.no_grad():
                        diag_q_max[ch_idx] = max(diag_q_max[ch_idx], float(q_channels_online[ch_idx].max()))
                        diag_q_min[ch_idx] = min(diag_q_min[ch_idx], float(q_channels_online[ch_idx].min()))
                        td_abs = float((q_pred - td_target).abs().max())
                        if td_abs > diag_td_abs_max:
                            diag_td_abs_max = td_abs

                with torch.no_grad():
                    diag_loss_max = max(diag_loss_max, float(total_loss))

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                optimizer.step()

            # Target network sync
            if global_step % cfg["target_update_freq"] == 0:
                target_net.load_state_dict(online_net.state_dict())

            if done:
                # Flush remaining transitions (shorter horizons; terminal within window)
                while len(n_step_queue) > 0:
                    push_n_step_transition(n_step_queue)
                    n_step_queue.popleft()

                pos = (env.row, env.col)
                if pos == env.GOAL:
                    ep_outcome = "goal"
                elif pos in env.HAZARDS:
                    ep_outcome = "hazard"
                # else: done from MAX_STEPS → keep "timeout"
                break

        eps = max(cfg["eps_min"], eps * cfg["eps_decay"])
        episode_returns.append(ep_return)
        diag_ep_outcomes.append(ep_outcome)

        if verbose:
            avg = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
            iterator.set_postfix({"avg100": f"{avg:.2f}", "eps": f"{eps:.3f}"})

        # --- Window checkpoint every `diag_window` episodes: log + best-model track ---
        if (ep + 1) % diag_window == 0:
            n_goal = diag_ep_outcomes.count("goal")
            n_haz = diag_ep_outcomes.count("hazard")
            n_to = diag_ep_outcomes.count("timeout")
            window_avg_return = float(np.mean(episode_returns[-diag_window:]))

            # Best-model: keep the snapshot with the highest window-average scalarised
            # return (what the agent optimises). Tie-break by latest.
            if window_avg_return >= best_avg_return:
                best_avg_return = window_avg_return
                best_state_dict = copy.deepcopy(online_net.state_dict())
                best_episode = ep + 1

            if verbose:
                q_ranges = " ".join(
                    f"{ch}:[{diag_q_min[i]:+.1f},{diag_q_max[i]:+.1f}]"
                    for i, ch in enumerate(REWARD_CHANNELS)
                )
                iterator.write(
                    f"  [ep {ep+1:4d}] outcomes: goal={n_goal:3d} haz={n_haz:3d} timeout={n_to:3d}  "
                    f"|TD|max={diag_td_abs_max:.2f}  loss_max={diag_loss_max:.2f}  "
                    f"best_return={best_avg_return:.2f}@ep{best_episode}  Q ranges: {q_ranges}"
                )
            # Reset window accumulators
            diag_q_max = [-1e18] * len(REWARD_CHANNELS)
            diag_q_min = [1e18] * len(REWARD_CHANNELS)
            diag_td_abs_max = 0.0
            diag_loss_max = 0.0
            diag_ep_outcomes = []

    # --- Save the BEST snapshot, not the final one ---
    # Fall back to current state only if training was shorter than one diag window.
    if best_state_dict is not None:
        state_to_save = best_state_dict
        online_net.load_state_dict(best_state_dict)  # mutate in place so returned model is the best
    else:
        state_to_save = online_net.state_dict()
        best_episode = cfg["n_episodes"]
        best_avg_return = float("nan")

    save_path = os.path.join(save_dir, f"{profile_name}.pt")
    torch.save({
        "model_state": state_to_save,
        "weights": weights,
        "profile_name": profile_name,
        "config": cfg,
        "episode_returns": episode_returns,
        "best_episode": best_episode,
        "best_avg_return": best_avg_return,
    }, save_path)

    if verbose:
        print(f"  Saved best snapshot from ep {best_episode} "
              f"(window-avg return {best_avg_return:.2f}) to {save_path}")

    return online_net, episode_returns


def load_agent(path: str) -> tuple[DecomposedDQN, dict]:
    """Load a trained agent from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DecomposedDQN(hidden_dim=ckpt["config"]["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_agent(
    model: DecomposedDQN,
    weights: list[float],
    n_eval: int = 100,
    seed: int = 42,
) -> dict:
    """
    Evaluate agent performance (mirrors assignment's evaluate_agent).
    Returns dict with success_rate, avg_R1, avg_R2, avg_length, etc.
    """
    env = MultiObjGridEnv(seed=seed)
    successes = 0
    safety_violations = 0
    R1s, R2s = [], []
    lengths = []
    channel_returns = {ch: [] for ch in REWARD_CHANNELS}

    for ep in range(n_eval):
        obs = env.reset()
        ep_r1, ep_r2 = 0.0, 0.0
        ep_channels = {ch: 0.0 for ch in REWARD_CHANNELS}
        steps = 0

        for _ in range(env.MAX_STEPS):
            action = model.select_action(obs, weights, epsilon=0.0)
            obs, _, done, info = env.step(action)
            ep_r1 += info["R1"]
            ep_r2 += info["R2"]
            for ch in REWARD_CHANNELS:
                ep_channels[ch] += info["reward_vector"][ch]
            steps += 1
            if done:
                break

        reached_goal = env.row == env.GOAL[0] and env.col == env.GOAL[1]
        hit_hazard = (env.row, env.col) in env.HAZARDS

        if reached_goal:
            successes += 1
            lengths.append(steps)
        if hit_hazard:
            safety_violations += 1

        R1s.append(ep_r1)
        R2s.append(ep_r2)
        for ch in REWARD_CHANNELS:
            channel_returns[ch].append(ep_channels[ch])

    return {
        "success_rate": successes / n_eval,
        "avg_R1": float(np.mean(R1s)),
        "avg_R2": float(np.mean(R2s)),
        "avg_length": float(np.mean(lengths)) if lengths else float("nan"),
        "safety_violations": safety_violations,
        "channel_returns": {ch: float(np.mean(v)) for ch, v in channel_returns.items()},
    }


def greedy_trajectory(
    model: DecomposedDQN,
    weights: list[float],
    max_steps: int = 200,
) -> dict:
    """
    Run a single greedy episode. Returns full trajectory with per-step data.
    """
    env = MultiObjGridEnv(seed=0)
    obs = env.reset()

    trajectory = {
        "states": [obs.copy()],
        "positions": [(env.row, env.col)],
        "actions": [],
        "reward_vectors": [],
        "q_decompositions": [],
        "features": [],
    }

    visited = set()
    visited.add(env.encode(env.row, env.col, env.collected))

    for _ in range(max_steps):
        # Get decomposed Q-values at this state
        q_info = model.get_decomposed_q(obs, weights)
        trajectory["q_decompositions"].append(q_info)
        trajectory["features"].append(q_info["features"])

        action = int(q_info["total"].argmax())
        trajectory["actions"].append(action)

        obs, _, done, info = env.step(action)
        trajectory["states"].append(obs.copy())
        trajectory["positions"].append((env.row, env.col))
        trajectory["reward_vectors"].append(info["reward_vector"])

        state_idx = env.encode(env.row, env.col, env.collected)
        if done or state_idx in visited:
            break
        visited.add(state_idx)

    return trajectory


if __name__ == "__main__":
    from agents.configs import PROFILES

    # Quick test: train balanced agent for 500 episodes
    print("Quick training test (500 episodes)...")
    model, returns = train_agent(
        "test_balanced",
        PROFILES["balanced"]["weights"],
        config={"n_episodes": 500},
        save_dir="/tmp/xrl_test",
    )
    metrics = evaluate_agent(model, PROFILES["balanced"]["weights"], n_eval=50)
    print(f"Success: {metrics['success_rate']:.0%}, "
          f"R1: {metrics['avg_R1']:.2f}, R2: {metrics['avg_R2']:.2f}")
