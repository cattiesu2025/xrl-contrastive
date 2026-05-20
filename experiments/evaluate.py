"""
Quantitative evaluation of explanation quality.

Metrics:
  1. Explanation Fidelity — mask the concept TCAV identifies as important,
     measure agent performance drop (bigger drop = more faithful explanation)
  2. Agent Distinguishability — given decompositions from two agents,
     can a simple classifier tell them apart?
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from envs.multi_obj_grid import MultiObjGridEnv
from agents.decomposed_dqn import DecomposedDQN
from agents.train import evaluate_agent
from agents.configs import REWARD_CHANNELS


def explanation_fidelity(
    model: DecomposedDQN,
    weights: list[float],
    important_channel_idx: int,
    n_eval: int = 100,
) -> dict:
    """
    Measure how much performance drops when the TCAV-identified
    important channel's Q-head is zeroed out.

    A large drop = the explanation is faithful (the channel truly matters).
    """
    # Baseline performance
    baseline = evaluate_agent(model, weights, n_eval=n_eval)

    # Create ablated weights (zero out the important channel)
    ablated_weights = list(weights)
    ablated_weights[important_channel_idx] = 0.0
    ablated = evaluate_agent(model, ablated_weights, n_eval=n_eval)

    return {
        "baseline_R1": baseline["avg_R1"],
        "ablated_R1": ablated["avg_R1"],
        "R1_drop": baseline["avg_R1"] - ablated["avg_R1"],
        "baseline_success": baseline["success_rate"],
        "ablated_success": ablated["success_rate"],
        "success_drop": baseline["success_rate"] - ablated["success_rate"],
        "channel": REWARD_CHANNELS[important_channel_idx],
    }


def agent_distinguishability(
    model_a: DecomposedDQN,
    weights_a: list[float],
    model_b: DecomposedDQN,
    weights_b: list[float],
    n_samples: int = 300,
    seed: int = 42,
) -> float:
    """
    Train a logistic regression classifier on decomposed Q-values
    to distinguish Agent A from Agent B. Return accuracy.

    High accuracy = the reward decomposition captures genuine
    differences between agent strategies.
    """
    rng = np.random.RandomState(seed)
    env = MultiObjGridEnv()

    features = []
    labels = []

    for _ in range(n_samples):
        row = rng.randint(0, env.GRID_SIZE)
        col = rng.randint(0, env.GRID_SIZE)
        collected = rng.randint(0, 256)
        state = env.features_from_pos(row, col, collected)

        # Get Q-decomposition for each agent
        q_a = model_a.get_decomposed_q(state, weights_a)
        q_b = model_b.get_decomposed_q(state, weights_b)

        # Feature = concatenation of per-channel Q-values for greedy action
        greedy_a = int(q_a["total"].argmax())
        greedy_b = int(q_b["total"].argmax())

        feat_a = np.array([q_a["channels"][i][greedy_a] for i in range(len(REWARD_CHANNELS))])
        feat_b = np.array([q_b["channels"][i][greedy_b] for i in range(len(REWARD_CHANNELS))])

        features.append(feat_a)
        labels.append(0)
        features.append(feat_b)
        labels.append(1)

    X = np.array(features)
    y = np.array(labels)

    # Shuffle and split
    idx = rng.permutation(len(y))
    split = int(0.7 * len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    return accuracy
