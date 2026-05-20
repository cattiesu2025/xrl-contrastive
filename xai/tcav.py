"""
TCAV (Testing with Concept Activation Vectors) for DecomposedDQN.

Concepts are defined by environmental states (e.g. "near_hazard"),
not images. We probe the shared encoder's 64-dim feature space.

Reference: Kim et al., "Interpretability Beyond Feature Attribution:
           Quantitative Testing with Concept Activation Vectors (TCAV)", ICML 2018.
Adapted for RL following Lee & Cruz (2025).
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from typing import Optional

from envs.multi_obj_grid import MultiObjGridEnv
from agents.decomposed_dqn import DecomposedDQN


# ---------- Concept dataset collection ----------

def collect_concept_states(
    concept_name: str,
    n_samples: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Collect state features where a concept is active.

    Concepts:
        near_hazard:   agent is within 1 cell of a hazard
        near_goal:     agent is within 2 cells of goal
        near_resource: agent is within 1 cell of uncollected resource
        open_space:    agent is >2 cells from any hazard

    Returns: (n_samples, state_dim) array
    """
    rng = np.random.RandomState(seed)
    env = MultiObjGridEnv(seed=seed)
    G = env.GRID_SIZE
    samples = []

    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 50:
        attempts += 1
        row = rng.randint(0, G)
        col = rng.randint(0, G)
        collected = rng.randint(0, 256)

        env.row, env.col, env.collected = row, col, collected
        neighbourhood = env.get_neighbourhood(radius=1)

        match = False
        if concept_name == "near_hazard" and "near_hazard" in neighbourhood:
            match = True
        elif concept_name == "near_goal":
            # within 2 cells of goal
            if abs(row - env.GOAL[0]) + abs(col - env.GOAL[1]) <= 2:
                match = True
        elif concept_name == "near_resource" and "near_resource" in neighbourhood:
            match = True
        elif concept_name == "open_space":
            # > 2 manhattan distance from all hazards
            min_dist = min(abs(row - hr) + abs(col - hc)
                          for hr, hc in env.HAZARDS)
            if min_dist > 2:
                match = True

        if match:
            feat = env.features_from_pos(row, col, collected)
            samples.append(feat)

    return np.array(samples[:n_samples], dtype=np.float32)


def collect_random_states(n_samples: int = 200, seed: int = 99) -> np.ndarray:
    """Collect uniformly random states as negative examples."""
    rng = np.random.RandomState(seed)
    env = MultiObjGridEnv()
    G = env.GRID_SIZE
    samples = []
    for _ in range(n_samples):
        row = rng.randint(0, G)
        col = rng.randint(0, G)
        collected = rng.randint(0, 256)
        samples.append(env.features_from_pos(row, col, collected))
    return np.array(samples, dtype=np.float32)


# ---------- CAV training ----------

def train_cav(
    model: DecomposedDQN,
    concept_states: np.ndarray,
    random_states: np.ndarray,
) -> np.ndarray:
    """
    Train a linear classifier to separate concept from random activations.
    Returns the CAV (normal vector of the decision boundary) as (64,) array.
    """
    # Get encoder features
    with torch.no_grad():
        concept_feats = model.encoder(
            torch.tensor(concept_states)).numpy()
        random_feats = model.encoder(
            torch.tensor(random_states[:len(concept_states)])).numpy()

    X = np.concatenate([concept_feats, random_feats], axis=0)
    y = np.concatenate([
        np.ones(len(concept_feats)),
        np.zeros(len(random_feats)),
    ])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)

    cav = clf.coef_[0]  # normal vector
    cav = cav / (np.linalg.norm(cav) + 1e-8)  # unit vector

    accuracy = clf.score(X, y)

    return cav, accuracy


# ---------- TCAV score ----------

def compute_tcav_score(
    model: DecomposedDQN,
    concept_states: np.ndarray,
    cav: np.ndarray,
    weights: Optional[list[float]] = None,
) -> float:
    """
    Compute TCAV score: fraction of concept states where the Q-value
    gradient w.r.t. encoder features aligns with the CAV direction.

    TCAV_score > 0.5 means the concept positively influences the Q-value.
    """
    w_tensor = torch.tensor(weights, dtype=torch.float32) if weights else None
    positive_count = 0

    for state in concept_states:
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        x.requires_grad_(False)

        # Need gradient through encoder
        features = model.encoder(x)
        features.retain_grad()

        # Compute Q_total for the greedy action
        q_channels = [head(features) for head in model.heads]
        if w_tensor is not None:
            q_total = sum(w * q for w, q in zip(w_tensor, q_channels))
        else:
            q_total = sum(q_channels)

        greedy_action = q_total.argmax(dim=1)
        q_value = q_total[0, greedy_action]

        # Backprop to get gradient w.r.t. features
        model.zero_grad()
        q_value.backward(retain_graph=False)

        if features.grad is not None:
            grad = features.grad.squeeze(0).numpy()
            # Directional derivative: dot product with CAV
            if np.dot(grad, cav) > 0:
                positive_count += 1

    return positive_count / len(concept_states)


# ---------- Full TCAV analysis ----------

CONCEPT_NAMES = ["near_hazard", "near_goal", "near_resource", "open_space"]


def run_tcav_analysis(
    model: DecomposedDQN,
    weights: list[float],
    n_samples: int = 150,
    seed: int = 42,
) -> dict:
    """
    Run full TCAV analysis for all concepts.

    Returns: {
        concept_name: {
            "tcav_score": float,
            "cav_accuracy": float,
            "cav": np.ndarray,
        }
    }
    """
    random_states = collect_random_states(n_samples=n_samples, seed=seed + 1000)
    results = {}

    for concept in CONCEPT_NAMES:
        concept_states = collect_concept_states(
            concept, n_samples=n_samples, seed=seed)

        if len(concept_states) < 20:
            print(f"  Warning: only {len(concept_states)} samples for '{concept}', skipping")
            continue

        cav, cav_acc = train_cav(model, concept_states, random_states)

        tcav_score = compute_tcav_score(
            model, concept_states[:50], cav, weights)

        results[concept] = {
            "tcav_score": tcav_score,
            "cav_accuracy": cav_acc,
            "cav": cav,
        }

    return results


if __name__ == "__main__":
    # Quick test with untrained model
    model = DecomposedDQN()
    results = run_tcav_analysis(model, weights=[1.0, 1.0, 1.0, 1.0], n_samples=100)
    for concept, info in results.items():
        print(f"  {concept:>15}: TCAV={info['tcav_score']:.3f}  "
              f"CAV_acc={info['cav_accuracy']:.3f}")
