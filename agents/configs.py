"""
Agent weight profiles for the 4-channel reward decomposition.

Channels: [goal, safety, coin, step]
Each profile creates a qualitatively different behavioural style.
"""

PROFILES = {
    "safety_first": {
        "name": "Agent α (safety-first)",
        "weights": [1.0, 3.0, 0.3, 0.5],
        "n_step": 5,  # high w_safety=3.0 needs longer horizon to ground the safety channel
        "description": "Avoids hazards aggressively, sacrifices speed and coins",
        "color": "#2980b9",
    },
    "speed_first": {
        "name": "Agent β (speed-first)",
        "weights": [2.0, 0.8, 0.3, 2.0],
        "n_step": 1,  # high w_step=2.0: large n over-amplifies step signal → premature suicide policy
        "description": "Reaches goal fast, tolerates risk, ignores coins",
        "color": "#e74c3c",
    },
    "balanced": {
        "name": "Agent γ (balanced)",
        "weights": [1.0, 1.5, 1.0, 0.8],
        "n_step": 3,  # balanced profile: n=3 converges to 100% and stays stable
        "description": "Moderate risk aversion with coin collection",
        "color": "#f39c12",
    },
}

# Hyperparameters (carried over from assignment)
TRAIN_CONFIG = {
    "n_episodes": 3000,
    "lr": 1e-3,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_min": 0.01,
    "eps_decay": 0.995,
    "batch_size": 64,
    "buffer_size": 50000,
    "target_update_freq": 500,  # steps between target net sync
    "hidden_dim": 128,
    "n_step": 3,  # n-step return horizon (reduces single-step bootstrap reliance / deadly triad)
    "seed": 42,
}

REWARD_CHANNELS = ["goal", "safety", "coin", "step"]
