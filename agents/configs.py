"""
Agent weight profiles for the 4-channel reward decomposition.

Channels: [goal, safety, coin, step]
Each profile creates a qualitatively different behavioural style.
"""

# Personality axis = navigation (goal) vs collection (coin). Safety is a SHARED
# concern (all avoid hazards), not a distinguishing dimension — in this grid the
# shortest path is also the safest, so a safety/speed axis produces no behavioural
# trade-off. Goal-vs-coin DOES create a real trade-off (detour for coins vs go
# straight), so the agents are genuinely distinguishable here.
PROFILES = {
    "navigator": {
        "name": "Agent α (navigator)",
        "weights": [3.0, 1.0, 0.0, 1.0],  # goal-driven, ignores coins
        "n_step": 3,  # n=1 let the high-w_goal channel diverge in Q magnitude; n=3 keeps it bounded
        "description": "Heads straight for the goal; ignores resources",
        "color": "#2980b9",
    },
    "collector": {
        "name": "Agent β (collector)",
        "weights": [1.2, 1.0, 5.0, 0.2],  # travels toward goal but detours hard for coins
        "n_step": 3,
        "description": "Detours to collect resources on its way to the goal",
        "color": "#e74c3c",
    },
    "balanced": {
        "name": "Agent γ (balanced)",
        "weights": [1.5, 1.0, 1.5, 0.4],  # collects some, still reaches the goal
        "n_step": 3,
        "description": "Collects some resources while still reaching the goal",
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
