"""
Decomposed DQN: shared encoder + per-channel Q-heads.

Architecture:
    Input (state_dim=10: row/9, col/9, bit0..bit7)
        → SharedEncoder (Linear→ReLU→Linear→ReLU)
        → features (64-dim)  ← TCAV probes this layer
        → Q_goal(s, a)       ← head 0
        → Q_safety(s, a)     ← head 1
        → Q_coin(s, a)       ← head 2
        → Q_step(s, a)       ← head 3
        → Q_total = Σ w_i · Q_i
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class DecomposedDQN(nn.Module):
    """Multi-head DQN with shared encoder and per-channel Q-value heads."""

    def __init__(
        self,
        state_dim: int = 10,
        n_actions: int = 4,
        n_channels: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_channels = n_channels

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )

        # Per-channel Q-heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_actions),
            )
            for _ in range(n_channels)
        ])

    def forward(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (batch, state_dim) normalised state features
            weights: (n_channels,) reward channel weights, default all-ones

        Returns:
            q_total:    (batch, n_actions) weighted sum of Q-heads
            q_channels: list of (batch, n_actions), one per channel
            features:   (batch, 64) shared encoder output (for TCAV)
        """
        features = self.encoder(x)

        q_channels = [head(features) for head in self.heads]

        if weights is None:
            weights = torch.ones(self.n_channels, device=x.device)

        q_total = sum(w * q for w, q in zip(weights, q_channels))

        return q_total, q_channels, features

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder features only (for TCAV probing)."""
        with torch.no_grad():
            return self.encoder(x)

    def select_action(
        self,
        state: np.ndarray,
        weights: Optional[List[float]] = None,
        epsilon: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> int:
        """Epsilon-greedy action selection."""
        if rng is None:
            rng = np.random.RandomState()

        if rng.rand() < epsilon:
            return rng.randint(0, self.n_actions)

        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            w = torch.tensor(weights, dtype=torch.float32) if weights else None
            q_total, _, _ = self.forward(x, w)
            return int(q_total.argmax(dim=1).item())

    def get_decomposed_q(
        self,
        state: np.ndarray,
        weights: Optional[List[float]] = None,
    ) -> dict:
        """
        Get decomposed Q-values for a single state.
        Returns dict with 'total', 'channels', 'features'.
        """
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            w = torch.tensor(weights, dtype=torch.float32) if weights else None
            q_total, q_channels, features = self.forward(x, w)

        return {
            "total": q_total.squeeze(0).numpy(),           # (n_actions,)
            "channels": [q.squeeze(0).numpy() for q in q_channels],  # list of (n_actions,)
            "features": features.squeeze(0).numpy(),        # (64,)
        }


if __name__ == "__main__":
    model = DecomposedDQN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.randn(8, 3)
    q_total, q_ch, feat = model(dummy, torch.tensor([1.0, 2.0, 0.5, 0.5]))
    print(f"q_total: {q_total.shape}")
    print(f"q_channels: {[q.shape for q in q_ch]}")
    print(f"features: {feat.shape}")
