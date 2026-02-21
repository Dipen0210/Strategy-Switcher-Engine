# layers/L5_bandit/global_bandit.py
"""
Bandit A — Global Regime Trust (Slow Learner)

Tracks how "trustworthy" each regime classification is via explicit weights.
Initializes with random unequal weights summing to 1.0.

Math: EXP3 (Exponential-weight) updates with slow temperature.
  weight *= exp(LR * return / temperature)
  Then normalize to sum to 1.0.

This is multiplicative — bad regimes decay exponentially, not linearly.
Prevents a single lucky return from dominating.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Default regimes
REGIMES = ["Bull-Quiet", "Bull-Volatile", "Sideways", "Crisis"]

# EXP3 hyperparameters for Global Bandit
LEARNING_RATE = 0.1     # Slow learner
TEMPERATURE = 0.05      # Controls exploration sensitivity
DECAY_FACTOR = 0.99     # Slowly pulls weights toward uniform over time


@dataclass
class GlobalBandit:
    """
    Master bandit that learns trustworthy regimes (Bandit A).
    Uses EXP3 multiplicative weight updates for stable learning.
    """
    regime_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.regime_weights:
            rands = np.random.rand(len(REGIMES))
            weights = rands / rands.sum()
            for i, r in enumerate(REGIMES):
                self.regime_weights[r] = float(weights[i])

    def _normalize(self) -> None:
        """Normalize weights to sum to exactly 1.0."""
        total = sum(self.regime_weights.values())
        if total > 0:
            for k in self.regime_weights:
                self.regime_weights[k] /= total
        else:
            n = len(self.regime_weights)
            if n > 0:
                for k in self.regime_weights:
                    self.regime_weights[k] = 1.0 / n

    def get_trust_weights(self) -> dict[str, float]:
        """Get the current explicit weights for all regimes."""
        return self.regime_weights

    def update_arm(self, regime: str, actual_return: float) -> None:
        """
        EXP3 update: weight *= exp(LR * return / temperature)
        
        Multiplicative update is more stable than additive:
        - Bad regimes decay exponentially (can't dominate from one lucky return)
        - Convergence is smoother over time
        """
        if regime in self.regime_weights:
            # EXP3: multiplicative update
            exponent = LEARNING_RATE * actual_return / TEMPERATURE
            # Clamp exponent to prevent overflow
            exponent = max(-5.0, min(5.0, exponent))
            self.regime_weights[regime] *= math.exp(exponent)
            self.regime_weights[regime] = max(1e-6, self.regime_weights[regime])
        self._normalize()

    def sample_trust(self, regime: str) -> float:
        """Legacy compatibility method."""
        return self.regime_weights.get(regime, 0.25)

    def decay_all(self) -> None:
        """
        Slowly forget old data by pulling weights slightly back toward a uniform distribution.
        This ensures no regime becomes permanently 0% or 100% just from ancient history.
        """
        if not self.regime_weights:
            return
            
        n = len(self.regime_weights)
        uniform_weight = 1.0 / n
        
        for k in self.regime_weights:
            # Pull current weight towards the uniform weight by the decay factor
            current = self.regime_weights[k]
            self.regime_weights[k] = (current * DECAY_FACTOR) + (uniform_weight * (1.0 - DECAY_FACTOR))
            
        self._normalize()

    def decay_and_update(self, regime: str, reward: float) -> None:
        self.decay_all()
        self.update_arm(regime, reward)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "GlobalBandit":
        if path.exists():
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                    if hasattr(obj, "regimes"):  # Legacy format
                        print("♻️ Migrating old GlobalBandit to EXP3 format")
                        return cls()
                    return obj
            except Exception as e:
                print(f"⚠️ Failed to load GlobalBandit: {e}")
        
        return cls()
