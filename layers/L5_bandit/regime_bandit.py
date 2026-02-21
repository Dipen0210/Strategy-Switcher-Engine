# layers/L5_bandit/regime_bandit.py
"""
Bandit B — Strategy Ranking per Regime (Medium Learner)

Each regime has its own bandit that learns which strategies work best.
Initializes with random unequal weights summing to 1.0.

Math:
  - EXP3 multiplicative weight updates (LR=0.5, temp=0.1)
  - ε-greedy exploration: 10% of the time, one random strategy
    replaces the lowest-ranked in the Top 5. This prevents the
    system from permanently ignoring potentially improved strategies.
"""

from __future__ import annotations

import pickle
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


REGIMES = ["Bull-Quiet", "Bull-Volatile", "Sideways", "Crisis"]

REGIME_NAME_MAP = {
    "Trend + Low Vol": "Bull-Quiet",
    "Trend + High Vol": "Bull-Volatile",
    "Range + Low Vol": "Sideways",
    "Bear + High Vol": "Crisis",
}

# EXP3 hyperparameters for Strategy Bandit
LEARNING_RATE = 0.5     # Mild learner
TEMPERATURE = 0.1       # Controls update sensitivity
EPSILON = 0.10          # 10% exploration rate
DECAY_FACTOR = 0.99     # Slowly pulls weights toward uniform over time


@dataclass
class RegimeBandit:
    """
    Regime-specific strategy bandit using EXP3 + ε-greedy exploration.
    """
    regime: str
    strategies: dict[str, float] = field(default_factory=dict)

    def _ensure_strategies(self, strategy_names: list[str]) -> None:
        """Ensure all strategies are initialized."""
        missing = [name for name in strategy_names if name not in self.strategies]
        if missing:
            if not self.strategies:
                rands = np.random.rand(len(missing))
                weights = rands / rands.sum()
                for i, name in enumerate(missing):
                    self.strategies[name] = float(weights[i])
            else:
                for name in missing:
                    self.strategies[name] = 0.1
                self._normalize()

    def _normalize(self) -> None:
        """Normalize weights to sum to exactly 1.0."""
        total = sum(self.strategies.values())
        if total > 0:
            for k in self.strategies:
                self.strategies[k] /= total
        else:
            n = len(self.strategies)
            if n > 0:
                for k in self.strategies:
                    self.strategies[k] = 1.0 / n

    def sample(self, strategy_name: str) -> float:
        """Return the current learned weight for the strategy."""
        return self.strategies.get(strategy_name, 0.0)

    def rank_strategies(self, strategy_names: list[str]) -> list[tuple[str, float]]:
        """
        Return the Top 5 strategies with ε-greedy exploration.
        
        90% of the time: pure exploitation (top 5 by weight)
        10% of the time: replace the lowest-ranked in top 5 with a 
                         random strategy from the remaining pool.
        """
        self._ensure_strategies(strategy_names)
        ranked = [(name, self.strategies[name]) for name in strategy_names]
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        top_5 = ranked[:5]
        remaining = ranked[5:]
        
        # ε-greedy: with probability EPSILON, swap the weakest in top 5
        # with a random strategy from the remaining pool
        if remaining and random.random() < EPSILON:
            explore_pick = random.choice(remaining)
            top_5[-1] = explore_pick  # Replace weakest with random explore
        
        return top_5

    def decay_all(self) -> None:
        """
        Slowly forget old data by pulling weights slightly back toward a uniform distribution.
        This ensures no strategy becomes permanently 0% or 100% just from ancient history.
        """
        if not self.strategies:
            return
            
        n = len(self.strategies)
        uniform_weight = 1.0 / n
        
        for k in self.strategies:
            current = self.strategies[k]
            self.strategies[k] = (current * DECAY_FACTOR) + (uniform_weight * (1.0 - DECAY_FACTOR))
            
        self._normalize()

    def update_arm(self, strategy_name: str, actual_return: float) -> None:
        """
        EXP3 update: weight *= exp(LR * return / temperature)
        
        Multiplicative update prevents single outlier returns from
        dominating. Bad strategies decay exponentially.
        """
        if strategy_name in self.strategies:
            exponent = LEARNING_RATE * actual_return / TEMPERATURE
            exponent = max(-5.0, min(5.0, exponent))
            self.strategies[strategy_name] *= math.exp(exponent)
            self.strategies[strategy_name] = max(1e-6, self.strategies[strategy_name])
        self._normalize()

    def decay_and_update(self, strategy_name: str, reward: float) -> None:
        """Combined decay then update wrapper."""
        self.decay_all()
        self.update_arm(strategy_name, reward)

    def get_all_weights(self, strategy_names: list[str]) -> dict[str, float]:
        """Get exploitation weights for strategies."""
        self._ensure_strategies(strategy_names)
        return {name: self.strategies[name] for name in strategy_names}


class RegimeBanditManager:
    """Manages per-regime strategy bandits."""

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path(".")
        self.bandits: dict[str, RegimeBandit] = {}

    def get_bandit(self, regime: str) -> RegimeBandit:
        """Get or create the bandit for a specific regime."""
        normalized_regime = REGIME_NAME_MAP.get(regime, regime)

        if normalized_regime not in self.bandits:
            self.bandits[normalized_regime] = RegimeBandit(regime=normalized_regime)

        return self.bandits[normalized_regime]

    def rank_strategies(self, regime: str, strategy_names: list[str]) -> list[tuple[str, float]]:
        return self.get_bandit(regime).rank_strategies(strategy_names)

    def update_strategy(self, regime: str, strategy_name: str, reward: float) -> None:
        self.get_bandit(regime).update_arm(strategy_name, reward)

    def sample(self, regime: str, strategy_name: str) -> float:
        return self.get_bandit(regime).sample(strategy_name)

    def decay_all(self) -> None:
        for bandit in self.bandits.values():
            bandit.decay_all()

    def save_all(self) -> None:
        if not self.save_dir:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        for regime_name, bandit in self.bandits.items():
            path = self.save_dir / f"{regime_name}.pkl"
            try:
                with open(path, "wb") as f:
                    pickle.dump(bandit, f)
            except Exception as e:
                print(f"⚠️ Failed to save bandit {regime_name}: {e}")

    @classmethod
    def load(cls, save_dir: Path) -> "RegimeBanditManager":
        manager = cls(save_dir=save_dir)
        if save_dir.exists():
            for p in save_dir.glob("*.pkl"):
                try:
                    with open(p, "rb") as f:
                        bandit = pickle.load(f)
                        manager.bandits[bandit.regime] = bandit
                except Exception:
                    pass
        return manager
