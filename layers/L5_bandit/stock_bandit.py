# layers/L5_bandit/stock_bandit.py
"""
Bandit C — Stock-Specific Adjustment (Fast Learner)

Per-Stock-Per-Regime models: each stock+regime combo gets its own bandit.
E.g., JNJ_Crisis.pkl only tracks the ~10 Crisis strategies,
     JNJ_Sideways.pkl only tracks the ~10 Sideways strategies.

Math: EXP3 multiplicative updates with harsh temperature.
  weight *= exp(LR * return / temperature)
  LR = 1.0 (harsh), temperature = 0.15

Fast adaptation to stock-level performance.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# EXP3 hyperparameters for Stock Bandit
LEARNING_RATE = 1.0     # Harsh learner
TEMPERATURE = 0.15      # Controls update sensitivity
DECAY_FACTOR = 0.99     # Slowly pulls weights toward uniform over time


@dataclass
class StockBandit:
    """
    Manages explicit strategy weights for a specific stock IN A SPECIFIC REGIME.
    Sum of weights equals 1.0.
    Max ~10 strategies per model → concentrated weights.
    """
    ticker: str
    regime: str = ""
    strategies: dict[str, float] = field(default_factory=dict)

    def _ensure_strategies(self, strategy_names: list[str]) -> None:
        """Ensure all candidates are initialized with random unequal weights."""
        missing = [name for name in strategy_names if name not in self.strategies]
        if missing:
            if not self.strategies:
                # First time: random unequal weights summing to 1.0
                rands = np.random.rand(len(missing))
                weights = rands / rands.sum()
                for i, name in enumerate(missing):
                    self.strategies[name] = float(weights[i])
            else:
                for name in missing:
                    self.strategies[name] = 0.1
                self._normalize()

    def _normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
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
        """Return the current learned explicit weight for the strategy."""
        return self.strategies.get(strategy_name, 0.0)

    def decay_all(self) -> None:
        """
        Slowly forget stock-specific history by pulling weights toward a uniform distribution.
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
        Harsh learner (LR=1.0) — fast adaptation at stock level.
        """
        if strategy_name in self.strategies:
            exponent = LEARNING_RATE * actual_return / TEMPERATURE
            exponent = max(-5.0, min(5.0, exponent))
            self.strategies[strategy_name] *= math.exp(exponent)
            self.strategies[strategy_name] = max(1e-6, self.strategies[strategy_name])
        self._normalize()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "StockBandit":
        if path.exists():
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                    if isinstance(obj, cls):
                        return obj
                    # Legacy migration: old format without regime
                    if hasattr(obj, "strategies") and hasattr(obj, "ticker"):
                        print(f"♻️ Migrating old StockBandit {path.name}")
                        return cls(ticker=obj.ticker, regime="")
            except Exception as e:
                print(f"⚠️ Failed to load StockBandit: {e}")
        return cls(ticker="UNKNOWN", regime="")


class StockBanditManager:
    """
    Manages per-stock-per-regime strategy bandits.
    
    Key format: "TICKER_REGIME" (e.g., "JNJ_Crisis", "NVDA_Bull-Quiet")
    Each combo has its own model with max ~10 strategies.
    
    File structure:
        stock_bandits/
        ├── JNJ/
        │   ├── JNJ_Bull_Quiet.pkl
        │   └── JNJ_Crisis.pkl
        ├── NVDA/
        │   └── NVDA_Sideways.pkl
        └── ...
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path(".")
        self.bandits: dict[str, StockBandit] = {}

    def _make_key(self, ticker: str, regime: str) -> str:
        """Create compound key: TICKER_REGIME."""
        return f"{ticker.upper()}_{regime}"

    def _safe_filename(self, key: str) -> str:
        """Convert key to safe filename."""
        return key.replace("/", "_").replace(".", "_").replace(" ", "-")

    def get_bandit(self, ticker: str, regime: str = "") -> StockBandit:
        """Get or lazily load/create a stock-regime bandit."""
        key = self._make_key(ticker, regime)

        if key not in self.bandits:
            safe_ticker = ticker.upper().replace("/", "_").replace(".", "_")
            safe_regime = regime.replace("/", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
            path = self.save_dir / safe_ticker / f"{safe_ticker}_{safe_regime}.pkl"
            
            if path.exists():
                bandit = StockBandit.load(path)
                bandit.ticker = ticker.upper()
                bandit.regime = regime
                self.bandits[key] = bandit
            else:
                self.bandits[key] = StockBandit(ticker=ticker.upper(), regime=regime)

        return self.bandits[key]

    def sample(self, ticker: str, strategy_name: str, regime: str = "") -> float:
        """Get the stock-regime-specific weight for a strategy."""
        bandit = self.get_bandit(ticker, regime)
        return bandit.sample(strategy_name)

    def update_strategy(self, ticker: str, strategy_name: str, reward: float, regime: str = "") -> None:
        """Update a specific strategy for a stock in a specific regime (Harsh update)."""
        self.get_bandit(ticker, regime).update_arm(strategy_name, reward)

    def decay_all(self) -> None:
        """Decay all models to slowly forget old history."""
        for bandit in self.bandits.values():
            bandit.decay_all()

    def save_all(self) -> None:
        if not self.save_dir:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        for key, bandit in self.bandits.items():
            safe_ticker = bandit.ticker.replace("/", "_").replace(".", "_")
            safe_regime = bandit.regime.replace("/", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
            ticker_dir = self.save_dir / safe_ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            path = ticker_dir / f"{safe_ticker}_{safe_regime}.pkl"
            try:
                bandit.save(path)
            except Exception as e:
                print(f"⚠️ Failed to save StockBandit {key}: {e}")
