# layers/L5_bandit/persistence.py
"""
Bandit Persistence Manager.

Organized folder structure:
layers/L5_bandit/data/
├── global_bandit.pkl           # Regime trust weights (Bandit A)
├── regime_bandits/             # Strategy ranking per regime (Bandit B)
│   ├── regime_bandit_bull_quiet.pkl
│   ├── regime_bandit_bull_volatile.pkl
│   ├── regime_bandit_sideways.pkl
│   └── regime_bandit_crisis.pkl
└── stock_bandits/              # Per-stock preferences (Bandit C)
    ├── stock_bandit_NVDA.pkl
    ├── stock_bandit_WMT.pkl
    └── ...
"""

from __future__ import annotations

import pickle
from pathlib import Path


# Default save directory - inside the L5_bandit module
DEFAULT_BANDIT_DIR = Path(__file__).parent / "data"


def get_bandit_dir() -> Path:
    """Get the bandit state directory, creating it if needed."""
    bandit_dir = DEFAULT_BANDIT_DIR
    bandit_dir.mkdir(parents=True, exist_ok=True)
    return bandit_dir


def get_regime_bandits_dir() -> Path:
    """Get the regime bandits subdirectory."""
    regime_dir = get_bandit_dir() / "regime_bandits"
    regime_dir.mkdir(parents=True, exist_ok=True)
    return regime_dir


def get_stock_bandits_dir() -> Path:
    """Get the stock bandits subdirectory."""
    stock_dir = get_bandit_dir() / "stock_bandits"
    stock_dir.mkdir(parents=True, exist_ok=True)
    return stock_dir


class BanditPersistenceManager:
    """
    Unified manager for all 3 bandit levels.

    Loads/saves:
        Bandit A (Global) → data/global_bandit.pkl
        Bandit B (Regime) → data/regime_bandits/*.pkl
        Bandit C (Stock)  → data/stock_bandits/*.pkl
    """

    def __init__(self):
        self.global_bandit = None     # Bandit A
        self.regime_bandits = None    # Bandit B (RegimeBanditManager)
        self.stock_bandits = None     # Bandit C (StockBanditManager)

    @classmethod
    def load(cls) -> "BanditPersistenceManager":
        """Load all bandits from disk."""
        from layers.L5_bandit.global_bandit import GlobalBandit
        from layers.L5_bandit.regime_bandit import RegimeBanditManager
        from layers.L5_bandit.stock_bandit import StockBanditManager

        manager = cls()

        # Load Bandit A (Global)
        global_path = get_bandit_dir() / "global_bandit.pkl"
        manager.global_bandit = GlobalBandit.load(global_path)

        # Load Bandit B (Regime)
        manager.regime_bandits = RegimeBanditManager(save_dir=get_regime_bandits_dir())

        # Load Bandit C (Stock)
        manager.stock_bandits = StockBanditManager(save_dir=get_stock_bandits_dir())

        print(f"📁 Loaded bandits from {get_bandit_dir()}")
        return manager

    def save_all(self) -> None:
        """Save all bandits to disk."""
        bandit_dir = get_bandit_dir()

        if self.global_bandit:
            self.global_bandit.save(bandit_dir / "global_bandit.pkl")

        if self.regime_bandits:
            self.regime_bandits.save_all()

        if self.stock_bandits:
            self.stock_bandits.save_all()

        print(f"💾 Saved all bandits to {bandit_dir}")

    def decay_all_bandits(self) -> None:
        """
        Decay ALL arms across all 3 bandit levels — call ONCE per cycle.
        
        Decays:
        1. Bandit A (Global): All regime trust weights
        2. Bandit B (Global): All regime strategy rankings
        3. Bandit C (Per-Stock): Handled per-stock during update, but we can't easily iterate all.
           Actually, Bandit C is usually decayed when we access/update it.
        """
        if self.global_bandit:
            # Decay all global trust scores
            self.global_bandit.decay_all()

        if self.regime_bandits:
            # Decay all global regime-strategy scores
            self.regime_bandits.decay_all()

    def update_arm(
        self,
        ticker: str,
        regime: str,
        strategy_name: str,
        rewards: dict[str, float],
    ) -> None:
        """
        Distribute rewards to the 3-level bandit hierarchy.
        
        Args:
            ticker: Stock symbol (for Bandit C)
            regime: Current market regime (for Bandit A & B)
            strategy_name: Strategy executed
            rewards: Dictionary of rewards keys "A", "B", "C"
        """
        # 1. Bandit A: Global Regime Trust
        # Learns: "Is 'Bull-Volatile' generally trustworthy right now?"
        if self.global_bandit:
            self.global_bandit.update_arm(regime, rewards["A"])

        # 2. Bandit B: Global Strategy Ranking per Regime
        # Learns: "Is 'TrendFollowing' good in 'Bull-Volatile'?"
        if self.regime_bandits:
            self.regime_bandits.update_strategy(regime, strategy_name, rewards["B"])

        # 3. Bandit C: Stock-Specific Preference (per-stock-per-regime)
        # Learns: "Does AAPL prefer 'MeanReversion' in this specific regime?"
        if self.stock_bandits:
            self.stock_bandits.update_strategy(ticker, strategy_name, rewards["C"], regime=regime)

    def update_from_trade(
        self,
        ticker: str,
        regime: str,
        strategy_name: str,
        rewards: dict[str, float],
    ) -> None:
        """
        Legacy: Decay + update combined. ⚠️ Over-decays with multi-stock.
        For multi-stock, use decay_all_bandits() once + update_arm() per stock.
        """
        if self.global_bandit:
            # Replicates decay + update for single stock flow
            self.global_bandit.decay_all()
            self.global_bandit.update_arm(ticker, regime, rewards["A"])

        if self.regime_bandits:
            self.regime_bandits.decay_all()
            self.regime_bandits.update_strategy(ticker, regime, strategy_name, rewards["B"])

        if self.stock_bandits:
            self.stock_bandits.decay_and_update(ticker, strategy_name, rewards["C"])

    def get_stats(self) -> dict:
        """Get statistics about all bandits."""
        stats = {
            "global_tickers": 0,
            "regime_tickers": 0,
            "stock_bandits": 0,
            "stock_tickers": [],
        }

        if self.global_bandit:
            stats["global_tickers"] = len(self.global_bandit.tickers)

        if self.regime_bandits:
            stats["regime_tickers"] = len(self.regime_bandits.bandits)

        if self.stock_bandits:
            stats["stock_bandits"] = len(self.stock_bandits.bandits)
            stats["stock_tickers"] = list(self.stock_bandits.bandits.keys())

        return stats
