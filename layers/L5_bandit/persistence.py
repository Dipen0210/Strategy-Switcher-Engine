# layers/L5_bandit/persistence.py
"""
Bandit Persistence Manager.

Organized folder structure:
layers/L5_bandit/data/
├── global_bandit.pkl           # Regime trust weights (Bandit A)
├── regime_bandits/             # Strategy ranking per regime (Bandit B)
│   ├── Bull-Quiet.pkl
│   ├── Bull-Volatile.pkl
│   ├── Sideways.pkl
│   └── Crisis.pkl
└── stock_bandits/              # Per-stock-per-regime preferences (Bandit C)
    ├── JNJ/
    │   ├── JNJ_Bull_Quiet.pkl
    │   └── JNJ_Crisis.pkl
    ├── NVDA/
    │   └── NVDA_Sideways.pkl
    └── ...
"""

from __future__ import annotations

import pickle
from pathlib import Path


# Default save directory - inside the L5_bandit module
DEFAULT_BANDIT_DIR = Path(__file__).parent / "data"


def get_bandit_dir(base: Path | None = None) -> Path:
    """
    Get the bandit state directory, creating it if needed.

    `base` lets a backtest point an isolated run at its own scratch directory
    so experiment runs never read or clobber the project's live learned state.
    """
    bandit_dir = Path(base) if base is not None else DEFAULT_BANDIT_DIR
    bandit_dir.mkdir(parents=True, exist_ok=True)
    return bandit_dir


def get_regime_bandits_dir(base: Path | None = None) -> Path:
    """Get the regime bandits subdirectory."""
    regime_dir = get_bandit_dir(base) / "regime_bandits"
    regime_dir.mkdir(parents=True, exist_ok=True)
    return regime_dir


def get_stock_bandits_dir(base: Path | None = None) -> Path:
    """Get the stock bandits subdirectory."""
    stock_dir = get_bandit_dir(base) / "stock_bandits"
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

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.global_bandit = None     # Bandit A
        self.regime_bandits = None    # Bandit B (RegimeBanditManager)
        self.stock_bandits = None     # Bandit C (StockBanditManager)

    @classmethod
    def load(cls, base_dir: Path | None = None) -> "BanditPersistenceManager":
        """
        Load all bandits from disk.

        Args:
            base_dir: optional isolated state directory. Backtests pass a
                per-run scratch path so experiments start from clean, seeded
                weights and never mutate the project's live learned state.
        """
        from layers.L5_bandit.global_bandit import GlobalBandit
        from layers.L5_bandit.regime_bandit import RegimeBanditManager
        from layers.L5_bandit.stock_bandit import StockBanditManager

        manager = cls(base_dir=base_dir)

        # Load Bandit A (Global)
        global_path = get_bandit_dir(base_dir) / "global_bandit.pkl"
        manager.global_bandit = GlobalBandit.load(global_path)

        # Load Bandit B (Regime)
        # NOTE: must go through .load() — constructing the manager directly
        # yields an EMPTY bandit dict, so every process start re-randomized
        # the regime weights and then overwrote the .pkl files on save_all().
        # Bandit B state did not survive restarts at all before this.
        manager.regime_bandits = RegimeBanditManager.load(get_regime_bandits_dir(base_dir))

        # Load Bandit C (Stock)
        manager.stock_bandits = StockBanditManager(save_dir=get_stock_bandits_dir(base_dir))

        print(f"📁 Loaded bandits from {get_bandit_dir(base_dir)}")
        return manager

    def save_all(self) -> None:
        """Save all bandits to disk."""
        bandit_dir = get_bandit_dir(self.base_dir)

        if self.global_bandit:
            self.global_bandit.save(bandit_dir / "global_bandit.pkl")

        if self.regime_bandits:
            self.regime_bandits.save_all()

        if self.stock_bandits:
            self.stock_bandits.save_all()

        print(f"💾 Saved all bandits to {bandit_dir}")

    def decay_all_bandits(self) -> None:
        """
        Decay arms across all 3 bandit levels — call ONCE per cycle.

        1. Bandit A (Global): all regime trust weights
        2. Bandit B (Global): all regime strategy rankings
        3. Bandit C (Per-Stock): every stock-regime model currently resident
           in memory.

        DECAY SEMANTICS FOR BANDIT C
        ----------------------------
        Bandit C models are loaded lazily, so only the stock-regime combos
        touched this session are resident and therefore decayed. This makes
        Bandit C decay "per active cycle" rather than "per calendar day":
        JNJ_Crisis does not decay while JNJ sits in Bull-Quiet.

        That is coherent — a Bandit C model's weights are only ever compared
        against each other, and all of its arms decay together, so its
        relative ranking is unaffected by cycles in which it was idle. But it
        does differ from a strict daily-decay reading. If calendar-day decay
        is required, each model needs a last-decayed timestamp and a catch-up
        applied on load.
        """
        if self.global_bandit:
            # Decay all global trust scores
            self.global_bandit.decay_all()

        if self.regime_bandits:
            # Decay all global regime-strategy scores
            self.regime_bandits.decay_all()

        if self.stock_bandits:
            # Decay resident per-stock-per-regime models
            self.stock_bandits.decay_all()

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
