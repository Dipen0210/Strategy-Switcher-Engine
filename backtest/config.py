# backtest/config.py
"""
Backtest configuration.

Every knob that affects a result lives here and is recorded with the run, so
a reported number can be traced back to the exact configuration that produced
it. Parameter SELECTION discipline (which window you are allowed to tune on)
is enforced by the split fields below.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "backtest" / "_cache"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"


@dataclass(frozen=True)
class BacktestConfig:
    """Fully specifies one backtest run."""

    # --- Universe & period -------------------------------------------------
    universe: tuple[str, ...] = ("WMT", "JNJ", "NVDA", "JPM", "NEE")
    start: str = "2015-01-01"
    end: str = "2024-12-31"

    # --- Capital & policy --------------------------------------------------
    initial_capital: float = 100_000.0
    risk_tolerance: str = "Medium"          # -> L0 RISK_LIMITS
    rebalance_frequency: str = "Weekly"     # Daily | Weekly | Monthly
    emergency_drawdown_threshold: float = 0.20

    # --- Transaction costs -------------------------------------------------
    # Reported results should sweep cost_bps rather than quote a single point.
    commission_per_trade: float = 0.0
    cost_bps: float = 5.0

    # --- Walk-forward HMM --------------------------------------------------
    # The model used to decide at date T is trained ONLY on data strictly
    # before its fold start. hmm_refit_months controls how often that model is
    # rebuilt; hmm_train_years is the rolling training window length.
    hmm_train_years: float = 10.0
    hmm_refit_months: int = 12
    hmm_min_train_samples: int = 250
    # True -> refit at EVERY rebalance date instead of every hmm_refit_months.
    # Strictest walk-forward reading; hmm_refit_months is then ignored.
    hmm_refit_at_rebalance: bool = False

    # --- Data window handed to the engine each cycle -----------------------
    # Must exceed the longest strategy TIMEFRAME (Death Cross = 200 trading
    # days) or that strategy silently scores 0. 500 calendar days ~ 345
    # trading days, which clears it with margin.
    lookback_days: int = 500

    # --- Market proxy ------------------------------------------------------
    # Fits the macro HMM and provides the index benchmark leg. Loaded
    # alongside the universe but NOT traded and NOT part of any equal-weight
    # baseline.
    macro_ticker: str = "SPY"

    # --- Data --------------------------------------------------------------
    # Dividend- and split-adjusted prices (total return). False reproduces the
    # earlier price-only behavior, which understates CAGR by ~2pp/yr and will
    # not reconcile against published index returns.
    use_adjusted_prices: bool = True

    # --- Performance -------------------------------------------------------
    use_garch: bool = False   # realized vol instead; see compute_all_features

    # --- Bandit hyperparameters (sensitivity sweeps) -----------------------
    # None -> the module defaults in layers.L5_bandit (delta=0.99, eps=0.10).
    # Set explicitly to sweep; applied per run via the bandit managers so
    # concurrent runs cannot contaminate each other.
    bandit_delta: float | None = None
    bandit_epsilon: float | None = None

    # --- Reproducibility ---------------------------------------------------
    seed: int = 0

    # --- Ablation ----------------------------------------------------------
    # Name from backtest.ablations.ABLATIONS. "full" is the complete engine.
    ablation: str = "full"

    # --- Paths -------------------------------------------------------------
    cache_dir: Path = DEFAULT_CACHE_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR

    # --- Experimental protocol (step 8) ------------------------------------
    # Parameter search is permitted ONLY inside the tuning window. The holdout
    # window must be run once, after parameters are frozen.
    tune_start: str = "2015-01-01"
    tune_end: str = "2020-12-31"
    holdout_start: str = "2021-01-01"
    holdout_end: str = "2024-12-31"

    def with_(self, **overrides) -> "BacktestConfig":
        """Return a copy with fields replaced (frozen dataclasses need this)."""
        data = {**asdict(self), **overrides}
        data["universe"] = tuple(data["universe"])
        data["cache_dir"] = Path(data["cache_dir"])
        data["results_dir"] = Path(data["results_dir"])
        return BacktestConfig(**data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["universe"] = list(self.universe)
        d["cache_dir"] = str(self.cache_dir)
        d["results_dir"] = str(self.results_dir)
        return d

    def fingerprint(self) -> str:
        """Short stable hash of the config, used to name result artifacts."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def label(self) -> str:
        return f"{self.ablation}_seed{self.seed}_{self.fingerprint()}"
