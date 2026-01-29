# layers/L3_strategy_universe/registry.py
"""
LAYER 3 — STRATEGY UNIVERSE (STATIC ACTION SPACE)

Define all possible actions in advance.

Each strategy must be:
- Deterministic
- Backtestable
- Asset-agnostic
- Risk-profiled (Low / Medium / High)

Strategies are FIXED. The system learns WHEN to use them, not how to invent them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd

from layers.L3_strategy_universe.momentum import run_momentum_strategy
from layers.L3_strategy_universe.mean_reversion import run_mean_reversion_strategy
from layers.L3_strategy_universe.breakout import run_breakout_strategy
from layers.L3_strategy_universe.defensive import run_defensive_strategy

StrategyRunner = Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


@dataclass
class StrategySpec:
    """Metadata describing a portfolio construction strategy."""
    name: str
    runner: StrategyRunner
    lookback_window: int
    risk_level: str = "Medium"
    expected_vol: float = 0.15
    max_drawdown: float = 0.10
    compatible_regimes: tuple = ()
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "risk_level": self.risk_level,
            "expected_vol": self.expected_vol,
            "max_drawdown": self.max_drawdown,
            "compatible_regimes": list(self.compatible_regimes),
            "lookback_window": self.lookback_window,
        }


STRATEGY_REGISTRY: Dict[str, StrategySpec] = {
    "Momentum": StrategySpec(
        name="Momentum",
        runner=run_momentum_strategy,
        lookback_window=60,
        risk_level="Medium",
        expected_vol=0.18,
        max_drawdown=0.12,
        compatible_regimes=("Trend + Low Vol", "Trend + High Vol"),
    ),
    "Mean Reversion": StrategySpec(
        name="Mean Reversion",
        runner=run_mean_reversion_strategy,
        lookback_window=120,
        risk_level="Low",
        expected_vol=0.10,
        max_drawdown=0.06,
        compatible_regimes=("Range + Low Vol", "Trend + Low Vol"),
    ),
    "Breakout": StrategySpec(
        name="Breakout",
        runner=run_breakout_strategy,
        lookback_window=60,
        risk_level="High",
        expected_vol=0.22,
        max_drawdown=0.18,
        compatible_regimes=("Trend + Low Vol", "Trend + High Vol"),
    ),
    "Defensive": StrategySpec(
        name="Defensive",
        runner=run_defensive_strategy,
        lookback_window=60,
        risk_level="Low",
        expected_vol=0.06,
        max_drawdown=0.04,
        compatible_regimes=("Range + Low Vol", "Trend + Low Vol", "Trend + High Vol", "Crisis"),
    ),
}


def get_strategy_spec(strategy_name: str) -> StrategySpec:
    """Get strategy specification by name."""
    normalized = strategy_name.strip() if strategy_name else ""
    spec = STRATEGY_REGISTRY.get(normalized)
    if not spec:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise NotImplementedError(
            f"Strategy '{strategy_name}' not found. Available: {available}"
        )
    return spec


def get_strategy_runner(strategy_name: str) -> StrategyRunner:
    """Get strategy runner function."""
    return get_strategy_spec(strategy_name).runner


def get_all_strategies() -> List[StrategySpec]:
    """Get all registered strategies."""
    return list(STRATEGY_REGISTRY.values())


def get_all_strategy_dicts() -> List[dict]:
    """Get all strategies as dicts for filtering."""
    return [spec.to_dict() for spec in STRATEGY_REGISTRY.values()]


def get_strategy_names() -> List[str]:
    """Get all strategy names."""
    return list(STRATEGY_REGISTRY.keys())
