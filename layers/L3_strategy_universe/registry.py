# layers/L3_strategy_universe/registry.py
"""
LAYER 3 — STRATEGY UNIVERSE (YAML-Driven Registry)

Loads strategies dynamically from regime_config.yaml.
Supports the multi-strategy ensemble architecture.

All strategies output StrategyOutput (Signal + Confidence).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import yaml

from layers.L3_strategy_universe.base_strategy import StrategyOutput


# Path to YAML config
CONFIG_PATH = Path(__file__).parent / "regime_config.yaml"

# Type alias for strategy runner
StrategyRunner = Callable[[dict[str, pd.DataFrame]], list[StrategyOutput]]


@dataclass
class StrategySpec:
    """Metadata describing a strategy from YAML config."""
    name: str
    module: str
    function: str
    pod: str
    regime: str
    legacy: bool = False  # True for old strategies (need wrapper)
    timeframe: int = 30   # Strategy's lookback period in trading days
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pod": self.pod,
            "regime": self.regime,
            "module": self.module,
            "timeframe": self.timeframe,
        }


# Cache for loaded config and strategy runners
_config_cache: Optional[dict] = None
_runner_cache: Dict[str, StrategyRunner] = {}


def load_regime_config() -> dict:
    """Load and cache the regime configuration from YAML."""
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Regime config not found: {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r") as f:
        _config_cache = yaml.safe_load(f)
    
    return _config_cache


def get_strategies_for_regime(regime: str) -> List[StrategySpec]:
    """
    Get all strategies compatible with a regime.
    
    Args:
        regime: Regime name ("Bull-Quiet", "Trend + Low Vol", etc.)
    
    Returns:
        List of StrategySpec for all strategies in that regime
    """
    config = load_regime_config()
    
    # Handle naming variations
    regime_map = {
        "Trend + Low Vol": "Bull-Quiet",
        "Trend + High Vol": "Bull-Volatile",
        "Range + Low Vol": "Sideways",
        "Bear + High Vol": "Crisis",
    }
    normalized_regime = regime_map.get(regime, regime)
    
    if normalized_regime not in config.get("regimes", {}):
        return []
    
    regime_config = config["regimes"][normalized_regime]
    specs = []
    
    for pod_name, strategies in regime_config.get("pods", {}).items():
        for strategy in strategies:
            # Read TIMEFRAME from the strategy's module
            timeframe = 30  # default
            try:
                mod_path = strategy["module"]
                full_module = f"layers.L3_strategy_universe.{mod_path}"
                mod = importlib.import_module(full_module)
                timeframe = getattr(mod, "TIMEFRAME", 30)
            except Exception:
                pass
            
            specs.append(StrategySpec(
                name=strategy["name"],
                module=strategy["module"],
                function=strategy["function"],
                pod=pod_name,
                regime=normalized_regime,
                legacy=strategy.get("legacy", False),
                timeframe=timeframe,
            ))
    
    return specs


def get_strategy_runner(strategy_spec: StrategySpec) -> StrategyRunner:
    """
    Get the runner function for a strategy.
    
    Dynamically imports the module and returns the function.
    """
    cache_key = f"{strategy_spec.module}.{strategy_spec.function}"
    
    if cache_key in _runner_cache:
        return _runner_cache[cache_key]
    
    # Build full module path
    if "." in strategy_spec.module:
        # New-style: "trend.trend_following"
        full_module = f"layers.L3_strategy_universe.{strategy_spec.module}"
    else:
        # Legacy: "momentum"
        full_module = f"layers.L3_strategy_universe.{strategy_spec.module}"
    
    try:
        module = importlib.import_module(full_module)
        runner = getattr(module, strategy_spec.function)
        _runner_cache[cache_key] = runner
        return runner
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to load strategy '{strategy_spec.name}' from {full_module}.{strategy_spec.function}: {e}"
        )


def run_strategies_for_regime(
    regime: str,
    stock_data_dict: dict[str, pd.DataFrame],
    strategy_filter: Optional[List[str]] = None,
) -> List[StrategyOutput]:
    """
    Run all strategies for a regime and collect outputs.
    
    Args:
        regime: Current regime
        stock_data_dict: Dict of ticker -> OHLCV DataFrame
        strategy_filter: Optional list of strategy names to run (for L4.5 filtering)
    
    Returns:
        List of StrategyOutput from all strategies
    """
    specs = get_strategies_for_regime(regime)
    
    if strategy_filter:
        specs = [s for s in specs if s.name in strategy_filter]
    
    all_outputs = []
    
    for spec in specs:
        try:
            runner = get_strategy_runner(spec)
            outputs = runner(stock_data_dict)
            
            # Handle legacy strategies that return DataFrame instead of StrategyOutput
            if spec.legacy and isinstance(outputs, pd.DataFrame):
                # Ensure Ticker is the index
                if "Ticker" in outputs.columns:
                    outputs = outputs.set_index("Ticker")
                
                # Convert legacy output to StrategyOutput
                for ticker in stock_data_dict.keys():
                    if ticker in outputs.index:
                        # Handle different score column names
                        score = 0.5
                        if "score" in outputs.columns:
                            score = outputs.loc[ticker, "score"]
                        elif "Strategy_Score" in outputs.columns:
                            raw_score = outputs.loc[ticker, "Strategy_Score"]
                            # Normalize -1 to 1 range to 0 to 1
                            score = (raw_score + 1) / 2
                        
                        # Generate signal based on normalized score (0 to 1)
                        # > 0.6 = BUY, < 0.4 = SELL, else HOLD
                        signal = 1 if score > 0.6 else (-1 if score < 0.4 else 0)
                        confidence = abs(score - 0.5) * 2
                        
                        all_outputs.append(StrategyOutput(
                            ticker=ticker,
                            signal=signal,
                            confidence=confidence,
                            strategy_name=spec.name,
                            pod=spec.pod,
                            regime=spec.regime,
                        ))
            else:
                # New-style strategies return List[StrategyOutput]
                all_outputs.extend(outputs)
                
        except Exception as e:
            print(f"Warning: Strategy '{spec.name}' failed: {e}")
            continue
    
    return all_outputs


def get_all_strategy_names() -> List[str]:
    """Get names of all registered strategies across all regimes."""
    config = load_regime_config()
    names = set()
    
    for regime_name, regime_config in config.get("regimes", {}).items():
        for pod_name, strategies in regime_config.get("pods", {}).items():
            for strategy in strategies:
                names.add(strategy["name"])
    
    return sorted(names)


def get_pods_for_regime(regime: str) -> List[str]:
    """Get pod names for a regime."""
    config = load_regime_config()
    
    regime_map = {
        "Trend + Low Vol": "Bull-Quiet",
        "Trend + High Vol": "Bull-Volatile",
        "Range + Low Vol": "Sideways",
        "Bear + High Vol": "Crisis",
    }
    normalized = regime_map.get(regime, regime)
    
    if normalized not in config.get("regimes", {}):
        return []
    
    return list(config["regimes"][normalized].get("pods", {}).keys())


# Legacy compatibility - old STRATEGY_REGISTRY for backward compatibility
from layers.L3_strategy_universe.bull_quiet.momentum import run_momentum_strategy
from layers.L3_strategy_universe.bull_volatile.mean_reversion import run_mean_reversion_strategy
from layers.L3_strategy_universe.bull_quiet.breakout import run_breakout_strategy
from layers.L3_strategy_universe.crisis.defensive import run_defensive_strategy

STRATEGY_REGISTRY = {
    "Momentum": {"runner": run_momentum_strategy, "compatible_regimes": ("Trend + Low Vol", "Trend + High Vol")},
    "Mean Reversion": {"runner": run_mean_reversion_strategy, "compatible_regimes": ("Range + Low Vol",)},
    "Breakout": {"runner": run_breakout_strategy, "compatible_regimes": ("Trend + Low Vol", "Trend + High Vol")},
    "Defensive": {"runner": run_defensive_strategy, "compatible_regimes": ("Crisis",)},
}


def get_all_strategies() -> List[dict]:
    """Legacy: Get all strategies as dicts."""
    return [{"name": k, **v} for k, v in STRATEGY_REGISTRY.items()]


def get_all_strategy_dicts() -> List[dict]:
    """Legacy: Get all strategies as dicts for filtering."""
    return [
        {
            "name": name,
            "compatible_regimes": list(spec.get("compatible_regimes", ())),
        }
        for name, spec in STRATEGY_REGISTRY.items()
    ]


def get_strategy_names() -> List[str]:
    """Legacy: Get all strategy names."""
    return list(STRATEGY_REGISTRY.keys())


def _get_strategy_runner_legacy(strategy_name: str):
    """Legacy: Get strategy runner function by name (for old code)."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise NotImplementedError(f"Strategy '{strategy_name}' not found")
    return STRATEGY_REGISTRY[strategy_name]["runner"]


def get_strategy_spec(strategy_name: str) -> dict:
    """Legacy: Get strategy specification by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise NotImplementedError(f"Strategy '{strategy_name}' not found")
    return {"name": strategy_name, **STRATEGY_REGISTRY[strategy_name]}
