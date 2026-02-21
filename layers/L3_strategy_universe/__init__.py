# layers/L3_strategy_universe/__init__.py
"""Layer 3: Strategy Universe (YAML-Driven Multi-Strategy Ensemble)"""

from layers.L3_strategy_universe.registry import (
    STRATEGY_REGISTRY,
    StrategySpec,
    get_all_strategies,
    get_strategies_for_regime,
    run_strategies_for_regime,
    get_all_strategy_names,
    get_pods_for_regime,
    # Legacy functions for backward compatibility
    get_all_strategy_dicts,
    get_strategy_names,
    get_strategy_runner,
    get_strategy_spec,
)
from layers.L3_strategy_universe.base_strategy import StrategyOutput

__all__ = [
    "STRATEGY_REGISTRY",
    "StrategySpec",
    "StrategyOutput",
    "get_all_strategies",
    "get_strategies_for_regime",
    "run_strategies_for_regime",
    "get_all_strategy_names",
    "get_pods_for_regime",
    # Legacy
    "get_all_strategy_dicts",
    "get_strategy_names",
    "get_strategy_runner",
    "get_strategy_spec",
]
