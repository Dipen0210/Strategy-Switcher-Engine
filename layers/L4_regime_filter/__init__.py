# layers/L4_regime_filter/__init__.py
"""
L4 Regime Filter - Filters strategies by pod weights from Global Bandit.

This layer uses learned pod preferences to prioritize strategies that belong
to pods that have historically performed well.

Usage:
    from layers.L4_regime_filter import filter_strategies_by_pod, PodFilterResult
    
    result = filter_strategies_by_pod(
        strategy_outputs=[...],
        pod_weights={'Trend': 0.6, 'Reversion': 0.3, ...},
        top_k=10,
    )
"""

from layers.L4_regime_filter.regime_filter import (
    filter_strategies_by_pod,
    get_pod_weights,
    PodFilterResult,
)

__all__ = [
    "filter_strategies_by_pod",
    "get_pod_weights",
    "PodFilterResult",
]
