# layers/L4_pod_filter/pod_filter.py
"""
Pod Filter - Filters strategies by learned pod weights.

This replaces the old L4_constraints_filter with a simpler, bandit-driven approach:
1. Get pod weights from GlobalBandit (learned from historical performance)
2. Prioritize strategies from high-weight pods
3. Flter to top-K strategies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from layers.L3_strategy_universe import StrategyOutput


@dataclass
class PodFilterResult:
    """Result from pod-based filtering."""
    filtered_strategies: List["StrategyOutput"]
    pod_weights: Dict[str, float]
    original_count: int
    filtered_count: int


def get_pod_weights(global_bandit) -> Dict[str, float]:
    """
    Get exploitation weights from the GlobalBandit.
    
    Returns a dict mapping pod name → weight (0-1).
    Higher weight = pod has performed better historically.
    """
    if global_bandit is None:
        # Default equal weights
        return {
            "Bull-Quiet": 0.25,
            "Bull-Volatile": 0.25,
            "Sideways": 0.25,
            "Crisis": 0.25,
            "Others": 0.0,
        }
    
    return global_bandit.get_exploitation_weights()


def filter_strategies_by_pod(
    strategy_outputs: List["StrategyOutput"],
    pod_weights: Dict[str, float],
    top_k: Optional[int] = None,
    min_weight: float = 0.05,
) -> PodFilterResult:
    """
    Filter strategies based on their pod's learned weight.
    
    Args:
        strategy_outputs: List of StrategyOutput from running strategies
        pod_weights: Dict mapping pod → weight (from GlobalBandit)
        top_k: If set, only keep top K strategies by pod weight
        min_weight: Minimum pod weight to include (default 0.05)
    
    Returns:
        PodFilterResult with filtered strategies and metadata
    """
    original_count = len(strategy_outputs)
    
    if not strategy_outputs:
        return PodFilterResult(
            filtered_strategies=[],
            pod_weights=pod_weights,
            original_count=0,
            filtered_count=0,
        )
    
    # Score each strategy by its pod's weight
    scored = []
    for output in strategy_outputs:
        pod = output.pod or "Bull-Quiet"
        weight = pod_weights.get(pod, 0.1)
        
        # Skip if pod weight is below minimum
        if weight < min_weight:
            continue
            
        scored.append((weight, output))
    
    # Sort by weight (descending)
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Apply top-K filter if specified
    if top_k is not None and len(scored) > top_k:
        scored = scored[:top_k]
    
    filtered = [output for _, output in scored]
    
    return PodFilterResult(
        filtered_strategies=filtered,
        pod_weights=pod_weights,
        original_count=original_count,
        filtered_count=len(filtered),
    )


def proportional_pod_selection(
    strategy_outputs: List["StrategyOutput"],
    pod_weights: Dict[str, float],
    target_count: int = 10,
) -> List["StrategyOutput"]:
    """
    Select strategies proportionally based on pod weights.
    
    Example: If Trend has weight 0.6 and Reversion 0.4, and target_count=10,
    select ~6 Trend strategies and ~4 Reversion strategies.
    
    Args:
        strategy_outputs: All strategy outputs
        pod_weights: Learned pod weights
        target_count: Total strategies to select
    
    Returns:
        List of selected StrategyOutput
    """
    if not strategy_outputs:
        return []
    
    # Group strategies by pod
    by_pod: Dict[str, List["StrategyOutput"]] = {}
    for output in strategy_outputs:
        pod = output.pod or "Bull-Quiet"
        if pod not in by_pod:
            by_pod[pod] = []
        by_pod[pod].append(output)
    
    # Calculate proportional allocation
    total_weight = sum(pod_weights.get(pod, 0.1) for pod in by_pod.keys())
    if total_weight == 0:
        total_weight = 1.0
    
    selected = []
    for pod, strategies in by_pod.items():
        weight = pod_weights.get(pod, 0.1)
        allocation = int(round(target_count * weight / total_weight))
        allocation = min(allocation, len(strategies))  # Can't select more than available
        
        # Sort by confidence (descending) and take top N
        strategies.sort(key=lambda x: x.confidence, reverse=True)
        selected.extend(strategies[:allocation])
    
    return selected
