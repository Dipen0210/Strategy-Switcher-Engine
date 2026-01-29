# risk/risk_filter.py
"""
Hard Constraint Risk Filter for SUP Flow 1.

Implements NON-NEGOTIABLE risk constraints:
1. User risk tolerance (Low/Medium/High) → max vol, max DD limits
2. Regime-strategy compatibility

Any strategy violating constraints is REMOVED before ML or ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Risk tolerance limits - HARD constraints
RISK_TOLERANCE_LIMITS = {
    "Low": {
        "max_vol": 0.08,     # 8% annualized volatility
        "max_drawdown": 0.05  # 5% max drawdown
    },
    "Medium": {
        "max_vol": 0.15,     # 15% annualized volatility
        "max_drawdown": 0.10  # 10% max drawdown
    },
    "High": {
        "max_vol": 0.25,     # 25% annualized volatility
        "max_drawdown": 0.20  # 20% max drawdown
    }
}


# Strategy risk profiles (must match registry)
STRATEGY_RISK_PROFILES = {
    "Momentum": {
        "risk_level": "Medium",
        "expected_vol": 0.18,
        "max_drawdown": 0.12,
        "compatible_regimes": ["Trend + Low Vol", "Trend + High Vol"]
    },
    "Mean Reversion": {
        "risk_level": "Low",
        "expected_vol": 0.10,
        "max_drawdown": 0.06,
        "compatible_regimes": ["Range + Low Vol", "Trend + Low Vol"]
    },
    "Breakout": {
        "risk_level": "High",
        "expected_vol": 0.22,
        "max_drawdown": 0.18,
        "compatible_regimes": ["Trend + Low Vol", "Trend + High Vol"]
    },
    "Defensive": {
        "risk_level": "Low",
        "expected_vol": 0.06,
        "max_drawdown": 0.04,
        "compatible_regimes": ["Range + Low Vol", "Trend + Low Vol", "Trend + High Vol", "Crisis"]
    }
}


@dataclass
class FilterResult:
    """Result of filtering operation."""
    allowed_strategies: list[dict]
    removed_strategies: list[dict]
    filter_reason: dict[str, str]  # strategy -> reason for removal


def get_risk_limits(risk_tolerance: str) -> dict:
    """Get limits for a risk tolerance level."""
    tolerance = risk_tolerance.strip().capitalize()
    if tolerance not in RISK_TOLERANCE_LIMITS:
        tolerance = "Medium"
    return RISK_TOLERANCE_LIMITS[tolerance]


def get_strategy_profile(strategy_name: str) -> Optional[dict]:
    """Get risk profile for a strategy."""
    return STRATEGY_RISK_PROFILES.get(strategy_name)


def filter_strategies_by_risk(
    strategies: list[dict],
    user_risk_tolerance: str,
) -> FilterResult:
    """
    Remove strategies violating user risk tolerance.
    
    This is a HARD filter - no exceptions.
    
    Args:
        strategies: List of strategy dicts with keys:
            - name: str
            - expected_vol: float (optional, uses profile if missing)
            - max_drawdown: float (optional, uses profile if missing)
        user_risk_tolerance: "Low", "Medium", or "High"
    
    Returns:
        FilterResult with allowed and removed strategies
    """
    limits = get_risk_limits(user_risk_tolerance)
    max_vol = limits["max_vol"]
    max_dd = limits["max_drawdown"]
    
    allowed = []
    removed = []
    reasons = {}
    
    for strat in strategies:
        name = strat.get("name", strat.get("strategy_name", "Unknown"))
        
        # Get risk metrics from strategy or profile
        profile = get_strategy_profile(name) or {}
        
        vol = strat.get("expected_vol", profile.get("expected_vol", 0.15))
        dd = strat.get("max_drawdown", profile.get("max_drawdown", 0.10))
        
        # Check violations
        violations = []
        
        if vol > max_vol:
            violations.append(f"vol {vol:.1%} > {max_vol:.1%}")
        
        if dd > max_dd:
            violations.append(f"DD {dd:.1%} > {max_dd:.1%}")
        
        if violations:
            removed.append(strat)
            reasons[name] = f"Violates {user_risk_tolerance} risk limits: {', '.join(violations)}"
        else:
            allowed.append(strat)
    
    return FilterResult(
        allowed_strategies=allowed,
        removed_strategies=removed,
        filter_reason=reasons
    )


def filter_strategies_by_regime(
    strategies: list[dict],
    current_regime: str,
) -> FilterResult:
    """
    Filter strategies by regime compatibility.
    
    Args:
        strategies: List of strategy dicts
        current_regime: Current detected regime name
    
    Returns:
        FilterResult with regime-compatible strategies
    """
    allowed = []
    removed = []
    reasons = {}
    
    for strat in strategies:
        name = strat.get("name", strat.get("strategy_name", "Unknown"))
        
        # Get compatible regimes from strategy or profile
        profile = get_strategy_profile(name) or {}
        compatible = strat.get("compatible_regimes", profile.get("compatible_regimes", []))
        
        if not compatible or current_regime in compatible:
            allowed.append(strat)
        else:
            removed.append(strat)
            reasons[name] = f"Not compatible with {current_regime} regime"
    
    return FilterResult(
        allowed_strategies=allowed,
        removed_strategies=removed,
        filter_reason=reasons
    )


def apply_all_filters(
    strategies: list[dict],
    user_risk_tolerance: str,
    current_regime: str,
) -> FilterResult:
    """
    Apply all hard constraint filters in order.
    
    Order:
    1. User risk tolerance (HARD)
    2. Regime compatibility
    
    Args:
        strategies: List of all available strategies
        user_risk_tolerance: User's risk level
        current_regime: Detected market regime
    
    Returns:
        FilterResult with final allowed strategies
    """
    all_removed = []
    all_reasons = {}
    
    # Step 1: Risk tolerance filter
    risk_result = filter_strategies_by_risk(strategies, user_risk_tolerance)
    all_removed.extend(risk_result.removed_strategies)
    all_reasons.update(risk_result.filter_reason)
    
    # Step 2: Regime compatibility filter
    regime_result = filter_strategies_by_regime(
        risk_result.allowed_strategies, 
        current_regime
    )
    all_removed.extend(regime_result.removed_strategies)
    all_reasons.update(regime_result.filter_reason)
    
    # If no strategies left, fall back to Defensive
    final_allowed = regime_result.allowed_strategies
    if not final_allowed:
        # Always allow Defensive as fallback
        defensive = {
            "name": "Defensive",
            "risk_level": "Low",
            "expected_vol": 0.06,
            "max_drawdown": 0.04
        }
        final_allowed = [defensive]
    
    return FilterResult(
        allowed_strategies=final_allowed,
        removed_strategies=all_removed,
        filter_reason=all_reasons
    )


def validate_strategy_risk(
    strategy_name: str,
    user_risk_tolerance: str,
) -> tuple[bool, str]:
    """
    Quick check if a single strategy is allowed for a risk level.
    
    Returns:
        Tuple of (is_allowed, reason)
    """
    profile = get_strategy_profile(strategy_name)
    
    if not profile:
        return True, "Unknown strategy - allowing by default"
    
    limits = get_risk_limits(user_risk_tolerance)
    
    if profile["expected_vol"] > limits["max_vol"]:
        return False, f"Volatility {profile['expected_vol']:.1%} exceeds {limits['max_vol']:.1%} limit"
    
    if profile["max_drawdown"] > limits["max_drawdown"]:
        return False, f"Max DD {profile['max_drawdown']:.1%} exceeds {limits['max_drawdown']:.1%} limit"
    
    return True, "Strategy within risk limits"


def get_all_strategy_profiles() -> dict[str, dict]:
    """Get all registered strategy risk profiles."""
    return STRATEGY_RISK_PROFILES.copy()


def register_strategy_profile(
    name: str,
    risk_level: str,
    expected_vol: float,
    max_drawdown: float,
    compatible_regimes: list[str]
) -> None:
    """Register a new strategy risk profile."""
    STRATEGY_RISK_PROFILES[name] = {
        "risk_level": risk_level,
        "expected_vol": expected_vol,
        "max_drawdown": max_drawdown,
        "compatible_regimes": compatible_regimes
    }
