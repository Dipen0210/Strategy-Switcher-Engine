# ml/strategy_selector.py
"""
Deterministic Strategy Selection Layer (Control Layer).

Final decision layer that combines:
1. Risk constraints (HARD filter - already applied)
2. Maximum expected return optimization
3. ML bandit score as tie-breaker
4. Lowest risk as final tie-breaker

Priority order is locked:
Risk constraint → Max return → Bandit score → Lowest risk

This layer never overrides risk constraints. ML is advisory only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StrategyDecision:
    """Result of strategy selection process."""
    selected_strategy: str
    expected_return: float
    risk_level: str
    bandit_score: float
    selection_reason: str
    candidates_considered: int
    

def select_best_strategy(
    allowed_strategies: list[dict],
    bandit_scores: dict[str, float],
    expected_returns: Optional[dict[str, float]] = None,
) -> StrategyDecision:
    """
    Select the best strategy using deterministic ranking.
    
    Priority order (lexicographic):
    1. Risk constraint (hard) - strategies already filtered
    2. Maximum expected return
    3. Tie-breaker: highest bandit score
    4. Final tie: lowest risk level
    
    Args:
        allowed_strategies: List of strategy dicts with keys:
            - name: str
            - expected_return: float (optional)
            - risk_level: str ("Low", "Medium", "High")
            - expected_vol: float (optional)
            - max_drawdown: float (optional)
        bandit_scores: Dict of strategy_name -> bandit score
        expected_returns: Optional override for expected returns
    
    Returns:
        StrategyDecision with selected strategy and reasoning
    """
    if not allowed_strategies:
        return StrategyDecision(
            selected_strategy="Defensive",
            expected_return=0.0,
            risk_level="Low",
            bandit_score=0.0,
            selection_reason="No strategies available after filtering",
            candidates_considered=0
        )
    
    # Build candidate list with all scores
    candidates = []
    for strat in allowed_strategies:
        # Handle both string and dict inputs
        if isinstance(strat, str):
            name = strat
            exp_ret = expected_returns.get(name, 0.0) if expected_returns else 0.0
            risk_level = "Medium"  # Default when only name provided
        else:
            name = strat.get("name", strat.get("strategy_name", "Unknown"))
            # Get expected return
            if expected_returns and name in expected_returns:
                exp_ret = expected_returns[name]
            else:
                exp_ret = strat.get("expected_return", 0.0)
            # Get risk info
            risk_level = strat.get("risk_level", "Medium")
        
        risk_order = {"Low": 0, "Medium": 1, "High": 2}.get(risk_level, 1)
        
        # Get bandit score
        bandit_score = bandit_scores.get(name, 0.0)
        
        candidates.append({
            "name": name,
            "expected_return": float(exp_ret) if exp_ret is not None else 0.0,
            "risk_level": risk_level,
            "risk_order": risk_order,
            "bandit_score": float(bandit_score),
            "strategy": strat
        })
    
    # Sort by priority:
    # 1. Higher expected return (descending)
    # 2. Higher bandit score (descending)
    # 3. Lower risk (ascending)
    candidates.sort(
        key=lambda x: (
            -x["expected_return"],  # Higher return first
            -x["bandit_score"],     # Higher bandit score second
            x["risk_order"]         # Lower risk third
        )
    )
    
    # Select best candidate
    best = candidates[0]
    
    # Determine selection reason
    if len(candidates) == 1:
        reason = "Only viable strategy after risk filtering"
    else:
        second = candidates[1]
        if best["expected_return"] > second["expected_return"]:
            reason = f"Highest expected return ({best['expected_return']:.2%})"
        elif best["bandit_score"] > second["bandit_score"]:
            reason = f"ML bandit tie-breaker (score: {best['bandit_score']:.3f})"
        elif best["risk_order"] < second["risk_order"]:
            reason = f"Lowest risk level ({best['risk_level']})"
        else:
            reason = "First in sorted order"
    
    return StrategyDecision(
        selected_strategy=best["name"],
        expected_return=best["expected_return"],
        risk_level=best["risk_level"],
        bandit_score=best["bandit_score"],
        selection_reason=reason,
        candidates_considered=len(candidates)
    )


def should_switch_strategy(
    current_strategy: str,
    new_strategy: str,
    new_probability: float,
    last_switch_date: Optional[str],
    current_date: str,
    cooldown_days: int = 5,
    probability_threshold: float = 0.6
) -> tuple[bool, str]:
    """
    Determine if strategy should be switched.
    
    Prevents overtrading by enforcing:
    - Cooldown timer (minimum days between switches)
    - Probability threshold (new strategy must be significantly better)
    
    Args:
        current_strategy: Currently active strategy
        new_strategy: Proposed new strategy
        new_probability: Bandit probability for new strategy
        last_switch_date: Date of last strategy switch (YYYY-MM-DD)
        current_date: Current date (YYYY-MM-DD)
        cooldown_days: Minimum days between switches
        probability_threshold: Minimum probability to trigger switch
    
    Returns:
        Tuple of (should_switch: bool, reason: str)
    """
    # Same strategy - no switch needed
    if current_strategy == new_strategy:
        return False, "Same strategy selected"
    
    # Check probability threshold
    if new_probability < probability_threshold:
        return False, f"New strategy probability ({new_probability:.2f}) below threshold ({probability_threshold})"
    
    # Check cooldown
    if last_switch_date:
        try:
            from datetime import datetime
            last_dt = datetime.strptime(last_switch_date, "%Y-%m-%d")
            curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
            days_since = (curr_dt - last_dt).days
            
            if days_since < cooldown_days:
                return False, f"Cooldown active ({days_since}/{cooldown_days} days)"
        except ValueError:
            pass  # Date parsing failed, proceed with switch
    
    return True, f"Strategy switch approved (probability: {new_probability:.2f})"


def build_context_vector(
    regime_probs: dict[str, float],
    volatility: float,
    momentum: float,
    drawdown: float,
    risk_score: float
) -> np.ndarray:
    """
    Build context vector for bandit selection.
    
    Components:
    - regime_encoded: Encoded regime probabilities (2D: trend_prob, vol_prob)
    - volatility: Current asset volatility
    - momentum: Recent momentum/trend
    - drawdown: Current drawdown from peak
    - risk_score: User risk tolerance (0=low, 1=high)
    
    Returns:
        numpy array of context features
    """
    # Encode regime as trend probability and volatility probability
    trend_prob = regime_probs.get("Trend + Low Vol", 0) + regime_probs.get("Trend + High Vol", 0)
    high_vol_prob = regime_probs.get("Trend + High Vol", 0) + regime_probs.get("Crisis", 0)
    
    context = np.array([
        trend_prob,
        high_vol_prob,
        np.clip(volatility, 0, 1),  # Normalize
        np.clip(momentum, -1, 1),   # Clip momentum
        np.clip(abs(drawdown), 0, 1),  # Absolute drawdown
        risk_score
    ], dtype=float)
    
    # Replace NaN with 0
    context = np.nan_to_num(context, nan=0.0)
    
    return context


def rank_strategies_for_display(
    allowed_strategies: list[dict],
    bandit_scores: dict[str, float],
    expected_returns: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    Rank all strategies for display in UI.
    
    Returns list of strategies with scores and rankings.
    """
    ranked = []
    
    for strat in allowed_strategies:
        # Handle both string and dict inputs
        if isinstance(strat, str):
            name = strat
            exp_ret = expected_returns.get(name, 0.0) if expected_returns else 0.0
            risk_level = "Medium"
        else:
            name = strat.get("name", strat.get("strategy_name", "Unknown"))
            exp_ret = 0.0
            if expected_returns and name in expected_returns:
                exp_ret = expected_returns[name]
            elif "expected_return" in strat:
                exp_ret = strat["expected_return"]
            risk_level = strat.get("risk_level", "Medium")
        
        ranked.append({
            "name": name,
            "expected_return": exp_ret,
            "risk_level": risk_level,
            "bandit_score": bandit_scores.get(name, 0.0),
            "allowed": True
        })
    
    # Sort by same priority as selection
    ranked.sort(
        key=lambda x: (
            -x["expected_return"],
            -x["bandit_score"],
            {"Low": 0, "Medium": 1, "High": 2}.get(x["risk_level"], 1)
        )
    )
    
    # Add rank
    for i, item in enumerate(ranked):
        item["rank"] = i + 1
    
    return ranked
