# optimization/weight_optimizer.py
"""
Weight optimization for SUP Flow 1.

Includes:
- Mean-Variance Optimization (Markowitz)
- Volatility-based position sizing
- Risk limit enforcement
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def weight_sum_constraint(w):
    """Constraint: weights sum to 1."""
    return np.sum(w) - 1.0


def mean_variance_optimize(mu, cov_matrix, risk_level="medium"):
    """
    Mean-Variance Optimization (Markowitz)
    Maximizes Sharpe Ratio for given risk preference.
    """
    tickers = mu.index
    n = len(tickers)

    risk_map = {"low": 2.0, "medium": 4.0, "high": 8.0}
    risk_aversion = risk_map.get(risk_level.lower(), 4.0)

    mu_vec = mu.values
    cov = cov_matrix.values

    def portfolio_return(w):
        return np.dot(w, mu_vec)

    def portfolio_volatility(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)))

    def objective(w):
        return -portfolio_return(w) + risk_aversion * portfolio_volatility(w)

    x0 = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': weight_sum_constraint}

    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    weights = pd.Series(result.x, index=tickers)
    return weights / weights.sum()


def volatility_adjusted_sizing(
    raw_weights: pd.Series,
    forecast_vol: pd.Series,
    stability_scores: pd.Series = None,
    target_vol: float = 0.15,
    max_position: float = 1.0,
    min_position: float = 0.01,
) -> pd.Series:
    """
    Adjust position sizes based on volatility and regime stability.
    
    Formula: Adjusted Weight = Raw Weight × (Target Vol / Forecast Vol) × Stability Score
    """
    if raw_weights.empty:
        return raw_weights
    
    common_idx = raw_weights.index.intersection(forecast_vol.index)
    if len(common_idx) == 0:
        return raw_weights
    
    raw = raw_weights.reindex(common_idx).fillna(0)
    vol = forecast_vol.reindex(common_idx).fillna(target_vol)
    vol = vol.replace(0, target_vol)
    
    # 1. Volatility Scalar
    vol_adjustment = target_vol / vol
    adjusted = raw * vol_adjustment
    
    # 2. Stability Scalar (Regime Confusion Penalty)
    if stability_scores is not None and not stability_scores.empty:
        # Align indices
        stab = stability_scores.reindex(common_idx).fillna(1.0) # Default to 1.0 (Stable) if missing
        adjusted = adjusted * stab
    
    adjusted = adjusted.clip(upper=max_position)
    
    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total
    else:
        adjusted = raw
    
    adjusted = adjusted[adjusted >= min_position]
    
    if adjusted.sum() > 0:
        adjusted = adjusted / adjusted.sum()
    
    return adjusted


def compute_position_sizes(
    user_weights: pd.Series,
    forecast_vol: pd.Series,
    total_capital: float,
    stability_scores: pd.Series = None,
    target_vol: float = 0.15,
    max_vol: float = 0.25,
    max_dd: float = 0.20,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """
    Compute final position sizes with all constraints.
    """
    adjusted = volatility_adjusted_sizing(
        raw_weights=user_weights,
        forecast_vol=forecast_vol,
        stability_scores=stability_scores,
        target_vol=target_vol,
    )
    
    if not forecast_vol.empty:
        common_idx = adjusted.index.intersection(forecast_vol.index)
        if len(common_idx) > 0:
            port_vol = (adjusted.reindex(common_idx) * forecast_vol.reindex(common_idx)).sum()
        else:
            port_vol = target_vol
    else:
        port_vol = target_vol
    
    if port_vol > max_vol:
        scale_factor = max_vol / port_vol
        adjusted = adjusted * scale_factor
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum()
    
    if adjusted.sum() > max_leverage:
        adjusted = adjusted * (max_leverage / adjusted.sum())
    
    capital_allocation = adjusted * total_capital
    
    result = pd.DataFrame({
        "Ticker": adjusted.index,
        "User_Weight": user_weights.reindex(adjusted.index).fillna(0),
        "Adjusted_Weight": adjusted.values,
        "Capital_Allocation": capital_allocation.values,
    })
    
    return result
