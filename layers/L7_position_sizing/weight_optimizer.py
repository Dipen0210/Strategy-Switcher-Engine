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


# --- Signal → capital participation -------------------------------------
# The execution engine (L10) is long/flat only: PortfolioState.update_position
# rejects selling more shares than are held, so a bearish view maps to FLAT
# (exit to cash) rather than to a short position.
BEARISH_PARTICIPATION = 0.0     # signal = -1  -> exit to cash
NEUTRAL_PARTICIPATION = 0.5     # signal =  0  -> half weight, no conviction
MIN_BULLISH_PARTICIPATION = 0.5  # signal = +1 floor, scaled up by confidence


def signal_participation(signal: int, confidence: float) -> float:
    """
    Translate a strategy's directional view into a capital participation
    multiplier in [0, 1], applied to the user's target weight before
    volatility sizing.

        signal = +1  ->  MIN_BULLISH + (1 - MIN_BULLISH) * confidence   [0.5, 1.0]
        signal =  0  ->  NEUTRAL_PARTICIPATION                          0.5
        signal = -1  ->  BEARISH_PARTICIPATION                          0.0

    This is the link that makes strategy selection consequential. Without it
    every strategy produces an identical portfolio (user weight x inverse
    vol), so which strategy the bandits pick cannot affect returns and no
    performance attribution or ablation of L5 is measurable.

    Args:
        signal: -1, 0, or +1 from StrategyOutput
        confidence: 0.0-1.0 pattern strength from StrategyOutput

    Returns:
        Participation multiplier in [0, 1].
    """
    conf = float(min(max(confidence, 0.0), 1.0))

    if signal > 0:
        return MIN_BULLISH_PARTICIPATION + (1.0 - MIN_BULLISH_PARTICIPATION) * conf
    if signal < 0:
        return BEARISH_PARTICIPATION
    return NEUTRAL_PARTICIPATION


def volatility_adjusted_sizing(
    raw_weights: pd.Series,
    forecast_vol: pd.Series,
    stability_scores: pd.Series = None,
    participation: pd.Series = None,
    target_vol: float = 0.15,
    max_position: float = 1.0,
    min_position: float = 0.01,
    fully_invested: bool = False,
) -> pd.Series:
    """
    Adjust position sizes by volatility, regime stability, and strategy conviction.

    Formula:
        Adjusted = Raw x (Target Vol / Forecast Vol) x Stability x Participation

    fully_invested:
        False (default) — the returned weights are an absolute exposure and
        may sum to less than 1.0, leaving the remainder in cash. This is what
        makes participation meaningful: if every strategy turns bearish, the
        book actually de-risks.

        True — renormalize so weights always sum to 1.0 (the previous
        behavior). Retained so an ablation can isolate the effect of variable
        exposure, but note that under this setting participation only shifts
        capital BETWEEN names and can never move the portfolio to cash.
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

    # 3. Strategy Conviction Scalar (signal x confidence of the winner)
    if participation is not None and not participation.empty:
        part = participation.reindex(common_idx).fillna(1.0)
        adjusted = adjusted * part

    adjusted = adjusted.clip(upper=max_position)

    if fully_invested:
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total
        else:
            adjusted = raw

    # Drop dust positions. A name scaled to ~0 (bearish signal, or Defensive)
    # falls out here, which makes L8 emit SELL and liquidate it.
    adjusted = adjusted[adjusted >= min_position]

    if fully_invested and adjusted.sum() > 0:
        adjusted = adjusted / adjusted.sum()

    return adjusted


def compute_position_sizes(
    user_weights: pd.Series,
    forecast_vol: pd.Series,
    total_capital: float,
    stability_scores: pd.Series = None,
    participation: pd.Series = None,
    target_vol: float = 0.15,
    max_vol: float = 0.25,
    max_dd: float = 0.20,
    max_leverage: float = 1.0,
    fully_invested: bool = False,
) -> pd.DataFrame:
    """
    Compute final position sizes with all constraints.

    Note on `max_dd`: accepted for interface completeness but NOT enforced
    here — drawdown is a realized-path property, not a point-in-time sizing
    input. It is enforced by the L0 kill switch in the pipeline
    (emergency_drawdown_threshold), which liquidates the book on breach.
    """
    adjusted = volatility_adjusted_sizing(
        raw_weights=user_weights,
        forecast_vol=forecast_vol,
        stability_scores=stability_scores,
        participation=participation,
        target_vol=target_vol,
        fully_invested=fully_invested,
    )

    if not forecast_vol.empty:
        common_idx = adjusted.index.intersection(forecast_vol.index)
        if len(common_idx) > 0:
            port_vol = (adjusted.reindex(common_idx) * forecast_vol.reindex(common_idx)).sum()
        else:
            port_vol = target_vol
    else:
        port_vol = target_vol

    # Portfolio volatility cap. The previous version rescaled and then
    # immediately renormalized back to sum 1.0, so the breach response was a
    # no-op — exposure was never actually reduced.
    if port_vol > max_vol:
        scale_factor = max_vol / port_vol
        adjusted = adjusted * scale_factor

    # Leverage cap. Also previously unreachable: volatility_adjusted_sizing
    # always returned weights summing to exactly 1.0, which never exceeds
    # max_leverage (1.0 for Low/Medium, 1.5 for High).
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
