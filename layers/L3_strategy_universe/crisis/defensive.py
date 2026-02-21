# strategies/defensive.py
"""
Defensive Strategy for SUP Flow 1.

Low-risk strategy optimized for capital preservation:
- Low volatility stocks
- Stable returns
- Drawdown resilience

Risk Level: Low
Compatible Regimes: All (especially Crisis)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Strategy-specific data requirements
LOOKBACK_DAYS = 60
TIMEFRAME = LOOKBACK_DAYS  # Common variable for pipeline past-return calculation  # 60 days for volatility and drawdown analysis
RISK_LEVEL = "Low"


def calculate_defensive_score(df: pd.DataFrame) -> float:
    """
    Compute defensive score for a stock.
    
    Components:
    - Low volatility (lower = better)
    - Stable returns (low variance)
    - Drawdown resilience (smaller max DD = better)
    - Price stability (low gap risk)
    
    Returns score from -1 (high risk) to +1 (defensive/stable)
    """
    if df is None or df.empty or len(df) < 60:
        return 0.0
    
    df = df.sort_values("Date").reset_index(drop=True) if "Date" in df.columns else df
    
    close = df["Close"]
    
    # Compute log returns
    log_returns = np.log(close / close.shift(1)).dropna()
    
    if len(log_returns) < 30:
        return 0.0
    
    # 1. Volatility score (lower vol = higher score)
    vol_20 = log_returns.tail(20).std() * np.sqrt(252)
    vol_60 = log_returns.tail(60).std() * np.sqrt(252)
    avg_vol = (vol_20 + vol_60) / 2
    
    if avg_vol < 0.10:
        score_vol = 1.0
    elif avg_vol < 0.15:
        score_vol = 0.8
    elif avg_vol < 0.20:
        score_vol = 0.5
    elif avg_vol < 0.30:
        score_vol = 0.0
    else:
        score_vol = -0.5
    
    # 2. Return stability (low variance of weekly returns)
    weekly_returns = log_returns.rolling(5).sum().dropna()
    if len(weekly_returns) > 5:
        return_variance = weekly_returns.var()
        score_stability = 1.0 - np.clip(return_variance * 100, 0, 1)
    else:
        score_stability = 0.5
    
    # 3. Drawdown resilience
    rolling_max = close.expanding().max()
    drawdown = (close - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())
    
    if max_dd < 0.10:
        score_dd = 1.0
    elif max_dd < 0.15:
        score_dd = 0.7
    elif max_dd < 0.25:
        score_dd = 0.3
    else:
        score_dd = -0.3
    
    # 4. Gap risk
    if len(df) > 1 and "Open" in df.columns:
        overnight_gaps = abs(df["Open"].shift(-1) / close - 1).dropna()
        avg_gap = overnight_gaps.mean() if len(overnight_gaps) > 0 else 0
        score_gap = 1.0 - np.clip(avg_gap * 20, 0, 1)
    else:
        score_gap = 0.5
    
    # Combine with defensive-focused weights
    aggregate = (
        0.35 * score_vol +
        0.25 * score_stability +
        0.25 * score_dd +
        0.15 * score_gap
    )
    
    aggregate = np.clip(aggregate, -1.0, 1.0)
    return float(round(aggregate, 4)) if not np.isnan(aggregate) else 0.0


def run_defensive_strategy(stock_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run defensive strategy across all stocks.
    
    Args:
        stock_data_dict: Dict of {ticker: OHLCV DataFrame}
    
    Returns:
        DataFrame with Ticker, Strategy_Score, Rank
    """
    results = []
    
    for ticker, df in stock_data_dict.items():
        try:
            score = calculate_defensive_score(df)
        except Exception:
            score = 0.0
        
        results.append({
            "Ticker": ticker,
            "Strategy_Score": score
        })
    
    result_df = pd.DataFrame(results)
    
    if result_df.empty:
        return result_df
    
    result_df["Strategy_Score"] = pd.to_numeric(
        result_df["Strategy_Score"], errors="coerce"
    ).fillna(0.0)
    
    result_df = result_df.sort_values("Strategy_Score", ascending=False).reset_index(drop=True)
    result_df["Rank"] = result_df.index + 1
    
    return result_df
