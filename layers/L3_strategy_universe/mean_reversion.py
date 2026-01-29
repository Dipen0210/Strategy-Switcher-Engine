# strategies/mean_reversion.py
"""
Mean Reversion Strategy for SUP Flow 1.

Low-risk strategy based on:
- RSI extremes (oversold = bullish)
- Bollinger Band deviation
- Price deviation from moving average

Risk Level: Low
Compatible Regimes: Range + Low Vol, Trend + Low Vol
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Strategy-specific data requirements
LOOKBACK_DAYS = 30  # 30 days for RSI and Bollinger Bands
RISK_LEVEL = "Low"


def calculate_rsi(close: pd.Series, period: int = 14) -> float:
    """Calculate RSI indicator."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    if len(gain) == 0 or len(loss) == 0:
        return 50.0
    
    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi if not np.isnan(rsi) else 50.0


def calculate_mean_reversion_score(df: pd.DataFrame) -> float:
    """
    Compute mean reversion score for a stock.
    
    Components:
    - RSI extremes (oversold = bullish, overbought = bearish)
    - Bollinger Band position (below lower = bullish)
    - Price deviation from SMA (negative = bullish)
    
    Returns score from -1 (bearish) to +1 (bullish reversion expected)
    """
    if df is None or df.empty or len(df) < 30:
        return 0.0
    
    df = df.sort_values("Date").reset_index(drop=True) if "Date" in df.columns else df
    close = df["Close"]
    
    # 1. RSI extremes (inverted for mean reversion)
    rsi = calculate_rsi(close)
    
    if rsi < 30:
        score_rsi = (30 - rsi) / 30  # Oversold = bullish (0 to 1)
    elif rsi > 70:
        score_rsi = (70 - rsi) / 30  # Overbought = bearish (-1 to 0)
    else:
        score_rsi = 0.0  # Neutral zone
    
    # 2. Bollinger Band position
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    current = close.iloc[-1]
    upper = upper_band.iloc[-1]
    lower = lower_band.iloc[-1]
    middle = sma_20.iloc[-1]
    
    if upper > lower:
        band_width = upper - lower
        position = (current - middle) / (band_width / 2 + 1e-10)
        score_bb = -np.clip(position, -1, 1)  # Inverted: below = bullish
    else:
        score_bb = 0.0
    
    # 3. Price deviation from SMA
    if middle > 0:
        deviation = (current - middle) / middle
        score_dev = -np.clip(deviation * 10, -1, 1)  # Inverted: below = bullish
    else:
        score_dev = 0.0
    
    # Combine with equal weights
    aggregate = (
        0.40 * score_rsi +
        0.35 * score_bb +
        0.25 * score_dev
    )
    
    aggregate = np.clip(aggregate, -1.0, 1.0)
    return float(round(aggregate, 4)) if not np.isnan(aggregate) else 0.0


def run_mean_reversion_strategy(stock_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run mean reversion strategy across all stocks.
    
    Args:
        stock_data_dict: Dict of {ticker: OHLCV DataFrame}
    
    Returns:
        DataFrame with Ticker, Strategy_Score, Rank
    """
    results = []
    
    for ticker, df in stock_data_dict.items():
        try:
            score = calculate_mean_reversion_score(df)
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
