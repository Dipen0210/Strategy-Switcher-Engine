# strategies/momentum.py
"""
Momentum Strategy for SUP Flow 1.

Medium-risk trend-following strategy based on:
- Price momentum (multi-period returns)
- RSI deviation from neutral
- MACD histogram
- Moving average relationship
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Strategy-specific data requirements
LOOKBACK_DAYS = 60
TIMEFRAME = LOOKBACK_DAYS  # Common variable for pipeline past-return calculation  # 60 days needed for momentum calculations
RISK_LEVEL = "Medium"


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


def calculate_macd(close: pd.Series) -> float:
    """Calculate MACD histogram."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return histogram.iloc[-1] if len(histogram) > 0 else 0.0


def calculate_momentum_score(df: pd.DataFrame) -> float:
    """
    Compute momentum score for a stock.
    
    Components:
    - 20-day return (medium-term momentum)
    - Price vs SMA-20 (trend position)
    - RSI deviation from 50 (overbought/oversold)
    - MACD histogram (momentum direction)
    
    Returns score from -1 (bearish) to +1 (bullish)
    """
    if df is None or df.empty or len(df) < 30:
        return 0.0
    
    df = df.sort_values("Date").reset_index(drop=True) if "Date" in df.columns else df
    close = df["Close"]
    
    # 1. 20-day return
    if len(close) >= 20:
        ret_20 = (close.iloc[-1] / close.iloc[-20]) - 1
        score_ret = np.clip(ret_20 * 5, -1, 1)  # Scale and clip
    else:
        score_ret = 0.0
    
    # 2. Price vs SMA-20
    sma_20 = close.rolling(20).mean().iloc[-1]
    if sma_20 > 0:
        pct_from_sma = (close.iloc[-1] - sma_20) / sma_20
        score_sma = np.clip(pct_from_sma * 10, -1, 1)
    else:
        score_sma = 0.0
    
    # 3. RSI deviation
    rsi = calculate_rsi(close)
    score_rsi = (rsi - 50) / 50  # -1 to +1 scale
    
    # 4. MACD histogram
    macd_hist = calculate_macd(close)
    score_macd = np.clip(macd_hist / (close.iloc[-1] * 0.02 + 1e-10), -1, 1)
    
    # Combine with weights
    aggregate = (
        0.30 * score_ret +   # Medium-term return
        0.25 * score_sma +   # Trend position
        0.25 * score_rsi +   # RSI
        0.20 * score_macd    # MACD momentum
    )
    
    aggregate = np.clip(aggregate, -1.0, 1.0)
    return float(round(aggregate, 4)) if not np.isnan(aggregate) else 0.0


def run_momentum_strategy(stock_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run momentum strategy across all stocks.
    
    Args:
        stock_data_dict: Dict of {ticker: OHLCV DataFrame}
    
    Returns:
        DataFrame with Ticker, Strategy_Score, Rank
    """
    results = []
    
    for ticker, df in stock_data_dict.items():
        try:
            score = calculate_momentum_score(df)
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
