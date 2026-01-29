# strategies/breakout.py
"""
Donchian Channel Breakout Strategy

High-risk trend-following strategy that identifies:
- Price breakouts from 20-day channels
- Volume confirmation
- Momentum acceleration

Risk Level: High
Compatible Regimes: Trend + Low Vol, Trend + High Vol
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Strategy-specific data requirements
LOOKBACK_DAYS = 20  # 20-day Donchian Channel (reduced from 252)
RISK_LEVEL = "High"


def calculate_donchian_breakout_score(df: pd.DataFrame) -> float:
    """
    Compute Donchian Channel breakout score for a stock.
    
    Components:
    - 20-day high/low channel position
    - Breakout confirmation (price vs channel)
    - Volume confirmation
    - Recent momentum
    
    Returns score from -1 (bearish) to +1 (bullish breakout)
    """
    if df is None or df.empty or len(df) < 25:
        return 0.0
    
    df = df.sort_values("Date").reset_index(drop=True) if "Date" in df.columns else df
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series([1] * len(df))
    
    # 1. Donchian Channel (20-day high/low)
    high_20 = high.rolling(window=20).max()
    low_20 = low.rolling(window=20).min()
    
    current_price = close.iloc[-1]
    channel_high = high_20.iloc[-1]
    channel_low = low_20.iloc[-1]
    channel_mid = (channel_high + channel_low) / 2
    channel_width = channel_high - channel_low
    
    if channel_width <= 0:
        return 0.0
    
    # Position within channel (-1 to +1)
    position = (current_price - channel_mid) / (channel_width / 2)
    
    # 2. Breakout detection
    # Upper breakout: price at or above 20-day high
    if current_price >= channel_high * 0.98:  # Within 2% of high
        score_breakout = 1.0
    elif current_price >= channel_mid:
        score_breakout = 0.5 * (current_price - channel_mid) / (channel_high - channel_mid)
    elif current_price <= channel_low * 1.02:  # Within 2% of low
        score_breakout = -1.0
    else:
        score_breakout = -0.5 * (channel_mid - current_price) / (channel_mid - channel_low)
    
    # 3. Volume confirmation
    avg_vol = volume.rolling(window=20).mean().iloc[-1]
    recent_vol = volume.iloc[-3:].mean() if len(volume) >= 3 else volume.iloc[-1]
    
    if avg_vol > 0:
        vol_ratio = recent_vol / avg_vol
    else:
        vol_ratio = 1.0
    
    # Volume boost if breaking out with high volume
    if vol_ratio > 1.2 and abs(score_breakout) > 0.5:
        vol_score = 0.3 * np.sign(score_breakout)
    elif vol_ratio > 1.0:
        vol_score = 0.1 * np.sign(score_breakout)
    else:
        vol_score = 0.0
    
    # 4. Recent momentum (5-day)
    if len(close) >= 5:
        mom_5 = (close.iloc[-1] / close.iloc[-5]) - 1
        score_mom = np.clip(mom_5 * 10, -0.3, 0.3)
    else:
        score_mom = 0.0
    
    # Combine scores
    aggregate = (
        0.50 * score_breakout +  # Primary: channel position
        0.25 * vol_score +       # Volume confirmation
        0.25 * score_mom         # Momentum
    )
    
    aggregate = np.clip(aggregate, -1.0, 1.0)
    return float(round(aggregate, 4)) if not np.isnan(aggregate) else 0.0


def run_breakout_strategy(stock_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run Donchian breakout strategy across all stocks.
    
    Args:
        stock_data_dict: Dict of {ticker: OHLCV DataFrame}
    
    Returns:
        DataFrame with Ticker, Strategy_Score, Rank
    """
    results = []
    
    for ticker, df in stock_data_dict.items():
        try:
            score = calculate_donchian_breakout_score(df)
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
