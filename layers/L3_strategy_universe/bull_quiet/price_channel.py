# layers/L3_strategy_universe/trend/price_channel.py
"""
Price Channel (Donchian) Strategy.

Pod: Trend
Regime: Bull-Quiet (Trend + Low Vol)

BUY when price breaks above the 20-period high.
This is the classic "Turtle Trading" entry signal.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Price Channel"
POD = "Bull-Quiet"
REGIME = "Bull-Quiet"
LOOKBACK = 25
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_donchian(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Donchian Channel."""
    high = df["High"]
    low = df["Low"]
    
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def run_price_channel(
    stock_data_dict: dict[str, pd.DataFrame],
    period: int = 20,
) -> list[StrategyOutput]:
    """Run Price Channel strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        upper, middle, lower = calculate_donchian(df, period)
        close = df["Close"]
        high = df["High"]
        
        close_now = close.iloc[-1]
        upper_now = upper.iloc[-1]
        upper_prev = upper.iloc[-2]
        middle_now = middle.iloc[-1]
        lower_now = lower.iloc[-1]
        
        # Check for breakout
        channel_width = (upper_now - lower_now) / middle_now
        position_in_channel = (close_now - lower_now) / (upper_now - lower_now + 1e-10)
        
        conditions = [
            close_now >= upper_now * 0.99,        # Near or above upper channel
            high.iloc[-1] > upper_prev,           # New high breakout
            position_in_channel > 0.8,            # In upper 20% of channel
            channel_width > 0.03,                 # Meaningful channel width
            channel_width < 0.15,                 # Not too volatile
        ]
        
        signal, confidence = compute_signal_and_confidence(
            conditions, buy_threshold=3, sell_threshold=4
        )
        
        outputs.append(StrategyOutput(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            strategy_name=STRATEGY_NAME,
            pod=POD,
            regime=REGIME,
            indicators={
                "upper": round(upper_now, 2),
                "middle": round(middle_now, 2),
                "lower": round(lower_now, 2),
            }
        ))
    
    return outputs
