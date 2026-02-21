# layers/L3_strategy_universe/reversion/bollinger_reversion.py
"""
Bollinger Band Reversion Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

BUY when price touches lower Bollinger Band and bounces.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Bollinger Reversion"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 25
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def run_bollinger_reversion(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Bollinger Reversion strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        low = df["Low"]
        upper, middle, lower = calculate_bollinger(close)
        
        close_now = close.iloc[-1]
        low_now = low.iloc[-1]
        lower_now = lower.iloc[-1]
        upper_now = upper.iloc[-1]
        middle_now = middle.iloc[-1]
        
        # Position in band (0 = lower, 1 = upper)
        band_position = (close_now - lower_now) / (upper_now - lower_now + 1e-10)
        
        conditions = [
            low_now <= lower_now * 1.01 or band_position < 0.1,  # At/below lower band
            close_now > close.iloc[-2],           # Bouncing
            close_now > lower_now,                # Didn't break down
            band_position < 0.5,                  # Still in lower half
            close_now < middle_now,               # Below middle (room to run)
        ]
        
        signal, confidence = compute_signal_and_confidence(
            conditions, buy_threshold=4, sell_threshold=4
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
                "lower": round(lower_now, 2),
                "band_pos": round(band_position, 2),
            }
        ))
    
    return outputs
