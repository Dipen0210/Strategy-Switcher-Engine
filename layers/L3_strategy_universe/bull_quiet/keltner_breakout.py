# layers/L3_strategy_universe/trend/keltner_breakout.py
"""
Keltner Channel Breakout Strategy.

Pod: Trend
Regime: Bull-Quiet (Trend + Low Vol)

BUY when price breaks above Keltner upper band.
Uses ATR for volatility-adjusted channels.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Keltner Breakout"
POD = "Bull-Quiet"
REGIME = "Bull-Quiet"
LOOKBACK = 25
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()


def calculate_keltner(df: pd.DataFrame, ema_period: int = 20, atr_mult: float = 2.0) -> tuple:
    """Calculate Keltner Channels."""
    close = df["Close"]
    atr = calculate_atr(df)
    
    middle = close.ewm(span=ema_period, adjust=False).mean()
    upper = middle + (atr_mult * atr)
    lower = middle - (atr_mult * atr)
    
    return upper, middle, lower, atr


def run_keltner_breakout(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Keltner Breakout strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        upper, middle, lower, atr = calculate_keltner(df)
        close = df["Close"]
        
        close_now = close.iloc[-1]
        close_prev = close.iloc[-2]
        upper_now = upper.iloc[-1]
        upper_prev = upper.iloc[-2]
        middle_now = middle.iloc[-1]
        atr_now = atr.iloc[-1]
        
        # Check for breakout
        breakout = close_now > upper_now and close_prev <= upper_prev
        above_upper = close_now > upper_now
        trend_up = middle_now > middle.iloc[-5]
        
        conditions = [
            above_upper or breakout,              # At or breaking upper band
            close_now > middle_now,               # Above middle (EMA)
            trend_up,                             # Middle line trending up
            atr_now / close_now < 0.03,          # Low volatility regime
            close_now > close_prev,               # Today's price higher
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
                "middle": round(middle_now, 2),
                "atr": round(atr_now, 2),
            }
        ))
    
    return outputs
