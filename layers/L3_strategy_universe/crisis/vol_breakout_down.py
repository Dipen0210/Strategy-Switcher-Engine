# layers/L3_strategy_universe/volatility/vol_breakout_down.py
"""
Volatility Breakout Down Strategy.

Pod: Short
Regime: Crisis (Bear + High Vol)

SELL when volatility spikes and price breaks down.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Vol Breakout Down"
POD = "Crisis"
REGIME = "Crisis"
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


def run_vol_breakout_down(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Volatility Breakout Down strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        atr = calculate_atr(df)
        close = df["Close"]
        low = df["Low"]
        
        atr_now = atr.iloc[-1]
        atr_avg = atr.iloc[-20:].mean()
        close_now = close.iloc[-1]
        low_now = low.iloc[-1]
        
        # Bearish conditions
        vol_spike = atr_now > atr_avg * 1.5
        breakdown = low_now < low.iloc[-5:-1].min()
        momentum_down = close_now < close.iloc[-2]
        
        bearish_conditions = [
            vol_spike,                            # Volatility spiking
            breakdown,                            # Breaking down
            momentum_down,                        # Falling
            close_now < close.rolling(20).mean().iloc[-1],  # Below MA
            close_now < close.iloc[-5],           # Lower than 5 days ago
        ]
        
        n_bearish = sum(bearish_conditions)
        
        if n_bearish >= 4:
            signal = -1  # SELL
            confidence = n_bearish / len(bearish_conditions)
        else:
            signal = 0
            confidence = 0.5
        
        outputs.append(StrategyOutput(
            ticker=ticker,
            signal=signal,
            confidence=round(confidence, 4),
            strategy_name=STRATEGY_NAME,
            pod=POD,
            regime=REGIME,
            indicators={"atr": round(atr_now, 2), "vol_spike": vol_spike}
        ))
    
    return outputs
