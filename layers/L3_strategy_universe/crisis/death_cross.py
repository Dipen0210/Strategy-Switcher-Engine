# layers/L3_strategy_universe/trend/death_cross.py
"""
Death Cross Strategy (Bearish).

Pod: Short
Regime: Crisis (Bear + High Vol)

SELL signal when 50-day SMA crosses below 200-day SMA.
This is a classic long-term bearish signal.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Death Cross"
POD = "Crisis"
REGIME = "Crisis"
LOOKBACK = 200
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_death_cross(
    stock_data_dict: dict[str, pd.DataFrame],
    short_period: int = 50,
    long_period: int = 200,
) -> list[StrategyOutput]:
    """Run Death Cross strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < long_period + 5:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        sma_50 = close.rolling(short_period).mean()
        sma_200 = close.rolling(long_period).mean()
        
        sma_50_now = sma_50.iloc[-1]
        sma_200_now = sma_200.iloc[-1]
        sma_50_prev = sma_50.iloc[-2]
        sma_200_prev = sma_200.iloc[-2]
        close_now = close.iloc[-1]
        
        # Bearish conditions (for SELL signal)
        bearish_conditions = [
            sma_50_now < sma_200_now,             # 50 below 200 (death cross)
            sma_50_prev >= sma_200_prev or sma_50_now < sma_50_prev,  # Recent cross or declining
            close_now < sma_50_now,               # Price below 50
            close_now < sma_200_now,              # Price below 200
            (sma_200_now - sma_50_now) / sma_200_now > 0.01,  # Meaningful spread
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
            indicators={
                "sma_50": round(sma_50_now, 2),
                "sma_200": round(sma_200_now, 2),
            }
        ))
    
    return outputs
