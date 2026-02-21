# layers/L3_strategy_universe/short/short_momentum.py
"""
Short Momentum Strategy.

Pod: Short
Regime: Crisis (Bear + High Vol)

SELL signal when negative momentum accelerates.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Short Momentum"
POD = "Crisis"
REGIME = "Crisis"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_short_momentum(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Short Momentum strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        
        # Momentum metrics
        mom_5d = (close.iloc[-1] / close.iloc[-5]) - 1
        mom_10d = (close.iloc[-1] / close.iloc[-10]) - 1
        mom_20d = (close.iloc[-1] / close.iloc[-20]) - 1
        
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        
        # Bearish momentum conditions
        bearish_conditions = [
            mom_5d < -0.03,                       # 5-day down 3%+
            mom_10d < -0.05,                      # 10-day down 5%+
            mom_20d < 0,                          # 20-day negative
            close.iloc[-1] < sma_20,              # Below short MA
            sma_20 < sma_50,                      # Bearish MA alignment
        ]
        
        n_bearish = sum(bearish_conditions)
        
        if n_bearish >= 4:
            signal = -1
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
                "mom_5d": round(mom_5d, 4),
                "mom_10d": round(mom_10d, 4),
            }
        ))
    
    return outputs
