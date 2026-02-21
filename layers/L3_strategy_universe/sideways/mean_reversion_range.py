# layers/L3_strategy_universe/range/mean_reversion_range.py
"""
Mean Reversion Range Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

Classic mean reversion in ranging markets.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Mean Reversion Range"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 25
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_mean_reversion_range(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Mean Reversion Range strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        
        current = close.iloc[-1]
        sma_now = sma_20.iloc[-1]
        std_now = std_20.iloc[-1]
        
        # Z-score: how many std devs from mean
        z_score = (current - sma_now) / (std_now + 1e-10)
        
        # Ranging check
        sma_10 = close.rolling(10).mean().iloc[-1]
        is_ranging = abs(sma_10 - sma_now) / sma_now < 0.02
        
        conditions = [
            z_score < -1.5,                       # Significantly below mean
            current > close.iloc[-2],             # Turning up
            is_ranging,                           # In range environment
            z_score > -3.0,                       # Not crashed
            current > sma_now * 0.9,             # Not too far from mean
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
            indicators={"z_score": round(z_score, 2), "sma": round(sma_now, 2)}
        ))
    
    return outputs
