# layers/L3_strategy_universe/range/stochastic_range.py
"""
Stochastic Range Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

Uses Stochastic for range-bound trading.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Stochastic Range"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 20
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic %K and %D."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(d_period).mean()
    
    return k, d


def run_stochastic_range(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Stochastic Range strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        k, d = calculate_stochastic(df)
        close = df["Close"]
        
        k_now = k.iloc[-1]
        d_now = d.iloc[-1]
        k_prev = k.iloc[-2]
        d_prev = d.iloc[-2]
        
        # Ranging market check
        sma_10 = close.rolling(10).mean().iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        is_ranging = abs(sma_10 - sma_20) / sma_20 < 0.02
        
        conditions = [
            k_now < 25,                           # Oversold
            k_now > k_prev,                       # Turning up
            k_now > d_now or (k_prev < d_prev and k_now >= d_now),  # Cross or crossed
            is_ranging,                           # In range environment
            close.iloc[-1] > close.iloc[-2],      # Price up
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
            indicators={"k": round(k_now, 2), "d": round(d_now, 2)}
        ))
    
    return outputs
