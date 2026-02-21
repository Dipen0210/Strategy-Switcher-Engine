# layers/L3_strategy_universe/reversion/stochastic_oversold.py
"""
Stochastic Oversold Strategy.

Pod: Reversion
Regime: Bull-Volatile (Trend + High Vol)

BUY when Stochastic %K crosses above %D from oversold territory.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Stochastic Oversold"
POD = "Bull-Volatile"
REGIME = "Bull-Volatile"
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


def run_stochastic_oversold(
    stock_data_dict: dict[str, pd.DataFrame],
    oversold: float = 20.0,
) -> list[StrategyOutput]:
    """Run Stochastic Oversold strategy."""
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
        sma_50 = close.rolling(50).mean()
        
        k_now = k.iloc[-1]
        k_prev = k.iloc[-2]
        d_now = d.iloc[-1]
        d_prev = d.iloc[-2]
        
        conditions = [
            k_now > d_now,                        # %K above %D
            k_prev <= d_prev,                     # Just crossed
            k_now < 50,                           # Still in lower half
            k_now > oversold or k_prev < oversold,  # Was oversold
            close.iloc[-1] > sma_50.iloc[-1] * 0.95,  # Not in downtrend
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
