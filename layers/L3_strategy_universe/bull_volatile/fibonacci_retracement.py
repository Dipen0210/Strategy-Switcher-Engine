# layers/L3_strategy_universe/reversion/fibonacci_retracement.py
"""
Fibonacci Retracement Strategy.

Pod: Reversion
Regime: Bull-Volatile (Trend + High Vol)

BUY at key Fibonacci levels (38.2%, 50%, 61.8%) during pullback.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Fibonacci Retracement"
POD = "Bull-Volatile"
REGIME = "Bull-Volatile"
LOOKBACK = 50
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_fibonacci_retracement(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Fibonacci Retracement strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        high_price = close.iloc[-LOOKBACK:].max()
        low_price = close.iloc[-LOOKBACK:].min()
        current = close.iloc[-1]
        
        # Fibonacci levels from swing high to swing low
        diff = high_price - low_price
        fib_382 = high_price - 0.382 * diff
        fib_500 = high_price - 0.500 * diff
        fib_618 = high_price - 0.618 * diff
        
        # Check if at Fib level
        at_382 = abs(current - fib_382) / fib_382 < 0.01
        at_500 = abs(current - fib_500) / fib_500 < 0.01
        at_618 = abs(current - fib_618) / fib_618 < 0.01
        
        # Uptrend confirmation
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        
        conditions = [
            at_382 or at_500 or at_618,           # At a Fib level
            current > low_price,                  # Not at lowest
            sma_20 > sma_50 * 0.98,               # Not in strong downtrend
            close.iloc[-1] > close.iloc[-2],      # Bouncing
            current < high_price * 0.95,          # Room to run
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
            indicators={"fib_level": "38.2" if at_382 else ("50" if at_500 else "61.8")}
        ))
    
    return outputs
