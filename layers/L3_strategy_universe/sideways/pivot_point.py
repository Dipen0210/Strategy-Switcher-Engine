# layers/L3_strategy_universe/range/pivot_point.py
"""
Pivot Point Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

Uses daily pivot points for support/resistance levels.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Pivot Point"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 5
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_pivot_point(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Pivot Point strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        # Previous day's OHLC for pivot calculation
        prev_high = df["High"].iloc[-2]
        prev_low = df["Low"].iloc[-2]
        prev_close = df["Close"].iloc[-2]
        
        # Calculate pivot levels
        pivot = (prev_high + prev_low + prev_close) / 3
        s1 = 2 * pivot - prev_high  # Support 1
        s2 = pivot - (prev_high - prev_low)  # Support 2
        r1 = 2 * pivot - prev_low  # Resistance 1
        r2 = pivot + (prev_high - prev_low)  # Resistance 2
        
        current = df["Close"].iloc[-1]
        current_low = df["Low"].iloc[-1]
        
        conditions = [
            current_low <= s1 * 1.01,             # At S1 support
            current > current_low,                # Closing above low
            current > df["Close"].iloc[-2],       # Bouncing
            current < pivot,                      # Below pivot (room up)
            current > s2,                         # Above S2 (not crashed)
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
            indicators={"pivot": round(pivot, 2), "s1": round(s1, 2), "r1": round(r1, 2)}
        ))
    
    return outputs
