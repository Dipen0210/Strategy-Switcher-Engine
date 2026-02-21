# layers/L3_strategy_universe/range/range_false_breakout.py
"""
False Breakout Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

BUY on false breakdowns (price dips below range then reverses).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "False Breakout"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_range_false_breakout(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run False Breakout strategy."""
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
        
        # Range boundaries
        range_low = low.iloc[-LOOKBACK:-1].min()
        range_high = close.iloc[-LOOKBACK:-1].max()
        
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        # False breakdown: dipped below range but closed back inside
        dipped_below = current_low < range_low * 0.99
        closed_inside = current_close > range_low
        reversal = current_close > close.iloc[-2]
        
        conditions = [
            dipped_below,                         # Went below range
            closed_inside,                        # Came back in
            reversal,                             # Bouncing
            current_close > current_low * 1.01,   # Good candle body
            current_close < range_high * 0.95,    # Room to run
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
            indicators={"range_low": round(range_low, 2), "dipped": dipped_below}
        ))
    
    return outputs
