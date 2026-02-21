# layers/L3_strategy_universe/range/support_resistance.py
"""
Support/Resistance Bounce Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

BUY at support levels, SELL at resistance.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Support Resistance"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 50
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_support_resistance(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Support/Resistance strategy."""
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
        high = df["High"]
        
        # Find support and resistance
        lows = low.iloc[-LOOKBACK:]
        highs = high.iloc[-LOOKBACK:]
        
        support = lows.nsmallest(5).mean()
        resistance = highs.nlargest(5).mean()
        
        current = close.iloc[-1]
        current_low = low.iloc[-1]
        
        # Check for support bounce
        at_support = current_low <= support * 1.02
        bouncing = current > close.iloc[-2]
        room_to_resistance = (resistance - current) / current > 0.02
        
        conditions = [
            at_support,                           # At support level
            bouncing,                             # Price turning up
            room_to_resistance,                   # Room to upside
            current > support,                    # Haven't broken support
            current < resistance * 0.95,          # Not at resistance
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
                "support": round(support, 2),
                "resistance": round(resistance, 2),
            }
        ))
    
    return outputs
