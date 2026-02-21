# layers/L3_strategy_universe/range/grid_trading.py
"""
Grid Trading Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

Places buy orders at regular intervals below current price.
Works in ranging markets.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Grid Trading"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_grid_trading(
    stock_data_dict: dict[str, pd.DataFrame],
    grid_levels: int = 5,
) -> list[StrategyOutput]:
    """
    Run Grid Trading strategy.
    
    BUY when price is at lower grid levels in a range-bound market.
    """
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        high_30d = close.iloc[-LOOKBACK:].max()
        low_30d = close.iloc[-LOOKBACK:].min()
        current = close.iloc[-1]
        
        # Calculate grid position (0 = bottom, 1 = top)
        range_size = high_30d - low_30d
        if range_size < 0.01:
            range_size = 0.01
        
        grid_position = (current - low_30d) / range_size
        
        # Check if market is ranging (not trending)
        sma_10 = close.rolling(10).mean().iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        is_ranging = abs(sma_10 - sma_20) / sma_20 < 0.02
        
        conditions = [
            grid_position < 0.3,                  # In lower part of range
            is_ranging,                           # Market is ranging
            current > low_30d * 1.01,            # Not breaking down
            current > close.iloc[-2],             # Bouncing
            range_size / current < 0.1,          # Not too volatile
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
            indicators={"grid_pos": round(grid_position, 2), "range": round(range_size, 2)}
        ))
    
    return outputs
