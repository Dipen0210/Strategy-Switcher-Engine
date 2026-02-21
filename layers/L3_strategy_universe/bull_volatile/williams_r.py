# layers/L3_strategy_universe/reversion/williams_r.py
"""
Williams %R Strategy.

Pod: Reversion
Regime: Bull-Volatile (Trend + High Vol)

BUY when Williams %R moves from oversold (< -80) back toward middle.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Williams %R"
POD = "Bull-Volatile"
REGIME = "Bull-Volatile"
LOOKBACK = 20
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr


def run_williams_r(
    stock_data_dict: dict[str, pd.DataFrame],
    oversold: float = -80.0,
) -> list[StrategyOutput]:
    """Run Williams %R strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        wr = calculate_williams_r(df)
        
        wr_now = wr.iloc[-1]
        wr_prev = wr.iloc[-2]
        wr_prev2 = wr.iloc[-3]
        
        conditions = [
            wr_now > oversold,                    # Coming out of oversold
            wr_prev < oversold or wr_prev2 < oversold,  # Was recently oversold
            wr_now > wr_prev,                     # Moving up
            wr_now < -20,                         # Not overbought yet
            wr_now - wr_prev > 5,                 # Momentum
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
            indicators={"williams_r": round(wr_now, 2)}
        ))
    
    return outputs
