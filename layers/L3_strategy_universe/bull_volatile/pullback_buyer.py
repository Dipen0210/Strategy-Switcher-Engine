# layers/L3_strategy_universe/reversion/pullback_buyer.py
"""
Pullback Buyer Strategy.

Pod: Reversion
Regime: Bull-Volatile (Trend + High Vol)

Buy the dip in an uptrend. Wait for pullback to support
before entering long position.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Pullback Buyer"
POD = "Bull-Volatile"
REGIME = "Bull-Volatile"
LOOKBACK = 50
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_pullback_buyer(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """
    Run Pullback Buyer strategy.
    
    BUY when price pulls back to 20-day SMA in an uptrend (50-day).
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
        low = df["Low"]
        
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        
        close_now = close.iloc[-1]
        low_now = low.iloc[-1]
        sma_20_now = sma_20.iloc[-1]
        sma_50_now = sma_50.iloc[-1]
        
        # Check for pullback
        was_above = close.iloc[-5:-1].min() > sma_20.iloc[-5:-1].min()
        touched_support = low_now <= sma_20_now * 1.02
        
        # Trend is up
        trend_up = sma_20_now > sma_50_now
        price_above_50 = close_now > sma_50_now
        
        # RSI oversold check
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_now = rsi.iloc[-1]
        
        conditions = [
            trend_up,                              # Overall uptrend
            price_above_50,                        # Not broken down
            touched_support or close_now <= sma_20_now * 1.01,  # At support
            rsi_now < 50,                          # Not overbought
            close_now > close.iloc[-1] * 0.95,     # Not crashing
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
                "sma_20": round(sma_20_now, 2),
                "rsi": round(rsi_now, 2),
            }
        ))
    
    return outputs
