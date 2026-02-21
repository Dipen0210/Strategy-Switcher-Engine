# layers/L3_strategy_universe/volatility/atr_trailing.py
"""
ATR Trailing Stop Strategy.

Pod: Volatility
Regime: Bull-Volatile (Trend + High Vol)

Uses ATR to set dynamic trailing stops and entry points.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "ATR Trailing"
POD = "Bull-Volatile"
REGIME = "Bull-Volatile"
LOOKBACK = 20
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()


def run_atr_trailing(
    stock_data_dict: dict[str, pd.DataFrame],
    atr_multiplier: float = 2.0,
) -> list[StrategyOutput]:
    """Run ATR Trailing Stop strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        atr = calculate_atr(df)
        close = df["Close"]
        high = df["High"]
        
        atr_now = atr.iloc[-1]
        close_now = close.iloc[-1]
        
        # Chandelier Exit style trailing stop
        highest_high_22 = high.iloc[-22:].max()
        trailing_stop = highest_high_22 - atr_multiplier * atr_now
        
        # Entry: price above trailing stop and trending
        sma_20 = close.rolling(20).mean().iloc[-1]
        
        conditions = [
            close_now > trailing_stop,            # Above stop (uptrend intact)
            close_now > sma_20,                   # Above SMA
            close_now > close.iloc[-2],           # Moving up
            (close_now - trailing_stop) / close_now < 0.05,  # Close to stop
            atr_now > atr.iloc[-10:].mean() * 0.8,  # Reasonable volatility
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
                "atr": round(atr_now, 2),
                "trailing_stop": round(trailing_stop, 2),
            }
        ))
    
    return outputs
