# layers/L3_strategy_universe/volatility/volatility_scalper.py
"""
Volatility Scalper Strategy.

Pod: Volatility
Regime: Bull-Volatile (Trend + High Vol)

Uses ATR to identify high volatility periods and trade breakouts.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Volatility Scalper"
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


def run_volatility_scalper(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Volatility Scalper strategy."""
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
        atr_avg = atr.iloc[-20:].mean()
        close_now = close.iloc[-1]
        close_prev = close.iloc[-2]
        high_now = high.iloc[-1]
        high_prev = high.iloc[-2]
        
        # High volatility + breakout
        vol_spike = atr_now > atr_avg * 1.2
        breakout = high_now > high.iloc[-5:-1].max()
        momentum_up = close_now > close_prev
        
        conditions = [
            vol_spike,                            # Volatility expanding
            breakout or close_now > close.iloc[-5:-1].max(),  # Breaking out
            momentum_up,                          # Moving up
            close_now > close.rolling(10).mean().iloc[-1],  # Above short MA
            atr_now / close_now < 0.05,          # Vol not extreme
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
            indicators={"atr": round(atr_now, 2), "atr_avg": round(atr_avg, 2)}
        ))
    
    return outputs
