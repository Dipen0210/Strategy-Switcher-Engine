# layers/L3_strategy_universe/trend/adx_trend.py
"""
ADX Trend Strength Strategy.

Pod: Trend
Regime: Bull-Quiet (Trend + Low Vol)

Uses ADX to measure trend strength and +DI/-DI for direction.
Only takes positions when trend is strong (ADX > 25).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "ADX Trend"
POD = "Bull-Quiet"
REGIME = "Bull-Quiet"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, +DI, and -DI."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DI
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di


def run_adx_trend(
    stock_data_dict: dict[str, pd.DataFrame],
    adx_threshold: float = 25.0,
) -> list[StrategyOutput]:
    """
    Run ADX Trend strategy on all tickers.
    """
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        adx, plus_di, minus_di = calculate_adx(df)
        
        adx_now = adx.iloc[-1]
        plus_now = plus_di.iloc[-1]
        minus_now = minus_di.iloc[-1]
        adx_prev = adx.iloc[-2]
        
        # Conditions for strong bullish trend
        conditions = [
            adx_now > adx_threshold,            # Trend is strong
            plus_now > minus_now,               # Bullish direction
            adx_now > adx_prev,                 # Trend strengthening
            plus_now > 20,                      # +DI meaningful
            (plus_now - minus_now) > 5,         # Clear directional edge
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
                "adx": round(adx_now, 2),
                "plus_di": round(plus_now, 2),
                "minus_di": round(minus_now, 2),
            }
        ))
    
    return outputs
