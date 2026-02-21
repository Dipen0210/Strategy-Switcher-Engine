# layers/L3_strategy_universe/trend/macd_trend.py
"""
MACD Trend Strategy.

Pod: Trend
Regime: Bull-Quiet (Trend + Low Vol)

Generates signals based on MACD line crossing signal line,
with histogram momentum confirmation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "MACD Trend"
POD = "Bull-Quiet"
REGIME = "Bull-Quiet"
LOOKBACK = 35
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal line, and Histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def run_macd_trend(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """
    Run MACD Trend strategy on all tickers.
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
        macd_line, signal_line, histogram = calculate_macd(close)
        
        # Current values
        macd_now = macd_line.iloc[-1]
        signal_now = signal_line.iloc[-1]
        hist_now = histogram.iloc[-1]
        hist_prev = histogram.iloc[-2]
        macd_prev = macd_line.iloc[-1]
        signal_prev = signal_line.iloc[-2]
        
        # Bullish conditions
        conditions = [
            macd_now > signal_now,              # MACD above signal
            macd_prev <= signal_prev or macd_now > macd_prev,  # Crossover or momentum
            hist_now > 0,                       # Positive histogram
            hist_now > hist_prev,               # Histogram increasing
            macd_now > 0,                       # MACD positive (above zero line)
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
                "macd": round(macd_now, 4),
                "signal": round(signal_now, 4),
                "histogram": round(hist_now, 4),
            }
        ))
    
    return outputs
