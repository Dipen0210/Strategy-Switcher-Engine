# layers/L3_strategy_universe/trend/ichimoku.py
"""
Ichimoku Cloud Strategy.

Pod: Trend
Regime: Bull-Quiet (Trend + Low Vol)

Uses Ichimoku Cloud (Kumo) for trend identification.
BUY when price above cloud and Tenkan > Kijun.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Ichimoku Cloud"
POD = "Bull-Quiet"
REGIME = "Bull-Quiet"
LOOKBACK = 52
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_ichimoku(df: pd.DataFrame) -> dict:
    """Calculate Ichimoku Cloud components."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    
    # Tenkan-sen (Conversion Line) - 9 period
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    
    # Kijun-sen (Base Line) - 26 period
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    
    # Senkou Span A (Leading Span A) - (Tenkan + Kijun) / 2, shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B) - 52 period high+low / 2, shifted 26
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    
    # Chikou Span (Lagging Span) - Close shifted back 26 periods
    chikou = close.shift(-26)
    
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }


def run_ichimoku(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Ichimoku Cloud strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK + 26:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        ichi = calculate_ichimoku(df)
        close = df["Close"].iloc[-1]
        
        tenkan = ichi["tenkan"].iloc[-1]
        kijun = ichi["kijun"].iloc[-1]
        senkou_a = ichi["senkou_a"].iloc[-1]
        senkou_b = ichi["senkou_b"].iloc[-1]
        
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        conditions = [
            close > cloud_top,                    # Price above cloud
            tenkan > kijun,                       # Tenkan above Kijun (bullish)
            close > tenkan,                       # Price above Tenkan
            senkou_a > senkou_b,                  # Green cloud (bullish)
            (close - cloud_top) / close < 0.05,  # Not too extended
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
                "tenkan": round(tenkan, 2),
                "kijun": round(kijun, 2),
                "cloud_top": round(cloud_top, 2),
            }
        ))
    
    return outputs
