# layers/L3_strategy_universe/reversion/vwap_bounce.py
"""
VWAP Bounce Strategy.

Pod: Reversion
Regime: Bull-Volatile (Trend + High Vol)

BUY when price bounces off VWAP from below.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "VWAP Bounce"
POD = "Bull-Volatile"
REGIME = "Bull-Volatile"
LOOKBACK = 20
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate VWAP (Volume Weighted Average Price)."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume = df["Volume"]
    
    # Rolling VWAP
    vwap = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
    return vwap


def run_vwap_bounce(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run VWAP Bounce strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK or "Volume" not in df.columns:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        vwap = calculate_vwap(df)
        close = df["Close"]
        low = df["Low"]
        
        vwap_now = vwap.iloc[-1]
        close_now = close.iloc[-1]
        close_prev = close.iloc[-2]
        low_now = low.iloc[-1]
        
        # Check for bounce
        touched_vwap = low_now <= vwap_now * 1.01
        bounced = close_now > close_prev and close_now > vwap_now * 0.99
        
        conditions = [
            touched_vwap or close_prev < vwap.iloc[-2],  # Touched or was below VWAP
            bounced or close_now > vwap_now,      # Bouncing
            close_now > close_prev,               # Price rising
            abs(close_now - vwap_now) / vwap_now < 0.02,  # Close to VWAP
            df["Volume"].iloc[-1] > df["Volume"].rolling(20).mean().iloc[-1],  # Volume
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
            indicators={"vwap": round(vwap_now, 2)}
        ))
    
    return outputs
