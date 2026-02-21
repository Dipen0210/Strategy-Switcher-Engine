# layers/L3_strategy_universe/trend/parabolic_sar.py
"""
Parabolic SAR Strategy.

Pod: Trend
Regime: Bull-Quiet (Trend + Low Vol)

Uses Parabolic SAR to identify trend direction and potential reversals.
BUY when price is above SAR (uptrend).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Parabolic SAR"
POD = "Bull-Quiet"
REGIME = "Bull-Quiet"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """Calculate Parabolic SAR."""
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)
    
    sar = np.zeros(n)
    ep = np.zeros(n)  # Extreme point
    af = np.zeros(n)  # Acceleration factor
    uptrend = np.ones(n, dtype=bool)
    
    # Initialize
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = af_start
    uptrend[0] = True
    
    for i in range(1, n):
        # Calculate SAR
        if uptrend[i-1]:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = min(sar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
        else:
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            sar[i] = max(sar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
        
        # Check for reversal
        if uptrend[i-1]:
            if low[i] < sar[i]:
                uptrend[i] = False
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                uptrend[i] = True
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:
            if high[i] > sar[i]:
                uptrend[i] = True
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                uptrend[i] = False
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    return pd.Series(sar, index=df.index), pd.Series(uptrend, index=df.index)


def run_parabolic_sar(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Parabolic SAR strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        sar, uptrend = calculate_parabolic_sar(df)
        close = df["Close"]
        
        sar_now = sar.iloc[-1]
        close_now = close.iloc[-1]
        uptrend_now = uptrend.iloc[-1]
        uptrend_prev = uptrend.iloc[-2]
        gap = abs(close_now - sar_now) / close_now
        
        conditions = [
            uptrend_now,                          # In uptrend
            close_now > sar_now,                  # Price above SAR
            not uptrend_prev or uptrend_now,      # Just flipped or continuing
            gap > 0.005,                          # Meaningful gap
            gap < 0.05,                           # Not too extended
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
            indicators={"sar": round(sar_now, 2), "uptrend": uptrend_now}
        ))
    
    return outputs
