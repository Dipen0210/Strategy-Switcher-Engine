# layers/L3_strategy_universe/reversion/rsi_exhaustion.py
"""
RSI Exhaustion Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

BUY when RSI shows exhaustion at lows (divergence or extreme oversold).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "RSI Exhaustion"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 25
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def run_rsi_exhaustion(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run RSI Exhaustion strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        rsi = calculate_rsi(close)
        
        rsi_now = rsi.iloc[-1]
        rsi_min_5d = rsi.iloc[-5:].min()
        close_now = close.iloc[-1]
        close_min_5d = close.iloc[-5:].min()
        
        # Check for divergence (price lower but RSI higher)
        divergence = close_now <= close_min_5d * 1.01 and rsi_now > rsi_min_5d
        
        conditions = [
            rsi_now < 35,                         # Oversold zone
            rsi_now > rsi.iloc[-2],               # Turning up
            divergence or rsi_min_5d < 25,        # Divergence or extreme
            close_now > close.iloc[-2],           # Price bouncing
            rsi_now < 50,                         # Room to run
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
            indicators={"rsi": round(rsi_now, 2), "divergence": divergence}
        ))
    
    return outputs
