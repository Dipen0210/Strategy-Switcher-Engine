# layers/L3_strategy_universe/reversion/dip_buyer_rsi.py
"""
Dip Buyer RSI Strategy.

Pod: Reversion
Regime: Bull-Volatile (Trend + High Vol)

BUY when RSI is oversold (< 30) and starts turning up.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Dip Buyer RSI"
POD = "Bull-Volatile"
REGIME = "Bull-Volatile"
LOOKBACK = 20
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def run_dip_buyer_rsi(
    stock_data_dict: dict[str, pd.DataFrame],
    oversold: float = 30.0,
) -> list[StrategyOutput]:
    """Run Dip Buyer RSI strategy."""
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
        sma_50 = close.rolling(50).mean()
        
        rsi_now = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2]
        rsi_prev2 = rsi.iloc[-3] if len(rsi) > 2 else rsi_prev
        
        conditions = [
            rsi_now > rsi_prev,                   # Turning up
            rsi_prev < oversold or rsi_prev2 < oversold,  # Was oversold
            rsi_now < 50,                         # Not overbought yet
            close.iloc[-1] > close.iloc[-2],      # Price up
            close.iloc[-1] > sma_50.iloc[-1] * 0.9,  # Not crashed
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
            indicators={"rsi": round(rsi_now, 2)}
        ))
    
    return outputs
