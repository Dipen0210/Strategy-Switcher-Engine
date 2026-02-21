# layers/L3_strategy_universe/reversion/cci_reversal.py
"""
CCI Reversal Strategy.

Pod: Reversion
Regime: Bull-Volatile (Trend + High Vol)

BUY when CCI comes back from extreme oversold (< -100).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "CCI Reversal"
POD = "Bull-Volatile"
REGIME = "Bull-Volatile"
LOOKBACK = 25
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return cci


def run_cci_reversal(
    stock_data_dict: dict[str, pd.DataFrame],
    oversold: float = -100.0,
) -> list[StrategyOutput]:
    """Run CCI Reversal strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        cci = calculate_cci(df)
        
        cci_now = cci.iloc[-1]
        cci_prev = cci.iloc[-2]
        cci_prev2 = cci.iloc[-3]
        
        conditions = [
            cci_now > oversold,                   # Above oversold
            cci_prev < oversold or cci_prev2 < oversold,  # Was oversold
            cci_now > cci_prev,                   # Turning up
            cci_now < 100,                        # Not overbought
            abs(cci_now - cci_prev) > 20,         # Momentum
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
            indicators={"cci": round(cci_now, 2)}
        ))
    
    return outputs
