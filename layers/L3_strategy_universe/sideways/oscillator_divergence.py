# layers/L3_strategy_universe/range/oscillator_divergence.py
"""
Oscillator Divergence Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

BUY when price makes lower low but RSI makes higher low (bullish divergence).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Oscillator Divergence"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def run_oscillator_divergence(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Oscillator Divergence strategy."""
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
        
        # Find recent lows (last 10 days vs previous 10 days)
        close_recent_low = close.iloc[-10:].min()
        close_prior_low = close.iloc[-20:-10].min()
        rsi_recent_low = rsi.iloc[-10:].min()
        rsi_prior_low = rsi.iloc[-20:-10].min()
        
        # Bullish divergence: lower price low, higher RSI low
        price_lower_low = close_recent_low < close_prior_low
        rsi_higher_low = rsi_recent_low > rsi_prior_low
        divergence = price_lower_low and rsi_higher_low
        
        current = close.iloc[-1]
        
        conditions = [
            divergence,                           # Bullish divergence
            rsi.iloc[-1] > rsi.iloc[-2],         # RSI turning up
            current > close.iloc[-2],             # Price turning up
            rsi.iloc[-1] < 50,                    # Not overbought
            current > close_recent_low,           # Off the low
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
            indicators={"divergence": divergence, "rsi": round(rsi.iloc[-1], 2)}
        ))
    
    return outputs
