# layers/L3_strategy_universe/trend/trend_following.py
"""
Trend Following Strategy - EMA Crossover.

Pod: Trend
Regime: Bull-Quiet (Trend + Low Vol)

Generates BUY signal when short EMA crosses above long EMA,
with price above both EMAs confirming the trend.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
    safe_get,
)


STRATEGY_NAME = "Trend Following"
POD = "Bull-Quiet"
REGIME = "Bull-Quiet"
LOOKBACK = 50
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def run_trend_following(
    stock_data_dict: dict[str, pd.DataFrame],
    short_period: int = 12,
    long_period: int = 26,
) -> list[StrategyOutput]:
    """
    Run Trend Following strategy on all tickers.
    
    Args:
        stock_data_dict: Dict of ticker -> OHLCV DataFrame
        short_period: Short EMA period (default 12)
        long_period: Long EMA period (default 26)
    
    Returns:
        List of StrategyOutput for each ticker
    """
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < long_period + 5:
            outputs.append(StrategyOutput(
                ticker=ticker,
                signal=0,
                confidence=0.0,
                strategy_name=STRATEGY_NAME,
                pod=POD,
                regime=REGIME,
            ))
            continue
        
        # Calculate EMAs
        close = df["Close"]
        ema_short = calculate_ema(close, short_period)
        ema_long = calculate_ema(close, long_period)
        
        # Current values
        current_price = close.iloc[-1]
        current_short = ema_short.iloc[-1]
        current_long = ema_long.iloc[-1]
        prev_short = ema_short.iloc[-2]
        prev_long = ema_long.iloc[-2]
        
        # Conditions for bullish trend
        conditions = [
            current_short > current_long,           # Short EMA above long
            prev_short <= prev_long or current_short > prev_short,  # Recent crossover or momentum
            current_price > current_short,          # Price above short EMA
            current_price > current_long,           # Price above long EMA
            (current_short - current_long) / current_long > 0.001,  # Meaningful spread
        ]
        
        signal, confidence = compute_signal_and_confidence(
            conditions, 
            buy_threshold=4,
            sell_threshold=4,
        )
        
        outputs.append(StrategyOutput(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            strategy_name=STRATEGY_NAME,
            pod=POD,
            regime=REGIME,
            indicators={
                "ema_short": round(current_short, 2),
                "ema_long": round(current_long, 2),
                "spread": round((current_short - current_long) / current_long * 100, 2),
            }
        ))
    
    return outputs
