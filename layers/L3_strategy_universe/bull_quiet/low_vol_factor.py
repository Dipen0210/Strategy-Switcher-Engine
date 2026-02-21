# layers/L3_strategy_universe/factor/low_vol_factor.py
"""
Low Volatility Factor Strategy.

Pod: Factor
Regime: Bull-Quiet (Trend + Low Vol)

Favors stocks with lower realized volatility.
The "low volatility anomaly" - lower vol stocks often outperform.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Low Vol Factor"
POD = "Bull-Quiet"
REGIME = "Bull-Quiet"
LOOKBACK = 60
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_low_vol_factor(
    stock_data_dict: dict[str, pd.DataFrame],
    vol_lookback: int = 20,
    vol_threshold: float = 0.20,  # Annualized vol threshold
) -> list[StrategyOutput]:
    """
    Run Low Volatility Factor strategy.
    
    BUY stocks with low realized volatility that are also trending up.
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
        returns = close.pct_change()
        
        # Calculate realized volatility (annualized)
        realized_vol = returns.iloc[-vol_lookback:].std() * np.sqrt(252)
        
        # Additional metrics
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        close_now = close.iloc[-1]
        momentum = (close_now / close.iloc[-20]) - 1
        
        # Low vol with positive trend = good
        conditions = [
            realized_vol < vol_threshold,          # Low volatility
            close_now > sma_20,                   # Above short-term MA
            close_now > sma_50,                   # Above medium-term MA
            momentum > 0,                          # Positive momentum
            realized_vol > 0.05,                   # Not dead (some vol)
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
                "realized_vol": round(realized_vol, 4),
                "momentum": round(momentum, 4),
            }
        ))
    
    return outputs
