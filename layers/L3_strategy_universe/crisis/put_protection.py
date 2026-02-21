# layers/L3_strategy_universe/hedge/put_protection.py
"""
Put Protection Strategy.

Pod: Hedge
Regime: Crisis (Bear + High Vol)

Signals when protective puts should be considered.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Put Protection"
POD = "Crisis"
REGIME = "Crisis"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_put_protection(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Put Protection strategy."""
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
        
        # Volatility and trend
        vol = returns.iloc[-20:].std() * np.sqrt(252)
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        current = close.iloc[-1]
        
        # Conditions for needing protection
        protection_conditions = [
            current < sma_20,                     # Below short MA
            sma_20 < sma_50,                      # Bearish alignment
            vol > 0.2,                            # Elevated vol
            returns.iloc[-5:].sum() < -0.03,      # Recent decline
            current < close.iloc[-10],            # 10-day lower
        ]
        
        n_protect = sum(protection_conditions)
        
        if n_protect >= 4:
            signal = -1  # Need protection / reduce exposure
            confidence = n_protect / len(protection_conditions)
        else:
            signal = 0
            confidence = 0.5
        
        outputs.append(StrategyOutput(
            ticker=ticker,
            signal=signal,
            confidence=round(confidence, 4),
            strategy_name=STRATEGY_NAME,
            pod=POD,
            regime=REGIME,
            indicators={"vol": round(vol, 4)}
        ))
    
    return outputs
