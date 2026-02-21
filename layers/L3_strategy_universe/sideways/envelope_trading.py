# layers/L3_strategy_universe/range/envelope_trading.py
"""
Envelope Trading Strategy.

Pod: Range
Regime: Sideways (Range + Low Vol)

Uses percentage envelopes around a moving average.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Envelope Trading"
POD = "Sideways"
REGIME = "Sideways"
LOOKBACK = 25
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_envelope_trading(
    stock_data_dict: dict[str, pd.DataFrame],
    envelope_pct: float = 0.025,  # 2.5% envelope
) -> list[StrategyOutput]:
    """Run Envelope Trading strategy."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        sma = close.rolling(20).mean()
        
        sma_now = sma.iloc[-1]
        upper_env = sma_now * (1 + envelope_pct)
        lower_env = sma_now * (1 - envelope_pct)
        
        current = close.iloc[-1]
        current_low = df["Low"].iloc[-1]
        
        # Position relative to envelope (0 = lower, 1 = upper)
        env_position = (current - lower_env) / (upper_env - lower_env + 1e-10)
        
        conditions = [
            current_low <= lower_env * 1.01,      # At lower envelope
            current > close.iloc[-2],             # Bouncing
            current > lower_env,                  # Haven't broken down
            env_position < 0.5,                   # Still in lower half
            current < sma_now,                    # Below middle
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
                "sma": round(sma_now, 2),
                "env_pos": round(env_position, 2),
            }
        ))
    
    return outputs
