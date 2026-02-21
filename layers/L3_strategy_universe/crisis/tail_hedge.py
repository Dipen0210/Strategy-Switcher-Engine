# layers/L3_strategy_universe/hedge/tail_hedge.py
"""
Tail Hedge Strategy.

Pod: Hedge
Regime: Crisis (Bear + High Vol)

Identifies tail risk events and recommends hedging.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Tail Hedge"
POD = "Crisis"
REGIME = "Crisis"
LOOKBACK = 60
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_tail_hedge(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Tail Hedge strategy."""
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
        
        # Tail risk metrics
        recent_return = returns.iloc[-1]
        vol = returns.iloc[-20:].std()
        z_score = recent_return / (vol + 1e-10)
        
        # Drawdown
        high_60d = close.iloc[-60:].max()
        current = close.iloc[-1]
        drawdown = (high_60d - current) / high_60d
        
        # Count negative days
        negative_days = (returns.iloc[-10:] < 0).sum()
        
        # Tail event conditions
        tail_conditions = [
            z_score < -2,                         # Tail event today
            drawdown > 0.15,                      # Significant drawdown
            negative_days >= 7,                   # Many down days
            returns.iloc[-5:].mean() < -0.01,     # Sustained decline
            current < close.rolling(50).mean().iloc[-1] if len(close) >= 50 else True,
        ]
        
        n_tail = sum(tail_conditions)
        
        if n_tail >= 3:
            signal = -1
            confidence = n_tail / len(tail_conditions)
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
            indicators={"z_score": round(z_score, 2), "drawdown": round(drawdown, 4)}
        ))
    
    return outputs
