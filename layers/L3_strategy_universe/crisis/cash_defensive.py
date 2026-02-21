# layers/L3_strategy_universe/hedge/cash_defensive.py
"""
Cash Defensive Strategy.

Pod: Defensive
Regime: Crisis (Bear + High Vol)

Recommends moving to cash/bonds during crisis.
Signal = SELL (reduce exposure).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Cash Defensive"
POD = "Crisis"
REGIME = "Crisis"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_cash_defensive(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Cash Defensive strategy - recommends reducing exposure."""
    outputs = []
    
    for ticker, df in stock_data_dict.items():
        if len(df) < LOOKBACK:
            outputs.append(StrategyOutput(
                ticker=ticker, signal=0, confidence=0.0,
                strategy_name=STRATEGY_NAME, pod=POD, regime=REGIME,
            ))
            continue
        
        close = df["Close"]
        
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        current = close.iloc[-1]
        
        # Drawdown from recent high
        high_20d = close.iloc[-20:].max()
        drawdown = (high_20d - current) / high_20d
        
        # Volatility spike
        returns = close.pct_change()
        vol = returns.iloc[-20:].std() * np.sqrt(252)
        
        # Bearish conditions → SELL signal
        bearish_conditions = [
            current < sma_20,                     # Below short MA
            current < sma_50,                     # Below medium MA
            sma_20 < sma_50,                      # Death cross forming
            drawdown > 0.1,                       # 10%+ drawdown
            vol > 0.25,                           # High volatility
        ]
        
        n_bearish = sum(bearish_conditions)
        
        if n_bearish >= 4:
            signal = -1  # SELL / go defensive
            confidence = n_bearish / len(bearish_conditions)
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
            indicators={"drawdown": round(drawdown, 4), "vol": round(vol, 4)}
        ))
    
    return outputs
