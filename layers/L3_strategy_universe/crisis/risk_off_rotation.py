# layers/L3_strategy_universe/hedge/risk_off_rotation.py
"""
Risk-Off Rotation Strategy.

Pod: Defensive
Regime: Crisis (Bear + High Vol)

Recommends rotating to defensive sectors/assets.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Risk-Off Rotation"
POD = "Crisis"
REGIME = "Crisis"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_risk_off_rotation(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Risk-Off Rotation strategy."""
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
        
        # Calculate metrics
        momentum = (close.iloc[-1] / close.iloc[-20]) - 1
        vol = returns.iloc[-20:].std() * np.sqrt(252)
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        
        # Beta proxy (higher vol = higher beta = more risk)
        high_beta = vol > 0.25
        
        # Bearish conditions
        bearish_conditions = [
            momentum < -0.05,                     # Negative momentum
            close.iloc[-1] < sma_50,              # Below MA
            high_beta,                            # High beta stock
            close.iloc[-1] < close.iloc[-5],      # 5-day decline
            vol > returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else True,  # Vol expanding
        ]
        
        n_bearish = sum(bearish_conditions)
        
        if n_bearish >= 4:
            signal = -1
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
            indicators={"momentum": round(momentum, 4), "vol": round(vol, 4)}
        ))
    
    return outputs
