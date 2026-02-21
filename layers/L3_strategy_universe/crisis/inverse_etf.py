# layers/L3_strategy_universe/short/inverse_etf.py
"""
Inverse ETF Strategy.

Pod: Short
Regime: Crisis (Bear + High Vol)

Signals when inverse/short exposure is appropriate.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "Inverse ETF"
POD = "Crisis"
REGIME = "Crisis"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_inverse_etf(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run Inverse ETF strategy."""
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
        
        # Trend and volatility
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        vol = returns.iloc[-20:].std() * np.sqrt(252)
        
        # Drawdown
        high_30d = close.iloc[-30:].max()
        drawdown = (high_30d - close.iloc[-1]) / high_30d
        
        # Inverse conditions
        inverse_conditions = [
            close.iloc[-1] < sma_20,              # Below short MA
            close.iloc[-1] < sma_50,              # Below medium MA
            sma_20 < sma_50,                      # Bearish cross
            drawdown > 0.1,                       # 10%+ drawdown
            vol > 0.25,                           # High volatility
        ]
        
        n_inverse = sum(inverse_conditions)
        
        if n_inverse >= 4:
            signal = -1  # SELL / short
            confidence = n_inverse / len(inverse_conditions)
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
