# layers/L3_strategy_universe/hedge/vix_spike.py
"""
VIX Spike Strategy.

Pod: Hedge
Regime: Crisis (Bear + High Vol)

Signals based on volatility spikes (proxy via returns volatility).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from layers.L3_strategy_universe.base_strategy import (
    StrategyOutput,
    compute_signal_and_confidence,
)


STRATEGY_NAME = "VIX Spike"
POD = "Crisis"
REGIME = "Crisis"
LOOKBACK = 30
TIMEFRAME = LOOKBACK  # Common variable for pipeline past-return calculation


def run_vix_spike(
    stock_data_dict: dict[str, pd.DataFrame],
) -> list[StrategyOutput]:
    """Run VIX Spike strategy (using realized vol as VIX proxy)."""
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
        
        # Realized vol as VIX proxy
        vol_5d = returns.iloc[-5:].std() * np.sqrt(252)
        vol_20d = returns.iloc[-20:].std() * np.sqrt(252)
        vol_60d = returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else vol_20d
        
        # VIX spike conditions
        vol_spike = vol_5d > vol_20d * 1.5
        vol_elevated = vol_20d > 0.25
        vol_expanding = vol_5d > vol_60d
        
        spike_conditions = [
            vol_spike,                            # Short-term vol spike
            vol_elevated,                         # Vol above threshold
            vol_expanding,                        # Vol regime shift
            close.iloc[-1] < close.iloc[-5],      # Price falling
            returns.iloc[-1] < 0,                 # Today negative
        ]
        
        n_spike = sum(spike_conditions)
        
        if n_spike >= 4:
            signal = -1
            confidence = n_spike / len(spike_conditions)
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
            indicators={"vol_5d": round(vol_5d, 4), "vol_20d": round(vol_20d, 4)}
        ))
    
    return outputs
