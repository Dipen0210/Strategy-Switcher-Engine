# layers/L3_strategy_universe/base_strategy.py
"""
Base Strategy Module - Standard Output Format.

All strategies MUST return StrategyOutput to ensure consistent
scoring across the multi-strategy ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class StrategyOutput:
    """
    Standard output format for all strategies.
    
    This ensures consistent scoring in L6 using the 4-factor formula:
    Score = Signal × Confidence × Stability × BanditWeight
    """
    ticker: str
    signal: int         # +1 (Buy), 0 (Hold), -1 (Sell)
    confidence: float   # 0.0 to 1.0 (pattern strength NOW)
    strategy_name: str
    pod: str            # Pod category: "Bull-Quiet", "Bull-Volatile", "Sideways", "Crisis", "Others"
    regime: str         # Compatible regime: "Bull-Quiet", "Bull-Volatile", etc.
    
    # Optional metadata
    indicators: Optional[dict] = None  # Raw indicator values for debugging
    
    def __post_init__(self):
        """Validate output values."""
        if self.signal not in (-1, 0, 1):
            raise ValueError(f"Signal must be -1, 0, or 1. Got: {self.signal}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0. Got: {self.confidence}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame compatibility."""
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "confidence": self.confidence,
            "strategy_name": self.strategy_name,
            "pod": self.pod,
            "regime": self.regime,
        }


def compute_signal_and_confidence(
    conditions: list[bool],
    buy_threshold: int = 3,
    sell_threshold: int = 3,
) -> tuple[int, float]:
    """
    Helper to compute signal and confidence from a list of conditions.
    
    Args:
        conditions: List of boolean conditions (True = bullish, False = bearish)
        buy_threshold: Minimum True conditions for BUY signal
        sell_threshold: Minimum False conditions for SELL signal
    
    Returns:
        (signal, confidence) tuple
    
    Example:
        conditions = [rsi > 50, macd > 0, price > sma, volume_spike]
        signal, confidence = compute_signal_and_confidence(conditions)
    """
    n_true = sum(conditions)
    n_total = len(conditions)
    
    # Confidence = proportion of aligned conditions
    confidence = n_true / n_total if n_total > 0 else 0.5
    
    # Signal based on thresholds
    if n_true >= buy_threshold:
        signal = 1  # BUY
    elif (n_total - n_true) >= sell_threshold:
        signal = -1  # SELL
        confidence = 1.0 - confidence  # Flip confidence for bearish
    else:
        signal = 0  # HOLD
        confidence = 0.5  # Neutral confidence
    
    return signal, round(confidence, 4)


def safe_get(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    """Safely get the last value from a DataFrame column."""
    if column in df.columns and len(df) > 0:
        val = df[column].iloc[-1]
        return float(val) if not pd.isna(val) else default
    return default
