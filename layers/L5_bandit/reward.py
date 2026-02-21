# layers/L5_bandit/reward.py
"""
Risk-Adjusted Reward Function for Bandit Updates.

Used by all three bandits (Global, Regime, Stock) after each trade.

Reward formula:
    R = sign(P) · ln(1 + |P|) · min(1/σ₆₀, 10)

Where:
    P = daily percentage return of the strategy
    σ₆₀ = 60-day rolling volatility (consistency factor)

The log transform compresses outliers so a random 4% jump doesn't
dominate the bandit's weights. The consistency factor penalizes
high-variance strategies and rewards stable performers.
"""

from __future__ import annotations

import math
from typing import Optional


def compute_reward(daily_return: float, vol_60d: float) -> float:
    """
    Compute risk-adjusted reward from trade P&L.

    R = sign(P) · ln(1 + |P|) · min(1/σ₆₀, 10)

    Args:
        daily_return: Daily percentage return of the strategy (e.g., 0.015 for 1.5%)
        vol_60d: 60-day rolling volatility (annualized)

    Returns:
        R_final: Risk-adjusted reward, typically in range [-3, 3]
    """
    sign = 1.0 if daily_return >= 0 else -1.0
    log_component = math.log(1 + abs(daily_return))

    # Consistency factor: inverse of volatility, capped at 10
    # Prevents explosion when σ is very small (e.g., σ=0.02 → 1/σ=50 → capped to 10)
    consistency = min(1.0 / max(vol_60d, 0.01), 10.0)

    return sign * log_component * consistency


def reward_to_beta(r_final: float) -> float:
    """
    Convert R_final to [0, 1] range for Beta bandit updates.

    Maps R ∈ [-3, 3] → [0, 1] linearly.
    - R = -3 → 0.0 (total failure)
    - R =  0 → 0.5 (neutral)
    - R = +3 → 1.0 (total success)

    Args:
        r_final: Raw reward from compute_reward()

    Returns:
        R_beta in [0, 1] suitable for Beta(α, β) updates
    """
    clipped = max(-3.0, min(3.0, r_final))
    return (clipped + 3.0) / 6.0


# --- Asymmetric Penalty Overrides ---

# Catastrophic loss (R < -2.5): strategy blew up
PENALTY_CATASTROPHIC = {"A": 0.3, "B": 0.1, "C": 0.0}

# Ambiguous regime + loss: HMM was uncertain and we lost
PENALTY_AMBIGUOUS = {"A": 0.0, "B": 0.2, "C": 0.2}

# Exceptional win (R > 2.0): everything worked
REWARD_EXCEPTIONAL = {"A": 0.7, "B": 0.9, "C": 1.0}


def get_differentiated_rewards(
    r_final: float,
    is_ambiguous_regime: bool = False,
) -> dict[str, float]:
    """
    Compute differentiated rewards for each bandit level.

    Normal case:
        Bandit A (Global):  R × 0.4 (dampened — regime calls need more evidence)
        Bandit B (Regime):  R × 0.7 (moderate)
        Bandit C (Stock):   R × 1.0 (full signal — most accountable)

    Catastrophic (R < -2.5):
        Force hard penalties: A=0.3, B=0.1, C=0.0

    Ambiguous regime + loss:
        Punish A hardest: A=0.0, B=0.2, C=0.2

    Exceptional (R > 2.0):
        Reward C most: A=0.7, B=0.9, C=1.0

    Args:
        r_final: Raw reward from compute_reward()
        is_ambiguous_regime: True if regime confidence was < 0.55

    Returns:
        Dict with keys "A", "B", "C" → reward values in [0, 1]
    """
    # Catastrophic loss override
    if r_final < -2.5:
        return dict(PENALTY_CATASTROPHIC)

    # Ambiguous regime + loss
    if is_ambiguous_regime and r_final < 0:
        return dict(PENALTY_AMBIGUOUS)

    # Exceptional win override
    if r_final > 2.0:
        return dict(REWARD_EXCEPTIONAL)

    # Normal case: differentiated dampening
    return {
        "A": reward_to_beta(r_final * 0.4),
        "B": reward_to_beta(r_final * 0.7),
        "C": reward_to_beta(r_final * 1.0),
    }
