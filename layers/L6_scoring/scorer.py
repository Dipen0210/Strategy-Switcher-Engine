# layers/L6_scoring/scorer.py
"""
Strategy Scoring — New 3-Factor Formula.

Score = 0.5 × θ_B + 0.3 × HMM_confidence + 0.2 × Stability

Each bandit acts at a DISTINCT step:
    Bandit A → Already acted at regime blending (Step 2)
    Bandit B → θ_B used here for strategy ranking (Step 6)
    Bandit C → Acts separately as stock-level filter (Step 7)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Scoring weights (hyperparameters — tune after 3 months of live data)
W_BANDIT_B = 0.5        # Bandit B signal (most adaptive, live-updated)
W_HMM_CONFIDENCE = 0.3  # HMM regime certainty
W_STABILITY = 0.2       # Regime stability (regime-flip penalty)


@dataclass
class ScoredStrategy:
    """Result of scoring a single strategy."""
    strategy_name: str
    theta_b: float           # Thompson sample from Bandit B
    hmm_confidence: float    # Max blended posterior
    stability: float         # Regime stability score
    final_score: float       # 0.5×θ_B + 0.3×conf + 0.2×stab


def score_strategy(
    strategy_name: str,
    theta_b: float,
    hmm_confidence: float,
    stability: float,
) -> ScoredStrategy:
    """
    Score a single strategy using the 3-factor formula.

    Args:
        strategy_name: Name of the strategy
        theta_b: Thompson sample from Bandit B (regime-level)
        hmm_confidence: Max blended posterior from HMM × Bandit A
        stability: Regime stability (1.0 = stable, 0.0 = flipping)

    Returns:
        ScoredStrategy with computed final_score
    """
    final_score = (
        W_BANDIT_B * theta_b
        + W_HMM_CONFIDENCE * hmm_confidence
        + W_STABILITY * stability
    )

    return ScoredStrategy(
        strategy_name=strategy_name,
        theta_b=theta_b,
        hmm_confidence=hmm_confidence,
        stability=stability,
        final_score=final_score,
    )


def rank_and_score(
    strategy_theta_pairs: list[tuple[str, float]],
    hmm_confidence: float,
    stability: float,
) -> list[ScoredStrategy]:
    """
    Score and rank multiple strategies.

    Args:
        strategy_theta_pairs: [(strategy_name, θ_B), ...] from Bandit B ranking
        hmm_confidence: Max blended posterior
        stability: Regime stability score

    Returns:
        List of ScoredStrategy sorted by final_score descending
    """
    scored = [
        score_strategy(name, theta_b, hmm_confidence, stability)
        for name, theta_b in strategy_theta_pairs
    ]
    scored.sort(key=lambda s: s.final_score, reverse=True)
    return scored


# Legacy compatibility aliases
def compute_strategy_score(strategy_output=None, **kwargs) -> Optional[ScoredStrategy]:
    """Legacy wrapper — returns a ScoredStrategy with basic defaults."""
    if strategy_output is None:
        return None
    return ScoredStrategy(
        strategy_name=getattr(strategy_output, 'strategy_name', 'Unknown'),
        theta_b=kwargs.get('regime_weight', 0.5),
        hmm_confidence=kwargs.get('stability', 0.5),
        stability=kwargs.get('stability', 0.5),
        final_score=0.5,
    )


def select_winner(scored_strategies, min_score_threshold=0.05):
    """Legacy wrapper — returns top strategy."""
    if scored_strategies:
        return scored_strategies[0]
    return None


def score_all_strategies(*args, **kwargs):
    """Legacy placeholder."""
    return []
