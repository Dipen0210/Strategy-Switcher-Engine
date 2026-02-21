# layers/L6_scoring/__init__.py
"""
Layer 6 - Strategy Scoring and Selection.

Implements the 4-factor scoring formula:
Score = Signal × Confidence × Stability × BanditWeight
"""

from layers.L6_scoring.scorer import (
    compute_strategy_score,
    select_winner,
    score_all_strategies,
)

__all__ = ["compute_strategy_score", "select_winner", "score_all_strategies"]
