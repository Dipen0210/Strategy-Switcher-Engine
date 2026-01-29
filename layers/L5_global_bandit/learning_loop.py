# layers/L6_online_learning/learning_loop.py
"""
LAYER 6 — ONLINE LEARNING LOOP

Always-on adaptation layer.

Frequency: Weekly (or user-defined rebalance)

Loop:
1. Observe context
2. Apply hard filters
3. Sample strategy via Thompson Sampling
4. Execute strategy
5. Observe realized reward
6. Update posterior immediately

Properties:
- Incremental
- Stateless
- O(1) updates
- No retraining cycles
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class LearningContext:
    """Context for a single learning step."""
    timestamp: datetime
    regime_probs: dict
    volatility: float
    momentum: float
    drawdown: float
    risk_score: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to context vector for bandit."""
        trend_prob = self.regime_probs.get("Trend + Low Vol", 0) + \
                     self.regime_probs.get("Trend + High Vol", 0)
        high_vol_prob = self.regime_probs.get("Trend + High Vol", 0) + \
                        self.regime_probs.get("Crisis", 0)
        
        return np.array([
            trend_prob,
            high_vol_prob,
            np.clip(self.volatility, 0, 1),
            np.clip(self.momentum, -1, 1),
            np.clip(abs(self.drawdown), 0, 1),
            self.risk_score,
        ], dtype=float)


@dataclass
class LearningOutcome:
    """Outcome of a learning step."""
    strategy_selected: str
    realized_return: float
    realized_volatility: float
    drawdown: float
    reward: float


class OnlineLearner:
    """
    Stateless online learning loop.
    
    Properties:
    - Each update is O(1)
    - No memory of past steps (stateless)
    - Immediate posterior updates
    """
    
    def __init__(self, bandit):
        """
        Initialize with bandit reference.
        
        Args:
            bandit: ContextualBandit instance from L5
        """
        self.bandit = bandit
        self.update_count = 0
        self.last_update: Optional[datetime] = None
    
    def observe_and_update(
        self,
        context: LearningContext,
        outcome: LearningOutcome,
    ) -> None:
        """
        Single learning step: observe outcome, update posterior.
        
        This is called AFTER a rebalance period completes.
        """
        # Convert context to vector
        context_vec = context.to_vector()
        
        # Update bandit posterior
        self.bandit.update(
            strategy=outcome.strategy_selected,
            reward=outcome.reward,
            context=context_vec,
        )
        
        self.update_count += 1
        self.last_update = context.timestamp
    
    def compute_reward(
        self,
        returns: float,
        volatility: float,
        drawdown: float,
        dd_penalty: float = 2.0,
    ) -> float:
        """
        Compute reward for bandit update.
        
        Reward = Risk-adjusted return with drawdown penalty.
        """
        if volatility > 0:
            risk_adj_return = returns / volatility
        else:
            risk_adj_return = returns
        
        dd_penalty_value = dd_penalty * abs(drawdown)
        
        return risk_adj_return - dd_penalty_value
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        return {
            "update_count": self.update_count,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "bandit_stats": self.bandit.get_stats(),
        }
