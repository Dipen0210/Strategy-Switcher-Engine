# ml/bandit.py
"""
Contextual Thompson Sampling Bandit for Strategy Selection.

Global bandit that learns which strategies perform best under different contexts.
- Arms: Strategy names
- Context: [regime, volatility, momentum, drawdown, risk_score]
- Reward: Risk-adjusted return with drawdown penalty

Key properties:
- Cold start with uniform priors (no warm start from backtest)
- Online learning with weekly updates
- O(1) posterior updates
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BanditState:
    """Posterior state for a single arm (strategy)."""
    # Parameters for Normal-Gamma posterior
    mu: float = 0.0          # Mean estimate
    lambda_: float = 1.0     # Precision of mean
    alpha: float = 1.0       # Shape of precision
    beta: float = 1.0        # Rate of precision
    n_pulls: int = 0         # Number of times selected
    total_reward: float = 0.0


class ContextualBandit:
    """
    Contextual Thompson Sampling for strategy selection.
    
    Uses a simplified linear contextual model where each arm has
    learned weights for context features.
    
    Cold start: All arms start with uniform priors.
    Learning: Online Bayesian updates after each reward observation.
    """
    
    def __init__(
        self, 
        strategy_names: list[str],
        context_dim: int = 5,
        prior_variance: float = 1.0
    ):
        """
        Initialize bandit with strategy arms.
        
        Args:
            strategy_names: List of strategy names (arms)
            context_dim: Dimension of context vector
            prior_variance: Prior variance for Thompson Sampling
        """
        self.strategy_names = list(strategy_names)
        self.context_dim = context_dim
        self.prior_variance = prior_variance
        
        # Initialize arm states with uniform priors (cold start)
        self.arm_states: dict[str, BanditState] = {
            name: BanditState() for name in strategy_names
        }
        
        # Context weights per arm (for contextual bandits)
        # Initialize near zero for cold start
        self.context_weights: dict[str, np.ndarray] = {
            name: np.zeros(context_dim) for name in strategy_names
        }
        
        # Precision matrices for context (identity = uninformative prior)
        self.context_precision: dict[str, np.ndarray] = {
            name: np.eye(context_dim) * 0.1 for name in strategy_names
        }
        
        self.total_updates = 0
    
    def _sample_from_posterior(self, arm: str) -> float:
        """
        Sample expected reward from posterior distribution.
        
        Uses Normal-Gamma posterior for Thompson Sampling.
        """
        state = self.arm_states[arm]
        
        # Sample precision from Gamma(alpha, beta)
        precision = np.random.gamma(state.alpha, 1.0 / state.beta)
        
        # Sample mean from Normal(mu, 1/(lambda * precision))
        variance = 1.0 / (state.lambda_ * precision + 1e-10)
        sampled_mean = np.random.normal(state.mu, np.sqrt(variance))
        
        return sampled_mean
    
    def _compute_context_score(self, arm: str, context: np.ndarray) -> float:
        """Compute context-dependent score adjustment."""
        weights = self.context_weights[arm]
        
        if context is None or len(context) == 0:
            return 0.0
        
        # Pad or truncate context to match weight dimension
        if len(context) < len(weights):
            context = np.pad(context, (0, len(weights) - len(context)))
        elif len(context) > len(weights):
            context = context[:len(weights)]
        
        return float(np.dot(weights, context))
    
    def select_strategy(
        self, 
        context: Optional[np.ndarray] = None,
        allowed_strategies: Optional[list[str]] = None
    ) -> tuple[str, dict[str, float]]:
        """
        Select strategy using Thompson Sampling.
        
        Args:
            context: Context vector (regime, vol, momentum, etc.)
            allowed_strategies: List of allowed strategies (after risk filter)
        
        Returns:
            Tuple of (selected_strategy, all_scores_dict)
        """
        if allowed_strategies is None:
            allowed_strategies = self.strategy_names
        
        # Filter to only allowed strategies
        candidates = [s for s in allowed_strategies if s in self.arm_states]
        
        if not candidates:
            candidates = self.strategy_names[:1] if self.strategy_names else ["Defensive"]
        
        # Sample from posterior for each candidate
        scores = {}
        for arm in candidates:
            # Base score from Thompson Sampling
            ts_score = self._sample_from_posterior(arm)
            
            # Context adjustment
            ctx_score = self._compute_context_score(arm, context) if context is not None else 0.0
            
            scores[arm] = ts_score + ctx_score
        
        # Select arm with highest sampled score
        selected = max(scores, key=scores.get)
        
        return selected, scores
    
    def get_strategy_scores(
        self, 
        context: Optional[np.ndarray] = None,
        allowed_strategies: Optional[list[str]] = None
    ) -> dict[str, float]:
        """
        Get expected scores for all strategies (mean, not sampled).
        
        Used for tie-breaking in deterministic ranking.
        """
        if allowed_strategies is None:
            allowed_strategies = self.strategy_names
        
        scores = {}
        for arm in allowed_strategies:
            if arm not in self.arm_states:
                scores[arm] = 0.0
                continue
            
            state = self.arm_states[arm]
            base_score = state.mu
            
            ctx_score = self._compute_context_score(arm, context) if context is not None else 0.0
            
            scores[arm] = base_score + ctx_score
        
        return scores
    
    def get_strategy_probabilities(
        self,
        context: Optional[np.ndarray] = None,
        allowed_strategies: Optional[list[str]] = None,
        n_samples: int = 100
    ) -> dict[str, float]:
        """
        Estimate selection probabilities via Monte Carlo sampling.
        """
        if allowed_strategies is None:
            allowed_strategies = self.strategy_names
        
        candidates = [s for s in allowed_strategies if s in self.arm_states]
        if not candidates:
            return {}
        
        counts = {arm: 0 for arm in candidates}
        
        for _ in range(n_samples):
            selected, _ = self.select_strategy(context, candidates)
            counts[selected] += 1
        
        return {arm: count / n_samples for arm, count in counts.items()}
    
    def update(
        self, 
        strategy: str, 
        reward: float,
        context: Optional[np.ndarray] = None
    ) -> None:
        """
        Update posterior after observing reward.
        
        Args:
            strategy: Strategy that was executed
            reward: Observed reward (risk-adjusted return)
            context: Context at time of selection
        """
        if strategy not in self.arm_states:
            return
        
        state = self.arm_states[strategy]
        
        # Bayesian update for Normal-Gamma posterior
        n = state.n_pulls
        old_mu = state.mu
        
        # Update mean
        new_lambda = state.lambda_ + 1
        new_mu = (state.lambda_ * old_mu + reward) / new_lambda
        
        # Update precision parameters
        new_alpha = state.alpha + 0.5
        new_beta = state.beta + 0.5 * state.lambda_ * (reward - old_mu) ** 2 / new_lambda
        
        # Store updates
        state.mu = new_mu
        state.lambda_ = new_lambda
        state.alpha = new_alpha
        state.beta = new_beta
        state.n_pulls = n + 1
        state.total_reward += reward
        
        # Update context weights if context provided
        if context is not None and len(context) > 0:
            self._update_context_weights(strategy, context, reward)
        
        self.total_updates += 1
    
    def _update_context_weights(
        self, 
        strategy: str, 
        context: np.ndarray, 
        reward: float,
        learning_rate: float = 0.1
    ) -> None:
        """
        Online update of context weights using gradient descent.
        """
        weights = self.context_weights[strategy]
        
        # Pad or truncate context
        if len(context) < len(weights):
            context = np.pad(context, (0, len(weights) - len(context)))
        elif len(context) > len(weights):
            context = context[:len(weights)]
        
        # Predicted score
        predicted = np.dot(weights, context)
        
        # Gradient update (minimize squared error)
        error = reward - predicted
        gradient = error * context
        
        # Apply decay to learning rate based on total updates
        effective_lr = learning_rate / (1 + 0.01 * self.total_updates)
        
        self.context_weights[strategy] = weights + effective_lr * gradient
    
    def compute_reward(
        self,
        returns: float,
        volatility: float,
        drawdown: float,
        risk_free_rate: float = 0.0,
        dd_penalty: float = 2.0
    ) -> float:
        """
        Compute reward for bandit update.
        
        Reward = Risk-adjusted return with drawdown penalty
        """
        # Risk-adjusted return (Sharpe-like)
        if volatility > 0:
            risk_adj_return = (returns - risk_free_rate) / volatility
        else:
            risk_adj_return = returns
        
        # Drawdown penalty (negative drawdown value made positive)
        dd_penalty_value = dd_penalty * abs(drawdown)
        
        reward = risk_adj_return - dd_penalty_value
        
        return reward
    
    def add_strategy(self, strategy_name: str) -> None:
        """Add a new strategy arm."""
        if strategy_name not in self.arm_states:
            self.arm_states[strategy_name] = BanditState()
            self.context_weights[strategy_name] = np.zeros(self.context_dim)
            self.context_precision[strategy_name] = np.eye(self.context_dim) * 0.1
            self.strategy_names.append(strategy_name)
    
    def save(self, path: str | Path) -> None:
        """Save bandit state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "strategy_names": self.strategy_names,
            "context_dim": self.context_dim,
            "prior_variance": self.prior_variance,
            "arm_states": self.arm_states,
            "context_weights": self.context_weights,
            "context_precision": self.context_precision,
            "total_updates": self.total_updates,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load(self, path: str | Path) -> "ContextualBandit":
        """Load bandit state from disk."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Bandit state not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.strategy_names = state["strategy_names"]
        self.context_dim = state["context_dim"]
        self.prior_variance = state["prior_variance"]
        self.arm_states = state["arm_states"]
        self.context_weights = state["context_weights"]
        self.context_precision = state["context_precision"]
        self.total_updates = state["total_updates"]
        
        return self
    
    def reset(self) -> None:
        """Reset to cold start (uniform priors)."""
        for name in self.strategy_names:
            self.arm_states[name] = BanditState()
            self.context_weights[name] = np.zeros(self.context_dim)
            self.context_precision[name] = np.eye(self.context_dim) * 0.1
        
        self.total_updates = 0
    
    def get_stats(self) -> dict:
        """Get statistics about bandit state."""
        stats = {
            "total_updates": self.total_updates,
            "arms": {}
        }
        
        for name, state in self.arm_states.items():
            stats["arms"][name] = {
                "n_pulls": state.n_pulls,
                "mean_reward": state.mu,
                "total_reward": state.total_reward,
            }
        
        return stats
