# layers/L10_monitoring/monitor.py
"""
LAYER 10 — MONITORING, EXPLANATION & PERFORMANCE

Trust & transparency layer.

Outputs:
- Strategy decisions with reasoning
- Regime attribution
- Risk metrics (Vol, DD, Sharpe)
- Performance attribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


@dataclass
class DecisionExplanation:
    """Explains a single strategy decision."""
    timestamp: datetime
    selected_strategy: str
    regime: str
    regime_probabilities: Dict[str, float]
    allowed_strategies: List[str]
    filtered_strategies: List[str]
    filter_reasons: Dict[str, str]
    bandit_scores: Dict[str, float]
    selection_reason: str
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "selected_strategy": self.selected_strategy,
            "regime": self.regime,
            "regime_probabilities": self.regime_probabilities,
            "allowed_strategies": self.allowed_strategies,
            "filtered_strategies": self.filtered_strategies,
            "filter_reasons": self.filter_reasons,
            "bandit_scores": self.bandit_scores,
            "selection_reason": self.selection_reason,
        }


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_return": f"{self.total_return:.2%}",
            "annualized_return": f"{self.annualized_return:.2%}",
            "volatility": f"{self.volatility:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "current_drawdown": f"{self.current_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.1%}",
        }


class PerformanceMonitor:
    """
    Tracks and reports system performance.
    """
    
    def __init__(self):
        self.decision_history: List[DecisionExplanation] = []
        self.returns_history: List[float] = []
        self.equity_curve: List[float] = [1.0]  # Starts at 1.0
        self.strategy_performance: Dict[str, List[float]] = {}
    
    def record_decision(self, explanation: DecisionExplanation) -> None:
        """Record a strategy decision."""
        self.decision_history.append(explanation)
    
    def record_return(self, ret: float, strategy: str) -> None:
        """Record a period return."""
        self.returns_history.append(ret)
        
        # Update equity curve
        new_equity = self.equity_curve[-1] * (1 + ret)
        self.equity_curve.append(new_equity)
        
        # Track per-strategy performance
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        self.strategy_performance[strategy].append(ret)
    
    def compute_metrics(self, risk_free_rate: float = 0.0) -> PerformanceMetrics:
        """Compute current performance metrics."""
        if not self.returns_history:
            return PerformanceMetrics()
        
        returns = np.array(self.returns_history)
        
        # Total return
        total_return = self.equity_curve[-1] / self.equity_curve[0] - 1
        
        # Annualized return (assuming weekly rebalance)
        n_periods = len(returns)
        periods_per_year = 52  # Weekly
        annualized = (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1
        
        # Volatility (annualized)
        vol = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        if vol > 0:
            sharpe = (annualized - risk_free_rate) / vol
        else:
            sharpe = 0.0
        
        # Drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_dd = abs(drawdowns.min())
        current_dd = abs(drawdowns[-1])
        
        # Win rate
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0.0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            win_rate=win_rate,
        )
    
    def get_strategy_attribution(self) -> pd.DataFrame:
        """Get performance attribution by strategy."""
        if not self.strategy_performance:
            return pd.DataFrame()
        
        rows = []
        for strategy, returns in self.strategy_performance.items():
            returns_arr = np.array(returns)
            rows.append({
                "Strategy": strategy,
                "Periods": len(returns),
                "Total Return": (1 + returns_arr).prod() - 1,
                "Avg Return": returns_arr.mean(),
                "Volatility": returns_arr.std() * np.sqrt(52),
                "Win Rate": (returns_arr > 0).mean(),
            })
        
        return pd.DataFrame(rows)
    
    def get_recent_decisions(self, n: int = 10) -> List[dict]:
        """Get last N decisions as dicts."""
        return [d.to_dict() for d in self.decision_history[-n:]]
    
    def check_kill_switch(
        self,
        dd_threshold: float = 0.15,
        vol_threshold: float = 0.30,
    ) -> tuple[bool, str]:
        """
        Check if kill-switch should be triggered.
        
        Returns:
            (triggered: bool, reason: str)
        """
        if len(self.equity_curve) < 2:
            return False, ""
        
        # Check drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        current_dd = abs((equity[-1] - running_max[-1]) / running_max[-1])
        
        if current_dd > dd_threshold:
            return True, f"Drawdown {current_dd:.1%} exceeds {dd_threshold:.1%} threshold"
        
        # Check recent volatility
        if len(self.returns_history) >= 5:
            recent_returns = np.array(self.returns_history[-5:])
            recent_vol = recent_returns.std() * np.sqrt(52)
            
            if recent_vol > vol_threshold:
                return True, f"Volatility {recent_vol:.1%} exceeds {vol_threshold:.1%} threshold"
        
        return False, ""
    
    def reset(self) -> None:
        """Reset all tracking."""
        self.decision_history = []
        self.returns_history = []
        self.equity_curve = [1.0]
        self.strategy_performance = {}
