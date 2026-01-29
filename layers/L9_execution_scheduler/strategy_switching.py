# execution/strategy_switching.py
"""
Strategy Switching Logic for SUP Flow 1.

Handles:
- Strategy switch decisions (cooldown, probability threshold)
- Position transitions (delta rebalance vs full exit)
- Switch history tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class SwitchDecision:
    """Result of strategy switch evaluation."""
    should_switch: bool
    reason: str
    current_strategy: str
    new_strategy: str
    new_probability: float
    days_since_last_switch: Optional[int] = None


@dataclass
class SwitchRecord:
    """Record of a strategy switch."""
    date: str
    from_strategy: str
    to_strategy: str
    probability: float
    reason: str


class StrategySwitchManager:
    """
    Manages strategy switching with cooldown and thresholds.
    
    Prevents overtrading by enforcing:
    - Minimum cooldown between switches
    - Probability threshold for new strategy
    """
    
    def __init__(
        self,
        cooldown_days: int = 5,
        probability_threshold: float = 0.6,
    ):
        """
        Initialize switch manager.
        
        Args:
            cooldown_days: Minimum days between strategy switches
            probability_threshold: Min probability to trigger switch
        """
        self.cooldown_days = cooldown_days
        self.probability_threshold = probability_threshold
        self.switch_history: list[SwitchRecord] = []
        self.current_strategy: Optional[str] = None
        self.last_switch_date: Optional[str] = None
    
    def evaluate_switch(
        self,
        new_strategy: str,
        new_probability: float,
        current_date: str,
    ) -> SwitchDecision:
        """
        Evaluate whether to switch to a new strategy.
        
        Args:
            new_strategy: Proposed new strategy
            new_probability: Bandit probability for new strategy
            current_date: Current date (YYYY-MM-DD)
        
        Returns:
            SwitchDecision with recommendation
        """
        current = self.current_strategy or "None"
        
        # Same strategy - no switch needed
        if new_strategy == self.current_strategy:
            return SwitchDecision(
                should_switch=False,
                reason="Same strategy selected",
                current_strategy=current,
                new_strategy=new_strategy,
                new_probability=new_probability,
            )
        
        # First time - always allow
        if self.current_strategy is None:
            return SwitchDecision(
                should_switch=True,
                reason="Initial strategy selection",
                current_strategy=current,
                new_strategy=new_strategy,
                new_probability=new_probability,
            )
        
        # Check probability threshold
        if new_probability < self.probability_threshold:
            return SwitchDecision(
                should_switch=False,
                reason=f"Probability ({new_probability:.2f}) below threshold ({self.probability_threshold})",
                current_strategy=current,
                new_strategy=new_strategy,
                new_probability=new_probability,
            )
        
        # Check cooldown
        if self.last_switch_date:
            try:
                last_dt = datetime.strptime(self.last_switch_date, "%Y-%m-%d")
                curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
                days_since = (curr_dt - last_dt).days
                
                if days_since < self.cooldown_days:
                    return SwitchDecision(
                        should_switch=False,
                        reason=f"Cooldown active ({days_since}/{self.cooldown_days} days)",
                        current_strategy=current,
                        new_strategy=new_strategy,
                        new_probability=new_probability,
                        days_since_last_switch=days_since,
                    )
            except ValueError:
                pass  # Date parsing failed, proceed with switch
        
        # All checks passed
        return SwitchDecision(
            should_switch=True,
            reason=f"Switch approved (probability: {new_probability:.2f})",
            current_strategy=current,
            new_strategy=new_strategy,
            new_probability=new_probability,
        )
    
    def execute_switch(
        self,
        new_strategy: str,
        probability: float,
        date: str,
        reason: str = "",
    ) -> None:
        """
        Record a strategy switch.
        
        Args:
            new_strategy: New strategy name
            probability: Probability at time of switch
            date: Switch date
            reason: Reason for switch
        """
        old_strategy = self.current_strategy or "None"
        
        record = SwitchRecord(
            date=date,
            from_strategy=old_strategy,
            to_strategy=new_strategy,
            probability=probability,
            reason=reason,
        )
        
        self.switch_history.append(record)
        self.current_strategy = new_strategy
        self.last_switch_date = date
    
    def get_history_df(self) -> pd.DataFrame:
        """Get switch history as DataFrame."""
        if not self.switch_history:
            return pd.DataFrame(columns=[
                "Date", "From", "To", "Probability", "Reason"
            ])
        
        return pd.DataFrame([
            {
                "Date": r.date,
                "From": r.from_strategy,
                "To": r.to_strategy,
                "Probability": r.probability,
                "Reason": r.reason,
            }
            for r in self.switch_history
        ])
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.switch_history = []
        self.current_strategy = None
        self.last_switch_date = None


def determine_rebalance_type(
    old_strategy: Optional[str],
    new_strategy: str,
) -> str:
    """
    Determine type of rebalance needed.
    
    Returns:
        "delta" - Same strategy, just adjust positions
        "full_exit" - Strategy change, exit all positions first
        "initial" - First time entry
    """
    if old_strategy is None:
        return "initial"
    
    if old_strategy == new_strategy:
        return "delta"
    
    return "full_exit"


def compute_delta_orders(
    current_positions: dict[str, float],
    target_positions: dict[str, float],
    prices: dict[str, float],
    min_trade_value: float = 10.0,
) -> list[dict]:
    """
    Compute delta orders for rebalancing within same strategy.
    
    Only trades the difference between current and target positions.
    
    Args:
        current_positions: Current shares per ticker
        target_positions: Target shares per ticker
        prices: Current prices
        min_trade_value: Minimum trade value to execute
    
    Returns:
        List of order dicts with Ticker, Side, Qty, NotionalValue
    """
    orders = []
    
    all_tickers = set(current_positions.keys()) | set(target_positions.keys())
    
    for ticker in all_tickers:
        current_qty = current_positions.get(ticker, 0.0)
        target_qty = target_positions.get(ticker, 0.0)
        delta = target_qty - current_qty
        
        price = prices.get(ticker, 0.0)
        if price <= 0:
            continue
        
        notional = abs(delta) * price
        
        # Skip tiny trades
        if notional < min_trade_value:
            continue
        
        if delta > 0:
            orders.append({
                "Ticker": ticker,
                "Side": "BUY",
                "Qty": delta,
                "NotionalValue": notional,
            })
        elif delta < 0:
            orders.append({
                "Ticker": ticker,
                "Side": "SELL",
                "Qty": abs(delta),
                "NotionalValue": notional,
            })
    
    return orders


def compute_full_exit_orders(
    current_positions: dict[str, float],
    prices: dict[str, float],
) -> list[dict]:
    """
    Compute orders to exit all current positions.
    
    Used when switching strategies (full exit before entering new positions).
    
    Returns:
        List of SELL orders for all current positions
    """
    orders = []
    
    for ticker, qty in current_positions.items():
        if qty <= 0:
            continue
        
        price = prices.get(ticker, 0.0)
        if price <= 0:
            continue
        
        orders.append({
            "Ticker": ticker,
            "Side": "SELL",
            "Qty": qty,
            "NotionalValue": qty * price,
        })
    
    return orders
