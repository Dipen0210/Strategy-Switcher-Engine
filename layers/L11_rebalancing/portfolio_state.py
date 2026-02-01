# layers/L11_rebalancing/portfolio_state.py
"""
LAYER 11 — REBALANCING & STATE UPDATE (Accounting Layer)

Maintains the complete portfolio state across trading cycles.

Purpose: Track all positions, cash, cost basis, realized/unrealized P&L,
fees, and trade history. Provides the source of truth for portfolio value.

State Tracked:
- Cash balance
- Positions (ticker -> quantity)
- Cost basis per position
- Strategy per position (for strategy-aware trading)
- Realized & unrealized P&L
- Cumulative fees paid
- Trade log (full history)
- Portfolio value history (for performance tracking)

Methods:
- update_position: Apply trade and update accounting
- current_equity: Get total portfolio value
- mark_to_market: Record point-in-time valuation
- get_summary: Return portable state snapshot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


def _normalize_date(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if value is None:
        return datetime.now()
    return pd.to_datetime(value).to_pydatetime()


def _truncate_value(value: float, decimals: int = 4) -> float:
    """Truncate a float to a specific number of decimal places."""
    multiplier = 10 ** decimals
    return int(value * multiplier) / multiplier


@dataclass
class PortfolioState:
    """
    Stateful portfolio container tracking positions, cash, and P&L.
    """
    cash: float = 100_000.0
    positions: Dict[str, float] = field(default_factory=dict)
    position_cost_basis: Dict[str, float] = field(default_factory=dict)
    position_strategies: Dict[str, str] = field(default_factory=dict)  # Ticker -> Strategy name
    initial_capital: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    peak_equity: float = 0.0  # Tracks High Water Mark for Kill Switch
    
    portfolio_value_history: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["Date", "Value", "HoldingsValue", "Cash", "RealizedPnL", "UnrealizedPnL", "FeesPaid"]
        )
    )
    trade_log: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "Date", "Ticker", "Side", "Qty", "Price",
                "Notional", "RealizedPnL", "Fee", "CashAfter",
            ]
        )
    )
    last_allocation: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["Ticker", "Weight"])
    )
    last_price_snapshot: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.initial_capital is None:
            self.initial_capital = float(self.cash)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    def position_qty(self, ticker: str) -> float:
        return float(self.positions.get(ticker, 0.0))

    def update_position(
        self, ticker: str, qty_delta: float, price: float, side: str
    ) -> float:
        """
        Apply a trade to holdings and update cash/cost basis.
        Returns the realized P&L contribution from the trade (sells only).
        """
        if qty_delta == 0:
            return 0.0

        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported side '{side}' for {ticker}.")

        realized_delta = 0.0
        if side == "BUY":
            cost = price * qty_delta
            if cost - 1e-9 > self.cash:
                raise ValueError("Insufficient cash for purchase.")
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0.0) + qty_delta
            self.position_cost_basis[ticker] = self.position_cost_basis.get(ticker, 0.0) + cost
        else:
            current_qty = self.positions.get(ticker, 0.0)
            shares_to_sell = abs(qty_delta)
            if shares_to_sell - current_qty > 1e-9:
                raise ValueError(f"Attempting to sell {shares_to_sell} shares of {ticker} but only {current_qty} held.")
            
            total_cost = self.position_cost_basis.get(ticker, 0.0)
            cost_per_share = total_cost / current_qty if current_qty else 0.0
            cost_removed = cost_per_share * shares_to_sell
            proceeds = shares_to_sell * price
            realized_delta = proceeds - cost_removed
            self.realized_pnl += realized_delta
            self.cash += proceeds
            
            remaining_cost = total_cost - cost_removed
            remaining_qty = current_qty - shares_to_sell
            if remaining_qty <= 1e-8:
                self.positions.pop(ticker, None)
                self.position_cost_basis.pop(ticker, None)
            else:
                self.positions[ticker] = remaining_qty
                self.position_cost_basis[ticker] = max(remaining_cost, 0.0)

        return realized_delta

    # ------------------------------------------------------------------
    # Fees & logging
    # ------------------------------------------------------------------
    def deduct_fee(self, amount: float):
        if amount <= 0:
            return
        if amount - 1e-9 > self.cash:
            raise ValueError("Insufficient cash to pay trading fee.")
        self.cash -= amount
        self.fees_paid += amount

    def append_trade(self, row: dict):
        entry = {col: row.get(col) for col in self.trade_log.columns}
        self.trade_log.loc[len(self.trade_log)] = entry

    # ------------------------------------------------------------------
    # Valuation utilities
    # ------------------------------------------------------------------
    def _require_price_snapshot(self, price_snapshot: Dict[str, float]):
        missing = [ticker for ticker in self.positions if ticker not in price_snapshot]
        if missing:
            raise ValueError(f"Missing prices for holdings: {', '.join(sorted(missing))}")

    def current_equity(self, price_snapshot: Dict[str, float]) -> float:
        self._require_price_snapshot(price_snapshot)
        holdings_value = sum(price_snapshot[t] * qty for t, qty in self.positions.items())
        return holdings_value + self.cash

    def mark_to_market(self, date, price_snapshot: Dict[str, float]) -> float:
        """
        Compute portfolio MV using provided price snapshot,
        update unrealized P&L, and record in history.
        """
        self._require_price_snapshot(price_snapshot)
        holdings_value = sum(price_snapshot[t] * qty for t, qty in self.positions.items())
        
        unrealized_components = []
        for ticker, qty in self.positions.items():
            basis = self.position_cost_basis.get(ticker, 0.0)
            unrealized_components.append(price_snapshot[ticker] * qty - basis)
        self.unrealized_pnl = float(sum(unrealized_components))
        
        total_value = holdings_value + self.cash
        
        # Update High Water Mark
        if total_value > self.peak_equity:
            self.peak_equity = total_value
            
        normalized_date = _normalize_date(date)
        
        self.portfolio_value_history.loc[len(self.portfolio_value_history)] = [
            normalized_date,
            _truncate_value(total_value),
            _truncate_value(holdings_value),
            _truncate_value(self.cash),
            _truncate_value(self.realized_pnl),
            _truncate_value(self.unrealized_pnl),
            _truncate_value(self.fees_paid),
        ]
        return total_value

    def get_summary(self) -> dict:
        """Return a summary of the current portfolio state."""
        return {
            "cash": self.cash,
            "positions": dict(self.positions),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees_paid": self.fees_paid,
            "num_trades": len(self.trade_log),
        }
