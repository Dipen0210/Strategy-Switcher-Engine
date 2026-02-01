# layers/L10_trade_execution/execution_engine.py
"""
LAYER 10 — TRADE EXECUTION / OMS (Order Management System)

Executes trades based on signals and maintains order lifecycle.

Purpose: Convert signals into actual orders, execute at market prices,
and update portfolio accounting with realized P&L and fees.

Inputs:
- Portfolio state (PortfolioState object)
- Price data dictionary (ticker -> OHLCV DataFrame)
- Signals DataFrame from Layer 8
- New target portfolio weights
- Commission per trade

Outputs:
- Orders DataFrame (planned trades)
- Fills DataFrame (executed trades with prices)
- Updated portfolio state (positions, cash, strategies)

Features:
- Handles LIQUIDATE signals (full position exit on strategy change)
- Updates position_strategies when entering new positions
- Supports BUY, SELL, REBALANCE order types
- Processes SELL orders first (frees up cash for BUY orders)
- Marks portfolio to market after execution
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from layers.L11_rebalancing.portfolio_state import PortfolioState


def _truncate_value(value: float, decimals: int = 4) -> float:
    """Truncate a float to a specific number of decimal places."""
    multiplier = 10 ** decimals
    return int(value * multiplier) / multiplier


def snapshot_prices(price_data_dict: Dict[str, pd.DataFrame], trading_date) -> Dict[str, float]:
    """
    Build an execution snapshot of opening prices for the supplied trading date.
    Falls back to Close if Open is not available.
    """
    ts = pd.Timestamp(trading_date).normalize()
    snapshot: Dict[str, float] = {}
    
    for ticker, df in price_data_dict.items():
        if df is None or df.empty:
            continue
        working = df.copy()
        
        if "Date" in working.columns:
            working["Date"] = pd.to_datetime(working["Date"], errors="coerce").dt.normalize()
            matches = working[working["Date"] == ts]
        else:
            working.index = pd.to_datetime(working.index)
            matches = working.loc[working.index.normalize() == ts]
        
        if matches.empty:
            # Fallback to last available price
            if "Close" in working.columns:
                snapshot[ticker] = float(working["Close"].iloc[-1])
            continue
            
        if "Open" in matches.columns:
            snapshot[ticker] = float(matches["Open"].iloc[-1])
        elif "Close" in matches.columns:
            snapshot[ticker] = float(matches["Close"].iloc[-1])
    
    if not snapshot:
        raise ValueError(f"No prices available on {ts.date()} for the provided tickers.")
    return snapshot


def target_shares_from_weights(
    target_weights: pd.DataFrame,
    prices: Dict[str, float],
    equity: float,
) -> Dict[str, float]:
    """
    Convert target weights to fractional share targets using execution-day prices.
    """
    targets: Dict[str, float] = {}
    for _, row in target_weights.iterrows():
        ticker = row["Ticker"]
        weight = float(row["Weight"])
        price = prices.get(ticker)
        if price is None or np.isnan(price) or price <= 0:
            continue
        notional = equity * weight
        qty = _truncate_value(notional / price, decimals=4)
        targets[ticker] = max(qty, 0.0)
    return targets


def reconcile_orders_from_signals(
    state: PortfolioState,
    signals_df: pd.DataFrame,
    prices: Dict[str, float],
    equity: float,
) -> pd.DataFrame:
    """
    Create order instructions from signal DataFrame.
    Handles LIQUIDATE, BUY, SELL, REBALANCE signals.
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    orders = []
    
    for _, signal in signals_df.iterrows():
        ticker = signal["Ticker"]
        signal_type = str(signal["Signal"]).upper()
        new_weight = float(signal.get("New_Weight", 0))
        old_weight = float(signal.get("Old_Weight", 0))
        new_strategy = signal.get("New_Strategy", "Unknown")
        
        price = prices.get(ticker)
        if price is None or np.isnan(price) or price <= 0:
            continue
        
        current_qty = state.positions.get(ticker, 0.0)
        
        if signal_type == "SELL":
            # Full exit (removed from portfolio)
            if current_qty > 1e-8:
                orders.append({
                    "Ticker": ticker,
                    "Side": "SELL",
                    "Qty": current_qty,
                    "Signal": "SELL",
                    "Strategy": None,
                })
                
        elif signal_type == "BUY":
            # New position or re-entry after liquidate
            notional = equity * new_weight
            target_qty = _truncate_value(notional / price, decimals=4)
            orders.append({
                "Ticker": ticker,
                "Side": "BUY",
                "Qty": target_qty,
                "Signal": "BUY",
                "Strategy": new_strategy,
            })
            
        elif signal_type == "REBALANCE":
            # Adjust existing position (same strategy)
            notional = equity * new_weight
            target_qty = _truncate_value(notional / price, decimals=4)
            delta = target_qty - current_qty
            
            if delta > 1e-8:
                orders.append({
                    "Ticker": ticker,
                    "Side": "BUY",
                    "Qty": delta,
                    "Signal": "REBALANCE",
                    "Strategy": new_strategy,
                })
            elif delta < -1e-8:
                orders.append({
                    "Ticker": ticker,
                    "Side": "SELL",
                    "Qty": abs(delta),
                    "Signal": "REBALANCE",
                    "Strategy": new_strategy,
                })
        # HOLD: no action needed

    if not orders:
        return pd.DataFrame()
    
    # Sort: SELL/LIQUIDATE first (frees up cash), then BUY
    orders_df = pd.DataFrame(orders)
    orders_df["_sort"] = orders_df["Side"].map({"SELL": 0, "BUY": 1})
    orders_df = orders_df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return orders_df


def execute_orders(
    state: PortfolioState,
    orders_df: pd.DataFrame,
    prices: Dict[str, float],
    date: Optional[str] = None,
    commission_per_trade: float = 1.0,
) -> pd.DataFrame:
    """
    Fill all orders at the execution-day price and update portfolio accounting.
    Updates position_strategies for new/changed positions.
    """
    if orders_df is None or orders_df.empty:
        return pd.DataFrame()

    exec_date = date or datetime.now().strftime("%Y-%m-%d")
    filled_orders = []

    for _, order in orders_df.iterrows():
        ticker = order["Ticker"]
        side = str(order["Side"]).upper()
        qty = float(order["Qty"])
        price = prices.get(ticker)
        signal_type = order.get("Signal", "UNKNOWN")
        strategy = order.get("Strategy")
        
        if price is None or np.isnan(price) or price <= 0 or qty <= 0:
            continue

        fee = commission_per_trade if commission_per_trade > 0 else 0.0

        if side == "BUY":
            available_cash = state.cash - fee
            if available_cash <= 0:
                continue
            max_affordable = available_cash / price
            max_affordable = _truncate_value(max_affordable, decimals=4)
            if max_affordable <= 0:
                continue
            qty = min(qty, max_affordable)
            if qty <= 0:
                continue

        qty_delta = qty if side == "BUY" else -qty
        realized = state.update_position(ticker, qty_delta, price=price, side=side)
        if fee > 0:
            state.deduct_fee(fee)
        
        # Update position_strategies
        if side == "BUY" and strategy:
            state.position_strategies[ticker] = strategy
        elif side == "SELL":
            # If fully exited, remove from strategies
            if state.positions.get(ticker, 0) <= 1e-8:
                state.position_strategies.pop(ticker, None)

        trade_record = {
            "Date": exec_date,
            "Ticker": ticker,
            "Side": side,
            "Signal": signal_type,
            "Strategy": strategy,
            "Qty": _truncate_value(qty, decimals=4),
            "Price": _truncate_value(price, decimals=4),
            "Notional": _truncate_value(price * qty, decimals=4),
            "RealizedPnL": _truncate_value(realized, decimals=4),
            "Fee": _truncate_value(fee, decimals=4),
            "CashAfter": _truncate_value(state.cash, decimals=4),
        }
        state.append_trade(trade_record)
        filled_orders.append(trade_record)

    return pd.DataFrame(filled_orders)


def run_execution_cycle(
    state: PortfolioState,
    price_data_dict: Dict[str, pd.DataFrame],
    signals_df: pd.DataFrame,
    new_portfolio_weights: pd.DataFrame,
    date: Optional[str] = None,
    commission_per_trade: float = 1.0,
) -> Dict:
    """
    Execute a full strategy-aware rebalance using signals.
    
    Handles:
    - LIQUIDATE: Full exit then re-entry with new strategy
    - REBALANCE: Adjust weights (same strategy)
    - BUY/SELL: Entry/exit from portfolio
    
    Returns a dict with orders, fills, and updated state summary.
    """
    if date is None:
        for df in price_data_dict.values():
            if df is not None and not df.empty:
                if "Date" in df.columns:
                    date = df["Date"].iloc[-1]
                else:
                    date = df.index[-1]
                break
    
    if date is None:
        raise ValueError("Execution date is required to perform a rebalance.")

    open_prices = snapshot_prices(price_data_dict, date)
    equity = state.current_equity(open_prices)

    # Generate orders from signals
    orders_df = reconcile_orders_from_signals(
        state=state,
        signals_df=signals_df,
        prices=open_prices,
        equity=equity,
    )
    
    fills_df = execute_orders(
        state=state,
        orders_df=orders_df,
        prices=open_prices,
        date=str(date),
        commission_per_trade=commission_per_trade,
    )

    state.mark_to_market(date, open_prices)
    state.last_price_snapshot = dict(open_prices)
    state.last_allocation = new_portfolio_weights.copy()

    # Calculate execution metrics for UI display
    orders_executed = len(fills_df) if fills_df is not None and not fills_df.empty else 0
    total_volume = float(fills_df["Notional"].sum()) if fills_df is not None and not fills_df.empty and "Notional" in fills_df.columns else 0.0
    fees_paid = float(fills_df["Fee"].sum()) if fills_df is not None and not fills_df.empty and "Fee" in fills_df.columns else 0.0
    cash_after = state.cash

    return {
        "orders": orders_df,
        "fills": fills_df,
        "signals": signals_df,
        "prices": open_prices,
        "equity": equity,
        "position_strategies": dict(state.position_strategies),
        "state": state.get_summary(),
        # UI-expected metrics
        "orders_executed": orders_executed,
        "total_volume": total_volume,
        "fees_paid": fees_paid,
        "cash_after": cash_after,
    }
