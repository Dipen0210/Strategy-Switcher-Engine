# layers/L11_rebalancing/cycle_logger.py
"""
Persistent logger for rebalance cycle summaries.
Tracks P/L, returns, position values at each rebalance cycle.
"""
import os
from datetime import datetime
from typing import Optional

import pandas as pd

LOG_PATH = "logs/rebalance_cycles.csv"


def log_cycle_summary(
    execution_date: str,
    rebalance_frequency: str,
    portfolio_value: float,
    cash: float,
    initial_capital: float,  # New Required Argument
    pnl: float = 0.0,        # Deprecated, calculated internally
    return_pct: float = 0.0, # Deprecated, calculated internally
    cycle_number: int = 1,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    cumulative_realized_pnl: float = 0.0,
    transaction_costs: float = 0.0,
    num_positions: int = 0,
) -> pd.DataFrame:
    """
    Log a rebalance cycle summary with strict accounting enforcement.
    P/L = (Current Holdings + Cash) - Initial Capital.
    """
    os.makedirs("logs", exist_ok=True)
    
    # STRICT ACCOUNTING IDENTITY CHECK
    # Note: portfolio_value passed from pipeline IS Total Equity (Holdings + Cash)
    total_value = portfolio_value
    actual_pnl = total_value - initial_capital
    actual_return_pct = (actual_pnl / initial_capital * 100) if initial_capital > 0 else 0.0
    
    record = {
        "Execution_Date": execution_date,
        "Rebalance_Cadence": f"{rebalance_frequency} Rebalance",
        "Current_Position": round(total_value, 4), # This serves as Total Equity
        "Cash": round(cash, 4),
        "Total_Value": round(total_value, 4),
        "Initial_Capital": round(initial_capital, 4),
        "P/L": round(actual_pnl, 4),
        "Return_%": f"{actual_return_pct:.4f}%",
        "Rebalances": cycle_number,
        "Realized_P/L": round(realized_pnl, 4),
        "Cumulative_Realized_P/L": round(cumulative_realized_pnl, 4),
        "Unrealized_P/L": round(unrealized_pnl, 4),
        "Latest_Unrealized_P/L": round(unrealized_pnl, 4),
        "Transaction_Costs": round(transaction_costs, 4),
        "Num_Positions": num_positions,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    record_df = pd.DataFrame([record])
    
    # Append to existing log
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        # Ensure new columns exist in old CSV
        if "Total_Value" not in existing.columns:
            existing["Total_Value"] = existing["Current_Position"] + existing.get("Cash", 0)
        if "Initial_Capital" not in existing.columns:
            existing["Initial_Capital"] = 10000.0 # fallback
            
        combined = pd.concat([existing, record_df], ignore_index=True)
    else:
        combined = record_df
    
    combined.to_csv(LOG_PATH, index=False)
    return record_df


def get_cycle_history() -> pd.DataFrame:
    """Read the full rebalance cycle history."""
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()


def get_latest_cycle_number() -> int:
    """Get the latest cycle number from history."""
    history = get_cycle_history()
    if not history.empty and "Rebalances" in history.columns:
        return int(history["Rebalances"].max())
    return 0


def clear_cycle_log() -> None:
    """Clear the rebalance cycle log."""
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
