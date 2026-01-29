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
    pnl: float = 0.0,
    return_pct: float = 0.0,
    cycle_number: int = 1,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    cumulative_realized_pnl: float = 0.0,
    transaction_costs: float = 0.0,
    num_positions: int = 0,
) -> pd.DataFrame:
    """
    Log a rebalance cycle summary.
    
    Parameters
    ----------
    execution_date : str
        Date of the rebalance
    rebalance_frequency : str
        Frequency setting (Daily, Weekly, Monthly)
    portfolio_value : float
        Total portfolio value at cycle end
    cash : float
        Cash balance at cycle end
    pnl : float
        Period P/L
    return_pct : float
        Period return percentage
    cycle_number : int
        Rebalance cycle number
    realized_pnl : float
        Realized P/L from closed positions
    unrealized_pnl : float
        Unrealized P/L from open positions
    cumulative_realized_pnl : float
        Cumulative realized P/L
    transaction_costs : float
        Total transaction costs this cycle
    num_positions : int
        Number of positions held
        
    Returns
    -------
    pd.DataFrame
        The cycle summary record
    """
    os.makedirs("logs", exist_ok=True)
    
    record = {
        "Execution_Date": execution_date,
        "Rebalance_Cadence": f"{rebalance_frequency} Rebalance",
        "Current_Position": round(portfolio_value, 4),
        "P/L": round(pnl, 4),
        "Return_%": f"{return_pct:.4f}%",
        "Rebalances": cycle_number,
        "Realized_P/L": round(realized_pnl, 4),
        "Cumulative_Realized_P/L": round(cumulative_realized_pnl, 4),
        "Unrealized_P/L": round(unrealized_pnl, 4),
        "Latest_Unrealized_P/L": round(unrealized_pnl, 4),
        "Transaction_Costs": round(transaction_costs, 4),
        "Cash": round(cash, 4),
        "Num_Positions": num_positions,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    record_df = pd.DataFrame([record])
    
    # Append to existing log
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
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
