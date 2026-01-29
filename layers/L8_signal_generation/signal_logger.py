# layers/L8_signal_generation/signal_logger.py
"""
Persistent logger for trade signals.
Appends each rebalancing cycle's signals (BUY, SELL, REBALANCE, HOLD) to a CSV log.
"""
import os
from datetime import datetime

import pandas as pd

LOG_PATH = "logs/trade_signals_log.csv"


def log_signals(signals_df: pd.DataFrame) -> None:
    """
    Append generated signals to the persistent log file.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with columns: Ticker, Signal, Old_Weight, New_Weight, 
        Old_Strategy, New_Strategy, Reason, Date, Timestamp
    """
    if signals_df is None or signals_df.empty:
        return
    
    os.makedirs("logs", exist_ok=True)
    signals_df = signals_df.copy()
    
    # Ensure Timestamp column exists
    if "Timestamp" not in signals_df.columns:
        signals_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Append to existing log or create new
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, signals_df], ignore_index=True)
    else:
        combined = signals_df
    
    combined.to_csv(LOG_PATH, index=False)


def get_signal_history() -> pd.DataFrame:
    """
    Read the full signal history log.
    
    Returns
    -------
    pd.DataFrame or empty DataFrame if no log exists
    """
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()


def clear_signal_log() -> None:
    """Clear the signal log file."""
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
