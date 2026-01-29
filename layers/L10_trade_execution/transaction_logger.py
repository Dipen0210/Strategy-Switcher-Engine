# layers/L10_trade_execution/transaction_logger.py
"""
Persistent logger for backtest transactions.
Logs detailed trade execution data: shares, price, trade value, cash flow, realized P/L.
"""
import os
from datetime import datetime
from typing import Optional

import pandas as pd

LOG_PATH = "logs/backtest_transactions.csv"


def log_transactions_from_fills(
    fills_df: pd.DataFrame,
    execution_date: str,
) -> pd.DataFrame:
    """
    Log executed trades (fills) to the transaction log.
    
    This uses the actual execution fills which contain exact shares,
    prices, realized P/L, and fees from trade execution.
    
    Parameters
    ----------
    fills_df : pd.DataFrame
        DataFrame from execute_orders with columns:
        Date, Ticker, Side, Signal, Strategy, Qty, Price, Notional, RealizedPnL, Fee, CashAfter
    execution_date : str
        Date of execution
        
    Returns
    -------
    pd.DataFrame
        Transaction records logged
    """
    if fills_df is None or fills_df.empty:
        return pd.DataFrame()
    
    os.makedirs("logs", exist_ok=True)
    
    transactions = []
    
    for _, row in fills_df.iterrows():
        ticker = row.get("Ticker", "")
        side = row.get("Side", "")  # BUY or SELL
        signal = row.get("Signal", side)
        if signal == "LIQUIDATE":
            signal = "SELL"  # Normalize to SELL for user display
        qty = float(row.get("Qty", 0))
        price = float(row.get("Price", 0))
        notional = float(row.get("Notional", qty * price))
        fee = float(row.get("Fee", 0))
        realized_pnl = float(row.get("RealizedPnL", 0))
        cash_after = float(row.get("CashAfter", 0))
        strategy = row.get("Strategy", "")
        
        # Cash flow: negative for BUY, positive for SELL
        if side == "BUY":
            cash_flow = -(notional + fee)
        else:  # SELL
            cash_flow = notional - fee
        
        transactions.append({
            "Date": execution_date,
            "Ticker": ticker,
            "Side": side,
            "Signal": signal,
            "Strategy": strategy or "",
            "Shares": round(qty, 4),
            "Price": round(price, 4),
            "TradeValue": round(notional, 2),
            "RealizedPnL": round(realized_pnl, 2),
            "Fee": round(fee, 2),
            "CashFlow": round(cash_flow, 2),
            "CashAfter": round(cash_after, 2),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    if not transactions:
        return pd.DataFrame()
    
    transactions_df = pd.DataFrame(transactions)
    
    # Append to existing log
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, transactions_df], ignore_index=True)
    else:
        combined = transactions_df
    
    combined.to_csv(LOG_PATH, index=False)
    return transactions_df


# Keep old function for backward compatibility but recommend using log_transactions_from_fills
def log_transactions(
    signals_df: pd.DataFrame,
    prices: dict,
    portfolio_value: float,
    remaining_cash: float,
    execution_date: str,
    commission_per_trade: float = 1.0,
) -> pd.DataFrame:
    """
    DEPRECATED: Use log_transactions_from_fills instead.
    This logs from signals, not actual fills.
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()
    
    os.makedirs("logs", exist_ok=True)
    
    transactions = []
    running_cash = remaining_cash + portfolio_value
    
    for _, row in signals_df.iterrows():
        ticker = row.get("Ticker", "")
        signal = row.get("Signal", "")
        old_weight = float(row.get("Old_Weight", 0) or 0)
        new_weight = float(row.get("New_Weight", 0) or 0)
        reason = row.get("Reason", "")
        
        price = prices.get(ticker, 0)
        if price == 0:
            continue
        
        # Determine side from signal
        if signal in ["SELL", "LIQUIDATE"]:
            side = "SELL"
        elif signal == "BUY":
            side = "BUY"
        elif signal == "REBALANCE":
            side = "SELL" if new_weight < old_weight else "BUY"
        else:
            side = "HOLD"
            continue  # Skip HOLD signals
        
        # Calculate trade details
        net_weight = new_weight - old_weight
        trade_value = abs(net_weight) * portfolio_value
        shares = trade_value / price if price > 0 else 0
        
        # Cash flow (negative for buys, positive for sells)
        if side == "BUY":
            cash_flow = -trade_value - commission_per_trade
        else:  # SELL
            cash_flow = trade_value - commission_per_trade
        
        running_cash += cash_flow
        
        transactions.append({
            "Date": execution_date,
            "Ticker": ticker,
            "Side": side,
            "Signal": signal,
            "Strategy": row.get("New_Strategy", "") or "",
            "Shares": round(shares, 4),
            "Price": round(price, 4),
            "TradeValue": round(trade_value, 2),
            "RealizedPnL": 0.0,  # Not available from signals
            "Fee": commission_per_trade,
            "CashFlow": round(cash_flow, 2),
            "CashAfter": round(running_cash, 2),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    if not transactions:
        return pd.DataFrame()
    
    transactions_df = pd.DataFrame(transactions)
    
    # Append to existing log
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, transactions_df], ignore_index=True)
    else:
        combined = transactions_df
    
    combined.to_csv(LOG_PATH, index=False)
    return transactions_df


def get_transaction_history() -> pd.DataFrame:
    """Read the full transaction history log."""
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()


def clear_transaction_log() -> None:
    """Clear the transaction log file."""
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
