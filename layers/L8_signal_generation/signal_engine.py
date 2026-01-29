# layers/L8_signal_generation/signal_engine.py
"""
LAYER 8 — SIGNAL GENERATION (Trade Instruction Layer)

Converts portfolio weight changes into actionable trade signals.

Purpose: Compare old vs new portfolios, detect strategy changes,
and generate precise execution instructions.

Inputs:
- Previous portfolio allocation (Ticker, Weight)
- New optimized portfolio (Ticker, Weight)
- Old position strategies (Ticker -> Strategy name)
- New selected strategy

Outputs:
- Signal DataFrame with columns:
  [Ticker, Signal, Old_Weight, New_Weight, Old_Strategy, New_Strategy, Reason, Date]

Signal Types:
- BUY: New position entry
- SELL: Full position exit (removed from portfolio)
- REBALANCE: Weight adjustment (same strategy)
- LIQUIDATE: Full exit before strategy change re-entry
- HOLD: No action needed

Strategy-Aware Logic:
- Same strategy → REBALANCE (adjust weights only)
- Strategy changed → LIQUIDATE + BUY (exit old, enter new at current price)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict

import pandas as pd


def _normalize_date(as_of_date) -> str:
    if as_of_date is None:
        return datetime.now().strftime("%Y-%m-%d")
    if isinstance(as_of_date, datetime):
        return as_of_date.strftime("%Y-%m-%d")
    return str(as_of_date)


def _prepare_portfolio(df: pd.DataFrame | None, strategy_col: str = "Strategy") -> pd.DataFrame:
    """Normalize portfolio DataFrame to have Ticker, Weight, and optionally Strategy columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Weight", "Strategy"])
    
    working = df.copy()
    if "Ticker" not in working.columns or "Weight" not in working.columns:
        return pd.DataFrame(columns=["Ticker", "Weight", "Strategy"])
    
    working["Ticker"] = working["Ticker"].astype(str).str.strip()
    working = working[working["Ticker"] != ""]
    working["Weight"] = pd.to_numeric(working["Weight"], errors="coerce")
    working = working.dropna(subset=["Weight"])
    
    # Ensure Strategy column exists
    if strategy_col not in working.columns:
        working["Strategy"] = "Unknown"
    else:
        working["Strategy"] = working[strategy_col].fillna("Unknown")
    
    # Deduplicate
    working = (
        working.sort_values(["Ticker"])
        .groupby("Ticker", as_index=False, sort=False)
        .last()
    )
    
    # Normalize weights
    total = working["Weight"].sum()
    if total > 0:
        working["Weight"] = working["Weight"] / total
    
    return working[["Ticker", "Weight", "Strategy"]]


def generate_portfolio_signals(
    old_portfolio_df: Optional[pd.DataFrame],
    new_portfolio_df: pd.DataFrame,
    old_strategies: Optional[Dict[str, str]] = None,
    new_strategy: str | Dict[str, str] = "Unknown",
    as_of_date=None,
) -> pd.DataFrame:
    """
    Compare old vs new portfolios with strategy awareness.

    Logic:
    - If NEW ticker: BUY
    - If REMOVED ticker: SELL (full liquidation)
    - If SAME ticker with SAME strategy: REBALANCE (adjust weights)
    - If SAME ticker with DIFFERENT strategy: LIQUIDATE + BUY (exit old, enter new)

    Parameters
    ----------
    old_portfolio_df : pd.DataFrame
        Previous portfolio (Ticker, Weight)
    new_portfolio_df : pd.DataFrame
        New optimized portfolio (Ticker, Weight)
    old_strategies : dict
        Mapping of Ticker -> Strategy name from previous cycle
    new_strategy : str | Dict[str, str]
        Either a global strategy name (str) or per-stock mapping (Dict[str, str])
        
    Returns
    -------
    signals_df : pd.DataFrame
        Columns: ['Ticker', 'Signal', 'Old_Weight', 'New_Weight', 'Old_Strategy', 
                  'New_Strategy', 'Reason', 'Date']
    """
    results = []
    date_str = _normalize_date(as_of_date)
    old_strategies = old_strategies or {}

    # Helper to get strategy for a ticker (handles both dict and str)
    def get_ticker_strategy(ticker: str) -> str:
        if isinstance(new_strategy, dict):
            return new_strategy.get(ticker, "Unknown")
        return new_strategy

    prepared_old = _prepare_portfolio(old_portfolio_df)
    prepared_new = _prepare_portfolio(new_portfolio_df)

    old_tickers = set(prepared_old["Ticker"])
    new_tickers = set(prepared_new["Ticker"])

    # 1️⃣ Initial entry (first portfolio → BUY all)
    if not old_tickers:
        for _, row in prepared_new.iterrows():
            ticker = row['Ticker']
            results.append({
                "Ticker": ticker,
                "Signal": "BUY",
                "Old_Weight": 0,
                "New_Weight": row['Weight'],
                "Old_Strategy": None,
                "New_Strategy": get_ticker_strategy(ticker),
                "Reason": "Initial portfolio allocation",
                "Date": date_str,
            })
        signals_df = pd.DataFrame(results)
        signals_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return signals_df

    old_weights = dict(zip(prepared_old['Ticker'], prepared_old['Weight']))
    new_weights = dict(zip(prepared_new['Ticker'], prepared_new['Weight']))

    tolerance = 1e-6

    # 2️⃣ SELL – stocks removed from portfolio
    for ticker in old_tickers - new_tickers:
        old_w = old_weights.get(ticker, 0.0)
        old_strat = old_strategies.get(ticker, "Unknown")
        if abs(old_w) <= tolerance:
            continue
        results.append({
            "Ticker": ticker,
            "Signal": "SELL",
            "Old_Weight": old_w,
            "New_Weight": 0,
            "Old_Strategy": old_strat,
            "New_Strategy": None,
            "Reason": "Removed from portfolio",
            "Date": date_str,
        })

    # 3️⃣ BUY – new stocks added to portfolio
    for ticker in new_tickers - old_tickers:
        ticker_strat = get_ticker_strategy(ticker)
        results.append({
            "Ticker": ticker,
            "Signal": "BUY",
            "Old_Weight": 0,
            "New_Weight": new_weights[ticker],
            "Old_Strategy": None,
            "New_Strategy": ticker_strat,
            "Reason": "Newly added to portfolio",
            "Date": date_str,
        })

    # 4️⃣ Existing stocks – check strategy change
    for ticker in old_tickers & new_tickers:
        old_w = old_weights[ticker]
        new_w = new_weights[ticker]
        old_strat = old_strategies.get(ticker, "Unknown")
        ticker_strat = get_ticker_strategy(ticker)
        delta = abs(new_w - old_w)
        
        # Check if strategy changed for THIS stock
        strategy_changed = old_strat != ticker_strat
        
        if strategy_changed:
            # LIQUIDATE (SELL all) then BUY with new strategy
            # First: SELL signal
            results.append({
                "Ticker": ticker,
                "Signal": "SELL",
                "Old_Weight": old_w,
                "New_Weight": 0,
                "Old_Strategy": old_strat,
                "New_Strategy": ticker_strat,
                "Reason": f"Strategy changed: {old_strat} → {ticker_strat}",
                "Date": date_str,
            })
            # Then: BUY signal at new weight
            results.append({
                "Ticker": ticker,
                "Signal": "BUY",
                "Old_Weight": 0,
                "New_Weight": new_w,
                "Old_Strategy": old_strat,
                "New_Strategy": ticker_strat,
                "Reason": f"Re-entry after strategy change",
                "Date": date_str,
            })
        elif delta <= tolerance:
            # Same strategy, no weight change → HOLD
            results.append({
                "Ticker": ticker,
                "Signal": "HOLD",
                "Old_Weight": old_w,
                "New_Weight": new_w,
                "Old_Strategy": old_strat,
                "New_Strategy": ticker_strat,
                "Reason": "Allocation unchanged",
                "Date": date_str,
            })
        else:
            # Same strategy, weight changed → REBALANCE
            results.append({
                "Ticker": ticker,
                "Signal": "REBALANCE",
                "Old_Weight": old_w,
                "New_Weight": new_w,
                "Old_Strategy": old_strat,
                "New_Strategy": ticker_strat,
                "Reason": "Weight adjusted (same strategy)",
                "Date": date_str,
            })

    signals_df = pd.DataFrame(results)
    signals_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return signals_df
