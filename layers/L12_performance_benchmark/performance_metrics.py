# layers/L12_performance_benchmark/performance_metrics.py
"""
LAYER 12 — PERFORMANCE & BENCHMARKING (Analytics Layer)

Measures and reports portfolio performance metrics.

Purpose: Compute risk-adjusted returns, track portfolio evolution,
and compare against market benchmarks to evaluate strategy effectiveness.

Inputs:
- Portfolio value history (Date, Value)
- Benchmark price history (Date, Close)
- Risk-free rate (default 4% annually)

Outputs:
- Daily/cumulative/annualized returns
- Risk-adjusted metrics (Sharpe, Sortino)
- Drawdown analysis
- Benchmark comparison with alpha calculation

Metrics Computed:
- Sharpe Ratio: Risk-adjusted excess return
- Sortino Ratio: Downside risk-adjusted return
- Max Drawdown: Largest peak-to-trough decline
- Cumulative Return: Total portfolio growth
- Annualized Return: CAGR equivalent
- Volatility: Annualized standard deviation
- Alpha: Excess return vs benchmark
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_daily_returns(value_history: pd.DataFrame) -> pd.Series:
    """
    Compute daily returns from portfolio value history.
    
    Args:
        value_history: DataFrame with 'Date' and 'Value' columns
        
    Returns:
        Series of daily returns indexed by date
    """
    if value_history is None or len(value_history) < 2:
        return pd.Series(dtype=float)

    value_history = value_history.sort_values("Date").reset_index(drop=True)
    returns = value_history["Value"].pct_change().dropna()
    returns.index = value_history["Date"].iloc[1:]
    return returns


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 4%)
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) == 0:
        return np.nan
    excess = returns - risk_free_rate / periods_per_year
    if excess.std(ddof=1) == 0:
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino Ratio (downside deviation only).
    
    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default 4%)
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) == 0:
        return np.nan
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return np.nan
    excess = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1)


def max_drawdown(value_history: pd.DataFrame) -> float:
    """
    Calculate maximum drawdown from peak to trough.
    
    Args:
        value_history: DataFrame with 'Value' column
        
    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.15 for 15% drawdown)
    """
    if value_history is None or "Value" not in value_history.columns:
        return np.nan
    if len(value_history) < 2:
        return 0.0

    cum_max = value_history["Value"].cummax()
    drawdowns = (value_history["Value"] - cum_max) / cum_max
    return drawdowns.min()


def cumulative_return(value_history: pd.DataFrame) -> float:
    """
    Calculate total cumulative portfolio return.
    
    Args:
        value_history: DataFrame with 'Value' column
        
    Returns:
        Cumulative return as a decimal (e.g., 0.25 for 25% return)
    """
    if value_history is None or len(value_history) < 2:
        return 0.0
    start_val = value_history["Value"].iloc[0]
    end_val = value_history["Value"].iloc[-1]
    if start_val == 0:
        return 0.0
    return (end_val - start_val) / start_val


def annualized_return(value_history: pd.DataFrame, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        value_history: DataFrame with 'Date' and 'Value' columns
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized return as a decimal
    """
    if value_history is None or len(value_history) < 2:
        return 0.0
    
    total_return = cumulative_return(value_history)
    n_periods = len(value_history) - 1
    if n_periods <= 0:
        return 0.0
    
    years = n_periods / periods_per_year
    if years <= 0:
        return total_return
    
    return (1 + total_return) ** (1 / years) - 1


def compute_all_metrics(value_history: pd.DataFrame) -> dict:
    """
    Compute all performance metrics for a portfolio.
    
    Args:
        value_history: DataFrame with 'Date' and 'Value' columns
        
    Returns:
        Dict of all computed metrics
    """
    returns = compute_daily_returns(value_history)
    
    return {
        "cumulative_return": cumulative_return(value_history),
        "annualized_return": annualized_return(value_history),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(value_history),
        "volatility": returns.std() * np.sqrt(252) if len(returns) > 0 else np.nan,
        "num_periods": len(value_history),
    }


def compare_to_benchmark(
    portfolio_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    benchmark_name: str = "S&P 500"
) -> dict:
    """
    Compare portfolio performance against a benchmark.
    
    Args:
        portfolio_history: DataFrame with 'Date' and 'Value' columns
        benchmark_history: DataFrame with 'Date' and 'Close' columns
        benchmark_name: Name of the benchmark
        
    Returns:
        Dict comparing portfolio vs benchmark metrics
    """
    portfolio_metrics = compute_all_metrics(portfolio_history)
    
    # Build benchmark value history from close prices
    if benchmark_history is None or benchmark_history.empty:
        return {
            "portfolio": portfolio_metrics,
            "benchmark": None,
            "alpha": np.nan,
        }
    
    bench = benchmark_history.copy()
    if "Close" in bench.columns:
        bench["Value"] = bench["Close"]
    elif "Adj Close" in bench.columns:
        bench["Value"] = bench["Adj Close"]
    else:
        return {
            "portfolio": portfolio_metrics,
            "benchmark": None,
            "alpha": np.nan,
        }
    
    if "Date" not in bench.columns and not isinstance(bench.index, pd.DatetimeIndex):
        bench = bench.reset_index().rename(columns={"index": "Date"})
    
    benchmark_metrics = compute_all_metrics(bench)
    
    # Calculate alpha (simple excess return)
    alpha = portfolio_metrics["annualized_return"] - benchmark_metrics["annualized_return"]
    
    return {
        "portfolio": portfolio_metrics,
        "benchmark": benchmark_metrics,
        "benchmark_name": benchmark_name,
        "alpha": alpha,
    }
