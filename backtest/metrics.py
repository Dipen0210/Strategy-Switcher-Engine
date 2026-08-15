# backtest/metrics.py
"""
Performance metrics computed from a daily equity curve.

All statistics derive from the DAILY series, not from rebalance-date
snapshots. Sampling only at rebalances understates both volatility and
drawdown, which flatters Sharpe and max-drawdown figures.

Sharpe here is excess of a configurable risk-free rate. Quoting a Sharpe of
1.15 over a period when cash paid 4-5% is not comparable to one computed
against zero, so the rate is explicit rather than implied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start_val, end_val = float(equity.iloc[0]), float(equity.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (end_val / start_val) ** (1.0 / years) - 1.0


def annualized_volatility(equity: pd.Series) -> float:
    r = daily_returns(equity)
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe of daily excess returns."""
    r = daily_returns(equity)
    if len(r) < 2:
        return 0.0
    excess = r - (risk_free_rate / TRADING_DAYS)
    sd = excess.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    r = daily_returns(equity)
    if len(r) < 2:
        return 0.0
    excess = r - (risk_free_rate / TRADING_DAYS)
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if len(downside) < 2 or dd == 0 or not np.isfinite(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough decline. Returns a NEGATIVE number."""
    if len(equity) < 2:
        return 0.0
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    return float(cagr(equity) / mdd) if mdd > 1e-12 else 0.0


def win_rate(equity: pd.Series) -> float:
    """Share of daily returns that are positive."""
    r = daily_returns(equity)
    return float((r > 0).mean()) if len(r) else 0.0


def turnover(cycles: pd.DataFrame) -> float:
    """Average traded notional per cycle as a fraction of equity."""
    if cycles is None or cycles.empty or "TradedNotional" not in cycles:
        return 0.0
    eq = cycles["Equity"].replace(0, np.nan)
    return float((cycles["TradedNotional"] / eq).replace([np.inf, -np.inf], np.nan).mean())


def total_costs(cycles: pd.DataFrame) -> float:
    if cycles is None or cycles.empty or "Fees" not in cycles:
        return 0.0
    return float(cycles["Fees"].sum())


def time_in_market(equity: pd.Series, cycles: pd.DataFrame) -> float:
    """Average share of equity actually deployed (1 - cash weight)."""
    if cycles is None or cycles.empty or "Cash" not in cycles:
        return float("nan")
    eq = cycles["Equity"].replace(0, np.nan)
    return float((1.0 - (cycles["Cash"] / eq)).clip(0, 1).mean())


def compute_metrics(
    equity: pd.Series,
    cycles: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    label: str = "",
) -> dict:
    """Full metric bundle for one run."""
    if equity is None or len(equity) < 2:
        return {"label": label, "n_days": 0, "error": "insufficient equity data"}

    return {
        "label": label,
        "n_days": int(len(equity)),
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr(equity),
        "volatility": annualized_volatility(equity),
        "sharpe": sharpe_ratio(equity, risk_free_rate),
        "sortino": sortino_ratio(equity, risk_free_rate),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "win_rate": win_rate(equity),
        "turnover_per_cycle": turnover(cycles),
        "total_costs": total_costs(cycles),
        "time_in_market": time_in_market(equity, cycles),
        "n_cycles": int(len(cycles)) if cycles is not None else 0,
        "risk_free_rate": risk_free_rate,
    }


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Format {label: metrics} into a comparison table."""
    rows = []
    for label, m in results.items():
        rows.append({
            "Strategy": label,
            "CAGR": f"{m.get('cagr', 0):.2%}",
            "Vol": f"{m.get('volatility', 0):.2%}",
            "Sharpe": f"{m.get('sharpe', 0):.2f}",
            "MaxDD": f"{m.get('max_drawdown', 0):.2%}",
            "Calmar": f"{m.get('calmar', 0):.2f}",
            "WinRate": f"{m.get('win_rate', 0):.1%}",
            "Turnover": f"{m.get('turnover_per_cycle', 0):.3f}",
            "InMkt": f"{m.get('time_in_market', float('nan')):.1%}",
        })
    return pd.DataFrame(rows)
