# backtest/data.py
"""
Point-in-time market data for walk-forward backtesting.

This module is the primary structural defense against look-ahead bias. The
engine never receives a DataFrame it could inspect for future information:
every slice handed out is truncated at an explicit cutoff, and the truncation
is ASSERTED rather than assumed.

Two distinct cutoffs exist per cycle:

    decision slice  : bars with Date <= T      (signal date)
    execution slice : bars with Date <= T+1    (fill date, T+1 open)

Keeping them separate is what stops the decision path from seeing the bar it
is about to trade into.

KNOWN LIMITATION — SURVIVORSHIP BIAS
------------------------------------
The universe is supplied as a fixed list of tickers that exist today. Running
that list back through history silently excludes companies that were delisted,
acquired, or went bankrupt, which biases returns upward. This is NOT corrected
here and must be disclosed with any result. Fixing it properly requires
point-in-time index constituents (e.g. a CRSP or Compustat membership file),
which yfinance cannot provide.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from utils.trading_calendar import is_us_business_day


OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output into a plain Date-indexed-by-column frame."""
    out = df.reset_index()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    if "Date" not in out.columns:
        for cand in ("index", "Datetime"):
            if cand in out.columns:
                out = out.rename(columns={cand: "Date"})
                break
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None).dt.normalize()
    keep = [c for c in OHLCV_COLUMNS if c in out.columns]
    out = out[keep].sort_values("Date").reset_index(drop=True)
    return out.dropna(subset=["Close"])


def download_history(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    cache_dir: Path,
    force_refresh: bool = False,
    adjusted: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch full OHLCV history once and cache it to disk.

    Downloading per-cycle would be both slow and non-reproducible, so the
    entire span is pulled up front and every backtest reads the same frozen
    snapshot.

    Args:
        adjusted: use dividend- and split-adjusted prices (total return).
            The previous behavior kept raw Close and DISCARDED Adj Close, so
            every return in the system was price-only. For SPY 2015-2024 that
            understates CAGR by roughly 2 percentage points a year (11.10% vs
            13.05%). It biases the engine and the baselines equally, so
            relative comparisons stayed fair, but absolute figures were low
            and would not reconcile against published index returns.

            Adjusting OHLC together keeps the T+1 OPEN — the price execution
            actually fills at — consistent with the return series.
    """
    import yfinance as yf

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    suffix = "adj" if adjusted else "raw"
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}_{start}_{end}_{suffix}.parquet"

        if cache_path.exists() and not force_refresh:
            data[ticker] = pd.read_parquet(cache_path)
            continue

        raw = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=adjusted,
        )
        if raw is None or raw.empty:
            print(f"  ! no data for {ticker}, excluded from universe")
            continue

        frame = _normalize_frame(raw)
        frame.to_parquet(cache_path, index=False)
        data[ticker] = frame

    return data


@dataclass
class PointInTimeData:
    """
    Immutable full-history store that only ever hands out truncated views.

    Args:
        frames: {ticker: full-history OHLCV frame}
        lookback_days: calendar-day window handed to the engine each cycle.
            Bounded rather than expanding so per-cycle feature cost stays flat;
            must exceed the longest strategy TIMEFRAME.
    """

    frames: dict[str, pd.DataFrame]
    lookback_days: int = 500
    # Market proxy used to fit the macro HMM and to run the benchmark leg.
    # Kept OUT of `frames` so it is never traded and never enters an
    # equal-weight baseline by accident — it describes the market, it is not
    # part of the portfolio.
    macro_frame: pd.DataFrame | None = None
    macro_ticker: str = "SPY"

    # ---------------------------------------------------------------- slices
    def _slice(self, cutoff: pd.Timestamp, window_start: pd.Timestamp) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for ticker, df in self.frames.items():
            mask = (df["Date"] <= cutoff) & (df["Date"] >= window_start)
            sliced = df.loc[mask].reset_index(drop=True)
            if sliced.empty:
                continue

            # Structural guarantee, not a comment: nothing after the cutoff
            # can reach the engine.
            assert sliced["Date"].max() <= cutoff, (
                f"{ticker}: slice leaked a bar after cutoff {cutoff.date()}"
            )
            out[ticker] = sliced
        return out

    def decision_slice(self, signal_date) -> dict[str, pd.DataFrame]:
        """Bars usable for deciding on `signal_date` (inclusive of T)."""
        cutoff = pd.Timestamp(signal_date).normalize()
        window_start = cutoff - pd.Timedelta(days=self.lookback_days)
        return self._slice(cutoff, window_start)

    def execution_slice(self, signal_date, execution_date) -> dict[str, pd.DataFrame]:
        """Bars usable for filling on `execution_date` (inclusive of T+1)."""
        cutoff = pd.Timestamp(execution_date).normalize()
        window_start = pd.Timestamp(signal_date).normalize() - pd.Timedelta(days=self.lookback_days)
        return self._slice(cutoff, window_start)

    def macro_training_slice(self, cutoff) -> pd.DataFrame | None:
        """
        Macro proxy history strictly BEFORE `cutoff`, for fitting the HMM.

        Same strict inequality as training_slice: the model that decides at T
        must not have seen T.
        """
        if self.macro_frame is None:
            return None
        cutoff = pd.Timestamp(cutoff).normalize()
        sliced = self.macro_frame.loc[self.macro_frame["Date"] < cutoff].reset_index(drop=True)
        if sliced.empty:
            return None
        assert sliced["Date"].max() < cutoff, "macro training slice leaked the cutoff date"
        return sliced

    def training_slice(self, cutoff) -> dict[str, pd.DataFrame]:
        """
        Full history strictly BEFORE `cutoff`, for model fitting.

        Strict inequality: a model used to decide at T must not have seen T.
        """
        cutoff = pd.Timestamp(cutoff).normalize()
        out = {}
        for ticker, df in self.frames.items():
            sliced = df.loc[df["Date"] < cutoff].reset_index(drop=True)
            if sliced.empty:
                continue
            assert sliced["Date"].max() < cutoff, f"{ticker}: training slice leaked {cutoff.date()}"
            out[ticker] = sliced
        return out

    # ------------------------------------------------------------- calendars
    def trading_days(self, start=None, end=None) -> pd.DatetimeIndex:
        """Union of all tickers' observed trading days (actual market data)."""
        all_dates = pd.DatetimeIndex([])
        for df in self.frames.values():
            all_dates = all_dates.union(pd.DatetimeIndex(df["Date"]))
        if start is not None:
            all_dates = all_dates[all_dates >= pd.Timestamp(start).normalize()]
        if end is not None:
            all_dates = all_dates[all_dates <= pd.Timestamp(end).normalize()]
        return all_dates.sort_values()

    def next_trading_day(self, date) -> pd.Timestamp | None:
        """First observed trading day strictly after `date`."""
        days = self.trading_days()
        after = days[days > pd.Timestamp(date).normalize()]
        return after[0] if len(after) else None

    def rebalance_schedule(self, start, end, frequency: str = "Weekly") -> list[pd.Timestamp]:
        """
        Signal dates on the observed trading calendar.

        Each returned date T requires a real T+1 bar to exist, so the final
        stub period (where no execution bar follows) is dropped rather than
        silently filled at a stale price.
        """
        step = {"Daily": 1, "Weekly": 7, "Monthly": 30}.get(frequency, 7)
        days = self.trading_days(start, end)
        if len(days) == 0:
            return []

        schedule: list[pd.Timestamp] = []
        cursor = days[0]
        last = days[-1]
        while cursor <= last:
            candidates = days[days >= cursor]
            if len(candidates) == 0:
                break
            signal_date = candidates[0]
            if self.next_trading_day(signal_date) is None:
                break
            schedule.append(signal_date)
            cursor = signal_date + pd.Timedelta(days=step)

        return schedule

    # --------------------------------------------------------------- pricing
    def close_matrix(self, start=None, end=None) -> pd.DataFrame:
        """Daily close prices, tickers as columns — used to value the book."""
        series = {}
        for ticker, df in self.frames.items():
            s = df.set_index("Date")["Close"]
            series[ticker] = s[~s.index.duplicated(keep="last")]
        matrix = pd.DataFrame(series).sort_index()
        if start is not None:
            matrix = matrix.loc[matrix.index >= pd.Timestamp(start).normalize()]
        if end is not None:
            matrix = matrix.loc[matrix.index <= pd.Timestamp(end).normalize()]
        return matrix


def load_point_in_time(config) -> PointInTimeData:
    """
    Build a PointInTimeData for a config.

    Extra history is pulled BEFORE config.start so the first rebalance already
    has a full feature warm-up and the HMM has a training window — without
    that buffer the first folds would train on almost nothing.
    """
    buffer_days = int(365 * config.hmm_train_years) + config.lookback_days + 60
    fetch_start = (pd.Timestamp(config.start) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")

    macro_ticker = getattr(config, "macro_ticker", "SPY")
    wanted = tuple(config.universe)
    if macro_ticker and macro_ticker not in wanted:
        wanted = wanted + (macro_ticker,)

    frames = download_history(
        tickers=wanted,
        start=fetch_start,
        end=config.end,
        cache_dir=config.cache_dir,
        adjusted=getattr(config, "use_adjusted_prices", True),
    )

    # Split the macro proxy out unless it is genuinely part of the traded
    # universe. Leaving it in frames would put it in the portfolio and in the
    # equal-weight benchmark.
    macro_frame = frames.get(macro_ticker)
    traded = {t: f for t, f in frames.items() if t in set(config.universe)}

    return PointInTimeData(
        frames=traded,
        lookback_days=config.lookback_days,
        macro_frame=macro_frame,
        macro_ticker=macro_ticker,
    )
