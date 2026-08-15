# backtest/runner.py
"""
Walk-forward backtest runner.

Executes the engine cycle by cycle over the evaluation span under strict
point-in-time discipline:

    1. Select the HMM fold whose training window closed before T.
    2. Hand the engine ONLY bars <= T for the decision.
    3. Hand execution the T+1 bar so fills happen at the next open.
    4. Record the resulting book, then value it daily until the next cycle.

Bandit learning is inherently sequential — each update uses only rewards
realized by T — so the bandits need no special treatment beyond starting from
clean, seeded state per run. That is handled by giving every run its own
isolated bandit directory.
"""

from __future__ import annotations

import contextlib
import io
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from layers.L0_user_policy import create_policy
from pipeline import StrategyEngine

from backtest.config import BacktestConfig
from backtest.data import PointInTimeData, load_point_in_time
from backtest.hmm_schedule import HMMFold, build_fold_schedule, select_fold, assert_no_lookahead


@dataclass
class BacktestResult:
    """Everything a run produced, plus the config that produced it."""

    config: BacktestConfig
    equity: pd.Series                      # daily portfolio value
    cycles: pd.DataFrame                   # one row per rebalance
    trades: pd.DataFrame                   # every fill
    decisions: pd.DataFrame                # per ticker per cycle
    metrics: dict = field(default_factory=dict)

    def daily_returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def _seed_everything(seed: int) -> None:
    """
    Seed both RNGs the engine touches.

    Bandit weights initialize via np.random.rand and epsilon-greedy
    exploration draws from random.random, so BOTH must be seeded or runs are
    not reproducible. Bandit init being random is also why single-seed results
    are meaningless — see backtest.statistics.
    """
    np.random.seed(seed)
    random.seed(seed)


def _build_engine(config: BacktestConfig, bandit_dir: Path) -> StrategyEngine:
    weights = [1.0 / len(config.universe)] * len(config.universe)
    policy = create_policy(
        tickers=list(config.universe),
        weights=weights,
        total_capital=config.initial_capital,
        risk_tolerance=config.risk_tolerance,
        rebalance_frequency=config.rebalance_frequency,
        emergency_drawdown_threshold=config.emergency_drawdown_threshold,
    )
    engine = StrategyEngine(
        policy=policy,
        bandit_dir=bandit_dir,
        use_garch=config.use_garch,
        logging_enabled=False,
    )

    # Sensitivity-sweep hyperparameters. Applied through the managers rather
    # than by rewriting module constants, which would leak across runs sharing
    # an interpreter and silently corrupt every other arm in the sweep.
    bandits = engine.ensemble_bandits
    if config.bandit_delta is not None:
        bandits.global_bandit.decay_factor = float(config.bandit_delta)
        bandits.regime_bandits.set_hyperparameters(decay_factor=config.bandit_delta)
        bandits.stock_bandits.set_hyperparameters(decay_factor=config.bandit_delta)
    if config.bandit_epsilon is not None:
        bandits.regime_bandits.set_hyperparameters(epsilon=config.bandit_epsilon)

    return engine


def _daily_equity(
    book_history: list[dict],
    closes: pd.DataFrame,
    initial_capital: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    Value the book every trading day between rebalances.

    Rebalance-date-only equity would understate both volatility and drawdown,
    so positions are held constant between cycles and marked to market daily.
    """
    days = closes.loc[(closes.index >= start) & (closes.index <= end)].index
    if len(days) == 0:
        return pd.Series(dtype=float)

    positions = pd.DataFrame(
        [h["positions"] for h in book_history],
        index=[h["date"] for h in book_history],
    ).reindex(columns=closes.columns).fillna(0.0)
    cash = pd.Series(
        [h["cash"] for h in book_history],
        index=[h["date"] for h in book_history],
    )

    # Hold the book steady until the next rebalance actually changes it.
    positions = positions.reindex(days, method="ffill").fillna(0.0)
    cash = cash.reindex(days, method="ffill")

    # Before the first fill the account is entirely cash.
    cash = cash.fillna(initial_capital)

    holdings = (positions * closes.reindex(days).ffill()).sum(axis=1)
    equity = holdings + cash
    return equity.astype(float)


def run_backtest(
    config: BacktestConfig,
    pit: PointInTimeData | None = None,
    folds: list[HMMFold] | None = None,
    ablation_hooks: dict | None = None,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run one full walk-forward backtest.

    Args:
        config: the run specification.
        pit: pre-loaded data (reused across ablations/seeds to avoid refetch).
        folds: pre-built HMM schedule (reused for the same reason).
        ablation_hooks: optional component overrides from backtest.ablations.
    """
    if pit is None:
        pit = load_point_in_time(config)
    if folds is None:
        folds = build_fold_schedule(pit, config, verbose=verbose)
    assert_no_lookahead(folds)

    _seed_everything(config.seed)

    bandit_dir = Path(tempfile.mkdtemp(prefix=f"bt_{config.label()}_")) / "bandits"
    engine = _build_engine(config, bandit_dir)

    if ablation_hooks:
        for name, fn in ablation_hooks.items():
            fn(engine, config)

    schedule = pit.rebalance_schedule(config.start, config.end, config.rebalance_frequency)
    closes = pit.close_matrix()

    if verbose:
        print(f"  {len(schedule)} rebalance cycles "
              f"{config.start}..{config.end} [{config.ablation}, seed={config.seed}]")

    book_history: list[dict] = []
    cycle_rows: list[dict] = []
    decision_rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    failures: list[tuple] = []
    active_fold_idx = None

    for cycle_num, signal_date in enumerate(schedule):
        execution_date = pit.next_trading_day(signal_date)
        if execution_date is None:
            break

        # --- select the fold trained strictly before this date -------------
        fold = select_fold(folds, signal_date)
        if fold is not None:
            engine.regime_manager.macro_detector = fold.detector
            active_fold_idx = fold.fold_index
        else:
            # No model has enough history yet — engine falls back to its
            # rule-based detector rather than borrowing a future model.
            engine.regime_manager.macro_detector = None
            active_fold_idx = None

        decision_data = pit.decision_slice(signal_date)
        execution_data = pit.execution_slice(signal_date, execution_date)
        if not decision_data:
            continue

        # Engine prints per-cycle diagnostics; silence them across thousands
        # of cycles but keep failures visible.
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink):
                result = engine.run(
                    decision_data,
                    current_date=signal_date.strftime("%Y-%m-%d"),
                    execution_data_dict=execution_data,
                    commission_per_trade=config.commission_per_trade,
                    cost_bps=config.cost_bps,
                    # Derived from the actual price data, not a rule calendar.
                    execution_date=execution_date.strftime("%Y-%m-%d"),
                )
        except Exception as exc:
            # Record rather than silently skip. A per-cycle exception used to
            # be swallowed here, so an arm that failed on EVERY cycle reported
            # a flat 0.00% equity curve indistinguishable from a real result.
            failures.append((signal_date, repr(exc)))
            if verbose:
                print(f"    ! cycle {signal_date.date()} failed: {exc}")
            continue

        state = engine.portfolio_state
        book_history.append({
            "date": pd.Timestamp(execution_date),
            "positions": dict(state.positions),
            "cash": float(state.cash),
        })

        fills = (result.execution_report or {}).get("fills")
        traded_notional = 0.0
        if fills is not None and not fills.empty:
            f = fills.copy()
            f["Cycle"] = cycle_num
            f["SignalDate"] = signal_date
            trade_frames.append(f)
            traded_notional = float(f["Notional"].abs().sum())

        prices_now = closes.reindex([pd.Timestamp(execution_date)]).ffill().iloc[0]
        equity_now = float(
            sum(prices_now.get(t, np.nan) * q for t, q in state.positions.items())
            + state.cash
        ) if state.positions else float(state.cash)

        cycle_rows.append({
            "Cycle": cycle_num,
            "SignalDate": signal_date,
            "ExecutionDate": pd.Timestamp(execution_date),
            "HMMFold": active_fold_idx,
            "DominantRegime": result.dominant_regime,
            "Equity": equity_now,
            "Cash": float(state.cash),
            "NumPositions": len(state.positions),
            "TradedNotional": traded_notional,
            "Fees": float((result.execution_report or {}).get("fees_paid", 0.0)),
            "EmergencyExit": bool(result.emergency_triggered),
        })

        for ticker, strategy in (result.per_stock_strategies or {}).items():
            det = (result.per_stock_details or {}).get(ticker, {})
            decision_rows.append({
                "SignalDate": signal_date,
                "Ticker": ticker,
                "Regime": det.get("regime"),
                "Strategy": strategy,
                "Signal": det.get("winner_signal"),
                "Confidence": det.get("winner_confidence"),
                "Participation": det.get("participation"),
                "HMMConfidence": det.get("hmm_confidence"),
                "Stability": det.get("stability"),
            })

    # --- fail loudly on a broken configuration --------------------------
    # A silently-failing arm is worse than a crash: it produces a plausible
    # looking flat equity curve that gets tabulated as a finding.
    attempted = len(schedule)
    failure_rate = len(failures) / attempted if attempted else 0.0

    if failures and failure_rate >= 0.5:
        first_date, first_exc = failures[0]
        raise RuntimeError(
            f"[{config.ablation}/seed{config.seed}] {len(failures)}/{attempted} "
            f"cycles failed ({failure_rate:.0%}). This configuration is broken, "
            f"not underperforming. First failure {first_date.date()}: {first_exc}"
        )
    if failures:
        print(f"    warning [{config.ablation}/seed{config.seed}]: "
              f"{len(failures)}/{attempted} cycles failed "
              f"({failure_rate:.1%}); first: {failures[0][1]}")

    equity = _daily_equity(
        book_history, closes, config.initial_capital,
        start=pd.Timestamp(config.start), end=pd.Timestamp(config.end),
    )

    # Clean up the isolated bandit state; artifacts we care about are returned.
    shutil.rmtree(bandit_dir.parent, ignore_errors=True)

    return BacktestResult(
        config=config,
        equity=equity,
        cycles=pd.DataFrame(cycle_rows),
        trades=pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
        decisions=pd.DataFrame(decision_rows),
    )
