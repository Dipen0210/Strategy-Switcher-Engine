# backtest/baselines.py
"""
Benchmark portfolios.

Baselines run through the SAME execution machinery as the engine — identical
T+1 open fills, identical cost model, identical point-in-time slicing — so a
performance difference reflects the decision logic rather than a difference in
simulation fidelity. Comparing an engine that pays costs against a benchmark
that does not is a common and badly misleading error.

Implemented
-----------
buy_and_hold                  equal-weight the universe once, then hold
static_equal_weight_strategies  the paper's "Static 40": every strategy funded
                              at all times regardless of regime, i.e. the
                              "Strategy Soup" the pod architecture is meant to
                              beat. This is the comparison that tests the pod
                              hypothesis, and it had no implementation before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from layers.L1_data_features import compute_all_features
from layers.L3_strategy_universe import run_strategies_for_regime
from layers.L7_position_sizing import signal_participation, compute_position_sizes
from layers.L8_signal_generation import generate_portfolio_signals
from layers.L10_trade_execution import run_execution_cycle
from layers.L11_rebalancing import PortfolioState

from backtest.config import BacktestConfig
from backtest.data import PointInTimeData
from backtest.runner import BacktestResult, _daily_equity


ALL_REGIMES = ("Bull-Quiet", "Bull-Volatile", "Sideways", "Crisis")


def _run_weight_schedule(
    config: BacktestConfig,
    pit: PointInTimeData,
    weight_fn,
    label: str,
    rebalance: bool = True,
) -> BacktestResult:
    """
    Shared driver: walk the schedule, ask `weight_fn` for target weights,
    execute through L8/L10 exactly as the engine does.

    Args:
        weight_fn: (signal_date, decision_data) -> {ticker: weight}
        rebalance: if False, only the first cycle trades (pure buy-and-hold).
    """
    state = PortfolioState(cash=config.initial_capital, initial_capital=config.initial_capital)
    schedule = pit.rebalance_schedule(config.start, config.end, config.rebalance_frequency)
    closes = pit.close_matrix()

    book_history: list[dict] = []
    cycle_rows: list[dict] = []
    strategies_held: dict[str, str] = {}

    for cycle_num, signal_date in enumerate(schedule):
        if not rebalance and cycle_num > 0:
            continue

        execution_date = pit.next_trading_day(signal_date)
        if execution_date is None:
            break

        decision_data = pit.decision_slice(signal_date)
        execution_data = pit.execution_slice(signal_date, execution_date)
        if not decision_data:
            continue

        weights = weight_fn(signal_date, decision_data)
        if not weights:
            continue

        new_portfolio_df = pd.DataFrame(
            {"Ticker": list(weights.keys()), "Weight": list(weights.values())}
        )

        signals = generate_portfolio_signals(
            old_portfolio_df=state.last_allocation,
            new_portfolio_df=new_portfolio_df,
            old_strategies=dict(strategies_held),
            new_strategy={t: label for t in weights},
            as_of_date=signal_date.strftime("%Y-%m-%d"),
        )
        if signals.empty:
            continue

        try:
            report = run_execution_cycle(
                state=state,
                price_data_dict=execution_data,
                signals_df=signals,
                new_portfolio_weights=new_portfolio_df,
                date=execution_date.strftime("%Y-%m-%d"),
                commission_per_trade=config.commission_per_trade,
                cost_bps=config.cost_bps,
            )
        except Exception:
            continue

        strategies_held = dict(state.position_strategies)
        book_history.append({
            "date": pd.Timestamp(execution_date),
            "positions": dict(state.positions),
            "cash": float(state.cash),
        })

        prices_now = closes.reindex([pd.Timestamp(execution_date)]).ffill().iloc[0]
        equity_now = float(
            sum(prices_now.get(t, np.nan) * q for t, q in state.positions.items()) + state.cash
        ) if state.positions else float(state.cash)

        fills = report.get("fills")
        cycle_rows.append({
            "Cycle": cycle_num,
            "SignalDate": signal_date,
            "ExecutionDate": pd.Timestamp(execution_date),
            "DominantRegime": label,
            "Equity": equity_now,
            "Cash": float(state.cash),
            "NumPositions": len(state.positions),
            "TradedNotional": float(fills["Notional"].abs().sum()) if fills is not None and not fills.empty else 0.0,
            "Fees": float(report.get("fees_paid", 0.0)),
            "EmergencyExit": False,
        })

    equity = _daily_equity(
        book_history, closes, config.initial_capital,
        start=pd.Timestamp(config.start), end=pd.Timestamp(config.end),
    )

    return BacktestResult(
        config=config,
        equity=equity,
        cycles=pd.DataFrame(cycle_rows),
        trades=pd.DataFrame(),
        decisions=pd.DataFrame(),
    )


def buy_and_hold(config: BacktestConfig, pit: PointInTimeData) -> BacktestResult:
    """Equal-weight the universe at the first cycle and hold to the end."""
    n = len(pit.frames)

    def weights(signal_date, decision_data):
        return {t: 1.0 / n for t in decision_data}

    return _run_weight_schedule(config, pit, weights, label="BuyHold", rebalance=False)


def static_equal_weight_strategies(config: BacktestConfig, pit: PointInTimeData) -> BacktestResult:
    """
    The paper's "Static 40" baseline.

    Every strategy in all four pods votes on every ticker at every rebalance,
    with equal weight and no regime gating and no bandit learning. A ticker's
    target weight is its equal-weight share scaled by the AVERAGE participation
    implied across all 40 strategies.

    This is the concrete form of the "Strategy Soup" problem: conflicting
    strategies cancel each other out, and the average conviction sits near
    neutral almost all the time.
    """
    n = len(pit.frames)

    def weights(signal_date, decision_data):
        out: dict[str, float] = {}
        for ticker, df in decision_data.items():
            enriched = compute_all_features(df, use_garch=config.use_garch)

            participations = []
            for regime in ALL_REGIMES:
                try:
                    outputs = run_strategies_for_regime(
                        regime=regime, stock_data_dict={ticker: enriched}
                    )
                except Exception:
                    continue
                for o in outputs:
                    if o.ticker == ticker:
                        participations.append(signal_participation(o.signal, o.confidence))

            avg_participation = float(np.mean(participations)) if participations else 0.0
            out[ticker] = (1.0 / n) * avg_participation

        # Apply the same L7 volatility/leverage treatment the engine gets, so
        # the comparison isolates strategy selection rather than sizing.
        vol = {}
        for ticker, df in decision_data.items():
            enriched = compute_all_features(df, use_garch=config.use_garch)
            vol[ticker] = (
                float(enriched["Realized_Vol"].iloc[-1])
                if "Realized_Vol" in enriched.columns and len(enriched)
                and np.isfinite(enriched["Realized_Vol"].iloc[-1]) else 0.15
            )

        sized = compute_position_sizes(
            user_weights=pd.Series(out),
            forecast_vol=pd.Series(vol),
            total_capital=config.initial_capital,
            target_vol=0.15,
            max_leverage=1.0,
        )
        if sized.empty:
            return {}
        return dict(zip(sized["Ticker"], sized["Adjusted_Weight"]))

    return _run_weight_schedule(config, pit, weights, label="Static40", rebalance=True)


def benchmark_buy_and_hold(config: BacktestConfig, pit: PointInTimeData) -> BacktestResult:
    """
    Buy-and-hold the market proxy (SPX via SPY) — the paper's index benchmark.

    Distinct from `buy_and_hold`, which equal-weights the traded universe.
    Both are reported: the universe leg says whether stock selection added
    anything, the index leg says whether the whole exercise beat the market.

    Runs through the same execution path (T+1 open, same cost model, same
    date grid) by handing the benchmark frame to the standard runner, so any
    difference against the engine cannot come from accounting.
    """
    if pit.macro_frame is None:
        raise ValueError(
            "no macro_frame on PointInTimeData; benchmark leg cannot run"
        )

    bench_pit = PointInTimeData(
        frames={pit.macro_ticker: pit.macro_frame},
        lookback_days=pit.lookback_days,
        macro_frame=pit.macro_frame,
        macro_ticker=pit.macro_ticker,
    )

    def weights(signal_date, decision_data):
        return {t: 1.0 for t in decision_data}

    return _run_weight_schedule(
        config.with_(universe=(pit.macro_ticker,)),
        bench_pit, weights, label="BenchmarkBuyHold", rebalance=False,
    )


BASELINES = {
    "buy_and_hold": buy_and_hold,
    "static_40": static_equal_weight_strategies,
    "benchmark_spx": benchmark_buy_and_hold,
}
