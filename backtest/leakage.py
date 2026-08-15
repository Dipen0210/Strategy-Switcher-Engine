# backtest/leakage.py
"""
Look-ahead bias tests for the walk-forward harness.

A backtest that claims to be out-of-sample has to prove it. These tests are
adversarial: they try to make the engine reveal that it saw the future, and
fail the run if it did.

    test_decision_slices_truncated  no bar after T reaches the decision path
    test_folds_trained_before_use   no HMM governs a date inside its training window
    test_execution_uses_next_open   fills occur at the T+1 open, not the T close
    test_future_perturbation        THE decisive test — corrupt everything after
                                    a cutoff and confirm prior decisions and the
                                    prior equity curve are bit-identical

The perturbation test is the one that matters. Structural assertions can be
defeated by a refactor; that test cannot, because if any future information
reaches a decision, scrambling the future must change that decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig
from backtest.data import PointInTimeData
from backtest.hmm_schedule import build_fold_schedule, select_fold
from backtest.runner import run_backtest


# --------------------------------------------------------------------------
# Structural tests
# --------------------------------------------------------------------------

def test_decision_slices_truncated(pit: PointInTimeData, config: BacktestConfig) -> dict:
    """Every decision slice must end at or before its signal date."""
    schedule = pit.rebalance_schedule(config.start, config.end, config.rebalance_frequency)
    violations = []

    for signal_date in schedule:
        for ticker, df in pit.decision_slice(signal_date).items():
            if df["Date"].max() > signal_date:
                violations.append((str(signal_date.date()), ticker, str(df["Date"].max().date())))

    return {
        "test": "decision_slices_truncated",
        "checked": len(schedule),
        "violations": violations,
        "passed": not violations,
    }


def test_folds_trained_before_use(pit: PointInTimeData, config: BacktestConfig, folds) -> dict:
    """No fold may govern a date its training window covered."""
    violations = []
    for fold in folds:
        if fold.train_end > fold.govern_start:
            violations.append(
                f"fold {fold.fold_index}: trains to {fold.train_end.date()}, "
                f"governs from {fold.govern_start.date()}"
            )

    # And the fold actually selected per date must never be trained on it.
    schedule = pit.rebalance_schedule(config.start, config.end, config.rebalance_frequency)
    for signal_date in schedule:
        fold = select_fold(folds, signal_date)
        if fold is not None and fold.train_end > signal_date:
            violations.append(
                f"{signal_date.date()} used fold {fold.fold_index} "
                f"trained through {fold.train_end.date()}"
            )

    return {
        "test": "folds_trained_before_use",
        "checked": len(folds),
        "violations": violations,
        "passed": not violations,
    }


def test_execution_uses_next_open(result, pit: PointInTimeData) -> dict:
    """Fill prices must equal the execution day's OPEN."""
    violations = []
    if result.trades is None or result.trades.empty:
        return {"test": "execution_uses_next_open", "checked": 0,
                "violations": [], "passed": True, "note": "no trades"}

    for _, trade in result.trades.iterrows():
        ticker = trade["Ticker"]
        frame = pit.frames.get(ticker)
        if frame is None:
            continue
        exec_date = pd.Timestamp(trade["Date"]).normalize()
        row = frame.loc[frame["Date"] == exec_date]
        if row.empty:
            continue
        expected = float(row["Open"].iloc[0])
        actual = float(trade["Price"])
        if abs(expected - actual) > max(0.01, abs(expected) * 1e-4):
            violations.append(
                f"{ticker} {exec_date.date()}: filled {actual:.4f}, T+1 open {expected:.4f}"
            )

    return {
        "test": "execution_uses_next_open",
        "checked": int(len(result.trades)),
        "violations": violations[:10],
        "passed": not violations,
    }


# --------------------------------------------------------------------------
# The decisive test
# --------------------------------------------------------------------------

def perturb_future(pit: PointInTimeData, cutoff, seed: int = 1234) -> PointInTimeData:
    """
    Return a copy of `pit` whose bars AFTER `cutoff` are replaced with noise.

    Dates and pre-cutoff bars are untouched. Post-cutoff prices become a
    random walk seeded independently of the engine, so any dependence of a
    pre-cutoff decision on post-cutoff data becomes observable.
    """
    cutoff = pd.Timestamp(cutoff).normalize()
    rng = np.random.default_rng(seed)

    def _randomize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        future = out["Date"] > cutoff
        n = int(future.sum())
        if n == 0:
            return out

        last_close = float(out.loc[~future, "Close"].iloc[-1]) if (~future).any() else 100.0
        shocks = rng.normal(0.0, 0.05, size=n)          # deliberately large
        path = last_close * np.exp(np.cumsum(shocks))

        for col in ("Open", "High", "Low", "Close"):
            if col in out.columns:
                out.loc[future, col] = path
        return out

    perturbed = {ticker: _randomize(df) for ticker, df in pit.frames.items()}

    # The macro proxy MUST be carried through and perturbed on the same terms.
    # Dropping it silently changed which series the HMM trained on between the
    # baseline and perturbed runs (SPY vs an equal-weight composite fallback),
    # which made the comparison meaningless and reported a leak that was
    # really just two different models. Perturbing it is also the point: if a
    # pre-cutoff decision depended on post-cutoff macro data, that IS leakage
    # and this test must be able to see it.
    macro = pit.macro_frame
    perturbed_macro = _randomize(macro) if macro is not None else None

    return PointInTimeData(
        frames=perturbed,
        lookback_days=pit.lookback_days,
        macro_frame=perturbed_macro,
        macro_ticker=pit.macro_ticker,
    )


def test_future_perturbation(
    config: BacktestConfig,
    pit: PointInTimeData,
    cutoff: str,
    verbose: bool = True,
) -> dict:
    """
    Run twice — once on real data, once with everything after `cutoff`
    randomized — and require decisions and equity BEFORE the cutoff to match
    exactly.

    Any mismatch is proof of look-ahead.
    """
    cutoff_ts = pd.Timestamp(cutoff).normalize()

    if verbose:
        print(f"  baseline run...")
    folds_base = build_fold_schedule(pit, config, verbose=False)
    base = run_backtest(config, pit=pit, folds=folds_base, verbose=False)

    if verbose:
        print(f"  perturbed run (data after {cutoff_ts.date()} randomized)...")
    pit_perturbed = perturb_future(pit, cutoff_ts)
    folds_pert = build_fold_schedule(pit_perturbed, config, verbose=False)
    pert = run_backtest(config, pit=pit_perturbed, folds=folds_pert, verbose=False)

    violations = []

    # --- decisions before the cutoff must be identical -------------------
    def _pre(df, col):
        if df is None or df.empty:
            return df
        return df[pd.to_datetime(df[col]) <= cutoff_ts].reset_index(drop=True)

    d_base = _pre(base.decisions, "SignalDate")
    d_pert = _pre(pert.decisions, "SignalDate")

    if len(d_base) != len(d_pert):
        violations.append(f"decision count differs: {len(d_base)} vs {len(d_pert)}")
    else:
        for col in ("Ticker", "Regime", "Strategy", "Signal", "Participation"):
            if col not in d_base.columns:
                continue
            a, b = d_base[col], d_pert[col]
            mismatch = int((a.fillna("~") != b.fillna("~")).sum())
            if mismatch:
                violations.append(f"{mismatch}/{len(a)} pre-cutoff '{col}' values differ")

    # --- equity before the cutoff must be identical -----------------------
    e_base = base.equity[base.equity.index <= cutoff_ts]
    e_pert = pert.equity[pert.equity.index <= cutoff_ts]

    if len(e_base) != len(e_pert):
        violations.append(f"equity length differs: {len(e_base)} vs {len(e_pert)}")
    elif len(e_base):
        max_diff = float((e_base - e_pert).abs().max())
        if max_diff > 1e-6:
            violations.append(f"pre-cutoff equity diverges by up to {max_diff:.6f}")

    # --- sanity: the perturbation must actually matter AFTER the cutoff ---
    e_base_post = base.equity[base.equity.index > cutoff_ts]
    e_pert_post = pert.equity[pert.equity.index > cutoff_ts]
    post_differs = False
    if len(e_base_post) and len(e_pert_post):
        n = min(len(e_base_post), len(e_pert_post))
        post_differs = float((e_base_post.iloc[:n] - e_pert_post.iloc[:n]).abs().max()) > 1e-6
    if not post_differs:
        violations.append(
            "post-cutoff equity is unchanged — the perturbation had no effect, "
            "so this test proves nothing (check the corruption is applied)"
        )

    return {
        "test": "future_perturbation",
        "cutoff": str(cutoff_ts.date()),
        "pre_cutoff_decisions": int(len(d_base)) if d_base is not None else 0,
        "pre_cutoff_days": int(len(e_base)),
        "post_cutoff_changed": post_differs,
        "violations": violations,
        "passed": not violations,
    }


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------

def run_leakage_suite(config: BacktestConfig, pit=None, cutoff: str | None = None,
                      verbose: bool = True) -> list[dict]:
    """Run every leakage test and return their reports."""
    from backtest.data import load_point_in_time

    if pit is None:
        pit = load_point_in_time(config)

    folds = build_fold_schedule(pit, config, verbose=False)

    if cutoff is None:
        # Midpoint of the evaluation span, so both halves are substantial.
        span = pd.Timestamp(config.end) - pd.Timestamp(config.start)
        cutoff = (pd.Timestamp(config.start) + span / 2).strftime("%Y-%m-%d")

    reports = []
    if verbose:
        print("running leakage suite")

    reports.append(test_decision_slices_truncated(pit, config))
    reports.append(test_folds_trained_before_use(pit, config, folds))

    base = run_backtest(config, pit=pit, folds=folds, verbose=False)
    reports.append(test_execution_uses_next_open(base, pit))

    reports.append(test_future_perturbation(config, pit, cutoff, verbose=verbose))

    return reports


def print_leakage_report(reports: list[dict]) -> bool:
    """Render the suite result. Returns True if everything passed."""
    all_passed = True
    print()
    print("=" * 68)
    print("LOOK-AHEAD BIAS TEST SUITE")
    print("=" * 68)
    for r in reports:
        status = "PASS" if r["passed"] else "FAIL"
        all_passed &= r["passed"]
        print(f"[{status}] {r['test']}")
        for k, v in r.items():
            if k in ("test", "passed", "violations"):
                continue
            print(f"         {k}: {v}")
        for v in r.get("violations", []):
            print(f"         !! {v}")
    print("=" * 68)
    print("RESULT:", "no look-ahead detected" if all_passed else "LEAKAGE DETECTED")
    print("=" * 68)
    return all_passed
