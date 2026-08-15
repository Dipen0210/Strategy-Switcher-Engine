# backtest/validate.py
"""
Pipeline validation against externally verifiable facts.

The point
--------
A backtest can only be trusted if its benchmark leg reproduces something you
can check independently. SPY's behavior over 2015-2024 is public record, so if
this harness cannot recover it, the bug is in the data or execution layer and
nothing downstream means anything.

These are ORACLES, not targets. They constrain the benchmark, which has a
known right answer. They say nothing about what the ENGINE should produce, and
they must never be used to decide whether the engine's own numbers are
"acceptable" — tuning until a strategy lands in an expected range is a
selection effect dressed up as validation.

Deliberately excluded: any expected range for engine CAGR, Sharpe, or ablation
deltas. If the engine returns 4%, the finding is 4%.
"""

from __future__ import annotations

import pandas as pd

from backtest.config import BacktestConfig
from backtest.data import load_point_in_time
from backtest.baselines import buy_and_hold
from backtest.metrics import compute_metrics, max_drawdown


# Externally verifiable reference points. Bands are WIDE on purpose — they are
# there to catch a broken pipeline (wrong prices, missed splits, bad execution
# accounting), not to certify a result to two decimal places.
SPY_REFERENCES = {
    "cagr_2015_2024_total_return": {
        "expected": 0.1305,
        "low": 0.11, "high": 0.15,
        "note": "SPY dividend-adjusted CAGR 2015-01-01..2024-12-31. "
                "Price-only is ~11.1%, so landing near that instead means "
                "dividends are being dropped.",
    },
    "covid_drawdown_2020": {
        "expected": -0.337,
        "low": -0.40, "high": -0.28,
        "note": "SPY peak-to-trough Feb-Mar 2020. Very robust; a large miss "
                "means the price series or the equity accounting is wrong.",
    },
}


def validate_spy_benchmark(cache_dir=None, verbose: bool = True) -> list[dict]:
    """
    Run buy-and-hold SPY through the harness and check it against public fact.

    This exercises the whole stack — download, point-in-time slicing, T+1 open
    execution, cost accounting, daily equity construction — so a failure here
    localizes the bug to infrastructure rather than strategy logic.
    """
    config = BacktestConfig(
        universe=("SPY",),
        start="2015-01-01", end="2024-12-31",
        initial_capital=100_000.0,
        cost_bps=0.0, commission_per_trade=0.0,   # benchmark pays no costs
        rebalance_frequency="Monthly",
        hmm_train_years=4,
    )
    if cache_dir is not None:
        config = config.with_(cache_dir=cache_dir)

    pit = load_point_in_time(config)
    if "SPY" not in pit.frames:
        return [{"check": "spy_download", "passed": False,
                 "detail": "SPY data unavailable"}]

    result = buy_and_hold(config, pit)
    metrics = compute_metrics(result.equity, result.cycles)

    checks = []

    ref = SPY_REFERENCES["cagr_2015_2024_total_return"]
    actual = metrics["cagr"]
    checks.append({
        "check": "SPY buy-hold CAGR 2015-2024",
        "actual": f"{actual:.2%}",
        "expected": f"{ref['expected']:.2%} (accept {ref['low']:.0%}-{ref['high']:.0%})",
        "passed": ref["low"] <= actual <= ref["high"],
        "note": ref["note"],
    })

    covid = result.equity.loc["2020-01-01":"2020-06-30"]
    ref = SPY_REFERENCES["covid_drawdown_2020"]
    actual_dd = max_drawdown(covid) if len(covid) > 5 else 0.0
    checks.append({
        "check": "SPY COVID drawdown (Feb-Mar 2020)",
        "actual": f"{actual_dd:.1%}",
        "expected": f"{ref['expected']:.1%} (accept {ref['low']:.0%}..{ref['high']:.0%})",
        "passed": ref["low"] <= actual_dd <= ref["high"],
        "note": ref["note"],
    })

    if verbose:
        print_validation(checks)
    return checks


def validate_seed_variance(metrics_by_seed: list[dict]) -> dict:
    """
    Confirm seeds actually produce different runs.

    Bandit weights initialize from np.random.rand and epsilon-greedy draws
    from random.random, so BOTH generators must be seeded. If they are not,
    every seed returns an identical curve and the reported "± std" is a
    fiction. Zero variance is a bug signal, not a stability result.
    """
    sharpes = [m.get("sharpe") for m in metrics_by_seed if m.get("sharpe") is not None]
    if len(sharpes) < 2:
        return {"check": "seed variance", "passed": False,
                "detail": "need >= 2 seeds"}

    spread = float(pd.Series(sharpes).std(ddof=1))
    identical = spread < 1e-9

    return {
        "check": "seed variance",
        "actual": f"Sharpe std = {spread:.4f} across {len(sharpes)} seeds",
        "passed": not identical,
        "note": ("ZERO variance across seeds — the RNG is not actually being "
                 "varied, so any reported ± is meaningless"
                 if identical else
                 "seeds produce distinct runs, so ± std is meaningful"),
    }


def validate_out_of_sample_degradation(in_sample: dict, out_sample: dict) -> dict:
    """
    Out-of-sample performance should be WORSE than in-sample.

    Improvement out-of-sample is the classic leakage signature: it usually
    means the test period informed the model. This complements the
    perturbation test rather than replacing it.
    """
    is_sharpe = in_sample.get("sharpe", float("nan"))
    oos_sharpe = out_sample.get("sharpe", float("nan"))
    improved = oos_sharpe > is_sharpe + 0.25

    return {
        "check": "out-of-sample degradation",
        "actual": f"in-sample Sharpe {is_sharpe:.2f} -> out-of-sample {oos_sharpe:.2f}",
        "passed": not improved,
        "note": ("out-of-sample IMPROVED materially — investigate leakage "
                 "before reporting" if improved else
                 "degrades or holds, as expected"),
    }


def print_validation(checks: list[dict]) -> bool:
    all_passed = True
    print()
    print("=" * 72)
    print("PIPELINE VALIDATION vs EXTERNALLY VERIFIABLE FACTS")
    print("=" * 72)
    for c in checks:
        status = "PASS" if c.get("passed") else "FAIL"
        all_passed &= bool(c.get("passed"))
        print(f"[{status}] {c['check']}")
        if "actual" in c:
            print(f"       actual   : {c['actual']}")
        if "expected" in c:
            print(f"       expected : {c['expected']}")
        if c.get("note"):
            print(f"       {c['note']}")
    print("=" * 72)
    print("RESULT:", "pipeline reproduces known facts" if all_passed
          else "PIPELINE MISMATCH — fix before trusting any result")
    print("=" * 72)
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate_spy_benchmark() else 1)
