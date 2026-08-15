# backtest/demo_run.py
"""
Demonstration experiment at a feasible scale.

This is NOT the final experiment for the paper — the universe is 5 tickers and
the span is 4 years, chosen so the whole ladder finishes in about an hour. It
exists to show the harness produces interpretable, honest output end to end.

For the paper, scale up via backtest.cli:
    python -m backtest.cli leakage  --universe ... --start ... --end ...
    python -m backtest.cli ablate   --seeds 20
    python -m backtest.cli protocol --seeds 20
    python -m backtest.cli costs    --seeds 10
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from backtest.config import BacktestConfig
from backtest.data import load_point_in_time
from backtest.experiment import run_experiment
from backtest.metrics import metrics_table
from backtest.statistics import (
    seed_summary_table, paired_bootstrap_test, subperiod_breakdown,
    bootstrap_ci, aggregate_seeds,
)
from backtest.tuning import grid_search, run_holdout, save_protocol_record


UNIVERSE = ("WMT", "JNJ", "NVDA", "JPM", "NEE")
SEEDS = 4


def main() -> None:
    config = BacktestConfig(
        universe=UNIVERSE,
        start="2019-01-01", end="2022-12-31",
        hmm_train_years=4, hmm_refit_months=12,
        cost_bps=5.0, initial_capital=100_000.0,
        tune_start="2019-01-01", tune_end="2020-12-31",
        holdout_start="2021-01-01", holdout_end="2022-12-31",
    )

    # ---------------------------------------------------------------- 0
    print("=" * 78)
    print("0. PIPELINE VALIDATION vs PUBLIC FACT (SPY 2015-2024)")
    print("=" * 78)
    from backtest.validate import validate_spy_benchmark, validate_seed_variance
    if not validate_spy_benchmark(cache_dir=config.cache_dir, verbose=True):
        print("ABORT: benchmark leg does not reproduce known SPY behavior.")
        return

    pit = load_point_in_time(config)
    print(f"universe loaded: {sorted(pit.frames)}")

    # ---------------------------------------------------------------- 1
    print("\n" + "=" * 78)
    print("1. LEAKAGE SUITE")
    print("=" * 78)
    from backtest.leakage import run_leakage_suite, print_leakage_report
    reports = run_leakage_suite(config, pit=pit, cutoff="2021-01-01", verbose=True)
    clean = print_leakage_report(reports)
    if not clean:
        print("ABORT: harness is leaking; downstream numbers would be meaningless.")
        return

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 78)
    print(f"2. ABLATION LADDER  ({SEEDS} seeds per arm)")
    print("=" * 78)
    results = run_experiment(
        config, seeds=list(range(SEEDS)), include_baselines=True, pit=pit, verbose=True
    )

    print("\n--- ablation arms (mean ± std across seeds) ---")
    print(seed_summary_table(results.metrics).to_string(index=False))

    # Zero seed variance would mean the RNG is not actually being varied,
    # which would make every reported ± meaningless.
    sv = validate_seed_variance(results.metrics["full"])
    print(f"\n[{'PASS' if sv['passed'] else 'FAIL'}] {sv['check']}: {sv.get('actual')}")
    print(f"       {sv.get('note')}")

    print("\n--- baselines ---")
    print(metrics_table(results.baseline_metrics).to_string(index=False))

    print("\n--- paired bootstrap vs FULL engine ---")
    full_returns = pd.concat(
        [r.equity for r in results.runs["full"]], axis=1
    ).mean(axis=1).pct_change().dropna()

    rows = []
    for arm, runs in results.runs.items():
        if arm == "full":
            continue
        arm_returns = pd.concat([r.equity for r in runs], axis=1).mean(axis=1).pct_change().dropna()
        test = paired_bootstrap_test(full_returns, arm_returns)
        rows.append({
            "Removed": arm,
            "dSharpe": f"{test['observed_diff']:+.3f}",
            "95% CI": f"[{test['ci_lower']:+.2f},{test['ci_upper']:+.2f}]",
            "p": f"{test['p_value']:.3f}",
            "Helps?": "yes" if test["p_value"] < 0.05 and test["observed_diff"] > 0 else "not shown",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- full engine Sharpe, block-bootstrap CI ---")
    ci = bootstrap_ci(full_returns, seed=7)
    print(f"  Sharpe {ci['point']:.3f}  95% CI [{ci['lower']:.3f}, {ci['upper']:.3f}]  (n={ci['n']} days)")

    print("\n--- episode breakdown (full engine, mean equity) ---")
    mean_equity = pd.concat([r.equity for r in results.runs["full"]], axis=1).mean(axis=1)
    breakdown = subperiod_breakdown(mean_equity)
    print(breakdown.to_string(index=False) if not breakdown.empty else "  (no episodes in window)")

    results.save()

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 78)
    print("3. TUNE -> FREEZE -> HOLDOUT")
    print("=" * 78)
    grid = {
        "blend_weights": [(0.6, 0.4), (0.3, 0.7), (0.0, 1.0)],
        "confidence_gate": [0.45, 0.55],
    }
    tuning = grid_search(config, grid=grid, seeds=[0, 1], pit=pit, verbose=True)
    holdout = run_holdout(config, tuning, seeds=list(range(SEEDS)), pit=pit, verbose=True)

    agg = holdout["aggregate"]
    deflated = holdout["deflated_sharpe"]
    print("\n--- HOLDOUT (parameters frozen before this window was evaluated) ---")
    print(f"  window           : {holdout['window'][0]} .. {holdout['window'][1]}")
    print(f"  frozen params    : {holdout['frozen_params']}")
    print(f"  CAGR             : {agg.get('cagr_mean', float('nan')):.2%} ± {agg.get('cagr_std', 0):.2%}")
    print(f"  Sharpe           : {agg.get('sharpe_mean', float('nan')):.2f} ± {agg.get('sharpe_std', 0):.2f}")
    print(f"  MaxDD            : {agg.get('max_drawdown_mean', float('nan')):.2%}")
    print(f"  configs searched : {holdout['n_tuning_trials']}")
    print(f"  E[max Sharpe] by chance : {deflated.get('expected_max_sharpe', float('nan')):.3f}")
    print(f"  DEFLATED PSR     : {deflated.get('deflated_psr', float('nan')):.3f}")

    save_protocol_record(tuning, holdout, config.results_dir / "protocol")
    print("\nDONE")


if __name__ == "__main__":
    main()
