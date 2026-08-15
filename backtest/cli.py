# backtest/cli.py
"""
Command-line entry point for the experimental harness.

    python -m backtest.cli leakage     # prove the harness cannot see the future
    python -m backtest.cli ablate      # component-removal ladder + baselines
    python -m backtest.cli protocol    # tune -> freeze -> single holdout
    python -m backtest.cli costs       # transaction-cost sensitivity sweep

Run `leakage` first and on every change to the data or scheduling code. If it
fails, nothing else the harness reports means anything.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from backtest.config import BacktestConfig
from backtest.data import load_point_in_time


def _base_config(args) -> BacktestConfig:
    return BacktestConfig(
        universe=tuple(args.universe.split(",")),
        start=args.start,
        end=args.end,
        initial_capital=args.capital,
        cost_bps=args.cost_bps,
        rebalance_frequency=args.frequency,
        hmm_train_years=args.hmm_train_years,
        hmm_refit_months=args.hmm_refit_months,
        tune_start=args.tune_start,
        tune_end=args.tune_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
    )


def cmd_leakage(args) -> int:
    from backtest.leakage import run_leakage_suite, print_leakage_report

    config = _base_config(args)
    pit = load_point_in_time(config)
    reports = run_leakage_suite(config, pit=pit, cutoff=args.cutoff)
    return 0 if print_leakage_report(reports) else 1


def cmd_ablate(args) -> int:
    from backtest.experiment import run_experiment
    from backtest.statistics import seed_summary_table, paired_bootstrap_test
    from backtest.metrics import metrics_table

    config = _base_config(args)
    seeds = list(range(args.seeds))
    arms = args.arms.split(",") if args.arms else None

    results = run_experiment(config, arms=arms, seeds=seeds, include_baselines=True)

    print("\n" + "=" * 78)
    print(f"ABLATION LADDER  ({len(seeds)} seeds, mean ± std across seeds)")
    print("=" * 78)
    print(seed_summary_table(results.metrics).to_string(index=False))

    print("\nBASELINES (deterministic, single run)")
    print(metrics_table(results.baseline_metrics).to_string(index=False))

    # Is each ablation significantly worse than the full engine?
    if "full" in results.runs:
        print("\n" + "=" * 78)
        print("PAIRED TEST vs FULL ENGINE  (H0: arm Sharpe >= full Sharpe)")
        print("=" * 78)
        full_returns = pd.concat(
            [r.equity for r in results.runs["full"]], axis=1
        ).mean(axis=1).pct_change().dropna()

        rows = []
        for arm, runs in results.runs.items():
            if arm == "full":
                continue
            arm_returns = pd.concat(
                [r.equity for r in runs], axis=1
            ).mean(axis=1).pct_change().dropna()
            test = paired_bootstrap_test(full_returns, arm_returns)
            rows.append({
                "Arm removed": arm,
                "ΔSharpe (full - arm)": f"{test['observed_diff']:+.3f}",
                "95% CI": f"[{test['ci_lower']:+.2f}, {test['ci_upper']:+.2f}]",
                "p": f"{test['p_value']:.3f}",
                "Component helps?": "yes" if test["p_value"] < 0.05 and test["observed_diff"] > 0 else "not shown",
            })
        print(pd.DataFrame(rows).to_string(index=False))

    out = results.save()
    print(f"\nartifacts written to {out}")
    print(f"elapsed {results.elapsed_seconds:.0f}s")
    return 0


def cmd_protocol(args) -> int:
    from backtest.tuning import grid_search, run_holdout, save_protocol_record

    config = _base_config(args)

    tuning = grid_search(config, seeds=list(range(args.tune_seeds)))
    holdout = run_holdout(config, tuning, seeds=list(range(args.seeds)))

    agg = holdout["aggregate"]
    deflated = holdout["deflated_sharpe"]

    print("\n" + "=" * 78)
    print("HOLDOUT RESULT (parameters frozen before this window was touched)")
    print("=" * 78)
    print(f"  window            : {holdout['window'][0]} .. {holdout['window'][1]}")
    print(f"  frozen params     : {holdout['frozen_params']}")
    print(f"  seeds             : {holdout['n_seeds']}")
    print(f"  CAGR              : {agg.get('cagr_mean', float('nan')):.2%} "
          f"± {agg.get('cagr_std', 0):.2%}")
    print(f"  Sharpe            : {agg.get('sharpe_mean', float('nan')):.2f} "
          f"± {agg.get('sharpe_std', 0):.2f}")
    print(f"  Max drawdown      : {agg.get('max_drawdown_mean', float('nan')):.2%}")
    print(f"  configs searched  : {holdout['n_tuning_trials']}")
    print(f"  expected max Sharpe by chance : {deflated.get('expected_max_sharpe', float('nan')):.3f}")
    print(f"  DEFLATED PSR      : {deflated.get('deflated_psr', float('nan')):.3f}")
    print()
    print("  Deflated PSR is the probability the true Sharpe exceeds what the")
    print("  best of that many random configurations would produce by luck.")
    print("  Below ~0.95, the result is not distinguishable from search noise.")

    out = save_protocol_record(tuning, holdout, config.results_dir / "protocol")
    print(f"\nprotocol record written to {out}")
    return 0


def cmd_costs(args) -> int:
    from backtest.experiment import run_experiment
    from backtest.statistics import cost_sensitivity_table

    base = _base_config(args)
    pit = load_point_in_time(base)

    table = {}
    for bps in [0.0, 5.0, 10.0, 25.0, 50.0]:
        config = base.with_(cost_bps=bps)
        results = run_experiment(
            config, arms=["full"], seeds=list(range(args.seeds)),
            include_baselines=False, pit=pit, verbose=False,
        )
        from backtest.statistics import aggregate_seeds
        agg = aggregate_seeds(results.metrics["full"])
        table[bps] = {
            "cagr": agg.get("cagr_mean"),
            "sharpe": agg.get("sharpe_mean"),
            "max_drawdown": agg.get("max_drawdown_mean"),
            "total_costs": sum(m.get("total_costs", 0) for m in results.metrics["full"]) / max(1, args.seeds),
        }
        print(f"  {bps:>5.1f} bps -> Sharpe {table[bps]['sharpe']:.2f}")

    print("\n" + "=" * 60)
    print("TRANSACTION COST SENSITIVITY")
    print("=" * 60)
    print(cost_sensitivity_table(table).to_string(index=False))
    print("\nAn edge that disappears between 5 and 25 bps is a cost assumption,")
    print("not an edge.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Strategy Engine experimental harness")
    parser.add_argument("command", choices=["leakage", "ablate", "protocol", "costs"])
    parser.add_argument("--universe", default="WMT,JNJ,NVDA,JPM,NEE")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--cost-bps", dest="cost_bps", type=float, default=5.0)
    parser.add_argument("--frequency", default="Weekly")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--tune-seeds", dest="tune_seeds", type=int, default=3)
    parser.add_argument("--arms", default=None, help="comma-separated ablation names")
    parser.add_argument("--cutoff", default=None, help="leakage perturbation cutoff date")
    parser.add_argument("--hmm-train-years", dest="hmm_train_years", type=float, default=10.0)
    parser.add_argument("--hmm-refit-months", dest="hmm_refit_months", type=int, default=12)
    parser.add_argument("--tune-start", dest="tune_start", default="2015-01-01")
    parser.add_argument("--tune-end", dest="tune_end", default="2020-12-31")
    parser.add_argument("--holdout-start", dest="holdout_start", default="2021-01-01")
    parser.add_argument("--holdout-end", dest="holdout_end", default="2024-12-31")

    args = parser.parse_args(argv)

    return {
        "leakage": cmd_leakage,
        "ablate": cmd_ablate,
        "protocol": cmd_protocol,
        "costs": cmd_costs,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
