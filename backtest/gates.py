# backtest/gates.py
"""
Plausibility gates from SUCCESS_CRITERIA.md.

These are BUG DETECTORS, not targets. A gate that trips means "stop and
investigate", never "adjust the engine until it passes". The distinction
matters: tuning until a result lands inside an expected band converts a
sanity check into a selection effect, which is precisely the failure the
experimental protocol exists to prevent.

Every gate therefore returns a verdict plus the observed value, and the
reporting layer prints tripped gates prominently WITHOUT changing anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Gate:
    name: str
    passed: bool
    observed: str
    expectation: str
    note: str = ""

    def status(self) -> str:
        return "PASS" if self.passed else "INVESTIGATE"


def gate_benchmark_leg(cagr: float, max_dd: float) -> list[Gate]:
    """Buy-Hold SPX must reproduce public record, or the pipeline is broken."""
    return [
        Gate(
            name="Buy-Hold SPX CAGR 2015-2025",
            passed=0.10 <= cagr <= 0.14,
            observed=f"{cagr:.2%}",
            expectation="~11-13% (accept 10-14%)",
            note="Outside this band means wrong prices, missed dividends, or "
                 "broken execution accounting — not a strategy finding.",
        ),
        Gate(
            name="Buy-Hold SPX MaxDD (COVID)",
            passed=-0.42 <= max_dd <= -0.28,
            observed=f"{max_dd:.1%}",
            expectation="~-34% (accept -42%..-28%)",
            note="Very robust historically; a large miss means the price "
                 "series or equity accounting is wrong.",
        ),
    ]


def gate_engine_out_of_sample(cagr: float, sharpe: float, max_dd: float,
                              win_rate: float | None) -> list[Gate]:
    """
    Look-ahead detectors. Each trips on results that are TOO GOOD.

    The prior full-sample figure was 11.8% CAGR. Out-of-sample performance
    exceeding an in-sample number is the classic leakage signature.
    """
    gates = [
        Gate(
            name="OOS CAGR vs prior full-sample 11.8%",
            passed=cagr <= 0.118,
            observed=f"{cagr:.2%}",
            expectation="<= 11.8%",
            note="Out-of-sample beating in-sample suggests leakage.",
        ),
        Gate(
            name="OOS Sharpe ceiling",
            passed=sharpe <= 1.4,
            observed=f"{sharpe:.2f}",
            expectation="<= 1.40",
            note="Suspect look-ahead above this for a weekly-rebalanced "
                 "long-only book.",
        ),
        Gate(
            name="OOS MaxDD floor",
            passed=max_dd <= -0.10,
            observed=f"{max_dd:.1%}",
            expectation="worse than -10%",
            note="A drawdown shallower than -10% across 2020 and 2022 is "
                 "implausible without foresight.",
        ),
    ]
    if win_rate is not None:
        gates.append(Gate(
            name="OOS win rate ceiling",
            passed=win_rate <= 0.58,
            observed=f"{win_rate:.1%}",
            expectation="<= 58%",
            note="Higher suggests the decision saw the outcome.",
        ))
    return gates


def gate_seed_variance(sharpes: list[float]) -> list[Gate]:
    """Zero variance means the RNG is not varying; huge variance means unstable."""
    if len(sharpes) < 2:
        return [Gate("Seed variance", False, f"n={len(sharpes)}",
                     ">= 2 seeds", "Cannot assess with fewer than 2 seeds.")]

    std = float(np.std(sharpes, ddof=1))
    return [
        Gate(
            name="Seed variance non-zero",
            passed=std > 1e-9,
            observed=f"Sharpe std = {std:.4f}",
            expectation="> 0",
            note="Zero means both RNGs are not actually being varied, so "
                 "every reported +/- is fiction.",
        ),
        Gate(
            name="Seed variance not excessive",
            passed=std <= 0.25,
            observed=f"Sharpe std = {std:.4f}",
            expectation="<= 0.25",
            note="Above this the configuration is unstable and per-seed "
                 "results should not be summarized by a mean alone.",
        ),
    ]


def gate_cost_sensitivity(cagr_by_bps: dict[float, float]) -> list[Gate]:
    """Turnover check: the paper's hysteresis claim fails if costs dominate."""
    if 5.0 not in cagr_by_bps or 20.0 not in cagr_by_bps:
        return [Gate("Cost sensitivity 5->20bps", False, "missing legs",
                     "need 5 and 20 bps", "Sweep did not produce both legs.")]

    drop = cagr_by_bps[5.0] - cagr_by_bps[20.0]
    return [Gate(
        name="Cost sensitivity 5->20bps",
        passed=drop <= 0.05,
        observed=f"{drop*100:.2f} pp drop",
        expectation="<= 5 pp",
        note="A larger drop means turnover is too high and the hysteresis "
             "claim does not hold.",
    )]


def print_gates(gates: list[Gate]) -> bool:
    print()
    print("=" * 78)
    print("PLAUSIBILITY GATES (bug detectors — NOT targets, NOT to be tuned to)")
    print("=" * 78)
    all_ok = True
    for g in gates:
        all_ok &= g.passed
        print(f"[{g.status():>12}] {g.name}")
        print(f"                observed : {g.observed}")
        print(f"                expected : {g.expectation}")
        if not g.passed and g.note:
            print(f"                -> {g.note}")
    print("=" * 78)
    print("RESULT:", "no gate tripped" if all_ok else
          "GATES TRIPPED — investigate before reporting; do NOT tune to fit")
    print("=" * 78)
    return all_ok
