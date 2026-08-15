# backtest/tuning.py
"""
Parameter-selection protocol: tune, freeze, then run the holdout ONCE.

The discipline this enforces
----------------------------
The engine ships with many hand-set constants — the 0.60/0.40 regime blend,
the 0.3/0.4/0.3 scoring weights, the 0.55 confidence gate — with no recorded
justification. Reporting holdout performance for values that were themselves
chosen by looking at holdout performance is circular, and it is the specific
failure that makes most published backtests unreproducible.

The protocol here:

    1. Search the grid ONLY on [tune_start, tune_end].
    2. Select by MEAN Sharpe across seeds, never a single run.
    3. Freeze the winner and record how many configurations were tried.
    4. Run the holdout exactly once.
    5. Deflate the holdout Sharpe by that trial count.

Step 5 is what makes the number honest: searching 48 configurations and
reporting the best inflates Sharpe even with no real edge, and the deflated
statistic prices that in.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig
from backtest.data import PointInTimeData, load_point_in_time
from backtest.hmm_schedule import build_fold_schedule
from backtest.metrics import compute_metrics
from concurrent.futures import ProcessPoolExecutor

from backtest.experiment import default_workers
from backtest.runner import run_backtest
from backtest.statistics import deflated_sharpe_ratio, aggregate_seeds


# Parameters exposed as engine attributes. Keep the grid SMALL — every extra
# combination raises the deflation penalty applied to the final result.
DEFAULT_GRID: dict[str, list] = {
    "blend_weights": [(0.6, 0.4), (0.3, 0.7), (0.0, 1.0)],
    "score_weights": [(0.3, 0.4, 0.3), (0.5, 0.2, 0.3), (0.2, 0.6, 0.2)],
    "confidence_gate": [0.45, 0.55],
    # Full deployment vs. letting conviction and the risk caps reduce
    # exposure. This drives results more than any other single switch, which
    # is exactly why it belongs in the grid rather than being fixed by
    # comparing test-period outcomes.
    "fully_invested": [True, False],
}


@dataclass
class TuningResult:
    best_params: dict
    n_trials: int
    trial_sharpes: list[float]
    trials: pd.DataFrame
    tune_window: tuple[str, str]
    elapsed_seconds: float = 0.0


_TUNE_WORKER: dict = {}


def _init_tuning_worker(pit, folds) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    _TUNE_WORKER["pit"] = pit
    _TUNE_WORKER["folds"] = folds


def _run_tuning_task(task):
    config, params, combo, seed = task
    result = run_backtest(
        config, pit=_TUNE_WORKER["pit"], folds=_TUNE_WORKER["folds"],
        ablation_hooks=_apply_params(params), verbose=False,
    )
    return combo, seed, compute_metrics(result.equity, result.cycles)


class ParamHook:
    """
    Picklable ablation-style hook that stamps tuned attributes on the engine.

    A top-level class rather than a closure on purpose: closures cannot be
    pickled, so a closure-based hook silently breaks the moment grid search
    runs across worker processes — the same failure mode that once made three
    ablation arms produce flat 0.00% equity curves.
    """

    __slots__ = ("params",)

    def __init__(self, params: dict):
        self.params = dict(params)

    def __call__(self, engine, config) -> None:
        for key, value in self.params.items():
            setattr(engine, key, value)


def _apply_params(params: dict):
    """Build the hook mapping for one parameter combination."""
    return {"tuned": ParamHook(params)}


def grid_search(
    config: BacktestConfig,
    grid: dict[str, list] | None = None,
    seeds: list[int] | None = None,
    pit: PointInTimeData | None = None,
    verbose: bool = True,
    workers: int | None = None,
) -> TuningResult:
    """
    Evaluate every grid combination on the TUNING window only.

    The returned config window is forced to [tune_start, tune_end] regardless
    of what `config` says, so this function cannot accidentally read holdout
    data.
    """
    grid = grid or DEFAULT_GRID
    seeds = seeds or [0, 1, 2]
    started = time.time()

    tune_config = config.with_(start=config.tune_start, end=config.tune_end)

    if pit is None:
        pit = load_point_in_time(tune_config)
    folds = build_fold_schedule(pit, tune_config, verbose=False)

    keys = list(grid.keys())
    combinations = list(itertools.product(*(grid[k] for k in keys)))

    if verbose:
        print(f"grid search: {len(combinations)} configurations x {len(seeds)} seeds "
              f"on {config.tune_start}..{config.tune_end}")

    # Every (combination, seed) pair is independent; run them across cores.
    tasks = [
        (tune_config.with_(seed=seed), dict(zip(keys, combo)), combo, seed)
        for combo in combinations for seed in seeds
    ]
    workers = default_workers() if workers is None else max(1, int(workers))

    metrics_by_combo: dict[tuple, list[dict]] = {combo: [] for combo in combinations}

    if workers == 1:
        _init_tuning_worker(pit, folds)
        for task in tasks:
            combo, seed, metrics = _run_tuning_task(task)
            metrics_by_combo[combo].append(metrics)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_tuning_worker,
            initargs=(pit, folds),
        ) as pool:
            for combo, seed, metrics in pool.map(_run_tuning_task, tasks):
                metrics_by_combo[combo].append(metrics)

    rows = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        seed_metrics = metrics_by_combo[combo]

        agg = aggregate_seeds(seed_metrics)
        row = {
            **{k: str(v) for k, v in params.items()},
            "sharpe_mean": agg.get("sharpe_mean", np.nan),
            "sharpe_std": agg.get("sharpe_std", np.nan),
            "cagr_mean": agg.get("cagr_mean", np.nan),
            "maxdd_mean": agg.get("max_drawdown_mean", np.nan),
            "_params": params,
        }
        rows.append(row)

        if verbose:
            print(f"  {params} -> Sharpe {row['sharpe_mean']:.3f} "
                  f"± {row['sharpe_std']:.3f}")

    trials = pd.DataFrame(rows).sort_values("sharpe_mean", ascending=False)
    best = trials.iloc[0]["_params"]
    trial_sharpes = [float(s) for s in trials["sharpe_mean"] if np.isfinite(s)]

    if verbose:
        print(f"\nselected on tuning window: {best}")

    return TuningResult(
        best_params=best,
        n_trials=len(combinations),
        trial_sharpes=trial_sharpes,
        trials=trials.drop(columns=["_params"]),
        tune_window=(config.tune_start, config.tune_end),
        elapsed_seconds=time.time() - started,
    )


def run_holdout(
    config: BacktestConfig,
    tuning: TuningResult,
    seeds: list[int] | None = None,
    pit: PointInTimeData | None = None,
    verbose: bool = True,
) -> dict:
    """
    Single evaluation on the holdout window with FROZEN parameters.

    Call this once. Re-running it after seeing the result and adjusting
    anything converts the holdout into a second tuning set and invalidates
    every number it produced.
    """
    seeds = seeds or list(range(20))
    holdout_config = config.with_(start=config.holdout_start, end=config.holdout_end)

    if pit is None:
        pit = load_point_in_time(holdout_config)
    folds = build_fold_schedule(pit, holdout_config, verbose=False)

    if verbose:
        print(f"\nHOLDOUT {config.holdout_start}..{config.holdout_end} "
              f"| frozen params: {tuning.best_params} | {len(seeds)} seeds")

    seed_metrics, equities = [], []
    for seed in seeds:
        run_config = holdout_config.with_(seed=seed)
        result = run_backtest(
            run_config, pit=pit, folds=folds,
            ablation_hooks=_apply_params(tuning.best_params), verbose=False,
        )
        seed_metrics.append(compute_metrics(result.equity, result.cycles))
        equities.append(result.equity)
        if verbose:
            print(f"  seed {seed:>2}: Sharpe {seed_metrics[-1].get('sharpe', 0):>6.2f} "
                  f"CAGR {seed_metrics[-1].get('cagr', 0):>7.2%}")

    agg = aggregate_seeds(seed_metrics)

    mean_equity = pd.concat(equities, axis=1).mean(axis=1) if equities else pd.Series(dtype=float)
    mean_returns = mean_equity.pct_change().dropna()

    deflated = deflated_sharpe_ratio(
        mean_returns,
        n_trials=tuning.n_trials,
        trial_sharpes=tuning.trial_sharpes,
    )

    return {
        "window": (config.holdout_start, config.holdout_end),
        "frozen_params": {k: str(v) for k, v in tuning.best_params.items()},
        "n_seeds": len(seeds),
        "n_tuning_trials": tuning.n_trials,
        "aggregate": agg,
        "deflated_sharpe": deflated,
        "mean_equity": mean_equity,
        "per_seed_metrics": seed_metrics,
    }


def save_protocol_record(tuning: TuningResult, holdout: dict, directory: Path) -> Path:
    """
    Persist the full protocol trail.

    Writing down what was searched, what was frozen, and when the holdout was
    run is what lets a reader verify the protocol was followed rather than
    take it on trust.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    tuning.trials.to_csv(directory / "tuning_trials.csv", index=False)

    record = {
        "protocol": "tune -> freeze -> single holdout evaluation",
        "tune_window": tuning.tune_window,
        "n_tuning_trials": tuning.n_trials,
        "selected_params": {k: str(v) for k, v in tuning.best_params.items()},
        "holdout_window": holdout["window"],
        "holdout_n_seeds": holdout["n_seeds"],
        "holdout_sharpe_mean": holdout["aggregate"].get("sharpe_mean"),
        "holdout_sharpe_std": holdout["aggregate"].get("sharpe_std"),
        "holdout_cagr_mean": holdout["aggregate"].get("cagr_mean"),
        "holdout_maxdd_mean": holdout["aggregate"].get("max_drawdown_mean"),
        "deflated_sharpe": holdout["deflated_sharpe"],
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(directory / "protocol_record.json", "w") as fh:
        json.dump(record, fh, indent=2, default=str)

    return directory
