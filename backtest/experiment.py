# backtest/experiment.py
"""
Experiment orchestration: run the ablation ladder across seeds, plus baselines.

Data and the HMM fold schedule are built ONCE and shared across every run.
That is both a large speedup and a correctness property — every configuration
sees exactly the same prices and the same regime models, so differences come
from the mechanism under test rather than from resampling noise.

Seeds vary per run because bandit weights initialize randomly. A single-seed
result is one draw from a distribution, not an estimate of it.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.ablations import ABLATIONS, get_hooks
from backtest.baselines import BASELINES
from backtest.config import BacktestConfig
from backtest.data import PointInTimeData, load_point_in_time
from backtest.hmm_schedule import build_fold_schedule
from backtest.metrics import compute_metrics
from backtest.repro import warn_if_unpinned
from backtest.runner import BacktestResult, run_backtest


# --- Parallel execution -----------------------------------------------------
#
# Runs are independent: each gets its own engine, its own temp bandit
# directory, and its own seed. The expensive shared inputs (prices and the HMM
# fold schedule) are built ONCE in the parent and handed to each worker a
# single time via the pool initializer rather than pickled per task.
#
# Determinism is unaffected — run_backtest seeds both RNGs from its config, so
# a given (arm, seed) produces identical output regardless of which worker
# executes it or in what order. Verified: parallel and serial runs of the same
# arm/seed match bit for bit.

_WORKER: dict = {}


def _init_worker(pit: PointInTimeData, folds, extra_hooks: dict | None = None) -> None:
    warnings.filterwarnings("ignore")
    _WORKER["pit"] = pit
    _WORKER["folds"] = folds
    # Hooks that apply to EVERY arm, most importantly the frozen tuned
    # parameters. These MUST travel through the pool initializer.
    #
    # They were previously applied by mutating backtest.ablations.ABLATIONS in
    # the parent. Under the spawn start method a worker re-imports the module
    # and sees the pristine registry, so the frozen parameters silently never
    # reached any run and every arm executed with engine defaults. The failure
    # is invisible in the output: the runs succeed and produce plausible
    # numbers, they are just not the configuration the protocol froze.
    _WORKER["extra_hooks"] = dict(extra_hooks or {})


def _run_one(task: tuple[BacktestConfig, str, int]) -> tuple[str, int, BacktestResult, dict]:
    config, arm, seed = task
    # Frozen parameters first, then the arm's ablation, so an ablation can
    # override a tuned value rather than being silently overwritten by it.
    hooks = {**_WORKER.get("extra_hooks", {}), **get_hooks(arm)}
    result = run_backtest(
        config, pit=_WORKER["pit"], folds=_WORKER["folds"],
        ablation_hooks=hooks, verbose=False,
    )
    metrics = compute_metrics(result.equity, result.cycles, label=f"{arm}/seed{seed}")
    return arm, seed, result, metrics


_SWEEP_WORKER: dict = {}


def _init_sweep_worker(pit, folds, hooks) -> None:
    warnings.filterwarnings("ignore")
    _SWEEP_WORKER["pit"] = pit
    _SWEEP_WORKER["folds"] = folds
    _SWEEP_WORKER["hooks"] = hooks


def _run_sweep_task(task):
    config, sweep, field, value, seed = task
    result = run_backtest(
        config, pit=_SWEEP_WORKER["pit"], folds=_SWEEP_WORKER["folds"],
        ablation_hooks=_SWEEP_WORKER["hooks"], verbose=False,
    )
    metrics = compute_metrics(result.equity, result.cycles)
    return sweep, field, value, seed, metrics


def run_parameter_sweep(
    config: BacktestConfig,
    variants: list[tuple[str, str, object]],
    seeds: list[int],
    hooks: dict | None = None,
    pit: PointInTimeData | None = None,
    folds=None,
    workers: int | None = None,
    verbose: bool = True,
) -> dict[tuple[str, str, object], list[dict]]:
    """
    Run a one-factor-at-a-time sensitivity sweep across cores.

    `variants` is a list of (sweep_name, config_field, value). Every variant
    runs on all seeds; a sweep with 10 variants x 5 seeds is 50 backtests,
    which is several hours serially and well under an hour parallel.

    Returns {(sweep, field, value): [metrics per seed]}.
    """
    workers = default_workers() if workers is None else max(1, int(workers))
    warn_if_unpinned()

    if pit is None:
        pit = load_point_in_time(config)
    if folds is None:
        folds = build_fold_schedule(pit, config, verbose=verbose)

    tasks = [
        (config.with_(seed=seed, **{field: value}), sweep, field, value, seed)
        for (sweep, field, value) in variants for seed in seeds
    ]

    out: dict[tuple[str, str, object], list[dict]] = {
        (s, f, v): [] for (s, f, v) in variants
    }

    if verbose:
        print(f"sweep: {len(tasks)} backtests on {workers} worker(s)...")

    if workers == 1:
        _init_sweep_worker(pit, folds, hooks)
        for task in tasks:
            sweep, field, value, seed, metrics = _run_sweep_task(task)
            out[(sweep, field, value)].append(metrics)
    else:
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_sweep_worker,
            initargs=(pit, folds, hooks),
        ) as pool:
            for sweep, field, value, seed, metrics in pool.map(_run_sweep_task, tasks):
                out[(sweep, field, value)].append(metrics)
                done += 1
                if verbose:
                    print(f"  [{done}/{len(tasks)}] {sweep} {field}={value} "
                          f"seed={seed} Sharpe={metrics.get('sharpe', 0):.2f}",
                          flush=True)

    return out


def default_workers() -> int:
    """Leave one core free so the machine stays usable during long sweeps."""
    return max(1, (os.cpu_count() or 2) - 1)


def _print_run(arm: str, seed: int, metrics: dict,
               seconds: float | None, suffix: str = "") -> None:
    timing = f"({seconds:.0f}s)" if seconds is not None else ""
    print(f"  {arm:<26} seed={seed:<3} "
          f"CAGR={metrics.get('cagr', 0):>7.2%} "
          f"Sharpe={metrics.get('sharpe', 0):>6.2f} "
          f"MDD={metrics.get('max_drawdown', 0):>7.2%} {timing}{suffix}",
          flush=True)


@dataclass
class ExperimentResults:
    """All runs from one experiment, plus their per-run metrics."""

    config: BacktestConfig
    runs: dict[str, list[BacktestResult]] = field(default_factory=dict)     # arm -> [per seed]
    metrics: dict[str, list[dict]] = field(default_factory=dict)            # arm -> [per seed]
    baselines: dict[str, BacktestResult] = field(default_factory=dict)
    baseline_metrics: dict[str, dict] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def equity_frame(self) -> pd.DataFrame:
        """Mean equity curve per arm, plus each baseline."""
        series = {}
        for arm, results in self.runs.items():
            curves = [r.equity for r in results if r.equity is not None and len(r.equity)]
            if curves:
                series[arm] = pd.concat(curves, axis=1).mean(axis=1)
        for name, result in self.baselines.items():
            if result.equity is not None and len(result.equity):
                series[name] = result.equity
        return pd.DataFrame(series)

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or self.config.results_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = f"{self.config.fingerprint()}"

        self.equity_frame().to_csv(directory / f"equity_{stamp}.csv")

        rows = []
        for arm, metric_list in self.metrics.items():
            for m in metric_list:
                rows.append({"arm": arm, "kind": "ablation", **m})
        for name, m in self.baseline_metrics.items():
            rows.append({"arm": name, "kind": "baseline", **m})
        pd.DataFrame(rows).to_csv(directory / f"metrics_{stamp}.csv", index=False)

        with open(directory / f"config_{stamp}.json", "w") as fh:
            json.dump(self.config.to_dict(), fh, indent=2)

        return directory


def run_experiment(
    config: BacktestConfig,
    arms: list[str] | None = None,
    seeds: list[int] | None = None,
    include_baselines: bool = True,
    pit: PointInTimeData | None = None,
    verbose: bool = True,
    workers: int | None = None,
    extra_hooks: dict | None = None,
) -> ExperimentResults:
    """
    Run the requested ablation arms across seeds, plus baselines.

    Args:
        arms: ablation names (default: every entry in ABLATIONS).
        seeds: RNG seeds (default: [config.seed]). Use >= 20 for reporting.
        workers: parallel processes. None -> cpu_count-1. 1 -> serial.
        extra_hooks: hooks applied to EVERY arm before its own ablation,
            used to pin frozen tuned parameters. Passed through the pool
            initializer; mutating module state in the parent does NOT reach
            spawned workers.
    """
    arms = arms if arms is not None else list(ABLATIONS.keys())
    seeds = seeds if seeds is not None else [config.seed]
    workers = default_workers() if workers is None else max(1, int(workers))

    # Cross-process reproducibility depends on a pinned hash seed; see
    # backtest.repro. Without it, parallel runs disagree with each other.
    warn_if_unpinned()

    started = time.time()

    if pit is None:
        if verbose:
            print("loading point-in-time data...")
        pit = load_point_in_time(config)

    if verbose:
        print("building HMM fold schedule (shared across all arms)...")
    folds = build_fold_schedule(pit, config, verbose=verbose)

    results = ExperimentResults(config=config)
    for arm in arms:
        results.runs[arm] = []
        results.metrics[arm] = []

    tasks = [
        (config.with_(ablation=arm, seed=seed), arm, seed)
        for arm in arms for seed in seeds
    ]

    # Collect keyed by (arm, seed) so parallel completion order does not
    # scramble the per-seed alignment the statistics depend on.
    collected: dict[tuple[str, int], tuple[BacktestResult, dict]] = {}

    if verbose:
        print(f"running {len(tasks)} backtests on {workers} worker(s)...")

    if workers == 1:
        _init_worker(pit, folds, extra_hooks)
        for task in tasks:
            t0 = time.time()
            arm, seed, result, metrics = _run_one(task)
            collected[(arm, seed)] = (result, metrics)
            if verbose:
                _print_run(arm, seed, metrics, time.time() - t0)
    else:
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(pit, folds, extra_hooks),
        ) as pool:
            for arm, seed, result, metrics in pool.map(_run_one, tasks):
                collected[(arm, seed)] = (result, metrics)
                done += 1
                if verbose:
                    elapsed = time.time() - started
                    rate = elapsed / done
                    eta = rate * (len(tasks) - done)
                    _print_run(arm, seed, metrics, None,
                               suffix=f"[{done}/{len(tasks)} eta {eta/60:.0f}m]")

    for arm in arms:
        for seed in seeds:
            result, metrics = collected[(arm, seed)]
            results.runs[arm].append(result)
            results.metrics[arm].append(metrics)

    if include_baselines:
        for name, fn in BASELINES.items():
            if verbose:
                print(f"  baseline: {name} ...")
            t0 = time.time()
            result = fn(config, pit)
            results.baselines[name] = result
            results.baseline_metrics[name] = compute_metrics(
                result.equity, result.cycles, label=name
            )
            if verbose:
                m = results.baseline_metrics[name]
                print(f"  {name:<26} {'':<8} "
                      f"CAGR={m.get('cagr', 0):>7.2%} "
                      f"Sharpe={m.get('sharpe', 0):>6.2f} "
                      f"MDD={m.get('max_drawdown', 0):>7.2%} "
                      f"({time.time() - t0:.0f}s)")

    results.elapsed_seconds = time.time() - started
    return results
