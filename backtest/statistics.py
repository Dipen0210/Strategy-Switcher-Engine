# backtest/statistics.py
"""
Statistical inference for backtest results.

Why this module is not optional
-------------------------------
Bandit weights initialize from np.random.rand, so every run is a draw from a
distribution over outcomes. Reporting one seed's Sharpe as "the" Sharpe is a
category error, and it is the single easiest thing for a reviewer to attack.
Everything here exists to answer: is the observed difference larger than the
noise, given how many configurations were tried?

Provided
--------
aggregate_seeds        mean / std / quantiles across seeds
bootstrap_ci           CI for a metric by resampling the daily return series
paired_bootstrap_test  is arm A better than arm B, on the same days?
deflated_sharpe_ratio  Sharpe corrected for the number of configurations tried
probabilistic_sharpe   probability the true Sharpe exceeds a threshold
subperiod_breakdown    performance within named market episodes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from backtest.metrics import TRADING_DAYS, sharpe_ratio, cagr, max_drawdown


# --------------------------------------------------------------------------
# Across seeds
# --------------------------------------------------------------------------

def aggregate_seeds(metric_list: list[dict], keys: tuple[str, ...] = (
    "cagr", "sharpe", "max_drawdown", "volatility", "win_rate",
    "turnover_per_cycle", "time_in_market",
)) -> dict:
    """Summarize one arm's metrics across seeds."""
    out: dict = {"n_seeds": len(metric_list)}
    for key in keys:
        values = np.array(
            [m.get(key, np.nan) for m in metric_list if m.get(key) is not None],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if len(values) == 0:
            out[f"{key}_mean"] = np.nan
            continue
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        out[f"{key}_min"] = float(values.min())
        out[f"{key}_max"] = float(values.max())
        out[f"{key}_p05"] = float(np.percentile(values, 5))
        out[f"{key}_p95"] = float(np.percentile(values, 95))
    return out


def seed_summary_table(metrics: dict[str, list[dict]]) -> pd.DataFrame:
    """Comparison table reporting mean +/- std across seeds, never a single run."""
    rows = []
    for arm, metric_list in metrics.items():
        agg = aggregate_seeds(metric_list)
        rows.append({
            "Arm": arm,
            "Seeds": agg["n_seeds"],
            "CAGR": f"{agg.get('cagr_mean', float('nan')):.2%} ± {agg.get('cagr_std', 0):.2%}",
            "Sharpe": f"{agg.get('sharpe_mean', float('nan')):.2f} ± {agg.get('sharpe_std', 0):.2f}",
            "MaxDD": f"{agg.get('max_drawdown_mean', float('nan')):.2%} ± {agg.get('max_drawdown_std', 0):.2%}",
            "Vol": f"{agg.get('volatility_mean', float('nan')):.2%}",
            "InMkt": f"{agg.get('time_in_market_mean', float('nan')):.1%}",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def stationary_bootstrap_indices(
    n: int,
    n_boot: int,
    mean_block: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resample indices via the Politis-Romano STATIONARY bootstrap.

    Block lengths are geometric with mean `mean_block` (restart probability
    p = 1/mean_block) and wrap circularly. Unlike a FIXED block length, the
    resampled series is stationary, which is the property the Sharpe-ratio
    inference below relies on.

    Returns an (n_boot, n) index matrix.
    """
    p = 1.0 / max(float(mean_block), 1.0)
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p

    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for j in range(1, n):
        cont = (idx[:, j - 1] + 1) % n
        idx[:, j] = np.where(restart[:, j], starts[:, j], cont)
    return idx


def _annualized_sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def stationary_bootstrap_ci(
    returns: pd.Series,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    mean_block: float = 21.0,
    batch: int = 500,
) -> dict:
    """
    Sharpe CI via stationary bootstrap, batched to bound peak memory.

    A 10,000 x 2,700 index matrix is ~216 MB in one allocation; batching keeps
    it near 11 MB without changing the draws, since the generator advances
    identically either way for a fixed batch size.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 30:
        return {"point": np.nan, "lower": np.nan, "upper": np.nan, "n": n}

    values = r.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    samples = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        size = min(batch, n_boot - done)
        idx = stationary_bootstrap_indices(n, size, mean_block, rng)
        drawn = values[idx]
        sd = drawn.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            samples[done:done + size] = np.where(
                sd > 0, drawn.mean(axis=1) / sd * np.sqrt(TRADING_DAYS), 0.0
            )
        done += size

    return {
        "point": _annualized_sharpe(values),
        "lower": float(np.percentile(samples, 100 * alpha / 2)),
        "upper": float(np.percentile(samples, 100 * (1 - alpha / 2))),
        "n": n,
        "n_boot": n_boot,
        "mean_block": mean_block,
        "method": "stationary bootstrap (Politis-Romano)",
    }


def bootstrap_ci(
    returns: pd.Series,
    statistic=None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    block_size: int = 5,
) -> dict:
    """
    Confidence interval via CIRCULAR FIXED-BLOCK bootstrap.

    NOTE: this resamples fixed-length contiguous blocks. That is the circular
    block bootstrap, NOT the stationary bootstrap — the docstring used to
    claim the latter. For reported results use stationary_bootstrap_ci, which
    implements Politis-Romano with geometric block lengths as required.

    Retained because paired_bootstrap_test and the earlier exploratory runs
    were computed with it.
    """
    if statistic is None:
        def statistic(r):
            sd = r.std(ddof=1)
            return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 20:
        return {"point": np.nan, "lower": np.nan, "upper": np.nan, "n": n}

    rng = np.random.default_rng(seed)
    values = r.to_numpy()
    n_blocks = int(np.ceil(n / block_size))

    samples = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([
            np.arange(s, s + block_size) % n for s in starts
        ])[:n]
        samples[i] = statistic(pd.Series(values[idx]))

    return {
        "point": float(statistic(r)),
        "lower": float(np.percentile(samples, 100 * alpha / 2)),
        "upper": float(np.percentile(samples, 100 * (1 - alpha / 2))),
        "n": n,
        "n_boot": n_boot,
    }


def paired_bootstrap_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    n_boot: int = 2000,
    seed: int = 0,
    block_size: int = 5,
) -> dict:
    """
    Test whether arm A's Sharpe exceeds arm B's, pairing on DATE.

    Pairing matters: both arms trade the same market, so the common market
    component should be differenced out rather than treated as independent
    noise. Returns a one-sided p-value for H0: Sharpe(A) <= Sharpe(B).
    """
    joined = pd.concat([pd.Series(returns_a), pd.Series(returns_b)], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return {"observed_diff": np.nan, "p_value": np.nan, "n": len(joined)}

    a = joined.iloc[:, 0].to_numpy()
    b = joined.iloc[:, 1].to_numpy()
    n = len(a)

    def _sharpe(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    observed = _sharpe(a) - _sharpe(b)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        diffs[i] = _sharpe(a[idx]) - _sharpe(b[idx])

    # p-value under H0 via the centered bootstrap distribution
    centered = diffs - diffs.mean()
    p_value = float((centered >= abs(observed)).mean())

    return {
        "observed_diff": observed,
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_value": p_value,
        "n": n,
    }


# --------------------------------------------------------------------------
# Multiple-testing corrections
# --------------------------------------------------------------------------

def _sharpe_diff_and_gradient(r1: np.ndarray, r2: np.ndarray):
    """Sharpe difference and its gradient w.r.t. the moment vector."""
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()

    v1, v2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    if v1 <= 0 or v2 <= 0:
        return np.nan, None

    diff = mu1 / np.sqrt(v1) - mu2 / np.sqrt(v2)
    grad = np.array([
        g1 / v1 ** 1.5,
        -g2 / v2 ** 1.5,
        -mu1 / (2 * v1 ** 1.5),
        mu2 / (2 * v2 ** 1.5),
    ])
    return float(diff), grad


def _hac_psi(moments: np.ndarray, bandwidth: int) -> np.ndarray:
    """Newey-West HAC covariance of the (centered) moment vector."""
    T = moments.shape[0]
    centered = moments - moments.mean(axis=0)

    psi = centered.T @ centered / T
    for lag in range(1, bandwidth + 1):
        gamma = centered[lag:].T @ centered[:-lag] / T
        weight = 1.0 - lag / (bandwidth + 1.0)
        psi += weight * (gamma + gamma.T)
    return psi


def _studentized_stat(r1: np.ndarray, r2: np.ndarray, bandwidth: int):
    diff, grad = _sharpe_diff_and_gradient(r1, r2)
    if grad is None:
        return np.nan, np.nan

    moments = np.column_stack([r1, r2, r1 ** 2, r2 ** 2])
    psi = _hac_psi(moments, bandwidth)
    var = float(grad @ psi @ grad) / len(r1)
    se = np.sqrt(var) if var > 0 else np.nan
    return diff, se


def ledoit_wolf_sharpe_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    n_boot: int = 10_000,
    mean_block: float = 21.0,
    bandwidth: int | None = None,
    seed: int = 0,
) -> dict:
    """
    Ledoit-Wolf (2008) test for a difference in Sharpe ratios.

    Why not a plain bootstrap
    -------------------------
    Sharpe ratios are ratios of dependent, heteroskedastic, autocorrelated
    moments. A naive test treats them as if they were independent means and
    badly understates the standard error. Ledoit-Wolf studentize the
    difference with a HAC (Newey-West) estimate of the moment covariance and
    obtain the null distribution from a studentized time-series bootstrap,
    which is what makes the p-value trustworthy at daily frequency.

    Returns the ANNUALIZED difference for readability; the test statistic and
    p-value are computed on the daily scale, where the theory applies.

    Two-sided p-value for H0: Sharpe(A) == Sharpe(B).
    """
    joined = pd.concat(
        [pd.Series(returns_a), pd.Series(returns_b)], axis=1, join="inner"
    ).dropna()
    if len(joined) < 60:
        return {"observed_diff": np.nan, "p_value": np.nan, "n": len(joined)}

    r1 = joined.iloc[:, 0].to_numpy(dtype=float)
    r2 = joined.iloc[:, 1].to_numpy(dtype=float)
    n = len(r1)

    if bandwidth is None:
        # Andrews-style rule of thumb; ample for daily data.
        bandwidth = int(np.ceil(4 * (n / 100.0) ** (2.0 / 9.0)))

    diff, se = _studentized_stat(r1, r2, bandwidth)

    if np.isfinite(diff) and abs(diff) < 1e-15:
        # Identical (or numerically identical) series: the studentized
        # statistic is 0/0. The difference is exactly zero, so H0 holds by
        # construction; report that rather than a nan a caller might misread.
        return {
            "sharpe_a": _annualized_sharpe(r1), "sharpe_b": _annualized_sharpe(r2),
            "observed_diff": 0.0, "std_error": 0.0, "statistic": 0.0,
            "p_value": 1.0, "n": n, "n_boot": 0, "bandwidth": bandwidth,
            "method": "degenerate (identical series)",
        }

    if not np.isfinite(diff) or not np.isfinite(se) or se == 0:
        return {"observed_diff": diff, "p_value": np.nan, "n": n}

    stat = diff / se

    rng = np.random.default_rng(seed)
    exceed = 0
    valid = 0
    batch = 500
    done = 0
    while done < n_boot:
        size = min(batch, n_boot - done)
        idx = stationary_bootstrap_indices(n, size, mean_block, rng)
        for row in idx:
            b1, b2 = r1[row], r2[row]
            d_b, se_b = _studentized_stat(b1, b2, bandwidth)
            if not np.isfinite(d_b) or not np.isfinite(se_b) or se_b == 0:
                continue
            valid += 1
            # Center on the observed difference: under H0 the bootstrap world
            # has true difference `diff`, so this is the null distribution.
            if abs((d_b - diff) / se_b) >= abs(stat):
                exceed += 1
        done += size

    p_value = (exceed + 1) / (valid + 1) if valid else np.nan
    ann = np.sqrt(TRADING_DAYS)

    return {
        "sharpe_a": _annualized_sharpe(r1),
        "sharpe_b": _annualized_sharpe(r2),
        "observed_diff": float(diff * ann),
        "std_error": float(se * ann),
        "statistic": float(stat),
        "p_value": float(p_value),
        "n": n,
        "n_boot": valid,
        "bandwidth": bandwidth,
        "method": "Ledoit-Wolf (2008) studentized stationary bootstrap",
    }


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    P(true Sharpe > benchmark), accounting for skew and kurtosis.

    Bailey & Lopez de Prado (2012). Daily returns are neither normal nor
    i.i.d., and non-normality materially changes the standard error of an
    estimated Sharpe.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 20:
        return np.nan

    sr = sharpe_ratio_from_returns(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    sr_daily = sr / np.sqrt(TRADING_DAYS)
    bench_daily = benchmark_sharpe / np.sqrt(TRADING_DAYS)

    denom = np.sqrt(
        max(1e-12, 1 - skew * sr_daily + ((kurt - 1) / 4.0) * sr_daily ** 2)
    )
    z = (sr_daily - bench_daily) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def sharpe_ratio_from_returns(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes: list[float] | None = None,
) -> dict:
    """
    Sharpe deflated for selection bias across `n_trials` configurations.

    Searching many parameter combinations and reporting the best inflates the
    apparent Sharpe even when no true edge exists. The deflated Sharpe asks
    whether the winner beats what the BEST OF n_trials random configurations
    would have produced by luck.

    Report this whenever the configuration was chosen by comparing runs.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 20 or n_trials < 1:
        return {"sharpe": np.nan, "deflated_psr": np.nan, "expected_max_sharpe": np.nan}

    observed = sharpe_ratio_from_returns(r)

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        variance = float(np.var(trial_sharpes, ddof=1))
    else:
        variance = 1.0 / len(r) * TRADING_DAYS  # crude fallback

    euler = 0.5772156649
    if n_trials > 1:
        expected_max = np.sqrt(variance) * (
            (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials)
            + euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    else:
        expected_max = 0.0

    return {
        "sharpe": observed,
        "n_trials": n_trials,
        "expected_max_sharpe": float(expected_max),
        "deflated_psr": probabilistic_sharpe_ratio(r, benchmark_sharpe=float(expected_max)),
    }


# --------------------------------------------------------------------------
# Regime / subperiod analysis
# --------------------------------------------------------------------------

DEFAULT_EPISODES = {
    "GFC 2008":        ("2007-10-01", "2009-03-31"),
    "Euro crisis 2011":("2011-05-01", "2011-11-30"),
    "Volmageddon 2018":("2018-01-15", "2018-04-15"),
    "COVID 2020":      ("2020-02-15", "2020-04-30"),
    "2022 bear":       ("2022-01-01", "2022-10-31"),
}

# Contiguous partition of the 2015-2025 evaluation window, as required by the
# reporting protocol. Unlike DEFAULT_EPISODES (isolated stress windows that
# skip the calm stretches between them), these tile the whole span, so the
# sub-period returns compose back to the headline figure.
EVAL_SUBPERIODS = {
    "2015-2019 (pre-COVID)": ("2015-01-01", "2019-12-31"),
    "2020 crash":            ("2020-01-01", "2020-03-31"),
    "2020-21 recovery":      ("2020-04-01", "2021-12-31"),
    "2022 bear":             ("2022-01-01", "2022-12-31"),
    "2023-2025":             ("2023-01-01", "2025-12-31"),
}


def subperiod_breakdown(
    equity: pd.Series,
    episodes: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Performance inside named market episodes.

    An aggregate CAGR can hide the thing the architecture claims to do —
    protect capital during crises — so report the episodes explicitly.
    """
    episodes = episodes or DEFAULT_EPISODES
    rows = []
    for name, (start, end) in episodes.items():
        segment = equity.loc[
            (equity.index >= pd.Timestamp(start)) & (equity.index <= pd.Timestamp(end))
        ]
        if len(segment) < 5:
            continue
        rows.append({
            "Episode": name,
            "Days": len(segment),
            "Return": segment.iloc[-1] / segment.iloc[0] - 1.0,
            "MaxDD": max_drawdown(segment),
            "Sharpe": sharpe_ratio(segment),
        })
    return pd.DataFrame(rows)


def cost_sensitivity_table(results_by_bps: dict[float, dict]) -> pd.DataFrame:
    """
    Show how conclusions move with transaction costs.

    A strategy whose edge vanishes between 5 and 10 bps does not have an edge;
    it has a cost assumption.
    """
    rows = []
    for bps, m in sorted(results_by_bps.items()):
        rows.append({
            "cost_bps": bps,
            "CAGR": f"{m.get('cagr', 0):.2%}",
            "Sharpe": f"{m.get('sharpe', 0):.2f}",
            "MaxDD": f"{m.get('max_drawdown', 0):.2%}",
            "TotalCosts": f"{m.get('total_costs', 0):,.0f}",
        })
    return pd.DataFrame(rows)
