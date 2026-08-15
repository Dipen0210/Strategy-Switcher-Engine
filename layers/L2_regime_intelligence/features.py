# layers/L2_regime_intelligence/features.py
"""
Canonical HMM feature construction — SHARED by training and inference.

Why this module exists
----------------------
Training (train_hmm.py) and inference (RegimeDetector) previously built their
own feature matrices independently, and they disagreed:

    training  : [Return_1D, Realized_Vol, Momentum]     (20d pct change, ~±0.10)
    inference : [Return_1D, GARCH_Vol,    MA_Slope]     (5d MA change,   ~±0.02)

Two of three columns differed in meaning and scale, so the fitted Gaussian
emissions were being evaluated well off their training support and the
posteriors were effectively arbitrary. Any model trained before this module
existed must be retrained.

Every feature here is CAUSAL — computable at time t using only data up to t:

    Return_1D    log(P_t / P_{t-1})
    Realized_Vol 20-day rolling std of log returns, annualized
    MA_Slope     5-day pct change of the 20-day moving average

GARCH_Vol is deliberately excluded. `arch_model.fit()` estimates parameters
over the whole sample, so the conditional volatility it reports at time t
embeds information from t+1..T. That is acceptable for a display metric but
not for a feature the regime model is trained and evaluated on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Canonical feature set. ORDER IS PART OF THE CONTRACT — the fitted model's
# means_/covars_ are indexed by this order, so changing it invalidates every
# previously trained model. Bump FEATURE_SCHEMA_VERSION if you ever do.
HMM_FEATURE_COLUMNS: tuple[str, ...] = ("Return_1D", "Realized_Vol", "MA_Slope")
FEATURE_SCHEMA_VERSION = 2

# Index of the return column within the feature matrix. Used by the regime
# label sorter to read each state's expected return and variance.
RETURN_FEATURE_IDX = 0
VOL_FEATURE_IDX = 1

# Rolling windows (kept here so training and inference cannot drift apart).
VOL_WINDOW = 20
MA_WINDOW = 20
MA_SLOPE_LAG = 5


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute any missing canonical feature columns from Close.

    Safe to call on a DataFrame that already went through
    L1's compute_all_features() — existing columns are left untouched.
    """
    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    if "Close" not in result.columns:
        raise ValueError("HMM features require a 'Close' column")

    close = result["Close"]

    if "Return_1D" not in result.columns:
        result["Return_1D"] = np.log(close / close.shift(1))

    if "Realized_Vol" not in result.columns:
        log_returns = np.log(close / close.shift(1))
        result["Realized_Vol"] = (
            log_returns.rolling(window=VOL_WINDOW).std() * np.sqrt(252)
        )

    if "MA_Slope" not in result.columns:
        ma = close.rolling(window=MA_WINDOW).mean()
        result["MA_Slope"] = (ma - ma.shift(MA_SLOPE_LAG)) / ma.shift(MA_SLOPE_LAG)

    return result


def build_hmm_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build the (n_samples, 3) feature matrix for the regime HMM.

    Rows containing NaN or inf in any feature are dropped, which trims the
    warm-up period the rolling windows need. Row order is preserved, so the
    LAST row always corresponds to the most recent observation in `df` that
    has a complete feature vector.

    Args:
        df: OHLCV DataFrame, optionally already enriched by L1.

    Returns:
        float64 array of shape (n_valid, len(HMM_FEATURE_COLUMNS)).
        May be empty if `df` is shorter than the rolling warm-up.
    """
    if df is None or df.empty:
        return np.empty((0, len(HMM_FEATURE_COLUMNS)), dtype=float)

    enriched = _ensure_feature_columns(df)

    X = enriched[list(HMM_FEATURE_COLUMNS)].to_numpy(dtype=float)

    valid = np.isfinite(X).all(axis=1)
    return X[valid]


def fit_normalization(X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute z-score statistics from a TRAINING matrix only.

    These get serialized with the model so inference standardizes with the
    training window's moments rather than recomputing them from whatever
    slice it happens to be looking at (which would leak test-set moments).
    """
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds = np.where(stds == 0, 1.0, stds)  # guard constant columns
    return {"means": means, "stds": stds}


def apply_normalization(
    X: np.ndarray,
    means: np.ndarray | None,
    stds: np.ndarray | None,
) -> np.ndarray:
    """Standardize with previously fitted statistics. No-op if unavailable."""
    if means is None or stds is None or len(X) == 0:
        return X
    return (X - means) / stds
