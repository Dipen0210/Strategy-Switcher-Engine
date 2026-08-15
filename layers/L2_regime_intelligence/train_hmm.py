# layers/L2_regime_intelligence/train_hmm.py
"""
Macro HMM Training Script for Strategy Engine

Trains a single Gaussian HMM on S&P 500 (SPY) for market regime detection.
This Macro HMM is shared across ALL stocks — individual stock HMMs are NOT used.

Usage:
    python layers/L2_regime_intelligence/train_hmm.py              # Train macro model on SPY
    python layers/L2_regime_intelligence/train_hmm.py --ticker QQQ  # Train on different index

Training Policy:
- 10-year rolling window on SPY
- Monthly retraining (external scheduler)
- 4 hidden states: Bull-Quiet, Bull-Volatile, Sideways, Crisis
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from layers.L1_data_features import compute_all_features
from layers.L2_regime_intelligence.features import (
    HMM_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    build_hmm_features,
    fit_normalization,
    apply_normalization,
)
from layers.L2_regime_intelligence.regime_labels import (
    derive_state_labels,
    describe_state_labels,
)


# === CONFIGURATION ===
DEFAULT_TICKER = "SPY"  # S&P 500 ETF — single Macro HMM for all stocks

# Default training period: 10 years. Overridable via --start/--end so a
# walk-forward harness can refit per fold without editing this file.
TRAINING_START_DATE = "2014-01-01"
TRAINING_END_DATE = "2024-01-01"

N_STATES = 4
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

REGIME_NAMES = {
    0: "Bull-Quiet",
    1: "Bull-Volatile",
    2: "Sideways",
    3: "Crisis",
}


def load_index_data(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    """Load historical OHLCV data for the macro index (SPY by default)."""
    print(f"  Downloading {ticker} data ({TRAINING_START_DATE} to {TRAINING_END_DATE})...")
    
    df = yf.download(
        ticker,
        start=TRAINING_START_DATE,
        end=TRAINING_END_DATE,
        progress=False,
        auto_adjust=False,
    )
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    df = df.reset_index()
    
    # Handle both old and new yfinance column formats
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Ensure we have required columns
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        expected_cols.append("Adj Close")
    
    df = df[[c for c in df.columns if c in expected_cols or c == "Date"]]
    
    print(f"  Downloaded {len(df)} days of data (10 years)")
    return df


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build the training feature matrix.

    Uses the SAME canonical builder as inference (features.py) — these two
    paths previously disagreed on two of three columns, which invalidated
    every posterior the served model produced.
    """
    features_df = compute_all_features(df)
    X = build_hmm_features(features_df)

    print(f"  Feature matrix shape: {X.shape}  columns={list(HMM_FEATURE_COLUMNS)}")
    return X


def train_hmm(X: np.ndarray, n_states: int = 4, n_iter: int = 100):
    """Train Gaussian HMM on feature matrix."""
    from hmmlearn.hmm import GaussianHMM
    
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
        verbose=False,
    )
    
    model.fit(X)
    
    # Get convergence info
    score = model.score(X)
    print(f"  Model log-likelihood: {score:.2f}")
    
    return model


def save_macro_model(
    model,
    ticker: str,
    metadata: dict,
    feature_stats: dict = None,
    state_labels: dict = None,
):
    """Save Macro HMM to disk as macro_hmm.pkl."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / "macro_hmm.pkl"
    
    # Save in format compatible with RegimeDetector.load_model()
    if feature_stats is None:
        raise ValueError(
            "feature_stats is required — without it the served model skips "
            "normalization entirely and scores raw features against "
            "standardized emissions."
        )

    model_data = {
        "n_states": model.n_components,
        "random_state": 42,
        "is_fitted": True,
        "feature_means": feature_stats["means"],
        "feature_stds": feature_stats["stds"],
        "model": model,
        "use_fallback": False,
        # Feature contract — RegimeDetector.load_model() validates these and
        # refuses models built against a different feature set.
        "feature_columns": list(HMM_FEATURE_COLUMNS),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        # Deterministic state -> regime mapping (label-switching fix).
        # RegimeDetector uses this instead of assuming EM's state ordering.
        "state_labels": state_labels,
        # Metadata
        "ticker": ticker,
        "type": "macro",
        "trained_at": datetime.now().isoformat(),
        "regime_names": REGIME_NAMES,
        **metadata,
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"  ✓ Saved Macro HMM to {model_path}")
    return model_path


def train_macro_hmm(ticker: str = DEFAULT_TICKER) -> bool:
    """Train single Macro HMM on S&P 500 (SPY) data."""
    print(f"\n{'='*50}")
    print(f"Training MACRO HMM on: {ticker}")
    print(f"Training period: {TRAINING_START_DATE} to {TRAINING_END_DATE}")
    print(f"This model is shared across ALL stocks.")
    print(f"{'='*50}")
    
    try:
        # 1. Load 10-year index data
        df = load_index_data(ticker)
        
        # 2. Prepare features (canonical, causal feature set)
        X = prepare_features(df)

        if len(X) < 100:
            print(f"  ⚠ Insufficient data ({len(X)} samples), cannot train")
            return False

        # 3. Fit normalization on the TRAINING window only, then standardize.
        #    These statistics are serialized with the model so inference
        #    reuses them instead of recomputing from the data being scored.
        feature_stats = fit_normalization(X)
        X_norm = apply_normalization(X, feature_stats["means"], feature_stats["stds"])
        print(f"  Normalization means={np.round(feature_stats['means'], 6)}")
        print(f"  Normalization stds ={np.round(feature_stats['stds'], 6)}")

        # 4. Train HMM on standardized features
        print(f"  Training {N_STATES}-state Macro HMM...")
        model = train_hmm(X_norm, n_states=N_STATES)

        # 4b. Resolve label switching. EM assigns state indices arbitrarily,
        #     so map them to canonical regime names by their own return
        #     moments. Bandit weights are keyed by these names, which is why
        #     the mapping must be stable across retrains.
        state_labels = derive_state_labels(model)
        print("  Deterministic state -> regime mapping:")
        print(describe_state_labels(model, state_labels))

        # 5. Save model
        metadata = {
            "data_start": TRAINING_START_DATE,
            "data_end": TRAINING_END_DATE,
            "n_samples": len(X),
            "training_period": "10 years",
            "index_ticker": ticker,
        }
        save_macro_model(
            model, ticker, metadata,
            feature_stats=feature_stats,
            state_labels=state_labels,
        )

        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train Macro HMM for market regime detection"
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"Index ticker to train on (default: {DEFAULT_TICKER})"
    )
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    print(f"\n🎯 Training Macro HMM on: {ticker}")
    print(f"📁 Model will be saved to: {MODELS_DIR / 'macro_hmm.pkl'}")
    
    success = train_macro_hmm(ticker)
    
    if success:
        print(f"\n✅ Macro HMM trained successfully on {ticker}")
    else:
        print(f"\n❌ Failed to train Macro HMM")


if __name__ == "__main__":
    main()
