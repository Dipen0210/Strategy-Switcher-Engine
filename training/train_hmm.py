# training/train_hmm.py
"""
HMM Training Script for Strategy Engine

This script trains a Gaussian HMM model for a specific stock ticker.
Trained models are saved to the models/ directory.

Usage:
    python training/train_hmm.py AAPL          # Train single ticker
    python training/train_hmm.py AAPL MSFT    # Train multiple tickers
    python training/train_hmm.py --all         # Train predefined list

Training Policy (from L2 spec):
- 5-year rolling window
- Fixed feature set
- 4 hidden states (Trend+LowVol, Trend+HighVol, Range+LowVol, Crisis)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
import pandas as pd
import yfinance as yf

from layers.L1_data_fabric.features import compute_all_features


# === CONFIGURATION ===
DEFAULT_TICKERS = ["WMT", "JNJ", "NVDA", "JPM", "NEE"]

# Fixed training period: 10 years (2015-2025)
TRAINING_START_DATE = "2014-01-01"
TRAINING_END_DATE = "2024-01-01"

N_STATES = 4
MODELS_DIR = PROJECT_ROOT / "models"

REGIME_NAMES = {
    0: "Trend + Low Vol",
    1: "Trend + High Vol",
    2: "Range + Low Vol",
    3: "Crisis",
}


def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load historical OHLCV data for a ticker using fixed 10-year period."""
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
    """Prepare feature matrix for HMM training."""
    features_df = compute_all_features(df)
    
    # Select features for HMM
    feature_cols = [
        "Return_1D",
        "Realized_Vol",
        "Momentum",
    ]
    
    available_cols = [c for c in feature_cols if c in features_df.columns]
    
    if not available_cols:
        # Fallback to simple returns and volatility
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        vol = returns.rolling(20).std().dropna() * np.sqrt(252)
        
        min_len = min(len(returns), len(vol))
        X = np.column_stack([
            returns.iloc[-min_len:].values,
            vol.iloc[-min_len:].values,
        ])
    else:
        X = features_df[available_cols].dropna().values
    
    print(f"  Feature matrix shape: {X.shape}")
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


def save_model(model, ticker: str, metadata: dict, feature_stats: dict = None):
    """Save trained model to disk in RegimeDetector format."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODELS_DIR / f"{ticker}_hmm.pkl"
    
    # Save in format compatible with RegimeDetector.load_model()
    model_data = {
        "n_states": model.n_components,
        "random_state": 42,
        "is_fitted": True,
        "feature_means": feature_stats.get("means") if feature_stats else None,
        "feature_stds": feature_stats.get("stds") if feature_stats else None,
        "model": model,
        "use_fallback": False,
        # Extra metadata
        "ticker": ticker,
        "trained_at": datetime.now().isoformat(),
        "regime_names": REGIME_NAMES,
        **metadata,
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"  ✓ Saved to {model_path}")
    return model_path


def train_ticker(ticker: str) -> bool:
    """Train HMM for a single ticker using 10-year data."""
    print(f"\n{'='*50}")
    print(f"Training HMM for: {ticker}")
    print(f"Training period: {TRAINING_START_DATE} to {TRAINING_END_DATE}")
    print(f"{'='*50}")
    
    try:
        # 1. Load 10-year data
        df = load_stock_data(ticker)
        
        # 2. Prepare features
        X = prepare_features(df)
        
        if len(X) < 100:
            print(f"  ⚠ Insufficient data ({len(X)} samples), skipping")
            return False
        
        # 3. Train HMM
        print(f"  Training {N_STATES}-state HMM...")
        model = train_hmm(X, n_states=N_STATES)
        
        # 4. Save model
        metadata = {
            "data_start": TRAINING_START_DATE,
            "data_end": TRAINING_END_DATE,
            "n_samples": len(X),
            "training_period": "10 years",
        }
        save_model(model, ticker, metadata)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train HMM models for stock regime detection"
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Ticker symbols to train (e.g., AAPL MSFT)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Train all default tickers: {DEFAULT_TICKERS}"
    )
    
    args = parser.parse_args()
    
    # Determine which tickers to train
    if args.all:
        tickers = DEFAULT_TICKERS
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        print("Usage:")
        print("  python training/train_hmm.py AAPL         # Single ticker")
        print("  python training/train_hmm.py AAPL MSFT   # Multiple tickers")
        print("  python training/train_hmm.py --all        # Default tickers")
        return
    
    print(f"\n🎯 Training HMM models for: {', '.join(tickers)}")
    print(f"📁 Models will be saved to: {MODELS_DIR}")
    
    # Train each ticker
    results = {}
    for ticker in tickers:
        success = train_ticker(ticker)
        results[ticker] = "✓" if success else "✗"
    
    # Summary
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    for ticker, status in results.items():
        print(f"  {status} {ticker}")
    
    success_count = sum(1 for s in results.values() if s == "✓")
    print(f"\nCompleted: {success_count}/{len(tickers)} models trained")


if __name__ == "__main__":
    main()
