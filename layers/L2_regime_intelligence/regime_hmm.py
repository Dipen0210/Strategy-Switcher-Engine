# ml/regime_hmm.py
"""
HMM-based Regime Detection for SUP Flow 1.

Asset-level Gaussian HMM to identify latent market regimes.
States:
- Bull (Trend + Low Vol): Strong upward momentum, stable
- Volatile Bull (Trend + High Vol): Upward with high uncertainty
- Sideways (Range + Low Vol): Mean-reverting, stable
- Crisis (Range + High Vol): Stress period, high uncertainty

Training:
- Data: 5 years daily data
- Method: Expectation-Maximization (EM)
- Retraining: Monthly rolling window
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Regime names for interpretation
REGIME_NAMES = {
    0: "Trend + Low Vol",
    1: "Trend + High Vol",
    2: "Range + Low Vol",
    3: "Crisis"
}

# Strategy compatibility per regime
REGIME_STRATEGY_COMPAT = {
    "Trend + Low Vol": ["Momentum", "Breakout"],
    "Trend + High Vol": ["Momentum", "Defensive"],
    "Range + Low Vol": ["Mean Reversion", "Defensive"],
    "Crisis": ["Defensive"]
}


@dataclass
class RegimeOutput:
    """Output from regime detection."""
    probabilities: dict[str, float]  # Regime name -> probability
    dominant_regime: str  # Most likely regime
    allowed_strategies: list[str]  # Strategies compatible with dominant regime


class RegimeDetector:
    """
    Gaussian Hidden Markov Model for market regime detection.
    
    Each asset gets its own HMM trained on:
    - Daily returns
    - Rolling volatility
    - Trend strength (MA slope)
    
    Outputs regime probabilities for strategy filtering.
    """
    
    def __init__(self, n_states: int = 4, random_state: int = 42):
        """
        Initialize regime detector.
        
        Args:
            n_states: Number of hidden states (default 4)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self._feature_means = None
        self._feature_stds = None
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for HMM.
        
        Uses:
        - Log returns
        - Realized volatility
        - MA slope (trend direction)
        """
        features = []
        
        # Log returns
        if "Return_1D" in df.columns:
            returns = df["Return_1D"].values
        else:
            returns = np.log(df["Close"] / df["Close"].shift(1)).values
        features.append(returns)
        
        # Volatility (Prioritize GARCH > Realized > Calculated)
        if "GARCH_Vol" in df.columns:
            vol = df["GARCH_Vol"].values
        elif "Realized_Vol" in df.columns:
            vol = df["Realized_Vol"].values
        else:
            log_ret = np.log(df["Close"] / df["Close"].shift(1))
            vol = log_ret.rolling(window=20).std().values * np.sqrt(252)
        features.append(vol)
        
        # Trend indicator (MA slope)
        if "MA_Slope" in df.columns:
            trend = df["MA_Slope"].values
        else:
            ma = df["Close"].rolling(window=20).mean()
            trend = ((ma - ma.shift(5)) / ma.shift(5)).values
        features.append(trend)
        
        # Stack and handle NaN
        X = np.column_stack(features)
        
        # Remove rows with NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        
        return X
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features for numerical stability."""
        if fit:
            self._feature_means = np.nanmean(X, axis=0)
            self._feature_stds = np.nanstd(X, axis=0)
            self._feature_stds[self._feature_stds == 0] = 1.0
        
        if self._feature_means is None:
            return X
        
        return (X - self._feature_means) / self._feature_stds
    
    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Fit HMM to historical data.
        
        Args:
            df: DataFrame with OHLCV data and computed features
                Expected columns: Close, Return_1D (optional), 
                Realized_Vol (optional), MA_Slope (optional)
        
        Returns:
            self
        """
        X = self._prepare_features(df)
        
        if len(X) < 100:
            raise ValueError("Not enough data to fit HMM (need at least 100 observations)")
        
        X_norm = self._normalize_features(X, fit=True)
        
        try:
            from hmmlearn.hmm import GaussianHMM
            
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=self.random_state,
                init_params="stmc",  # Initialize all parameters
            )
            
            self.model.fit(X_norm)
            self.is_fitted = True
            
        except ImportError:
            # hmmlearn not installed, use simple rule-based fallback
            self._use_fallback = True
            self.is_fitted = True
        
        return self
    
    def predict_regime(self, df: pd.DataFrame) -> RegimeOutput:
        """
        Predict current regime probabilities.
        
        Args:
            df: DataFrame with recent price data
        
        Returns:
            RegimeOutput with probabilities and allowed strategies
        """
        if not self.is_fitted:
            # Return uniform probabilities if not fitted
            probs = {name: 1.0 / self.n_states for name in REGIME_NAMES.values()}
            return RegimeOutput(
                probabilities=probs,
                dominant_regime="Range + Low Vol",
                allowed_strategies=["Mean Reversion", "Defensive", "Momentum"]
            )
        
        # Check for fallback mode
        if getattr(self, "_use_fallback", False):
            return self._fallback_regime_detection(df)
        
        X = self._prepare_features(df)
        
        if len(X) == 0:
            return self._default_output()
        
        X_norm = self._normalize_features(X, fit=False)
        
        try:
            # Get state probabilities for the latest observation
            log_prob, posteriors = self.model.score_samples(X_norm)
            
            # Get probabilities for the latest observation
            latest_probs = posteriors[-1]
            
            # Map to regime names
            probs = {REGIME_NAMES[i]: float(latest_probs[i]) for i in range(self.n_states)}
            
            # Find dominant regime
            dominant_idx = np.argmax(latest_probs)
            dominant_regime = REGIME_NAMES[dominant_idx]
            
            # Get allowed strategies
            allowed = REGIME_STRATEGY_COMPAT.get(dominant_regime, ["Defensive"])
            
            return RegimeOutput(
                probabilities=probs,
                dominant_regime=dominant_regime,
                allowed_strategies=allowed
            )
            
        except Exception:
            return self._default_output()
    
    def _fallback_regime_detection(self, df: pd.DataFrame) -> RegimeOutput:
        """
        Simple rule-based regime detection when HMM is not available.
        
        Uses volatility and trend to classify regimes.
        """
        if df.empty or len(df) < 30:
            return self._default_output()
        
        # Compute simple metrics
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        vol = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
        momentum = df["Close"].pct_change(20).iloc[-1]
        
        # Simple rule-based classification
        high_vol = vol > 0.25
        trending = abs(momentum) > 0.05
        bullish = momentum > 0
        
        if trending and not high_vol:
            if bullish:
                regime = "Trend + Low Vol"
                probs = {"Trend + Low Vol": 0.7, "Trend + High Vol": 0.15, 
                         "Range + Low Vol": 0.1, "Crisis": 0.05}
            else:
                regime = "Range + Low Vol"
                probs = {"Trend + Low Vol": 0.1, "Trend + High Vol": 0.1,
                         "Range + Low Vol": 0.7, "Crisis": 0.1}
        elif trending and high_vol:
            regime = "Trend + High Vol"
            probs = {"Trend + Low Vol": 0.1, "Trend + High Vol": 0.7,
                     "Range + Low Vol": 0.05, "Crisis": 0.15}
        elif high_vol:
            regime = "Crisis"
            probs = {"Trend + Low Vol": 0.05, "Trend + High Vol": 0.15,
                     "Range + Low Vol": 0.1, "Crisis": 0.7}
        else:
            regime = "Range + Low Vol"
            probs = {"Trend + Low Vol": 0.15, "Trend + High Vol": 0.05,
                     "Range + Low Vol": 0.7, "Crisis": 0.1}
        
        allowed = REGIME_STRATEGY_COMPAT.get(regime, ["Defensive"])
        
        return RegimeOutput(
            probabilities=probs,
            dominant_regime=regime,
            allowed_strategies=allowed
        )
    
    def _default_output(self) -> RegimeOutput:
        """Return default output when prediction fails."""
        return RegimeOutput(
            probabilities={name: 0.25 for name in REGIME_NAMES.values()},
            dominant_regime="Range + Low Vol",
            allowed_strategies=["Mean Reversion", "Defensive"]
        )
    
    def save_model(self, path: str | Path) -> None:
        """Save fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "n_states": self.n_states,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "model": self.model,
            "use_fallback": getattr(self, "_use_fallback", False)
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load_model(self, path: str | Path) -> "RegimeDetector":
        """Load model from disk."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.n_states = state["n_states"]
        self.random_state = state["random_state"]
        self.is_fitted = state["is_fitted"]
        self._feature_means = state["feature_means"]
        self._feature_stds = state["feature_stds"]
        self.model = state["model"]
        self._use_fallback = state.get("use_fallback", False)
        
        return self


class RegimeManager:
    """
    Manages regime detectors for multiple assets.
    
    Handles:
    - Per-asset HMM training
    - Monthly retraining schedule
    - Model persistence in models/ directory
    
    Flow:
    - Initial run: Load from models/{ticker}_hmm.pkl if exists
    - Daily runs: Use cached model, predict regime
    - Monthly: Retrain if model is >30 days old
    """
    
    def __init__(self, model_dir: str | Path = None):
        # Use project-level models/ folder
        if model_dir is None:
            # Find project root (where main.py is)
            current = Path(__file__).resolve()
            project_root = current.parent.parent.parent  # layers/L2_regime_intelligence -> layers -> project
            self.model_dir = project_root / "models"
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.detectors: dict[str, RegimeDetector] = {}
        self.last_trained: dict[str, pd.Timestamp] = {}
    
    def get_model_path(self, ticker: str) -> Path:
        """Get path to model file for a ticker."""
        return self.model_dir / f"{ticker}_hmm.pkl"
    
    def get_or_create(self, ticker: str) -> RegimeDetector:
        """
        Get existing detector or create/load one.
        
        Priority:
        1. Return cached detector if exists
        2. Load from disk if model file exists
        3. Create new (unfitted) detector
        """
        if ticker in self.detectors:
            return self.detectors[ticker]
        
        detector = RegimeDetector()
        model_path = self.get_model_path(ticker)
        
        if model_path.exists():
            try:
                detector.load_model(model_path)
                # Read last modified time as training date
                import os
                mtime = os.path.getmtime(model_path)
                self.last_trained[ticker] = pd.Timestamp.fromtimestamp(mtime)
            except Exception:
                pass  # Will use unfitted detector
        
        self.detectors[ticker] = detector
        return detector
    
    # Alias for backward compatibility
    get_or_create_detector = get_or_create
    
    def train_detector(self, ticker: str, df: pd.DataFrame, force: bool = False) -> RegimeDetector:
        """
        Train or retrain detector for a ticker.
        
        Args:
            ticker: Stock ticker
            df: DataFrame with OHLCV data
            force: Force retrain even if model exists
        
        Returns:
            Trained RegimeDetector
        """
        detector = self.get_or_create(ticker)
        
        # Check if we need to retrain
        if not force and detector.is_fitted and not self.needs_retraining(ticker):
            return detector
        
        # Train
        detector.fit(df)
        
        # Save to disk
        model_path = self.get_model_path(ticker)
        detector.save_model(model_path)
        
        self.last_trained[ticker] = pd.Timestamp.now()
        
        return detector
    
    def needs_retraining(self, ticker: str, retrain_days: int = 30) -> bool:
        """
        Check if detector needs retraining.
        
        Retraining triggers:
        - Model doesn't exist
        - Model is older than retrain_days (default 30)
        """
        if ticker not in self.last_trained:
            # Check if model file exists
            model_path = self.get_model_path(ticker)
            if not model_path.exists():
                return True
            
            # Get file modification time
            import os
            mtime = os.path.getmtime(model_path)
            self.last_trained[ticker] = pd.Timestamp.fromtimestamp(mtime)
        
        days_since = (pd.Timestamp.now() - self.last_trained[ticker]).days
        return days_since >= retrain_days
    
    def predict_regime(self, ticker: str, df: pd.DataFrame) -> RegimeOutput:
        """
        Predict regime for a single asset.
        
        Loads model if exists, uses fallback if not trained.
        """
        detector = self.get_or_create(ticker)
        return detector.predict_regime(df)
    
    def predict_all_regimes(
        self, 
        stock_data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, RegimeOutput]:
        """
        Predict regimes for all assets.
        
        NOTE: This does NOT train models automatically.
        Use train_detector() explicitly for training.
        
        Returns dict of ticker -> RegimeOutput
        """
        results = {}
        
        for ticker, df in stock_data_dict.items():
            if df is None or df.empty:
                continue
            
            results[ticker] = self.predict_regime(ticker, df)
        
        return results
    
    def get_model_info(self, ticker: str) -> dict:
        """Get info about a saved model."""
        model_path = self.get_model_path(ticker)
        
        if not model_path.exists():
            return {"exists": False, "ticker": ticker}
        
        import os
        mtime = os.path.getmtime(model_path)
        trained_at = pd.Timestamp.fromtimestamp(mtime)
        days_ago = (pd.Timestamp.now() - trained_at).days
        
        return {
            "exists": True,
            "ticker": ticker,
            "path": str(model_path),
            "trained_at": trained_at.isoformat(),
            "days_ago": days_ago,
            "needs_retraining": days_ago >= 30,
        }

