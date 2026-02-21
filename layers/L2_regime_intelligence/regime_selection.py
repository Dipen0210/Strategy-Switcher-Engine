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
    0: "Bull-Quiet",
    1: "Bull-Volatile",
    2: "Sideways",
    3: "Crisis"
}

# Reverse mapping
REGIME_NAME_TO_IDX = {v: k for k, v in REGIME_NAMES.items()}

# Strategy compatibility per regime (which strategies CAN run)
REGIME_STRATEGY_COMPAT = {
    "Bull-Quiet": ["Momentum", "Trend Following", "Breakout", "Envelope Trading", "MACD Trend"],
    "Bull-Volatile": ["Momentum", "Breakout", "Defensive", "MACD Trend", "Mean Reversion"],
    "Sideways": ["Mean Reversion", "Envelope Trading", "Defensive", "MACD Trend", "Trend Following"],
    "Crisis": ["Defensive", "Mean Reversion", "Envelope Trading"],
}

# Legacy name mapping
LEGACY_REGIME_MAP = {
    "Trend + Low Vol": "Bull-Quiet",
    "Trend + High Vol": "Bull-Volatile",
    "Range + Low Vol": "Sideways",
    "Bear + High Vol": "Crisis",
}


@dataclass
class RegimeOutput:
    """Output from regime detection."""
    probabilities: dict[str, float]  # Regime name -> probability
    dominant_regime: str  # Most likely regime
    allowed_strategies: list[str]  # Strategies compatible with dominant regime
    stability_score: float = 1.0  # 1.0 = Stable, 0.4 = Transition/Unstable
    hmm_confidence: float = 0.0  # Max blended posterior (used for scoring)
    is_ambiguous: bool = False  # True if confidence < 0.55
    transition_flag: bool = False  # True if regime changed from last cycle


# ===== REGIME BLENDING FUNCTIONS =====

def blend_regime(
    hmm_posteriors: dict[str, float],
    bandit_a_weights: dict[str, float],
    w_global: float = 0.60,
    w_hmm: float = 0.40,
) -> dict[str, float]:
    """
    Blend HMM posteriors with Bandit A trust weights using a weighted sum.

    Final Score = (W_Global * Normalized_BanditA) + (W_HMM * HMM_Posterior)

    Args:
        hmm_posteriors: Raw posteriors from HMM {regime: probability}
        bandit_a_weights: Trust weights from Bandit A {regime: weight}
        w_global: Weight given to the global bandit's regime selection
        w_hmm: Weight given to the HMM model output

    Returns:
        Renormalized blended scores {regime: blended_probability}
    """
    # Normalize Bandit A weights so they sum to 1.0 (comparable to HMM probabilities)
    total_trust = sum(bandit_a_weights.values())
    if total_trust > 0:
        norm_trust = {r: v / total_trust for r, v in bandit_a_weights.items()}
    else:
        n = len(bandit_a_weights) if bandit_a_weights else 4
        norm_trust = {r: 1.0 / n for r in bandit_a_weights}

    blended = {}
    for regime, posterior in hmm_posteriors.items():
        # Normalize legacy names
        regime_key = LEGACY_REGIME_MAP.get(regime, regime)
        trust = norm_trust.get(regime_key, 0.25)
        
        # Explicit weighted sum formula
        blended[regime_key] = (w_global * trust) + (w_hmm * posterior)

    # Re-normalize just in case
    total = sum(blended.values())
    if total > 0:
        blended = {r: v / total for r, v in blended.items()}
    else:
        # Uniform if everything is 0
        n = len(blended)
        blended = {r: 1.0 / n for r in blended}

    return blended


def compute_stability(regime_history: list[str], window: int = 5) -> float:
    """
    Compute regime stability from recent history.

    Stability = 1 - (number of regime changes in last N days) / (N - 1)

    - 1.0 = Same regime every day (perfectly stable)
    - 0.0 = Regime changed every single day (extremely unstable)

    Args:
        regime_history: List of last N regime labels (most recent last)
        window: Number of days to consider

    Returns:
        Stability score in [0, 1]
    """
    if len(regime_history) < 2:
        return 0.5  # neutral if insufficient data

    history = regime_history[-window:]
    changes = sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])
    max_changes = len(history) - 1

    return 1.0 - (changes / max_changes) if max_changes > 0 else 1.0


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
            
            # --- STABILITY SCORE CALCULATION ---
            # 1. Calculate Entropy: Sum(-p * log2(p))
            # Entropy measures "confusion". High entropy = probabilities spread out = Unstable.
            entropy = 0.0
            for p in probs.values():
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # 2. Map Entropy to Stability Score (1.0 to 0.4)
            # Max entropy for 4 states is -log2(1/4) = 2.0
            # Threshold: Entropy > 1.2 is "Unstable"
            if entropy < 0.6:
                stability = 1.0
            elif entropy > 1.2:
                stability = 0.4  # Transition / Unstable
            else:
                # Linear interpolation between 0.6 and 1.2
                # 0.6 -> 1.0, 1.2 -> 0.4
                pct = (entropy - 0.6) / (1.2 - 0.6)
                stability = 1.0 - (pct * 0.6)
            
            return RegimeOutput(
                probabilities=probs,
                dominant_regime=dominant_regime,
                allowed_strategies=allowed,
                stability_score=round(stability, 2)
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
            allowed_strategies=["Mean Reversion", "Defensive"],
            stability_score=0.4  # Default to unstable if prediction fails
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
    Manages the Macro HMM for regime detection.

    Uses a SINGLE macro model (trained on SPY) shared across all stocks.
    The macro model is loaded once and reused for all predictions.

    Flow:
    - Load macro model from models/macro_hmm.pkl
    - Use it for ALL stocks (market climate is the same for everyone)
    - Never retrain during pipeline runs (retrain monthly via train_hmm.py)
    """

    def __init__(self, model_dir: str | Path = None):
        # Save models alongside the layer code
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.model_dir = Path(model_dir)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.macro_detector: Optional[RegimeDetector] = None
        
        # Per-ticker state for stability calculation
        self.ticker_histories: dict[str, list[str]] = {} 
        self.ticker_last_regimes: dict[str, Optional[str]] = {}

    def get_macro_model_path(self) -> Path:
        """Get path to macro HMM model."""
        return self.model_dir / "macro_hmm.pkl"

    def _load_macro_model(self) -> RegimeDetector:
        """Load the single Macro HMM (trained on SPY)."""
        if self.macro_detector is not None:
            return self.macro_detector

        detector = RegimeDetector()
        model_path = self.get_macro_model_path()

        if model_path.exists():
            try:
                detector.load_model(model_path)
                print(f"  📊 Loaded Macro HMM from {model_path}")
            except Exception as e:
                print(f"  ⚠ Failed to load Macro HMM: {e}, using fallback")
        else:
            print(f"  ⚠ No Macro HMM found at {model_path}, using rule-based fallback")

        self.macro_detector = detector
        return detector

    # Legacy compatibility
    def get_or_create(self, ticker: str) -> RegimeDetector:
        return self._load_macro_model()

    get_or_create_detector = get_or_create

    def predict_regime(self, ticker: str, df: pd.DataFrame) -> RegimeOutput:
        """
        Predict regime using Macro HMM.

        Note: The same macro model is applied to this stock's data.
        State (history) is tracked per-ticker to ensure correct stability scores.
        """
        detector = self._load_macro_model()
        output = detector.predict_regime(df)

        # Initialize ticker history if needed
        if ticker not in self.ticker_histories:
            self.ticker_histories[ticker] = []
            self.ticker_last_regimes[ticker] = None

        # Track regime history for THIS ticker
        self.ticker_histories[ticker].append(output.dominant_regime)
        if len(self.ticker_histories[ticker]) > 20:
            self.ticker_histories[ticker] = self.ticker_histories[ticker][-20:]

        # Detect transition for THIS ticker
        last_r = self.ticker_last_regimes[ticker]
        output.transition_flag = (
            last_r is not None
            and output.dominant_regime != last_r
        )

        # Compute stability from THIS ticker's history
        output.stability_score = compute_stability(self.ticker_histories[ticker])

        # Set confidence
        output.hmm_confidence = max(output.probabilities.values())
        output.is_ambiguous = output.hmm_confidence < 0.55

        self.ticker_last_regimes[ticker] = output.dominant_regime
        return output

    def predict_all_regimes(
        self,
        stock_data_dict: dict[str, pd.DataFrame],
    ) -> dict[str, RegimeOutput]:
        """
        Predict regimes for all assets using the shared Macro HMM.

        Since we use a single macro model, the regime is the same
        for all stocks. We still return per-ticker results for
        API compatibility.
        """
        results = {}

        for ticker, df in stock_data_dict.items():
            if df is None or df.empty:
                continue
            results[ticker] = self.predict_regime(ticker, df)

        return results

    def get_model_info(self) -> dict:
        """Get info about the macro model."""
        model_path = self.get_macro_model_path()

        if not model_path.exists():
            return {"exists": False, "type": "macro"}

        import os
        mtime = os.path.getmtime(model_path)
        trained_at = pd.Timestamp.fromtimestamp(mtime)
        days_ago = (pd.Timestamp.now() - trained_at).days

        return {
            "exists": True,
            "type": "macro",
            "path": str(model_path),
            "trained_at": trained_at.isoformat(),
            "days_ago": days_ago,
            "needs_retraining": days_ago >= 30,
        }

