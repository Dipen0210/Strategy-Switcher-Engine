# ml/features.py
"""
Feature Engineering for SUP Flow 1.

Transforms raw OHLCV data into model-ready features:
- Multi-period returns (1D, 5D, 20D)
- Realized volatility
- GARCH volatility forecast
- Trend indicators (ADX, MA slope)
- Autocorrelation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_returns(df: pd.DataFrame, periods: list[int] = [1, 5, 20]) -> pd.DataFrame:
    """
    Compute log returns over multiple periods.
    
    Args:
        df: DataFrame with 'Close' column
        periods: List of lookback periods (e.g., [1, 5, 20] for 1D, 5D, 20D)
    
    Returns:
        DataFrame with return columns: Return_1D, Return_5D, etc.
    """
    result = df.copy()
    close = result["Close"]
    
    for period in periods:
        col_name = f"Return_{period}D"
        result[col_name] = np.log(close / close.shift(period))
    
    return result


def compute_realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling realized volatility from log returns.
    
    Args:
        df: DataFrame with 'Close' column
        window: Rolling window in days (default 20 ~ 1 month)
    
    Returns:
        DataFrame with 'Realized_Vol' column (annualized)
    """
    result = df.copy()
    log_returns = np.log(result["Close"] / result["Close"].shift(1))
    
    # Annualize: sqrt(252) for daily data
    result["Realized_Vol"] = log_returns.rolling(window=window).std() * np.sqrt(252)
    
    return result


def compute_garch_volatility(df: pd.DataFrame, p: int = 1, q: int = 1) -> pd.DataFrame:
    """
    Compute GARCH(p,q) volatility forecast.
    
    Uses arch library for GARCH fitting. Falls back to realized vol if fitting fails.
    
    Args:
        df: DataFrame with 'Close' column
        p: GARCH lag order for volatility
        q: ARCH lag order for returns
    
    Returns:
        DataFrame with 'GARCH_Vol' column
    """
    result = df.copy()
    result["GARCH_Vol"] = np.nan
    
    try:
        from arch import arch_model
        
        log_returns = np.log(result["Close"] / result["Close"].shift(1)).dropna()
        
        if len(log_returns) < 100:
            # Not enough data for GARCH, use realized vol
            result["GARCH_Vol"] = result.get("Realized_Vol", np.nan)
            return result
        
        # Scale returns for numerical stability
        scaled_returns = log_returns * 100
        
        model = arch_model(scaled_returns, vol="Garch", p=p, q=q, rescale=False)
        fitted = model.fit(disp="off", show_warning=False)
        
        # Get conditional volatility and scale back, annualize
        cond_vol = fitted.conditional_volatility / 100 * np.sqrt(252)
        
        # Align with original index
        result.loc[cond_vol.index, "GARCH_Vol"] = cond_vol.values
        
    except ImportError:
        # arch not installed, fall back to realized vol
        if "Realized_Vol" not in result.columns:
            result = compute_realized_volatility(result)
        result["GARCH_Vol"] = result["Realized_Vol"]
    except Exception:
        # GARCH fitting failed, use realized vol
        if "Realized_Vol" not in result.columns:
            result = compute_realized_volatility(result)
        result["GARCH_Vol"] = result["Realized_Vol"]
    
    return result


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average Directional Index (ADX) for trend strength.
    
    Args:
        df: DataFrame with High, Low, Close columns
        period: ADX period (default 14)
    
    Returns:
        DataFrame with 'ADX' column (0-100 scale)
    """
    result = df.copy()
    
    # Flatten columns if multi-index (from yfinance)
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    
    high = result["High"].values.flatten()
    low = result["Low"].values.flatten()
    close = result["Close"].values.flatten()
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = np.nan  # First value invalid due to shift
    
    # Directional Movement
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(-low, prepend=-low[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # Smoothed averages using pandas for rolling
    atr = pd.Series(tr).rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / (atr + 1e-10)
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    result["ADX"] = dx.rolling(window=period).mean().values
    
    return result


def compute_ma_slope(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Compute slope of moving average as trend indicator.
    
    Args:
        df: DataFrame with 'Close' column
        period: MA period (default 20)
    
    Returns:
        DataFrame with 'MA_Slope' column (normalized by price)
    """
    result = df.copy()
    
    ma = result["Close"].rolling(window=period).mean()
    
    # Slope as percent change of MA over 5 days
    slope = (ma - ma.shift(5)) / ma.shift(5)
    result["MA_Slope"] = slope
    
    return result


def compute_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all trend-related indicators.
    
    Returns DataFrame with:
    - ADX: Average Directional Index (trend strength)
    - MA_Slope: Moving average slope (trend direction)
    """
    result = compute_adx(df)
    result = compute_ma_slope(result)
    return result


def compute_autocorrelation(df: pd.DataFrame, lags: list[int] = [1, 5, 10]) -> pd.DataFrame:
    """
    Compute autocorrelation of returns at specified lags.
    
    Args:
        df: DataFrame with 'Close' column
        lags: List of lag periods
    
    Returns:
        DataFrame with 'Autocorr_1', 'Autocorr_5', etc. columns
    """
    result = df.copy()
    log_returns = np.log(result["Close"] / result["Close"].shift(1))
    
    for lag in lags:
        col_name = f"Autocorr_{lag}"
        result[col_name] = log_returns.rolling(window=60).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
            raw=False
        )
    
    return result


def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling drawdown from peak.
    
    Returns:
        DataFrame with 'Drawdown' column (negative values, 0 = at peak)
    """
    result = df.copy()
    
    rolling_max = result["Close"].expanding().max()
    result["Drawdown"] = (result["Close"] - rolling_max) / rolling_max
    
    return result


def compute_momentum(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Compute momentum as rate of change.
    
    Returns:
        DataFrame with 'Momentum' column
    """
    result = df.copy()
    result["Momentum"] = result["Close"].pct_change(periods=period)
    return result


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for a single asset.
    
    Input: DataFrame with OHLCV columns (Date optional as index or column)
    Output: DataFrame with all feature columns added
    """
    if df.empty or len(df) < 30:
        return df
    
    result = df.copy()
    
    # Ensure sorted by date
    if "Date" in result.columns:
        result = result.sort_values("Date").reset_index(drop=True)
    
    # Core features
    result = compute_returns(result, periods=[1, 5, 20])
    result = compute_realized_volatility(result, window=20)
    result = compute_garch_volatility(result)
    result = compute_trend_indicators(result)
    result = compute_autocorrelation(result, lags=[1, 5, 10])
    result = compute_drawdown(result)
    result = compute_momentum(result)
    
    return result


def build_feature_matrix(stock_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build feature matrix for all assets.
    
    Args:
        stock_data_dict: Dict of {ticker: OHLCV DataFrame}
    
    Returns:
        DataFrame with one row per asset containing latest features:
        - Ticker
        - Return_1D, Return_5D, Return_20D
        - Realized_Vol, GARCH_Vol
        - ADX, MA_Slope
        - Autocorr_1, Autocorr_5, Autocorr_10
        - Drawdown, Momentum
    """
    rows = []
    
    feature_cols = [
        "Return_1D", "Return_5D", "Return_20D",
        "Realized_Vol", "GARCH_Vol",
        "ADX", "MA_Slope",
        "Autocorr_1", "Autocorr_5", "Autocorr_10",
        "Drawdown", "Momentum"
    ]
    
    for ticker, df in stock_data_dict.items():
        if df is None or df.empty:
            continue
        
        try:
            enriched = compute_all_features(df)
            
            if enriched.empty:
                continue
            
            # Get latest values
            latest = enriched.iloc[-1]
            
            row = {"Ticker": ticker}
            for col in feature_cols:
                row[col] = latest[col] if col in latest.index else np.nan
            
            rows.append(row)
            
        except Exception:
            continue
    
    if not rows:
        return pd.DataFrame(columns=["Ticker"] + feature_cols)
    
    return pd.DataFrame(rows)


def get_feature_vector(df: pd.DataFrame, feature_names: Optional[list[str]] = None) -> np.ndarray:
    """
    Extract feature vector for ML model input.
    
    Args:
        df: DataFrame with computed features (from compute_all_features)
        feature_names: Optional list of feature names to extract
    
    Returns:
        numpy array of feature values (latest row)
    """
    if feature_names is None:
        feature_names = [
            "Return_1D", "Return_5D", "Return_20D",
            "Realized_Vol", "GARCH_Vol",
            "ADX", "MA_Slope",
            "Drawdown", "Momentum"
        ]
    
    if df.empty:
        return np.zeros(len(feature_names))
    
    latest = df.iloc[-1]
    values = [latest[col] if col in latest.index else 0.0 for col in feature_names]
    
    # Replace NaN with 0
    return np.nan_to_num(np.array(values, dtype=float), nan=0.0)
