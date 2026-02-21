# layers/L0_user_policy/policy.py
"""
LAYER 0 — USER & POLICY DEFINITION (AUTHORITY LAYER)

Nothing below can override this layer.

Purpose: Define immutable constraints and intent.

Inputs:
- Selected assets (tickers, asset type)
- Asset weights (sum = 100%)
- Total capital
- Risk tolerance per asset (Low / Medium / High)
- Rebalance frequency (Daily / Weekly / Monthly)

Outputs:
- User policy object
- Hard risk limits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

RiskTolerance = Literal["Low", "Medium", "High"]
RebalanceFrequency = Literal["Daily", "Weekly", "Monthly"]


# Hard risk limits per tolerance level
RISK_LIMITS = {
    "Low": {
        "max_volatility": 0.08,    # 8%
        "max_drawdown": 0.05,      # 5%
        "max_leverage": 1.0,
    },
    "Medium": {
        "max_volatility": 0.15,    # 15%
        "max_drawdown": 0.10,      # 10%
        "max_leverage": 1.0,
    },
    "High": {
        "max_volatility": 0.25,    # 25%
        "max_drawdown": 0.20,      # 20%
        "max_leverage": 1.5,
    },
}


@dataclass
class AssetPolicy:
    """Policy for a single asset."""
    ticker: str
    weight: float  # 0.0 to 1.0
    risk_tolerance: RiskTolerance = "Medium"
    
    @property
    def risk_limits(self) -> dict:
        return RISK_LIMITS[self.risk_tolerance]


@dataclass
class UserPolicy:
    """
    Immutable user policy object.
    
    ALL downstream layers must comply. No learning, ranking, 
    or execution can bypass this.
    """
    assets: List[AssetPolicy] = field(default_factory=list)
    total_capital: float = 10000.0
    rebalance_frequency: RebalanceFrequency = "Weekly"
    
    # Kill-switch thresholds
    emergency_drawdown_threshold: float = 0.15  # Trigger defensive mode
    emergency_volatility_threshold: float = 0.30  # Trigger defensive mode
    
    def __post_init__(self):
        self._validate()
    
    def _validate(self):
        """Validate policy constraints."""
        if self.total_capital <= 0:
            raise ValueError("Total capital must be positive")
        
        if self.assets:
            total_weight = sum(a.weight for a in self.assets)
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError(f"Asset weights must sum to 1.0, got {total_weight}")
    
    @property
    def tickers(self) -> List[str]:
        return [a.ticker for a in self.assets]
    
    @property
    def weights(self) -> Dict[str, float]:
        return {a.ticker: a.weight for a in self.assets}
    
    def get_asset_policy(self, ticker: str) -> AssetPolicy:
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
        raise KeyError(f"Asset {ticker} not in policy")
    
    def get_risk_limits(self, ticker: str) -> dict:
        return self.get_asset_policy(ticker).risk_limits
    
    def get_strictest_limits(self) -> dict:
        """Get the most restrictive limits across all assets."""
        if not self.assets:
            return RISK_LIMITS["Medium"]
        
        min_vol = min(a.risk_limits["max_volatility"] for a in self.assets)
        min_dd = min(a.risk_limits["max_drawdown"] for a in self.assets)
        min_lev = min(a.risk_limits["max_leverage"] for a in self.assets)
        
        return {
            "max_volatility": min_vol,
            "max_drawdown": min_dd,
            "max_leverage": min_lev,
        }
    
    def to_dict(self) -> dict:
        return {
            "assets": [
                {
                    "ticker": a.ticker,
                    "weight": a.weight,
                    "risk_tolerance": a.risk_tolerance,
                }
                for a in self.assets
            ],
            "total_capital": self.total_capital,
            "rebalance_frequency": self.rebalance_frequency,
            "emergency_dd_threshold": self.emergency_drawdown_threshold,
            "emergency_vol_threshold": self.emergency_volatility_threshold,
        }


def create_policy(
    tickers: List[str],
    weights: List[float],
    total_capital: float = 10000.0,
    risk_tolerance: RiskTolerance = "Medium",
    rebalance_frequency: RebalanceFrequency = "Weekly",
    emergency_drawdown_threshold: float = 0.20,
) -> UserPolicy:
    """
    Factory function to create a UserPolicy.
    
    Args:
        tickers: List of asset tickers
        weights: List of weights (same order as tickers)
        total_capital: Total investment capital
        risk_tolerance: Applied to all assets
        rebalance_frequency: How often to rebalance
    
    Returns:
        UserPolicy object
    """
    if len(tickers) != len(weights):
        raise ValueError("Tickers and weights must have same length")
    
    # Normalize weights
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    
    assets = [
        AssetPolicy(ticker=t, weight=w, risk_tolerance=risk_tolerance)
        for t, w in zip(tickers, weights)
    ]
    
    return UserPolicy(
        assets=assets,
        total_capital=total_capital,
        rebalance_frequency=rebalance_frequency,
        emergency_drawdown_threshold=emergency_drawdown_threshold,
    )
