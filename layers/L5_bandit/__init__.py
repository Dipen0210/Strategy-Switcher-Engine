# layers/L5_bandit/__init__.py
"""
Hierarchical Bandit System — 3 Levels.

Bandit A (Global)  → Regime Trust (slow, γ=0.99)
Bandit B (Regime)  → Strategy Ranking (medium, γ=0.97)
Bandit C (Stock)   → Stock-Specific Filter (fast, γ=0.94)

All bandits persist to disk for long-term learning.
"""

from layers.L5_bandit.global_bandit import GlobalBandit
from layers.L5_bandit.regime_bandit import RegimeBandit, RegimeBanditManager
from layers.L5_bandit.stock_bandit import StockBandit, StockBanditManager
from layers.L5_bandit.persistence import BanditPersistenceManager, get_bandit_dir
from layers.L5_bandit.reward import compute_reward, reward_to_beta, get_differentiated_rewards

__all__ = [
    "GlobalBandit",
    "RegimeBandit",
    "RegimeBanditManager",
    "StockBandit",
    "StockBanditManager",
    "BanditPersistenceManager",
    "get_bandit_dir",
    "compute_reward",
    "reward_to_beta",
    "get_differentiated_rewards",
]
