# Multi-Strategy Ensemble Implementation Plan

## Naming Convention

| Display Name | Internal Name |
|--------------|---------------|
| Bull-Quiet | Trend + Low Vol |
| Bull-Volatile | Trend + High Vol |
| Sideways | Range + Low Vol |
| Crisis | Bear + High Vol |

---

## Folder Structure (By Style)

```
layers/L3_strategy_universe/
├── base_strategy.py          # StrategyOutput dataclass
├── registry.py               # Loads YAML config
├── regime_config.yaml        # Regime → Strategy mapping
├── trend/                    # 10 files
├── reversion/                # 10 files
├── volatility/               # 3 files
├── range/                    # 7 files
├── hedge/                    # 6 files
├── short/                    # 2 files
└── factor/                   # 2 files
```

---

## Strategy Output Format

```python
@dataclass
class StrategyOutput:
    ticker: str
    signal: int        # +1 Buy, 0 Hold, -1 Sell
    confidence: float  # 0.0 - 1.0 (pattern strength)
    strategy_name: str
    pod: str
    regime: str
```

---

## 40 Strategies (10 per Regime)

### Bull-Quiet (Trend + Low Vol)
| Strategy | File | Pod |
|----------|------|-----|
| Momentum | trend/momentum.py | Trend |
| Trend Following | trend/trend_following.py | Trend |
| Breakout | trend/breakout.py | Trend |
| MACD Trend | trend/macd_trend.py | Trend |
| ADX Trend | trend/adx_trend.py | Trend |
| Parabolic SAR | trend/parabolic_sar.py | Trend |
| Ichimoku | trend/ichimoku.py | Trend |
| Price Channel | trend/price_channel.py | Trend |
| Keltner Breakout | trend/keltner_breakout.py | Trend |
| Low Vol Factor | factor/low_vol_factor.py | Factor |

### Bull-Volatile (Trend + High Vol)
| Strategy | File | Pod |
|----------|------|-----|
| Mean Reversion | reversion/mean_reversion.py | Reversion |
| Pullback Buyer | reversion/pullback_buyer.py | Reversion |
| Stochastic Oversold | reversion/stochastic_oversold.py | Reversion |
| Williams %R | reversion/williams_r.py | Reversion |
| CCI Reversal | reversion/cci_reversal.py | Reversion |
| VWAP Bounce | reversion/vwap_bounce.py | Reversion |
| Fibonacci Retracement | reversion/fibonacci_retracement.py | Reversion |
| Dip Buyer RSI | reversion/dip_buyer_rsi.py | Reversion |
| Volatility Scalper | volatility/volatility_scalper.py | Volatility |
| ATR Trailing | volatility/atr_trailing.py | Volatility |

### Sideways (Range + Low Vol)
| Strategy | File | Pod |
|----------|------|-----|
| Bollinger Reversion | reversion/bollinger_reversion.py | Range |
| RSI Exhaustion | reversion/rsi_exhaustion.py | Range |
| Grid Trading | range/grid_trading.py | Range |
| Support/Resistance | range/support_resistance.py | Range |
| Envelope Trading | range/envelope_trading.py | Range |
| Pivot Point | range/pivot_point.py | Range |
| False Breakout | range/range_false_breakout.py | Range |
| Oscillator Divergence | range/oscillator_divergence.py | Range |
| Stochastic Range | range/stochastic_range.py | Range |
| Mean Reversion Range | range/mean_reversion_range.py | Range |

### Crisis (Bear + High Vol)
| Strategy | File | Pod |
|----------|------|-----|
| Defensive | hedge/defensive.py | Defensive |
| Cash Defensive | hedge/cash_defensive.py | Defensive |
| Risk-Off Rotation | hedge/risk_off_rotation.py | Defensive |
| Tail Hedge | hedge/tail_hedge.py | Hedge |
| Put Protection | hedge/put_protection.py | Hedge |
| VIX Spike | hedge/vix_spike.py | Hedge |
| Short Momentum | short/short_momentum.py | Short |
| Inverse ETF | short/inverse_etf.py | Short |
| Vol Breakout Down | volatility/vol_breakout_down.py | Short |
| Death Cross | trend/death_cross.py | Short |

---

## 4-Factor Scoring Formula

```
Strategy Score = Signal × Confidence × Stability × BanditWeight
```

| Factor | Source | Range | Purpose |
|--------|--------|-------|---------|
| Signal | Strategy (L3) | +1, 0, -1 | Direction (Buy/Hold/Sell) |
| Confidence | Strategy (L3) | 0.0 - 1.0 | How strong is the pattern NOW? |
| Stability | HMM (L2) | 0.4 - 1.0 | Is the regime certain? Reduce size if confused |
| BanditWeight | L5 | 0.0 - 1.0 | ML-learned trust from HISTORICAL performance |

**BanditWeight = GlobalW × RegimeW × StockW**

| Bandit Level | Purpose |
|--------------|---------|
| GlobalW | "Is this Pod type working in current market conditions?" |
| RegimeW | "Does this strategy work in this regime historically?" |
| StockW | "Does this strategy work for THIS specific stock?" |

> **VolScalar** (GARCH) used in **L7 Position Sizing only** (not scoring).

---

## L4.5 Pod Filtering (100 Strategy Example)

```
Step 1: Global Bandit scores Pods → Trend: 0.7, Factor: 0.3
Step 2: Proportional selection → 15 strategies from 100
Step 3: Run only filtered strategies
```

---

## Pipeline Flow (10 Steps)

| Step | Layer | Action |
|------|-------|--------|
| 1 | L2 | Detect regime + Stability Score |
| 2 | L4 | Load YAML → Get regime strategies |
| 3 | L4.5 | Global Bandit → Filter by Pod |
| 4 | L3 | Execute → Signal + Confidence |
| 5 | L5 | Get BanditWeight (G × R × S) |
| 6 | L6 | Compute 4-factor score |
| 7 | L6 | Winner = Highest Score |
| 8 | L7 | Position Sizing (User × Vol × Stability) |
| 9 | L10/L11 | Execute → Update Portfolio |
| 10 | L12 | Feedback → Update L5 Bandits |

---

## Hierarchical Bandit (3 Levels)

| Level | Model | Purpose |
|-------|-------|---------|
| Global (1) | Pod weights | L4.5 filtering |
| Regime (4) | Strategy weights per regime | L5 scoring |
| Stock (N) | Per-stock preferences | L5 scoring |

---

## Files Summary

| Type | Count |
|------|-------|
| NEW strategies | 36 |
| UPDATE strategies | 4 |
| NEW bandit files | 3 |
| NEW config | 1 |
| UPDATE existing | 5 |
| **Total** | **49** |
