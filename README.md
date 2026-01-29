# 🧱 Strategy Engine — Layered System Architecture

**Automated Strategy Switching with Regime Awareness**

> **Core philosophy:**
> *ML provides probabilistic intelligence. Rules enforce safety. Execution remains deterministic.*

---

## 🧠 One-Line Mental Model

**User rules constrain → HMM contextualizes → Bandit learns → Rules decide → Execution scales exposure**

---

## Project Structure

```
strategyEngine/
├── main.py                    # Streamlit dashboard
├── pipeline.py                # Core orchestrator (all 10 layers)
├── layers/                    # 10-layer architecture
│   ├── L0_user_policy/        # Authority layer (immutable)
│   ├── L1_data_fabric/        # Feature engineering
│   ├── L2_regime_intelligence/# Asset-level HMMs
│   ├── L3_strategy_universe/  # Strategy definitions
│   ├── L4_risk_filter/        # Hard constraint filter
│   ├── L5_bandit_learning/    # Contextual Thompson Sampling
│   ├── L6_online_learning/    # O(1) update loop
│   ├── L7_decision_ranking/   # Deterministic final authority
│   ├── L8_position_sizing/    # Volatility-adjusted sizing
│   ├── L9_execution/          # Rebalancing & switching
│   └── L10_monitoring/        # Performance & explanations
└── requirements.txt
```

---

## Layer Overview

| Layer | Name | Purpose |
|-------|------|---------|
| **L0** | User Policy | Immutable constraints (weights, risk limits) |
| **L1** | Data Fabric | Feature engineering (returns, vol, trend) |
| **L2** | Regime Intelligence | Per-asset HMM (4 states) |
| **L3** | Strategy Universe | Static action space |
| **L4** | Risk Filter | Safety gate (non-negotiable) |
| **L5** | Bandit Learning | Global Thompson Sampling |
| **L6** | Online Learning | Incremental updates (O(1)) |
| **L7** | Decision Ranking | Final authority layer |
| **L8** | Position Sizing | Exposure control |
| **L9** | Execution | Rebalancing & switch logic |
| **L10** | Monitoring | Trust & transparency |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run main.py
```

---

## Risk Tolerance Levels

| Level | Max Volatility | Max Drawdown |
|-------|---------------|--------------|
| Low   | 8%            | 5%           |
| Medium| 15%           | 10%          |
| High  | 25%           | 20%          |

---

## Strategies

| Strategy | Risk | Expected Vol | Regimes |
|----------|------|--------------|---------|
| Momentum | Medium | 18% | Trend |
| Mean Reversion | Low | 10% | Range |
| Breakout | High | 22% | Trend |
| Defensive | Low | 6% | All |

---

## Key Design Principles

1. **L0 is Authority** — No downstream layer can override user policy
2. **HMM = Context Only** — Never drives selection directly
3. **Strategies are Fixed** — System learns WHEN, not WHAT
4. **ML ≠ Decisions** — Deterministic ranking (L7) is final authority
5. **Cold Start = Uniform** — No backtest injection
6. **Online Only** — Incremental, stateless updates

---

## Tech Stack

- Python 3.10+
- Streamlit (UI)
- hmmlearn (HMM)
- arch (GARCH)
- scipy (Optimization)
- yfinance (Data)
