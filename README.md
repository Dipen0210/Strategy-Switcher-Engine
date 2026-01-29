# 🧠 Strategy Switcher Engine
## *AI-Driven Adaptive Portfolio Management*

**A 13-Layer Institutional-Grade Trading System** that dynamically switches strategies based on market regimes identified by Hidden Markov Models (HMM) and reinforced by Multi-Armed Bandit (MAB) learning.

---

## 🚀 One-Line Mental Model
**User Rules Constrain → HMM Contextualizes → Bandit Learns → Strategies Compete → Execution Scales.**

---

## ✨ Key Features
- **Adaptive Strategy Switching**: Automatically toggles between Momentum, Mean Reversion, Breakout, and Defensive strategies.
- **Regime Detection (HMM)**: Uses Gaussian HMM to identify 4 latent market states (Bull, Volatile Bull, Sideways, Crisis).
- **Reinforcement Learning (MAB)**: Uses Thompson Sampling to learn which strategies perform best in each regime over time.
- **Risk-First Architecture**: "L0 User Policy" is immutable—risk limits are never violated, regardless of AI predictions.
- **Institutional Execution**: T+1 execution logic, volatility-adjusted position sizing, and transaction cost modeling.

---

## 🏗️ 13-Layer Architecture

| Layer | Name | Responsibility |
|-------|------|----------------|
| **L0** | **User Policy** | The Constitution. Immutable constraints (Risk Limits, Capital). |
| **L1** | **Data Features** | Feature engineering (Returns, Volatility, Trend Strength). |
| **L2** | **Regime Intel** | **HMM Model**. Detects: *Trend+LowVol*, *Trend+HighVol*, *Range*, *Crisis*. |
| **L3** | **Strategy Universe** | The "Menu". Defines logic for Momentum, MeanRev, Breakout, Defensive. |
| **L4** | **Risk Filter** | Hard gate. Blocks strategies exceeding Vol/Drawdown limits. |
| **L5** | **Global Bandit** | **Thompson Sampling**. Learns strategy probability per regime. |
| **L6** | **Deterministic Rank** | Selects the single best strategy for each asset. |
| **L7** | **Position Sizing** | Inverse-volatility sizing to equalize risk contribution. |
| **L8** | **Signal Gen** | Translates decisions into signals (BUY, SELL, LIQUIDATE, REBALANCE). |
| **L9** | **Scheduler** | Switch vs. Drift logic. Prevents excessive churning (hysteresis). |
| **L10** | **Execution** | OMS simulation. Manages orders, fills, and cost basis. |
| **L11** | **Rebalancing** | Portfolio state management and cycle tracking. |
| **L12** | **Performance** | real-time analytics, Sharpe ratios, and attribution. |

---

## 📊 Strategies & Regimes

### Detected Market Regimes (HMM)
| Regime | Description | Best Strategies |
| :--- | :--- | :--- |
| **Trend + Low Vol** | **"Bull Market"** | Momentum, Breakout |
| **Trend + High Vol** | **"Volatile Bull"** | Momentum, Defensive |
| **Range + Low Vol** | **"Sideways"** | Mean Reversion, Defensive |
| **Crisis** | **"Bear / Crash"** | Defensive (Cash/Bonds) |

### Available Strategies
1.  **Momentum**: Rides strong trends (EMA crossovers).
2.  **Mean Reversion**: Buys oversold / Sells overbought (Bollinger Bands + RSI).
3.  **Breakout**: Enters on volatility expansion (Donchian Channels).
4.  **Defensive**: Preserves capital (Low-volatility + Cash).

---

## 🛠️ Tech Stack
-   **Core**: Python 3.10+, Pandas, NumPy
-   **ML/AI**: `hmmlearn` (HMM), `scipy` (Optimization)
-   **UI**: Streamlit (Dashboard)
-   **Data**: yfinance (Real-time market data)

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Dipen0210/Strategy-Switcher-Engine.git
cd Strategy-Switcher-Engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the engine
streamlit run main.py
```

---

## ⚠️ Disclaimer
*This project is for educational and research purposes only. It is not financial advice. Trading involves significant risk.*
