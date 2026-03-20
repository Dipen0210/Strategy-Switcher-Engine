# 🧠 Strategy Switcher Engine
## *AI-Driven Adaptive Portfolio Management*

**A 13-Layer Institutional-Grade Trading System** that dynamically switches between **40 specialized strategies** across market regimes identified by Hidden Markov Models (HMM) and refined through a **3-Layer Multi-Armed Bandit** learning hierarchy.

---

## 🚀 One-Line Mental Model
**User Rules Constrain → HMM Detects Regime → 3-Layer Bandit Ranks, Evaluates & Learns → Best Strategy Wins → Execution Scales.**

---

## ✨ Key Features
- **40 Specialized Strategies**: 10 per regime across Bull-Quiet, Bull-Volatile, Sideways, and Crisis — each with its own `TIMEFRAME` lookback window.
- **Regime Detection (HMM)**: Gaussian HMM identifies 4 latent market states with confidence scoring and stability tracking.
- **3-Layer Bandit System**: Hierarchical RL with Global → Strategy → Stock bandits.
  - **EXP3 Math**: Multiplicative weight updates prevent outlier dominance (`weight *= exp(LR * Return / Temp)`).
  - **Memory Decay**: A `0.99` decay factor pulls all weights toward uniform distribution over time, allowing the system to gradually "forget" old regimes/strategies and adapt to new ones.
  - **ε-greedy Exploration**: 10% chance to test underperforming strategies to ensure they aren't permanently ignored.
- **Per-Strategy Timeframes**: Each strategy defines its own lookback (5–200 days). Past returns use each strategy's native window.
- **Per-Stock-Per-Regime Models**: Stock Bandit stores separate models for each ticker × regime combo (max ~10 strategies each), preventing weight dilution.
- **Risk-First Architecture**: L0 User Policy is immutable — risk limits are never violated regardless of AI predictions.
- **Institutional Execution**: T+1 execution, volatility-adjusted position sizing, and transaction cost modeling.

---

## 🏗️ 13-Layer Architecture

| Layer | Name | Responsibility |
|-------|------|----------------|
| **L0** | **User Policy** | The Constitution. Immutable constraints (Risk Limits, Capital). |
| **L1** | **Data Features** | Feature engineering (Returns, Volatility, Trend Strength). |
| **L2** | **Regime Intel** | **HMM Model**. Detects: *Bull-Quiet*, *Bull-Volatile*, *Sideways*, *Crisis*. |
| **L3** | **Strategy Universe** | 40 strategies organized into 4 regime pods (10 each), each with its own `TIMEFRAME`. |
| **L4** | **Risk Filter** | Hard gate. Blocks strategies exceeding Vol/Drawdown limits. |
| **L5** | **Bandit System** | **3-Layer Hierarchy** — see below. |
| **L6** | **Deterministic Rank** | Selects the single best strategy per asset from Bandit C output. |
| **L7** | **Position Sizing** | Inverse-volatility sizing to equalize risk contribution. |
| **L8** | **Signal Gen** | Translates decisions into signals (BUY, SELL, LIQUIDATE, REBALANCE). |
| **L9** | **Scheduler** | Switch vs. Drift logic. Prevents excessive churning (hysteresis). |
| **L10** | **Execution** | OMS simulation. Manages orders, fills, and cost basis. |
| **L11** | **Rebalancing** | Portfolio state management and cycle tracking. |
| **L12** | **Performance** | Real-time analytics, Sharpe ratios, and attribution. |

---

## 🎰 3-Layer Bandit Hierarchy (L5)

```
┌─────────────────────────────────────────────────────────┐
│  Bandit A (Global)         LR = 0.1 (Slow)             │
│  "Trust this regime?"      W_Global=0.60, W_HMM=0.40   │
│  Blends HMM + Bandit scores                            │
├─────────────────────────────────────────────────────────┤
│  Bandit B (Strategy)       LR = 0.5 (Mild)             │
│  "Which strategy is best   Random unequal θ_B weights   │
│   in this regime?"         summing to 100%, Top 5 → C   │
├─────────────────────────────────────────────────────────┤
│  Bandit C (Stock)          LR = 1.0 (Harsh)            │
│  "How does this strategy   Per-stock-per-regime models   │
│   perform on THIS stock?"  θ_C weights, per-strategy TF │
│                            Final Score = θ_B × Return × θ_C   │
└─────────────────────────────────────────────────────────┘
```

### Bandit A — Global Regime Trust
- Blends HMM confidence and Bandit's learned regime score
- `Final Score = (0.60 × Global Bandit) + (0.40 × HMM Score)`
- **EXP3 learning** (0.1 LR) — multiplicative updates for stable evolution
- **Decay Factor** (0.99) — slowly pulls trust weights toward uniform (25% each) if a regime is inactive

### Bandit B — Strategy Selection
- Ranks **all** strategies in the detected regime by learned weights (θ_B)
- **ε-greedy exploration**: 10% of the time, tests an outside strategy
- Passes **Top 5** strategies to Bandit C for stock-level evaluation
- **EXP3 learning** (0.5 LR) — mild learner

### Bandit C — Stock Preference
- Evaluates Top 5 from Bandit B on each specific stock
- **Per-stock-per-regime models**: `JNJ_Crisis.pkl`, `NVDA_Sideways.pkl`, etc.
- Max ~10 strategies per model → concentrated weights → faster learning
- **Sharpe-like Returns**: `Mean / StdDev` of daily returns over each strategy's `TIMEFRAME`
- **Linear Combination Scoring**: `Final Score = (0.3 × θ_B) + (0.4 × Norm Ret) + (0.3 × θ_C)`
- **EXP3 learning** (1.0 LR) — fast adaptation to stock-level performance
- **Decay Factor** (0.99) — slowly forgets old stock-specific history

### Feedback Loop
```
Actual P&L → Stock Bandit C (LR=1.0) → Strategy Bandit B (LR=0.5) → Global Bandit A (LR=0.1)
```

---

## 📊 Strategies & Regimes

### Detected Market Regimes (HMM)
| Regime | Description | # Strategies | Example Strategies |
| :--- | :--- | :---: | :--- |
| **Bull-Quiet** | Steady uptrend, low vol | 10 | Momentum, Trend Following, Ichimoku Cloud |
| **Bull-Volatile** | Uptrend with swings | 10 | Mean Reversion, Pullback Buyer, Fibonacci |
| **Sideways** | Range-bound market | 10 | Grid Trading, Bollinger Reversion, Pivot Point |
| **Crisis** | Bear market / crash | 10 | Cash Defensive, Tail Hedge, Death Cross |

### Strategy Timeframes
Each strategy defines its own `TIMEFRAME` lookback period:

| Range | Examples |
|-------|----------|
| **5 days** | Pivot Point |
| **20-25 days** | Williams %R, Stochastic, ATR Trailing, Bollinger |
| **30-35 days** | Grid Trading, Cash Defensive, MACD, VIX Spike |
| **50-60 days** | Trend Following, Momentum, Fibonacci, Tail Hedge |
| **200 days** | Death Cross |

---

### Demo

<img width="1470" height="752" alt="Screenshot 2026-03-20 at 5 59 32 PM" src="https://github.com/user-attachments/assets/10cd3d2c-c4e8-4648-86f5-87d922d1811c" />

<img width="1248" height="530" alt="Screenshot 2026-03-20 at 5 56 21 PM" src="https://github.com/user-attachments/assets/746c500d-77df-4c79-866e-35929dcc2e18" />

<img width="1246" height="580" alt="Screenshot 2026-03-20 at 5 56 43 PM" src="https://github.com/user-attachments/assets/37249ecf-ec3c-47a7-8907-5d42608b2bca" />

<img width="1249" height="679" alt="Screenshot 2026-03-20 at 5 57 17 PM" src="https://github.com/user-attachments/assets/9bbabb79-be16-4958-b126-e51496f01024" />

<img width="1250" height="662" alt="Screenshot 2026-03-20 at 5 57 27 PM" src="https://github.com/user-attachments/assets/566f6879-2493-4000-ad52-2782d960a128" />

<img width="1253" height="274" alt="Screenshot 2026-03-20 at 5 57 35 PM" src="https://github.com/user-attachments/assets/f4479d24-fbc0-452a-9eb9-601d4c48795e" />

<img width="1250" height="636" alt="Screenshot 2026-03-20 at 5 57 42 PM" src="https://github.com/user-attachments/assets/787002fa-2b8c-412e-85dd-0c2ef9ed7261" />

<img width="1250" height="721" alt="Screenshot 2026-03-20 at 5 57 51 PM" src="https://github.com/user-attachments/assets/bb2f3970-e01d-4e8c-913f-f25c5a519b2d" />

<img width="1243" height="432" alt="Screenshot 2026-03-20 at 5 58 01 PM" src="https://github.com/user-attachments/assets/9bc809c7-f30a-4796-985f-9227b74806a2" />

<img width="1254" height="618" alt="Screenshot 2026-03-20 at 5 58 33 PM" src="https://github.com/user-attachments/assets/12ba3f30-8b3c-4f60-8f4e-8f70feff1dbb" />

<img width="1251" height="303" alt="Screenshot 2026-03-20 at 5 58 40 PM" src="https://github.com/user-attachments/assets/bddcea8b-4c02-47c7-9c42-9254e463e339" />












---

## 📁 Model Persistence

```
layers/L5_bandit/data/
├── global_bandit.pkl              # Bandit A
├── regime_bandits/
│   ├── Bull-Quiet.pkl             # Bandit B per regime
│   ├── Sideways.pkl
│   └── Crisis.pkl
└── stock_bandits/
    ├── JNJ/
    │   ├── JNJ_Bull_Quiet.pkl     # Bandit C per stock × regime
    │   └── JNJ_Sideways.pkl
    ├── NVDA/
    │   └── NVDA_Crisis.pkl
    └── ...
```

Bandit models are **preserved across sessions** — they learn gradually over time. Only trading logs (in-memory) are cleared on session restart.

---

## 🛠️ Tech Stack
-   **Core**: Python 3.10+, Pandas, NumPy
-   **ML/AI**: `hmmlearn` (HMM), Multi-Armed Bandit (explicit weight RL)
-   **UI**: Streamlit (Dashboard with Plotly charts)
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
