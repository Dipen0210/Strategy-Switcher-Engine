# main.py
"""
Strategy Engine — Streamlit Dashboard

Hierarchical Bandit Architecture:
  Bandit A (regime trust) → Bandit B (strategy ranking) → Bandit C (stock filter)
  HMM detects regimes. Bandits learn from P&L feedback.
"""

from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Layer imports
from pipeline import StrategyEngine, run_engine, PipelineResult
from layers.L0_user_policy import create_policy, UserPolicy, RISK_LIMITS
from layers.L3_strategy_universe import STRATEGY_REGISTRY
from layers.L8_signal_generation import get_signal_history, clear_signal_log
from layers.L10_trade_execution import get_transaction_history, clear_transaction_log
from layers.L11_rebalancing import get_cycle_history, clear_cycle_log

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Strategy Engine",
    page_icon="🎯",
    layout="wide"
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = True
    # Auto-clear trading logs on fresh session (browser refresh / new session)
    # Bandit models are PRESERVED — only in-memory logs are cleared
    clear_signal_log()
    clear_transaction_log()
    clear_cycle_log()


st.title("🎯 Strategy Engine")
st.caption("Hierarchical bandit system with HMM regime detection")

# Fixed stock universe
AVAILABLE_STOCKS = {
    "WMT": "Walmart",
    "JNJ": "Johnson & Johnson",
    "NVDA": "NVIDIA",
    "JPM": "JPMorgan Chase",
    "NEE": "NextEra Energy",
}

# ---------------------------------------------------------
# Sidebar — Configuration Only (compact)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")

# Stock selection - dropdown
selected_tickers = st.sidebar.multiselect(
    "Select Stocks",
    options=list(AVAILABLE_STOCKS.keys()),
    default=[],
    format_func=lambda x: f"{x} - {AVAILABLE_STOCKS[x]}",
    key="stock_selector",
)

if not selected_tickers:
    st.sidebar.warning("Select at least one stock")
    st.info("👈 Select stocks from the sidebar to get started")
    st.stop()

tickers = selected_tickers

# Capital & Risk in sidebar
capital = st.sidebar.number_input("Capital ($)", min_value=1000, value=10000, step=1000)
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=1)
rebalance_freq = st.sidebar.selectbox("Rebalance", ["Daily", "Weekly", "Monthly"], index=1)

# ---------------------------------------------------------
# Date Selection with Trading Day Validation
# ---------------------------------------------------------
from utils.trading_calendar import is_us_business_day, next_trading_day, previous_trading_day, get_trading_dates

# Date input from calendar widget
raw_as_of = st.sidebar.date_input("As of Date", datetime.today())

# Convert date to datetime
if isinstance(raw_as_of, datetime):
    as_of_date = raw_as_of
else:
    as_of_date = datetime.combine(raw_as_of, datetime.min.time())

# Validate: No future dates allowed
today = datetime.today()
if as_of_date.date() > today.date():
    st.sidebar.warning("⚠️ Future dates not supported. Using today.")
    as_of_date = datetime.combine(today.date(), datetime.min.time())

# Get execution and analysis trading days
execution_date, analysis_date = get_trading_dates(as_of_date)

# Show info popup if selected date is NOT a trading day
if execution_date.date() != as_of_date.date():
    st.sidebar.info(
        f"📅 {as_of_date.strftime('%Y-%m-%d')} is not a trading day. "
        f"Trades execute on {execution_date.strftime('%Y-%m-%d')}."
    )

# Show data date being used
st.sidebar.caption(
    f"📊 Data through {analysis_date.strftime('%Y-%m-%d')} (prev trading close)"
)

# Unified data fetch: 60 days covers all strategy requirements
DATA_LOOKBACK_DAYS = 60

# Session state
if "result" not in st.session_state:
    st.session_state.result = None
if "stock_data" not in st.session_state:
    st.session_state.stock_data = {}
if "run_requested" not in st.session_state:
    st.session_state.run_requested = False
if "rebalance_requested" not in st.session_state:
    st.session_state.rebalance_requested = False
if "current_as_of_date" not in st.session_state:
    st.session_state.current_as_of_date = None
if "strategy_engine" not in st.session_state:
    st.session_state.strategy_engine = None

# --- Sidebar Action Buttons ---
st.sidebar.divider()
is_first_run = st.session_state.current_as_of_date is None
button_label = "🚀 Run Strategy" if is_first_run else "🔄 Rebalance (Next Cycle)"
button_type = "primary" if is_first_run else "secondary"

if st.sidebar.button(button_label, width="stretch", type=button_type):
    # Validate Allocation (Must be 100%)
    if "user_weights" in st.session_state and st.session_state.user_weights:
        current_total = sum(st.session_state.user_weights.values())
        if abs(current_total - 100.0) > 0.1:
            st.sidebar.error(f"⚠️ Total Allocation is {current_total:.1f}%!")
            st.sidebar.error("Must be 100%. Please check 'Allocation' tab.")
            st.stop()

    today = date.today()
    
    if st.session_state.current_as_of_date is None:
        st.session_state.current_as_of_date = execution_date
        st.session_state.run_requested = True
    else:
        freq_days = {"Daily": 1, "Weekly": 7, "Monthly": 30}
        days_to_add = freq_days.get(rebalance_freq, 7)
        new_date = st.session_state.current_as_of_date + timedelta(days=days_to_add)
        
        while not is_us_business_day(new_date):
            new_date += timedelta(days=1)
            
        if new_date.date() > today:
            st.sidebar.error(f"⚠️ Cannot advance to {new_date.strftime('%Y-%m-%d')} - no market data for future dates!")
            st.stop()
            
        st.session_state.current_as_of_date = new_date
        st.session_state.rebalance_requested = True
        
    st.rerun()

# Show next rebalance date
if st.session_state.current_as_of_date is not None:
    st.sidebar.caption(f"📆 Current date: **{st.session_state.current_as_of_date.strftime('%Y-%m-%d')}**")
    freq_days = {"Daily": 1, "Weekly": 7, "Monthly": 30}
    next_date = st.session_state.current_as_of_date + timedelta(days=freq_days.get(rebalance_freq, 7))
    while not is_us_business_day(next_date):
        next_date += timedelta(days=1)
    st.sidebar.caption(f"⏭️ Next rebalance: {next_date.strftime('%Y-%m-%d')}")

# ---------------------------------------------------------
# Main Screen — 7 Tabs
# ---------------------------------------------------------
tab_allocation, tab_regime, tab_bandits, tab_strategy, tab_sizing, tab_execution, tab_history = st.tabs([
    "📊 Allocation",
    "🌊 Regime",
    "🧠 Bandits",
    "🎯 Strategy",
    "💰 Sizing",
    "⚡ Execution",
    "📜 History",
])

# ---------------------------------------------------------
# Tab 1: Asset Allocation
# ---------------------------------------------------------
with tab_allocation:
    st.header("Asset Allocation")
    st.markdown(f"**Selected:** {', '.join([f'{t} ({AVAILABLE_STOCKS[t]})' for t in tickers])}")
    
    st.divider()
    
    cols = st.columns(min(len(tickers), 3))
    
    if "user_weights" not in st.session_state:
        st.session_state.user_weights = {}
    
    default_per_stock = 100.0 / len(tickers) if tickers else 0.0
    for ticker in tickers:
        if ticker not in st.session_state.user_weights:
            st.session_state.user_weights[ticker] = default_per_stock
    
    # Remove weights for deselected tickers
    current_tickers = set(tickers)
    stored_tickers = set(st.session_state.user_weights.keys())
    for old_ticker in stored_tickers - current_tickers:
        del st.session_state.user_weights[old_ticker]
    
    def update_weight(ticker):
        key = f"slider_{ticker}"
        if key in st.session_state:
            st.session_state.user_weights[ticker] = st.session_state[key]
    
    for i, ticker in enumerate(tickers):
        col_idx = i % 3
        with cols[col_idx]:
            st.slider(
                f"{ticker} - {AVAILABLE_STOCKS.get(ticker, ticker)}",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.user_weights[ticker],
                step=1.0,
                key=f"slider_{ticker}",
                on_change=update_weight,
                args=(ticker,),
                help="Target allocation %"
            )
    
    weights = st.session_state.user_weights.copy()
    total_allocated = sum(weights.values())
    
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        total = sum(weights.values())
        if abs(total - 100) < 0.1:
            st.success(f"✅ Total: {total:.0f}%")
        else:
            st.error(f"⚠️ Total: {total:.1f}% (Must be 100%)")
            if total > 0:
                if st.button("⚖️ Auto-Normalize to 100%"):
                    factor = 100.0 / total
                    for t in tickers:
                        key = f"weight_v2_{t}"
                        if key in st.session_state:
                            st.session_state[key] = st.session_state[key] * factor
                    st.rerun()
            else:
                st.error("Please allocate weights.")
        
        st.metric("Total Capital", f"${capital:,.0f}")
        
        for ticker, w in weights.items():
            st.write(f"**{ticker}**: ${capital * w / 100:,.0f} ({w:.0f}%)")
    
    with col2:
        if weights:
            alloc_df = pd.DataFrame({
                "Ticker": list(weights.keys()),
                "Allocation": list(weights.values())
            })
            fig = px.pie(
                alloc_df, 
                names="Ticker", 
                values="Allocation",
                title="Portfolio Allocation",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")
    
    st.divider()
    
    # Run logic triggered by sidebar buttons
    if st.session_state.run_requested or st.session_state.rebalance_requested:
        is_rebalance = st.session_state.rebalance_requested
        st.session_state.run_requested = False
        st.session_state.rebalance_requested = False
        
        run_date = st.session_state.current_as_of_date
        st.info(f"📅 Running for date: **{run_date.strftime('%Y-%m-%d')}**")
        
        with st.spinner("Loading market data..."):
            end_date = run_date.strftime("%Y-%m-%d")
            start_date = (run_date - timedelta(days=DATA_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            
            stock_data = {}
            for ticker in tickers:
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
                    if not df.empty:
                        df = df.reset_index()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                        stock_data[ticker] = df
                except Exception as e:
                    st.warning(f"Failed to load {ticker}: {e}")
            
            st.session_state.stock_data = stock_data
        
        if stock_data:
            action = "Rebalancing" if is_rebalance else "Running Strategy Engine"
            with st.spinner(f"{action}..."):
                try:
                    current_weights = []
                    for t in tickers:
                        w = st.session_state.user_weights.get(t, 0.0)
                        current_weights.append(w / 100.0)
                    
                    weight_list = current_weights
                    
                    policy = create_policy(
                        tickers=tickers,
                        weights=weight_list,
                        total_capital=capital,
                        risk_tolerance=risk_tolerance,
                        rebalance_frequency=rebalance_freq,
                    )
                    
                    if not is_rebalance or st.session_state.strategy_engine is None:
                        st.session_state.strategy_engine = StrategyEngine(policy)
                    else:
                        engine = st.session_state.strategy_engine
                        engine.policy = policy
                    
                    pipeline_input_date = previous_trading_day(run_date)
                    
                    result = st.session_state.strategy_engine.run(
                        stock_data, 
                        current_date=pipeline_input_date.strftime("%Y-%m-%d")
                    )
                    
                    st.session_state.result = result
                    action_done = "Rebalance" if is_rebalance else "Strategy"
                    st.success(f"✅ {action_done} completed for {run_date.strftime('%Y-%m-%d')} in {result.execution_time_ms:.0f}ms")
                    
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
        else:
            st.error("No data loaded. Please check tickers.")
    
    if st.session_state.result is None:
        st.info("👈 Click 'Run Strategy' or 'Rebalance' in the sidebar to execute")

# ---------------------------------------------------------
# Tab 2: Regime Detection
# ---------------------------------------------------------
with tab_regime:
    st.header("🌊 Regime Detection")
    st.caption("HMM runs on each stock's data → Bandit A blends with global trust weights → per-stock regime")
    
    result = st.session_state.result
    
    if result and result.regime_output:
        regime_colors = {
            "Bull-Quiet": "🟢", "Bull-Volatile": "🟡",
            "Sideways": "🔵", "Crisis": "🔴"
        }
        color_map = {
            "Bull-Quiet": "#22c55e", "Bull-Volatile": "#eab308",
            "Sideways": "#3b82f6", "Crisis": "#ef4444"
        }
        
        # Summary: count how many stocks in each regime
        regime_counts = {}
        for ticker, info in result.regime_output.items():
            r = info.get("dominant_regime", "Unknown")
            regime_counts[r] = regime_counts.get(r, 0) + 1
        
        summary_cols = st.columns(len(regime_counts))
        for i, (regime, count) in enumerate(regime_counts.items()):
            summary_cols[i].metric(
                f"{regime_colors.get(regime, '⚪')} {regime}",
                f"{count} stock{'s' if count > 1 else ''}"
            )
        
        st.divider()
        
        # Per-stock regime details
        for ticker, regime_info in result.regime_output.items():
            dominant = regime_info.get("dominant_regime", "Unknown")
            hmm_conf = regime_info.get("hmm_confidence", 0.0)
            stability = regime_info.get("stability_score", 0.0)
            is_ambiguous = regime_info.get("is_ambiguous", False)
            transition = regime_info.get("transition_flag", False)
            probs = regime_info.get("probabilities", {})
            
            icon = regime_colors.get(dominant, "⚪")
            status = "⚠️ Ambiguous" if is_ambiguous else ("🔄 Transition" if transition else "✅ Stable")
            
            with st.expander(f"{icon} **{ticker}** — {dominant}  |  Conf: {hmm_conf:.1%}  |  {status}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Regime", f"{icon} {dominant}")
                c2.metric("HMM Confidence", f"{hmm_conf:.1%}")
                c3.metric("Stability", f"{stability:.2f}")
                c4.metric("Status", status)
                
                if is_ambiguous:
                    st.warning("⚠️ Confidence < 0.55 — ambiguous regime, defensive bias applied.")
                
                if probs:
                    prob_df = pd.DataFrame({
                        "Regime": list(probs.keys()),
                        "Probability": list(probs.values())
                    }).sort_values("Probability", ascending=True)
                    
                    fig = px.bar(
                        prob_df, y="Regime", x="Probability",
                        orientation="h",
                        text=prob_df["Probability"].apply(lambda x: f"{x:.1%}"),
                        color="Regime",
                        color_discrete_map=color_map,
                    )
                    fig.update_layout(
                        height=200, showlegend=False,
                        xaxis_title="Blended Probability (HMM + Global Trust)",
                        xaxis=dict(tickformat=".0%", range=[0, 1]),
                        margin=dict(l=0, r=0, t=0, b=0),
                    )
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True, key=f"regime_chart_{ticker}")
    else:
        st.info("👈 Run the strategy engine to see regime detection results")

# ---------------------------------------------------------
# Tab 3: Bandits (Hierarchical 3-Level)
# ---------------------------------------------------------
# Tab 3: Bandits (Hierarchical 3-Level)
# ---------------------------------------------------------
with tab_bandits:
    st.header("🧠 Hierarchical Bandit System")
    st.caption("Three bandits learn at different speeds from P&L feedback")
    
    engine = st.session_state.get("strategy_engine")
    result = st.session_state.result
    
    # helper to get available tickers
    available_tickers = []
    if result and result.per_stock_strategies:
        available_tickers = sorted(list(result.per_stock_strategies.keys()))
    elif engine and engine.ensemble_bandits.stock_bandits:
        available_tickers = sorted(list(engine.ensemble_bandits.stock_bandits.bandits.keys()))
        
    selected_ticker = st.selectbox(
        "Select Stock to Inspect:", 
        available_tickers if available_tickers else ["No Data"],
        index=0 if available_tickers else 0
    )
    
    # Context: What is this stock doing?
    current_regime = "Sideways"
    confidence = 0.0
    stability = 0.0
    
    if result and result.regime_output and selected_ticker in result.regime_output:
        info = result.regime_output[selected_ticker]
        current_regime = info["dominant_regime"]
        confidence = info.get("hmm_confidence", 0.0)
        stability = info.get("stability_score", 0.0)
        
    st.info(f"📊 **Context for {selected_ticker}**: Regime = **{current_regime}** (Conf: {confidence:.1%}, Stability: {stability:.2f})")
    
    st.divider()

    candidates = []
    all_b_weights = {}
    if result and result.per_stock_details and selected_ticker in result.per_stock_details:
        candidates = result.per_stock_details[selected_ticker].get("candidates", [])
        all_b_weights = result.per_stock_details[selected_ticker].get("all_bandit_b_weights", {})

    # ---- BANDIT B: ALL Strategy Weights in Regime ----
    st.subheader(f"🅱️ Bandit B (Strategy Weights) for {current_regime}")
    st.caption(f"All strategies in the {current_regime} regime with their learned RL weights (sum = 100%). Top 5 are sent to Bandit C.")
    
    if all_b_weights:
        df_b = pd.DataFrame([
            {"Strategy": name, "Weight (θ_B)": weight}
            for name, weight in all_b_weights.items()
        ])
        df_b = df_b.sort_values("Weight (θ_B)", ascending=False).reset_index(drop=True)
        
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            fig = px.bar(
                df_b, x="Strategy", y="Weight (θ_B)",
                text=df_b["Weight (θ_B)"].apply(lambda x: f"{x:.4f}"), 
                color="Weight (θ_B)",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True, key=f"bandit_b_chart_{selected_ticker}")
        with col_b2:
            display_b = df_b.copy()
            display_b["Weight (θ_B)"] = display_b["Weight (θ_B)"].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_b, hide_index=True, use_container_width=True)
            st.caption(f"Total: {sum(all_b_weights.values()):.4f}")
    else:
        st.info("No Strategy Bandit data found. Run the engine.")
    
    st.divider()
    
    # ---- BANDIT C: Top 5 from B → Historical Evaluation ----
    st.subheader(f"©️ Bandit C (Stock Preference & Final Eval) for {selected_ticker}")
    st.caption(f"Top 5 strategies from Bandit B with ε-greedy. Final Score = 30% θ_B + 40% Norm Risk-Adj Ret + 30% θ_C.")
    
    if candidates:
        df_c = pd.DataFrame(candidates)[["Strategy", "θ_B", "Past_Return", "θ_C", "Score"]]
        df_c = df_c.rename(columns={
            "θ_B": "Bandit B Weight (θ_B)",
            "Past_Return": "Risk-Adj Return (Sharpe)",
            "θ_C": "Stock Preference (θ_C)",
            "Score": "Final Score (Linear Combo)"
        })
        # Sort by Final Score (highest to lowest)
        df_c = df_c.sort_values("Final Score (Linear Combo)", ascending=False).reset_index(drop=True)
        st.dataframe(df_c, hide_index=True, use_container_width=True)
    else:
        st.info("No Stock Bandit evaluation data found. Run the engine.")
    
    st.divider()
    
    # ---- BANDIT C: Learned Weights Visualization ----
    st.subheader(f"📊 Bandit C Learned Weights: {selected_ticker} × {current_regime}")
    st.caption(f"All strategy weights stored in the {selected_ticker}_{current_regime.replace('-','_')} model (sum = 100%). Updated after each feedback cycle.")
    
    engine = st.session_state.get("strategy_engine")
    if engine and engine.ensemble_bandits.stock_bandits:
        sb = engine.ensemble_bandits.stock_bandits.get_bandit(selected_ticker, current_regime)
        if sb.strategies:
            df_weights = pd.DataFrame([
                {"Strategy": name, "Learned Weight (θ_C)": weight}
                for name, weight in sb.strategies.items()
            ])
            df_weights = df_weights.sort_values("Learned Weight (θ_C)", ascending=False).reset_index(drop=True)
            
            col_c1, col_c2 = st.columns([2, 1])
            with col_c1:
                fig = px.bar(
                    df_weights, x="Strategy", y="Learned Weight (θ_C)",
                    text=df_weights["Learned Weight (θ_C)"].apply(lambda x: f"{x:.4f}"),
                    color="Learned Weight (θ_C)",
                    color_continuous_scale="Plasma"
                )
                fig.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True, key=f"bandit_c_weights_{selected_ticker}_{current_regime}")
            with col_c2:
                display_w = df_weights.copy()
                display_w["Learned Weight (θ_C)"] = display_w["Learned Weight (θ_C)"].apply(lambda x: f"{x:.4f}")
                st.dataframe(display_w, hide_index=True, use_container_width=True)
                st.caption(f"Total: {sum(sb.strategies.values()):.4f}")
        else:
            st.info(f"No learned weights yet for {selected_ticker} in {current_regime}. Run the engine.")
    else:
        st.info("Engine not loaded. Run the engine to see learned weights.")

# ---------------------------------------------------------
# Tab 4: Strategy Selection (per-stock summary)
# ---------------------------------------------------------
with tab_strategy:
    st.header("🎯 Strategy Selection")
    
    result = st.session_state.result
    
    if result:
        # Header metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Primary Strategy", result.selected_strategy)
        col2.metric("Regime", result.dominant_regime)
        col3.metric("Execution Time", f"{result.execution_time_ms:.0f}ms")
        
        st.divider()
        
        # Per-stock strategy assignment table
        if result.per_stock_strategies:
            st.subheader("Per-Stock Strategy Assignment")
            
            strategy_data = []
            for ticker, strategy in result.per_stock_strategies.items():
                details = (result.per_stock_details or {}).get(ticker, {})
                scores = details.get("scores", {})
                current_score = scores.get(strategy, 0.0)
                theta_c = details.get("winner_theta_c", 0.0)
                hmm_conf = details.get("hmm_confidence", 0.0)
                stab = details.get("stability", 0.0)
                regime = details.get("regime", result.dominant_regime)
                allowed = details.get("allowed", [])
                
                strategy_data.append({
                    "Ticker": ticker,
                    "Regime": regime,
                    "Winner": strategy,
                    "Score": f"{current_score:.3f}",
                    "θ_C": f"{theta_c:.3f}",
                    "HMM Conf": f"{hmm_conf:.1%}",
                    "Stability": f"{stab:.2f}",
                    "# Available": len(allowed),
                })
            
            strategy_df = pd.DataFrame(strategy_data)
            st.dataframe(strategy_df, hide_index=True, use_container_width=True)
        
        # Strategy switch decision
        if result.switch_decision:
            st.divider()
            st.subheader("🔄 Strategy Switch Decision")
            switch = result.switch_decision
            if hasattr(switch, 'should_switch'):
                if switch.should_switch:
                    st.success(f"✅ Strategy switch approved: {switch.reason if hasattr(switch, 'reason') else ''}")
                else:
                    st.warning(f"⏸️ Strategy switch blocked: {switch.reason if hasattr(switch, 'reason') else 'Cooldown/threshold'}")
    else:
        st.info("👈 Run the engine to see strategy selection")

# ---------------------------------------------------------
# Tab 5: Position Sizing
# ---------------------------------------------------------
with tab_sizing:
    st.header("💰 Position Sizing")
    
    result = st.session_state.result
    
    if result:
        if result.position_sizes is not None and not result.position_sizes.empty:
            pos_df = result.position_sizes.copy()
            
            display_df = pos_df.copy()
            rename_map = {
                "User_Weight": "Your Input",
                "Adjusted_Weight": "Volatility Sized",
                "Capital_Allocation": "Final $ Allocation"
            }
            display_df = display_df.rename(columns=rename_map)
            
            for col in ["Your Input", "Volatility Sized"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
            if "Final $ Allocation" in display_df.columns:
                display_df["Final $ Allocation"] = display_df["Final $ Allocation"].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            st.divider()
            st.subheader("⚖️ Sizing Impact: Input vs. Risk-Adjusted")
            
            if "Ticker" in pos_df.columns:
                chart_data = pos_df.melt(
                    id_vars=["Ticker"], 
                    value_vars=["User_Weight", "Adjusted_Weight"],
                    var_name="Type",
                    value_name="Weight"
                )
                chart_data["Type"] = chart_data["Type"].map({
                    "User_Weight": "Your Input", 
                    "Adjusted_Weight": "Volatility Sized"
                })
                
                fig = px.bar(
                    chart_data,
                    x="Ticker", y="Weight", color="Type",
                    barmode="group",
                    title="Comparison: User Intent vs. Risk Sizing",
                    text_auto=".1%",
                    color_discrete_map={"Your Input": "#3366CC", "Volatility Sized": "#DC3912"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No position sizing data available")
    else:
        st.info("👈 Run the engine to see position sizing")

# ---------------------------------------------------------
# Tab 6: Execution (Signals + Trade Execution)
# ---------------------------------------------------------
with tab_execution:
    st.header("⚡ Execution")
    
    result = st.session_state.result
    
    if result:
        # --- Signals Section ---
        if result.signals_df is not None and not result.signals_df.empty:
            st.subheader("📡 Trade Signals")
            
            def signal_color(signal):
                colors = {
                    "BUY": "🟢", "SELL": "🔴", "REBALANCE": "🔵",
                    "HOLD": "⚪"
                }
                return colors.get(str(signal).upper(), "⚫")
            
            signals_display = result.signals_df.copy()
            if "Signal" in signals_display.columns:
                signals_display["Action"] = signals_display["Signal"].apply(signal_color) + " " + signals_display["Signal"].astype(str)
            
            # Signal summary metrics
            if "Signal" in result.signals_df.columns:
                signal_counts = result.signals_df["Signal"].value_counts().to_dict()
                cols = st.columns(len(signal_counts))
                for i, (signal, count) in enumerate(signal_counts.items()):
                    cols[i].metric(signal, count)
            
            st.dataframe(signals_display, hide_index=True, use_container_width=True)
        else:
            st.info("No trade signals generated (portfolio unchanged)")
        
        st.divider()
        
        # --- Execution Report ---
        if result.execution_report:
            st.subheader("📋 Execution Report")
            report = result.execution_report
            
            if "error" in report:
                st.error(f"Execution Error: {report['error']}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Orders Executed", report.get("orders_executed", 0))
                col2.metric("Total Volume", f"${report.get('total_volume', 0):,.2f}")
                col3.metric("Fees Paid", f"${report.get('fees_paid', 0):,.2f}")
                col4.metric("Cash After", f"${report.get('cash_after', 0):,.2f}")
                
                orders = report.get("orders")
                if orders is not None:
                    if hasattr(orders, 'empty'):
                        has_orders = not orders.empty
                        orders_df = orders
                    else:
                        has_orders = bool(orders)
                        orders_df = pd.DataFrame(orders) if orders else pd.DataFrame()
                    
                    if has_orders:
                        st.dataframe(orders_df, hide_index=True, use_container_width=True)
        else:
            st.info("No execution report available")
    else:
        st.info("👈 Run the engine to see execution details")

# ---------------------------------------------------------
# Tab 7: History (Portfolio + Performance + Trading Logs)
# ---------------------------------------------------------
with tab_history:
    st.header("📜 History & Performance")
    
    result = st.session_state.result
    
    # --- Portfolio State ---
    if result and result.portfolio_state:
        state = result.portfolio_state
        
        col1, col2, col3, col4 = st.columns(4)
        
        realized_pnl = state.get("realized_pnl", 0)
        pnl_pct = (realized_pnl / capital) * 100 if capital else 0
        
        col1.metric("💵 Cash", f"${state.get('cash', 0):,.2f}")
        col2.metric("📈 Realized P&L", f"${realized_pnl:,.2f}", f"{pnl_pct:+.2f}%")
        col3.metric("📊 Unrealized P&L", f"${state.get('unrealized_pnl', 0):,.2f}")
        col4.metric("🔄 Total Trades", state.get("num_trades", 0))
        
        # Current positions
        positions = state.get("positions", {})
        if positions:
            pos_items = [{"Ticker": t, "Shares": q} for t, q in positions.items() if q > 0.001]
            if pos_items:
                st.divider()
                st.subheader("Current Positions")
                st.dataframe(pd.DataFrame(pos_items), hide_index=True, use_container_width=True)
        
        fees = state.get("fees_paid", 0)
        if fees > 0:
            st.caption(f"Total fees paid: ${fees:,.2f}")
    
    st.divider()
    
    # --- Trading Logs (3 tables) ---
    st.subheader("📋 Trading Logs")
    
    # Signal History
    with st.expander("📊 Signal History", expanded=False):
        history_df = get_signal_history()
        if not history_df.empty:
            st.dataframe(
                history_df.sort_values("Timestamp", ascending=True) if "Timestamp" in history_df.columns else history_df,
                use_container_width=True, height=250
            )
        else:
            st.info("No signal history yet.")
    
    # Transaction History
    with st.expander("💰 Transactions", expanded=False):
        transactions_df = get_transaction_history()
        if not transactions_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", len(transactions_df))
            total_cost = transactions_df["Transaction_Cost"].sum() if "Transaction_Cost" in transactions_df.columns else 0
            col2.metric("Total Costs", f"${total_cost:.2f}")
            act_col = "Side" if "Side" in transactions_df.columns else "Action"
            buy_count = len(transactions_df[transactions_df[act_col] == "BUY"]) if act_col in transactions_df.columns else 0
            col3.metric("Buys", buy_count)
            sell_count = len(transactions_df[transactions_df[act_col] == "SELL"]) if act_col in transactions_df.columns else 0
            col4.metric("Sells", sell_count)
            
            st.dataframe(
                transactions_df.sort_values("Timestamp", ascending=True) if "Timestamp" in transactions_df.columns else transactions_df,
                use_container_width=True, height=250
            )
        else:
            st.info("No transactions yet.")
    
    # Rebalance Cycles
    with st.expander("🔄 Rebalance Cycles", expanded=False):
        cycles_df = get_cycle_history()
        if not cycles_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Cycles", len(cycles_df))
            total_costs = cycles_df["Transaction_Costs"].sum() if "Transaction_Costs" in cycles_df.columns else 0
            col2.metric("Total Costs", f"${total_costs:.2f}")
            latest_value = cycles_df["Current_Position"].iloc[-1] if "Current_Position" in cycles_df.columns else 0
            col3.metric("Latest Position", f"${latest_value:,.2f}")
            total_pnl = cycles_df["P/L"].sum() if "P/L" in cycles_df.columns else 0
            col4.metric("Total P/L", f"${total_pnl:,.2f}")
            
            st.dataframe(cycles_df, use_container_width=True, height=200)
        else:
            st.info("No rebalance cycles yet.")

# ---------------------------------------------------------
# Sidebar — Reset Button
# ---------------------------------------------------------
st.sidebar.divider()
if st.sidebar.button("🔄 Reset All", width="stretch"):
    # Clear trading logs
    clear_signal_log()
    clear_transaction_log()
    clear_cycle_log()
    
    import shutil, os
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    
    # All bandit state (A, B, C) is PRESERVED — learned knowledge survives resets
    
    # Clear session state and caches
    st.session_state.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

