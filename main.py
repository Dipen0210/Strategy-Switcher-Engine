# main.py
"""
Strategy Engine — Streamlit Dashboard

10-Layer automated strategy switching system with:
- HMM regime detection
- Contextual bandit learning
- Risk-constrained strategy selection
- Volatility-based position sizing
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
# Auto-Reset Logs on Page Refresh (New Session)
# ---------------------------------------------------------
if "session_initialized" not in st.session_state:
    # This block runs only once when the user opens/refreshes the page
    clear_cycle_log()
    clear_transaction_log()
    clear_signal_log()
    st.session_state.session_initialized = True
    # Initial clear feedback (optional, logging to console)
    print("Logs cleared for new session.")

st.title("🎯 Strategy Engine")
st.caption("Automated strategy switching with HMM regime detection")

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
    key="stock_selector",  # Explicit key for reset
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
# Momentum: 60, Mean Reversion: 30, Breakout: 20, Defensive: 60
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
    st.session_state.current_as_of_date = None  # Will be set on first Run
if "strategy_engine" not in st.session_state:
    st.session_state.strategy_engine = None  # Persists PortfolioState across cycles

# --- Sidebar Action Buttons ---
st.sidebar.divider()
# Determine button label based on state
is_first_run = st.session_state.current_as_of_date is None
button_label = "🚀 Run Strategy" if is_first_run else "🔄 Rebalance (Next Cycle)"
button_type = "primary" if is_first_run else "secondary"

if st.sidebar.button(button_label, use_container_width=True, type=button_type):
    # 0. Validate Allocation (Must be 100%)
    # Read from the authoritative user_weights dict
    if "user_weights" in st.session_state and st.session_state.user_weights:
        current_total = sum(st.session_state.user_weights.values())
        
        if abs(current_total - 100.0) > 0.1:
            st.sidebar.error(f"⚠️ Total Allocation is {current_total:.1f}%!")
            st.sidebar.error("Must be 100%. Please check 'Allocation' tab.")
            st.stop()

    today = date.today()
    
    if st.session_state.current_as_of_date is None:
        # FIRST RUN: Use selected "As of Date" (Execution Date)
        # Fix: Start on the actual trading day (e.g., Monday 12th), not the data day (Friday 9th)
        st.session_state.current_as_of_date = execution_date
        st.session_state.run_requested = True
    else:
        # REBALANCE: Advance date
        freq_days = {"Daily": 1, "Weekly": 7, "Monthly": 30}
        days_to_add = freq_days.get(rebalance_freq, 7)
        new_date = st.session_state.current_as_of_date + timedelta(days=days_to_add)
        
        # Skip non-business days
        while not is_us_business_day(new_date):
            new_date += timedelta(days=1)
            
        # Stop if future date
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
# Main Screen — Tabbed Interface
# ---------------------------------------------------------
tab_allocation, tab_strategy, tab_regime, tab_positions, tab_signals, tab_execution, tab_portfolio, tab_performance, tab_history = st.tabs([
    "📊 Allocation",
    "🎯 Strategy", 
    "🌊 Regime",
    "💰 Sizing",
    "📡 Signals",
    "⚡ Execution",
    "📋 Portfolio",
    "📈 Performance",
    "📜 History"
])

# ---------------------------------------------------------
# Tab 1: Asset Allocation (moved from sidebar)
# ---------------------------------------------------------
with tab_allocation:
    st.header("Asset Allocation")
    st.markdown(f"**Selected:** {', '.join([f'{t} ({AVAILABLE_STOCKS[t]})' for t in tickers])}")
    
    st.divider()
    
    # Create columns for sliders
    cols = st.columns(min(len(tickers), 3))
    
    # =======================================================
    # WEIGHT STORAGE: Use a dedicated session state dict
    # =======================================================
    if "user_weights" not in st.session_state:
        st.session_state.user_weights = {}
    
    # Initialize defaults for any new tickers
    default_per_stock = 100.0 / len(tickers) if tickers else 0.0
    for ticker in tickers:
        if ticker not in st.session_state.user_weights:
            st.session_state.user_weights[ticker] = default_per_stock
    
    # Remove weights for deselected tickers
    current_tickers = set(tickers)
    stored_tickers = set(st.session_state.user_weights.keys())
    for old_ticker in stored_tickers - current_tickers:
        del st.session_state.user_weights[old_ticker]
    
    # Callback function for slider changes
    def update_weight(ticker):
        key = f"slider_{ticker}"
        if key in st.session_state:
            st.session_state.user_weights[ticker] = st.session_state[key]
    
    # Render sliders with callbacks
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
    
    # Read weights from the authoritative source
    weights = st.session_state.user_weights.copy()
    total_allocated = sum(weights.values())
    
    st.divider()
    
    # Allocation summary with pie chart
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
                        # Update session state for next rerun
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
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Run logic triggered by sidebar buttons
    if st.session_state.run_requested or st.session_state.rebalance_requested:
        is_rebalance = st.session_state.rebalance_requested
        st.session_state.run_requested = False
        st.session_state.rebalance_requested = False
        
        # Use the tracked date (advances on rebalance)
        run_date = st.session_state.current_as_of_date
        
        # Display which date we're running on
        st.info(f"📅 Running for date: **{run_date.strftime('%Y-%m-%d')}**")
        
        with st.spinner("Loading market data..."):
            # Use run_date for data end point
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
                    # RE-FETCH weights from session state to ensure fresh inputs
                    # NOW READING STRICTLY FROM THE 'user_weights' DICT
                    current_weights = []
                    
                    # Ensure we iterate in the same order as 'tickers' passed to create_policy
                    for t in tickers:
                        # Default to 0 if not found (should be caught by 100% check earlier)
                        w = st.session_state.user_weights.get(t, 0.0)
                        current_weights.append(w / 100.0)
                    
                    weight_list = current_weights
                    
                    # Create policy
                    policy = create_policy(
                        tickers=tickers,
                        weights=weight_list,
                        total_capital=capital,
                        risk_tolerance=risk_tolerance,
                        rebalance_frequency=rebalance_freq,
                    )
                    
                    # On RUN: Create new engine (fresh start)
                    # On REBALANCE: Reuse existing engine (preserves PortfolioState for P/L)
                    if not is_rebalance or st.session_state.strategy_engine is None:
                        # Create fresh engine on Run button
                        st.session_state.strategy_engine = StrategyEngine(policy)
                    else:
                        # Rebalance: update policy but keep existing PortfolioState
                        engine = st.session_state.strategy_engine
                        engine.policy = policy
                    
                    # Run the engine
                    # Pipeline calculates execution as T+1 from 'current_date'.
                    # User picked 'run_date' as the EXECUTION date.
                    # So we pass T-1 to the pipeline.
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
    
    # Show hint if no result
    if st.session_state.result is None:
        st.info("👈 Click 'Run Strategy' or 'Rebalance' in the sidebar to execute")

# ---------------------------------------------------------
# Tab 2: Strategy Selection
# ---------------------------------------------------------
with tab_strategy:
    st.header("Strategy Selection")
    
    result = st.session_state.result
    
    if result:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Selected Strategy",
                result.selected_strategy,
                help=getattr(result.strategy_decision, 'selection_reason', '')
            )
        
        with col2:
            exp_ret = getattr(result.strategy_decision, 'expected_return', 0.1)
            st.metric("Expected Return", f"{exp_ret:.1%}" if isinstance(exp_ret, float) else "N/A")
        
        with col3:
            st.metric("Risk Level", getattr(result.strategy_decision, 'risk_level', risk_tolerance))
        
        st.divider()
        
        st.divider()
        st.info("ℹ️ Strategy selection, filtering, and scoring is performed independently for each stock based on its regime.")
        
        
        # Per-stock strategy assignment table
        if result.per_stock_strategies:
            st.divider()
            st.subheader("📊 Per-Stock Strategy Assignment")
            
            strategy_data = []
            for ticker, strategy in result.per_stock_strategies.items():
                regime_info = result.regime_output.get(ticker, {})
                regime = regime_info.get("dominant_regime", "Unknown") if isinstance(regime_info, dict) else "Unknown"
                
                # Fetch details
                details = getattr(result, "per_stock_details", {}).get(ticker, {})
                allowed = details.get("allowed", [])
                removed = details.get("removed", [])
                scores = details.get("scores", {})
                current_score = scores.get(strategy, 0.0)
                
                strategy_data.append({
                    "Ticker": ticker,
                    "Regime": regime,
                    "Selected Strategy": f"{strategy} ({current_score:.3f})",
                    "Allowed Strategies": ", ".join(allowed),
                    "Filtered Out": ", ".join(removed)
                })
            
            strategy_df = pd.DataFrame(strategy_data)
            st.dataframe(strategy_df, hide_index=True, use_container_width=True)
    else:
        st.info("👈 Configure allocation and run the engine to see strategy selection")

# ---------------------------------------------------------
# Tab 3: Regime Detection
# ---------------------------------------------------------
with tab_regime:
    st.header("Regime Detection")
    
    result = st.session_state.result
    
    if result:
        st.subheader(f"🎯 Dominant Regime: {result.dominant_regime}")
        
        if result.regime_output:
            for ticker, regime_info in result.regime_output.items():
                with st.expander(f"📈 {ticker} - {AVAILABLE_STOCKS.get(ticker, ticker)}", expanded=True):
                    probs = regime_info.get("probabilities", {})
                    
                    if probs:
                        prob_df = pd.DataFrame({
                            "Regime": list(probs.keys()),
                            "Probability": list(probs.values())
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x="Regime",
                            y="Probability",
                            color="Probability",
                            color_continuous_scale="RdYlGn",
                            title=f"{ticker} Regime Probabilities"
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No regime data available")
    else:
        st.info("👈 Run the strategy engine to see regime detection results")

# ---------------------------------------------------------
# Tab 4: Position Sizing
# ---------------------------------------------------------
with tab_positions:
    st.header("Position Sizing")
    
    result = st.session_state.result
    
    if result:
        if result.position_sizes is not None and not result.position_sizes.empty:
            pos_df = result.position_sizes.copy()
            
            # Format display
            # Format display
            display_df = pos_df.copy()
            
            # Explicit rename for user clarity
            rename_map = {
                "User_Weight": "Your Input",
                "Adjusted_Weight": "Volatility Sized",
                "Capital_Allocation": "Final $ Allocation"
            }
            display_df = display_df.rename(columns=rename_map)
            
            # Format percentages and currency
            for col in ["Your Input", "Volatility Sized"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
            if "Final $ Allocation" in display_df.columns:
                display_df["Final $ Allocation"] = display_df["Final $ Allocation"].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            st.divider()
            st.subheader("⚖️ Sizing Impact: Input vs. Risk-Adjusted")
            
            # Comparison Chart
            if "Ticker" in pos_df.columns:
                # Melt for grouped bar chart
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
                    x="Ticker",
                    y="Weight",
                    color="Type",
                    barmode="group",
                    title="Comparison: User Intent vs. Risk Sizing",
                    text_auto=".1%",
                    color_discrete_map={"Your Input": "#3366CC", "Volatility Sized": "#DC3912"}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No position sizing data available")
    else:
        st.info("👈 Run the strategy engine to see position sizing results")

# ---------------------------------------------------------
# Tab 5: Trade Signals (Layer 8)
# ---------------------------------------------------------
with tab_signals:
    st.header("📡 Trade Signals")
    
    result = st.session_state.result
    
    if result:
        if result.signals_df is not None and not result.signals_df.empty:
            st.subheader("Generated Signals")
            
            # Color-code signals
            def signal_color(signal):
                colors = {
                    "BUY": "🟢", "SELL": "🔴", "REBALANCE": "🔵",
                    "HOLD": "⚪"
                }
                return colors.get(str(signal).upper(), "⚫")
            
            signals_display = result.signals_df.copy()
            if "Signal" in signals_display.columns:
                signals_display["Action"] = signals_display["Signal"].apply(signal_color) + " " + signals_display["Signal"].astype(str)
            
            st.dataframe(signals_display, hide_index=True, use_container_width=True)
            
            # Signal summary
            if "Signal" in result.signals_df.columns:
                signal_counts = result.signals_df["Signal"].value_counts().to_dict()
                cols = st.columns(len(signal_counts))
                for i, (signal, count) in enumerate(signal_counts.items()):
                    cols[i].metric(signal, count)
        else:
            st.info("No trade signals generated (portfolio unchanged)")
        
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
        st.info("👈 Run the strategy engine to see trade signals")

# ---------------------------------------------------------
# Tab 6: Execution (Layer 9-10)
# ---------------------------------------------------------
with tab_execution:
    st.header("⚡ Trade Execution")
    
    result = st.session_state.result
    
    if result:
        if result.execution_report:
            report = result.execution_report
            
            if "error" in report:
                st.error(f"Execution Error: {report['error']}")
            else:
                # Execution metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Orders Executed", report.get("orders_executed", 0))
                col2.metric("Total Volume", f"${report.get('total_volume', 0):,.2f}")
                col3.metric("Fees Paid", f"${report.get('fees_paid', 0):,.2f}")
                col4.metric("Cash After", f"${report.get('cash_after', 0):,.2f}")
                
                # Order details
                orders = report.get("orders")
                if orders is not None:
                    # Handle both DataFrame and list
                    if hasattr(orders, 'empty'):
                        has_orders = not orders.empty
                        orders_df = orders
                    else:
                        has_orders = bool(orders)
                        orders_df = pd.DataFrame(orders) if orders else pd.DataFrame()
                    
                    if has_orders:
                        st.divider()
                        st.subheader("Order Details")
                        st.dataframe(orders_df, hide_index=True, use_container_width=True)
        else:
            st.info("No execution report available")
    else:
        st.info("👈 Run the strategy engine to see execution details")

# ---------------------------------------------------------
# Tab 7: Portfolio State (Layer 11)
# ---------------------------------------------------------
with tab_portfolio:
    st.header("📋 Portfolio State")
    
    result = st.session_state.result
    
    if result:
        if result.portfolio_state:
            state = result.portfolio_state
            
            # Portfolio overview
            col1, col2, col3 = st.columns(3)
            col1.metric("💵 Cash", f"${state.get('cash', 0):,.2f}")
            col2.metric("📈 Realized P&L", f"${state.get('realized_pnl', 0):,.2f}")
            col3.metric("📊 # Trades", state.get("num_trades", 0))
            
            # Positions
            st.divider()
            st.subheader("Current Positions")
            positions = state.get("positions", {})
            if positions:
                pos_df = pd.DataFrame([
                    {"Ticker": ticker, "Shares": qty}
                    for ticker, qty in positions.items()
                    if qty > 0.001
                ])
                if not pos_df.empty:
                    st.dataframe(pos_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No open positions")
            else:
                st.info("No positions held")
            
            # Fees
            fees = state.get("fees_paid", 0)
            if fees > 0:
                st.caption(f"Total fees paid: ${fees:,.2f}")
        else:
            st.info("No portfolio state available")
    else:
        st.info("👈 Run the strategy engine to see portfolio state")

# ---------------------------------------------------------
# Tab 8: Performance (Layer 12)
# ---------------------------------------------------------
with tab_performance:
    st.header("📈 Performance Metrics")
    
    result = st.session_state.result
    
    if result:
        if result.portfolio_state:
            state = result.portfolio_state
            
            # Performance summary
            col1, col2, col3 = st.columns(3)
            
            realized_pnl = state.get("realized_pnl", 0)
            initial_capital = capital  # From sidebar input
            pnl_pct = (realized_pnl / initial_capital) * 100 if initial_capital else 0
            
            col1.metric("Realized P&L", f"${realized_pnl:,.2f}", f"{pnl_pct:+.2f}%")
            col2.metric("Unrealized P&L", f"${state.get('unrealized_pnl', 0):,.2f}")
            col3.metric("Total Trades", state.get("num_trades", 0))
            
            st.divider()
            
            # Strategy info
            st.subheader("Strategy Summary")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | **Selected Strategy** | {result.selected_strategy} |
            | **Dominant Regime** | {result.dominant_regime} |
            | **Execution Time** | {result.execution_time_ms:.0f}ms |
            """)
            
            # Bandit scores
            if result.bandit_scores:
                st.divider()
                st.subheader("Strategy Scores (Bandit)")
                scores_df = pd.DataFrame([
                    {"Strategy": k, "Score": v} 
                    for k, v in result.bandit_scores.items()
                ]).sort_values("Score", ascending=False)
                
                fig = px.bar(
                    scores_df, x="Strategy", y="Score", 
                    color="Score", color_continuous_scale="Viridis",
                    title="Bandit Strategy Scores"
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Performance metrics require portfolio data")
    else:
        st.info("👈 Run the strategy engine to see performance metrics")

# ---------------------------------------------------------
# Tab 9: History (3 Tables: Signals, Transactions, Cycles)
# ---------------------------------------------------------
with tab_history:
    st.header("📜 Trading History")
    st.caption("Persistent logs of all trading activity")
    
    # --- Table 1: Generated Signal History ---
    with st.expander("📊 Signal History (Generated Signals)", expanded=True):
        history_df = get_signal_history()
        if not history_df.empty:
            st.dataframe(
                history_df.sort_values("Timestamp", ascending=True) if "Timestamp" in history_df.columns else history_df,
                use_container_width=True,
                height=300
            )
        else:
            st.info("No signal history yet. Run the engine to generate signals.")
    
    # --- Table 2: Backtest Transactions ---
    with st.expander("💰 Backtest Transactions (Trade Details)", expanded=True):
        transactions_df = get_transaction_history()
        if not transactions_df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", len(transactions_df))
            with col2:
                total_cost = transactions_df["Transaction_Cost"].sum() if "Transaction_Cost" in transactions_df.columns else 0
                st.metric("Total Costs", f"${total_cost:.2f}")
            with col3:
                # Support both "Side" and "Action" columns
                act_col = "Side" if "Side" in transactions_df.columns else "Action"
                buy_count = len(transactions_df[transactions_df[act_col] == "BUY"]) if act_col in transactions_df.columns else 0
                st.metric("Buys", buy_count)
            with col4:
                act_col = "Side" if "Side" in transactions_df.columns else "Action"
                sell_count = len(transactions_df[transactions_df[act_col] == "SELL"]) if act_col in transactions_df.columns else 0
                st.metric("Sells", sell_count)
            
            st.dataframe(
                transactions_df.sort_values("Timestamp", ascending=True) if "Timestamp" in transactions_df.columns else transactions_df,
                use_container_width=True,
                height=300
            )
        else:
            st.info("No transaction history yet.")
    
    # --- Table 3: Rebalance Cycle Summary ---
    with st.expander("🔄 Rebalance Cycles (P&L Summary)", expanded=True):
        cycles_df = get_cycle_history()
        if not cycles_df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cycles", len(cycles_df))
            with col2:
                total_costs = cycles_df["Transaction_Costs"].sum() if "Transaction_Costs" in cycles_df.columns else 0
                st.metric("Total Costs", f"${total_costs:.2f}")
            with col3:
                latest_value = cycles_df["Current_Position"].iloc[-1] if "Current_Position" in cycles_df.columns else 0
                st.metric("Latest Position", f"${latest_value:,.2f}")
            with col4:
                total_pnl = cycles_df["P/L"].sum() if "P/L" in cycles_df.columns else 0
                st.metric("Total P/L", f"${total_pnl:,.2f}")
            
            st.dataframe(
                cycles_df,
                use_container_width=True,
                height=200
            )
        else:
            st.info("No rebalance cycles yet.")

# ---------------------------------------------------------
# Sidebar — Action buttons at bottom
# ---------------------------------------------------------
st.sidebar.divider()
if st.sidebar.button("🔄 Reset All", use_container_width=True):
    # 1. Clear persist logs using module functions (guarantees correct path)
    clear_signal_log()
    clear_transaction_log()
    clear_cycle_log()
    
    # 2. Also try to clean up the logs dir physically as a fallback
    import shutil
    import os
    import time
    try:
        if os.path.exists("logs"):
            shutil.rmtree("logs")
            time.sleep(0.2)
    except Exception:
        pass
    
    # 3. Clear all session state
    st.session_state.clear()
    
    # 4. Clear cache to prevent stale data re-load
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # 5. Rerun
    st.rerun()

