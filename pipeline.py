"""
Strategy Engine — Core Pipeline Orchestrator (Refactored)

Implements the 13-layer flow (0-12):
0. User & Policy
1. Market Data & Features
2. Regime Intelligence
3. Strategy Universe
4. Constraints Filter
5. Global Bandit
6. Deterministic Ranking
7. Position Sizing
8. Signal Generation
9. Execution Scheduler
10. Trade Execution
11. Rebalancing
12. Performance
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
from utils.trading_calendar import next_trading_day

# Layer imports (New Structure)
from layers.L0_user_policy import UserPolicy, create_policy, RISK_LIMITS
from layers.L1_data_features import compute_all_features
from layers.L2_regime_intelligence import RegimeManager
from layers.L3_strategy_universe import STRATEGY_REGISTRY, get_all_strategy_dicts
from layers.L4_constraints_filter import apply_all_filters
from layers.L5_global_bandit import ContextualBandit, OnlineLearner
from layers.L6_deterministic_ranking import select_best_strategy, build_context_vector
from layers.L7_position_sizing import compute_position_sizes
from layers.L8_signal_generation import generate_portfolio_signals, log_signals
from layers.L9_execution_scheduler import StrategySwitchManager
from layers.L10_trade_execution import run_execution_cycle, log_transactions_from_fills, snapshot_prices
from layers.L11_rebalancing import PortfolioState, log_cycle_summary, get_latest_cycle_number
from layers.L12_performance_benchmark import PerformanceMonitor, DecisionExplanation


def get_next_trading_day(date_str: str) -> str:
    """
    Get next trading day using robust calendar (skips weekends + holidays).
    Signal generation happens on Day T closing, execution on Day T+1 opening.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Start search from tomorrow (T+1)
    start_search = dt + timedelta(days=1)
    # Find next valid business day on or after T+1
    valid_next = next_trading_day(start_search)
    return valid_next.strftime("%Y-%m-%d")


@dataclass
class PipelineResult:
    """Result from a single pipeline run."""
    selected_strategy: str  # Dominant/fallback strategy
    strategy_decision: object
    dominant_regime: str
    regime_output: Dict[str, dict]
    allowed_strategies: List[str]
    removed_strategies: List[str]
    bandit_scores: Dict[str, float]
    per_stock_strategies: Dict[str, str] = None  # Ticker → Strategy (per-stock selection)
    per_stock_details: Dict[str, dict] = None    # Ticker → {allowed, removed, scores}
    position_sizes: Optional[pd.DataFrame] = None
    signals_df: Optional[pd.DataFrame] = None  # Layer 8 trade signals
    execution_report: Optional[dict] = None
    portfolio_state: Optional[dict] = None
    execution_time_ms: float = 0.0
    switch_decision: Optional[object] = None


class StrategyEngine:
    """
    Main Strategy Engine orchestrating all 13 layers (0-12).
    """
    
    def __init__(self, policy: Optional[UserPolicy] = None):
        self.policy = policy
        
        # State-persistent layers
        self.regime_manager = RegimeManager()                  # L2
        strategy_names = list(STRATEGY_REGISTRY.keys())
        self.bandit = ContextualBandit(strategy_names)         # L5
        self.learner = OnlineLearner(self.bandit)              # L5 (Learning Loop)
        self.switch_manager = StrategySwitchManager()          # L9
        # Initialize with policy capital, or default to 10K if no policy
        initial_capital = policy.total_capital if policy else 10_000.0
        self.portfolio_state = PortfolioState(cash=initial_capital, initial_capital=initial_capital)  # L11 State
        self.monitor = PerformanceMonitor()                     # L12
        
        self.current_strategy: Optional[str] = None
        
        # Learning State (Persists across rebalances)
        self.last_decisions: Dict[str, str] = {}    # Ticker → Strategy Name
        self.last_contexts: Dict[str, np.ndarray] = {}  # Ticker → Context Vector
    
    def set_policy(self, policy: UserPolicy) -> None:
        self.policy = policy
        # Initialize portfolio state with policy capital
        self.portfolio_state = PortfolioState(cash=policy.total_capital)
    
    def run(
        self,
        stock_data_dict: Dict[str, pd.DataFrame],
        user_weights: Optional[pd.Series] = None,
        current_date: Optional[str] = None,
    ) -> PipelineResult:
        start_time = time.time()
        current_date_str = current_date or datetime.now().strftime("%Y-%m-%d")
        
        if self.policy is None:
            raise ValueError("Policy not set. Call set_policy() first.")
        
        # === 0. USER & POLICY ===
        if user_weights is None:
            user_weights = pd.Series(self.policy.weights)
        risk_limits = self.policy.get_strictest_limits()
        risk_tolerance = self._infer_tolerance(risk_limits)
        
        # === 1. MARKET DATA & FEATURES ===
        enriched_data = {}
        for ticker, df in stock_data_dict.items():
            if df is not None and not df.empty:
                enriched_data[ticker] = compute_all_features(df)
                
        # === 0.5 EMERGENCY KILL SWITCH ===
        # Independent check before any strategy logic
        current_equity = self.portfolio_state.current_equity(
            snapshot_prices(stock_data_dict, current_date_str)
        )
        peak_equity = self.portfolio_state.peak_equity
        
        # Calculate Drawdown from High Water Mark
        current_drawdown = 0.0
        if peak_equity > 0:
            current_drawdown = (peak_equity - current_equity) / peak_equity
            
        emergency_triggered = False
        if current_drawdown > self.policy.emergency_drawdown_threshold:
            emergency_triggered = True
            print(f"!!! KILL SWITCH TRIGGERED !!! Drawdown {current_drawdown:.2%} > Limit {self.policy.emergency_drawdown_threshold:.2%}")
            
            # Force Liquidation
            selected_strategy = "EMERGENCY_EXIT"
            dominant_regime = "CRISIS"
            regime_outputs = {}
            allowed_strategies = ["Defensive"]
            removed_strategies = ["ALL_OTHERS"]
            bandit_scores = {}
            per_stock_strategies = {t: "Defensive" for t in stock_data_dict.keys()}
            per_stock_details = {}
            
            # Position Sizing -> Force 0 (Cash)
            position_sizes = pd.Series({t: 0.0 for t in stock_data_dict.keys()})
            switch_decision = SwitchDecision(
                should_switch=True, reason="Emergency Protocol", 
                current_strategy=self.current_strategy or "None",
                new_strategy="EMERGENCY_EXIT", new_probability=1.0
            ) 
            strategy_decision = None # Special case
            
            # BYPASS REST OF LOGIC -> Jump to Signal Generation
            # We mock the variables needed for Signal Generation
            
        else:
            # === 2. REGIME INTELLIGENCE ===
            regime_outputs = {}
        for ticker, df in enriched_data.items():
            try:
                detector = self.regime_manager.get_or_create(ticker)
                regime_out = detector.predict_regime(df)
                regime_outputs[ticker] = {
                    "dominant_regime": regime_out.dominant_regime,
                    "probabilities": regime_out.probabilities,
                }
            except Exception:
                regime_outputs[ticker] = {
                    "dominant_regime": "Range + Low Vol",
                    "probabilities": {"Range + Low Vol": 1.0},
                }
        
        # Aggregate regime
        regime_counts = {}
        for info in regime_outputs.values():
            if isinstance(info, dict):
                r = info.get("dominant_regime", "Range + Low Vol")
                regime_counts[r] = regime_counts.get(r, 0) + 1
        dominant_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "Range + Low Vol"
        
        # === 3. STRATEGY UNIVERSE ===
        all_strategies = get_all_strategy_dicts()
        
        # === 4-6. PER-STOCK STRATEGY SELECTION ===
        # Run filter → bandit → ranking for EACH stock based on its individual regime
        per_stock_strategies = {}
        per_stock_details = {}
        per_stock_allowed = {}
        all_removed = set()
        risk_score_val = {"Low": 0.3, "Medium": 0.5, "High": 0.7}.get(risk_tolerance, 0.5)
        
        for ticker in enriched_data.keys():
            # Get this stock's regime
            ticker_regime_info = regime_outputs.get(ticker, {})
            if not isinstance(ticker_regime_info, dict):
                ticker_regime_info = {"dominant_regime": "Range + Low Vol", "probabilities": {"Range + Low Vol": 1.0}}
            
            stock_regime = ticker_regime_info.get("dominant_regime", "Range + Low Vol")
            stock_regime_probs = ticker_regime_info.get("probabilities", {stock_regime: 1.0})
            
            # L4: Filter strategies for THIS stock's regime
            stock_filter = apply_all_filters(
                strategies=all_strategies,
                user_risk_tolerance=risk_tolerance,
                current_regime=stock_regime,
            )
            stock_allowed = [s if isinstance(s, str) else s.get("name") for s in stock_filter.allowed_strategies]
            stock_removed = [s if isinstance(s, str) else s.get("name") for s in stock_filter.removed_strategies]
            all_removed.update(stock_removed)
            per_stock_allowed[ticker] = stock_allowed
            
            # L5: Bandit for THIS stock's context
            stock_data = enriched_data.get(ticker)
            if stock_data is not None and hasattr(stock_data, "columns"):
                # Use GARCH if available (Predictive), else Realized
                if "GARCH_Vol" in stock_data.columns:
                    stock_vol = stock_data["GARCH_Vol"].iloc[-1]
                elif "Realized_Vol" in stock_data.columns:
                    stock_vol = stock_data["Realized_Vol"].iloc[-1]
                else:
                    stock_vol = 0.15
                    
                stock_momentum = stock_data["Momentum"].iloc[-1] if "Momentum" in stock_data.columns else 0.0
                stock_dd = stock_data["Drawdown"].iloc[-1] if "Drawdown" in stock_data.columns else 0.0
            else:
                stock_vol, stock_momentum, stock_dd = 0.15, 0.0, 0.0
            
            stock_context = build_context_vector(
                regime_probs=stock_regime_probs,
                volatility=float(stock_vol) if hasattr(stock_vol, 'item') else stock_vol,
                momentum=float(stock_momentum) if hasattr(stock_momentum, 'item') else stock_momentum,
                drawdown=float(stock_dd) if hasattr(stock_dd, 'item') else stock_dd,
                risk_score=risk_score_val,
            )
            
            bandit_strategies = stock_allowed if stock_allowed else ["Defensive"]
            _, stock_bandit_scores = self.bandit.select_strategy(
                context=stock_context,
                allowed_strategies=bandit_strategies
            )
            
            # L6: Ranking for THIS stock
            stock_decision = select_best_strategy(
                allowed_strategies=stock_allowed,
                bandit_scores=stock_bandit_scores,
                expected_returns={s["name"]: 0.10 for s in all_strategies if s.get("name") in stock_allowed},
            )
            
            per_stock_strategies[ticker] = stock_decision.selected_strategy
            
            # Capture details for UI
            per_stock_details[ticker] = {
                "allowed": stock_allowed,
                "removed": stock_removed,
                "scores": stock_bandit_scores,
            }
            
            # Save context for next update
            self.last_contexts[ticker] = stock_context
            
        # Save decisions for next update
        self.last_decisions = per_stock_strategies.copy()
        
        # Aggregate: most common strategy for display (fallback)
        if per_stock_strategies:
            from collections import Counter
            strategy_counts = Counter(per_stock_strategies.values())
            selected_strategy = strategy_counts.most_common(1)[0][0]
        else:
            selected_strategy = "Defensive"
        
        # Keep bandit_scores as aggregate for UI (use last stock's scores for simplicity)
        bandit_scores = stock_bandit_scores if 'stock_bandit_scores' in dir() else {}
        
        # Aggregate allowed/removed for UI
        allowed_strategies = list(set().union(*per_stock_allowed.values())) if per_stock_allowed else ["Defensive"]
        removed_strategies = list(all_removed)
        
        # Create a summary decision object
        # Create a summary decision object (for normal path)
        strategy_decision = select_best_strategy(
            allowed_strategies=allowed_strategies,
            bandit_scores=bandit_scores,
            expected_returns={s["name"]: 0.10 for s in all_strategies if s.get("name") in allowed_strategies},
        )
        
        # === 7. POSITION SIZING ===
        vol_series = pd.Series({
            ticker: (
                (data["GARCH_Vol"].iloc[-1] if "GARCH_Vol" in data.columns else data["Realized_Vol"].iloc[-1])
                if "Realized_Vol" in data.columns else 0.15
            ) if hasattr(data, "columns") else 0.15
            for ticker, data in enriched_data.items()
        })
        forecast_vol = vol_series # Simplification
        
        position_sizes = compute_position_sizes(
            user_weights=user_weights,
            forecast_vol=forecast_vol,
            total_capital=self.portfolio_state.cash,
            target_vol=0.15,
            max_vol=risk_limits["max_volatility"],
            max_dd=risk_limits["max_drawdown"],
            max_leverage=risk_limits["max_leverage"],
        )
        
        # === 8. SIGNAL GENERATION ===
        # Convert position sizes to portfolio weights for signal generation
        if isinstance(position_sizes, pd.Series):
            new_portfolio_df = pd.DataFrame({
                "Ticker": position_sizes.index.tolist(),
                "Weight": position_sizes.values.tolist()
            })
        else:
            new_portfolio_df = position_sizes.reset_index()
            if len(new_portfolio_df.columns) == 2:
                new_portfolio_df.columns = ["Ticker", "Weight"]
            else:
                # Handle multi-column case: first col is ticker, last is weight
                new_portfolio_df = new_portfolio_df.iloc[:, [0, -1]]
                new_portfolio_df.columns = ["Ticker", "Weight"]
        
        # Get old strategies from portfolio state (tracks which strategy was used for each position)
        old_strategies = dict(self.portfolio_state.position_strategies)
        
        # Pass per-stock strategies instead of single global strategy
        signals = generate_portfolio_signals(
            old_portfolio_df=self.portfolio_state.last_allocation,
            new_portfolio_df=new_portfolio_df,
            old_strategies=old_strategies,
            new_strategy=per_stock_strategies,  # Now a dict: Ticker → Strategy
            as_of_date=current_date_str,
        )
        
        # Log signals to persistent CSV
        log_signals(signals)
        
        # === 9. EXECUTION SCHEDULER ===
        switch_decision = self.switch_manager.evaluate_switch(
            new_strategy=selected_strategy,
            new_probability=strategy_decision.bandit_score,
            current_date=current_date_str,
        )
        if switch_decision.should_switch:
            self.current_strategy = selected_strategy
        
        # === 10. TRADE EXECUTION ===
        # Signal generation uses Day T closing prices
        # Trade execution happens on Day T+1 opening prices
        execution_date_str = get_next_trading_day(current_date_str)
        
        execution_report = {}
        if not signals.empty:
            try:
                execution_report = run_execution_cycle(
                    state=self.portfolio_state,
                    price_data_dict=stock_data_dict,
                    signals_df=signals,
                    new_portfolio_weights=new_portfolio_df,
                    date=execution_date_str,  # T+1 execution date
                    commission_per_trade=1.0,
                )
                
                # Get current prices for transaction logging and P/L calculation
                prices = snapshot_prices(stock_data_dict, current_date_str)
                
                # Calculate actual portfolio value after trades
                portfolio_value = self.portfolio_state.current_equity(prices)
                
                # Get P/L from portfolio state (updated by run_execution_cycle)
                realized_pnl = self.portfolio_state.realized_pnl
                unrealized_pnl = self.portfolio_state.unrealized_pnl
                
                # Calculate return percentage
                initial_capital = self.policy.total_capital
                pnl = portfolio_value - initial_capital
                return_pct = (pnl / initial_capital * 100) if initial_capital > 0 else 0.0
                
                # Log detailed transactions from actual fills (includes BUY + SELL)
                fills_df = execution_report.get("fills")
                if fills_df is not None and not fills_df.empty:
                    log_transactions_from_fills(
                        fills_df=fills_df,
                        execution_date=execution_date_str,  # T+1 execution date
                    )
                
                # Log cycle summary with actual P/L values
                cycle_num = get_latest_cycle_number() + 1
                
                # Count positions with non-zero qty
                num_positions = len(self.portfolio_state.positions)
                
                # Extract marginal fees from this execution only
                fees_this_cycle = 0.0
                if execution_report and "fees_paid" in execution_report:
                    fees_this_cycle = execution_report["fees_paid"]

                log_cycle_summary(
                    execution_date=execution_date_str,  # T+1 execution date
                    rebalance_frequency=self.policy.rebalance_frequency,
                    portfolio_value=portfolio_value,
                    cash=self.portfolio_state.cash,
                    initial_capital=self.policy.total_capital,  # Pass strict initial capital
                    pnl=pnl,
                    return_pct=return_pct,
                    cycle_number=cycle_num,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    cumulative_realized_pnl=realized_pnl,
                    transaction_costs=fees_this_cycle,  # Log MARGINAL fees, not cumulative
                    num_positions=num_positions,
                )
            except Exception as e:
                execution_report = {"error": str(e), "signals": signals.to_dict()}
        
        # === 11. REBALANCING ===
        # PortfolioState is updated in-place by run_execution_cycle
        
        # === 12. PERFORMANCE BENCHMARK ===
        # Safe monitor probs
        monitor_probs = {}
        if regime_outputs and isinstance(regime_outputs, dict):
            first_val = regime_outputs.get(list(regime_outputs.keys())[0], {})
            if isinstance(first_val, dict):
                monitor_probs = first_val.get("probabilities", {})
                
        explanation = DecisionExplanation(
            timestamp=datetime.now(),
            selected_strategy=selected_strategy,
            regime=dominant_regime,
            regime_probabilities=monitor_probs,
            allowed_strategies=allowed_strategies,
            filtered_strategies=removed_strategies,
            filter_reasons={},  # Filter runs per-stock now
            bandit_scores=bandit_scores,
            selection_reason=strategy_decision.selection_reason,
        )
        self.monitor.record_decision(explanation)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return PipelineResult(
            selected_strategy=selected_strategy,
            strategy_decision=strategy_decision,
            dominant_regime=dominant_regime,
            regime_output=regime_outputs,
            allowed_strategies=allowed_strategies,
            removed_strategies=removed_strategies,
            bandit_scores=bandit_scores,
            per_stock_strategies=per_stock_strategies,
            per_stock_details=per_stock_details,
            position_sizes=position_sizes,
            signals_df=signals,
            execution_report=execution_report,
            portfolio_state=self.portfolio_state.get_summary(),
            execution_time_ms=elapsed_ms,
            switch_decision=switch_decision,
        )

    def _infer_tolerance(self, limits: dict) -> str:
        max_vol = limits["max_volatility"]
        if max_vol <= 0.08: return "Low"
        elif max_vol <= 0.15: return "Medium"
        else: return "High"

    def _compute_avg_volatility(self, enriched_data: dict) -> float:
        vols = []
        for df in enriched_data.values():
            if isinstance(df, pd.DataFrame) and "Realized_Vol" in df.columns:
                vols.append(df["Realized_Vol"].iloc[-1])
        return np.mean(vols) if vols else 0.15
        
    def _compute_avg_momentum(self, enriched_data: dict) -> float:
        moms = []
        for df in enriched_data.values():
            if isinstance(df, pd.DataFrame) and "Momentum" in df.columns:
                moms.append(df["Momentum"].iloc[-1])
        return np.mean(moms) if moms else 0.0

    def _update_bandit(self, enriched_data: Dict[str, pd.DataFrame]):
        """
        Update bandit posteriors based on realized rewards from the previous cycle.
        Reward = Risk-Adj Return (Sharpe) if position was held, else 0 (or penalty).
        """
        for ticker, strategy in self.last_decisions.items():
            if ticker not in enriched_data: continue
            
            df = enriched_data[ticker]
            if df.empty or "Returns" not in df.columns: continue
            
            # 1. Did we participate? (Check T-1 holdings)
            # self.portfolio_state has holdings from START of this run (end of prev cycle)
            qty = self.portfolio_state.positions.get(ticker, 0)
            
            # 2. Calculate Strategy Return
            # If Qty > 0, we captured the market return (simplified)
            # If Qty == 0, we got 0 return (Cash)
            mkt_return = df["Returns"].iloc[-1]
            mkt_vol = df["Realized_Vol"].iloc[-1] if "Realized_Vol" in df.columns else 0.15
            mkt_dd = df["Max_Drawdown"].iloc[-1] if "Max_Drawdown" in df.columns else 0.0
            
            if abs(qty) > 0:
                strategy_return = mkt_return
            else:
                strategy_return = 0.0
            
            # 3. Compute Bandit Reward
            reward = self.bandit.compute_reward(
                returns=strategy_return,
                volatility=mkt_vol,
                drawdown=mkt_dd
            )
            
            # 4. Update
            context = self.last_contexts.get(ticker)
            self.bandit.update(strategy, reward, context)

    def _compute_avg_drawdown(self, enriched_data: dict) -> float:
        dds = []
        for df in enriched_data.values():
            if isinstance(df, pd.DataFrame) and "Max_Drawdown" in df.columns:
                dds.append(df["Max_Drawdown"].iloc[-1])
        return np.mean(dds) if dds else 0.0

    def get_performance(self): return self.monitor.compute_metrics()
    def get_decision_history(self, n: int = 10): return self.monitor.get_recent_decisions(n)
    def get_bandit_stats(self): return self.bandit.get_stats()


def run_engine(
    stock_data_dict: Dict[str, pd.DataFrame],
    tickers: List[str],
    weights: List[float],
    total_capital: float = 10000.0,
    risk_tolerance: str = "Medium",
    rebalance_frequency: str = "Weekly",
    as_of_date: Optional[str] = None,
) -> PipelineResult:
    policy = create_policy(
        tickers=tickers,
        weights=weights,
        total_capital=total_capital,
        risk_tolerance=risk_tolerance,
        rebalance_frequency=rebalance_frequency,
    )
    
    engine = StrategyEngine(policy)
    return engine.run(stock_data_dict, current_date=as_of_date)
