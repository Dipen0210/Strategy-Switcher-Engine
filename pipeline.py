"""
Strategy Engine — Core Pipeline Orchestrator

New Architecture:
  PRE-MARKET (per stock):
    1. HMM → regime posteriors
    2. Bandit A: blend posteriors × trust → regime label
    3. Confidence gate (> 0.55)
    4. Load strategies for regime
    5. Bandit B: rank strategies by Thompson sampling
    6. Bandit C: evaluate top 3 for this stock → pick winner
    7. Score = 0.5×θ_B + 0.3×HMM_conf + 0.2×stability
  EXECUTE:
    8. Position sizing + signal generation + trade execution
  POST-MARKET:
    9. Compute R_final, update all 3 bandits with differentiated rewards
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
from utils.trading_calendar import next_trading_day

# Layer imports
from layers.L0_user_policy import UserPolicy, create_policy, RISK_LIMITS
from layers.L1_data_features import compute_all_features
from layers.L2_regime_intelligence import RegimeManager
from layers.L2_regime_intelligence.regime_selection import (
    blend_regime, compute_stability, REGIME_STRATEGY_COMPAT
)
from layers.L3_strategy_universe import STRATEGY_REGISTRY, get_all_strategy_dicts, get_strategies_for_regime, run_strategies_for_regime
from layers.L7_position_sizing import compute_position_sizes
from layers.L8_signal_generation import generate_portfolio_signals, log_signals
from layers.L9_execution_scheduler import StrategySwitchManager
from layers.L10_trade_execution import run_execution_cycle, log_transactions_from_fills, snapshot_prices
from layers.L11_rebalancing import PortfolioState, log_cycle_summary, get_latest_cycle_number
from layers.L12_performance_benchmark import PerformanceMonitor, DecisionExplanation

# Hierarchical Bandit System (persistent across restarts)
from layers.L5_bandit import BanditPersistenceManager
from layers.L5_bandit.reward import compute_reward


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
        self.switch_manager = StrategySwitchManager()          # L9
        # Initialize with policy capital, or default to 10K if no policy
        initial_capital = policy.total_capital if policy else 10_000.0
        self.portfolio_state = PortfolioState(cash=initial_capital, initial_capital=initial_capital)  # L11 State
        self.monitor = PerformanceMonitor()                     # L12
        
        # Hierarchical Bandit System (persistent across restarts)
        self.ensemble_bandits = BanditPersistenceManager.load()  # L5 (Global + Regime + Stock)
        
        self.current_strategy: Optional[str] = None
        
        # Tracking for post-trade feedback
        self.last_decisions: Dict[str, str] = {}    # Ticker → Strategy Name
        self.last_regime: Optional[str] = None       # For transition detection
    
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
                
        # === 1.5 POST-MARKET FEEDBACK (from previous cycle) ===
        # Update bandits based on how our LAST decisions performed
        if self.last_decisions:
            self._update_bandit(enriched_data)
                
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
            
        else:
            # ==== PRE-MARKET: PER-STOCK FLOW ====
            # Each stock gets its own HMM regime detection from its own data,
            # blended with its own Bandit A trust weights.
            
            per_stock_strategies = {}
            per_stock_details = {}
            per_stock_allowed = {}
            regime_outputs = {}
            all_removed = set()
            
            for ticker, df in enriched_data.items():
                # === STEP 1: HMM → regime posteriors (per stock) ===
                try:
                    regime_output = self.regime_manager.predict_regime(ticker, df)
                    hmm_posteriors = regime_output.probabilities
                except Exception as e:
                    print(f"  ⚠ HMM failed for {ticker}: {e}, fallback to Sideways")
                    hmm_posteriors = {"Sideways": 1.0}
                    regime_output = None
                
                # === STEP 2: BANDIT A — Blend posteriors × GLOBAL trust ===
                # Get GLOBAL trust weights (learned from all stocks)
                bandit_a_weights = self.ensemble_bandits.global_bandit.get_trust_weights()
                blended = blend_regime(hmm_posteriors, bandit_a_weights)
                
                dominant_regime = max(blended, key=blended.get)
                hmm_confidence = max(blended.values())
                stability = regime_output.stability_score if regime_output else 0.5
                
                # Confidence gate
                is_ambiguous = hmm_confidence < 0.55
                if is_ambiguous:
                    print(f"  ⚠ {ticker}: Ambiguous regime (confidence={hmm_confidence:.2f} < 0.55)")
                
                # Transition detection (per stock vs global last regime for now)
                transition_flag = (
                    self.last_regime is not None
                    and dominant_regime != self.last_regime
                )
                
                print(f"  📊 {ticker}: regime={dominant_regime} (conf={hmm_confidence:.1%}, stab={stability:.2f})")
                
                # Save regime output for UI
                regime_outputs[ticker] = {
                    "dominant_regime": dominant_regime,
                    "probabilities": blended,
                    "stability_score": stability,
                    "hmm_confidence": hmm_confidence,
                    "is_ambiguous": is_ambiguous,
                    "transition_flag": transition_flag,
                }
                
                # === STEP 3: Load strategies for this stock's regime ===
                allowed_strategies_for_regime = REGIME_STRATEGY_COMPAT.get(dominant_regime, ["Defensive"])
                
                try:
                    strategy_outputs = run_strategies_for_regime(
                        regime=dominant_regime,
                        stock_data_dict={ticker: df},
                    )
                except Exception as e:
                    print(f"⚠️ Strategy execution failed for {ticker} (regime={dominant_regime}): {e}")
                    strategy_outputs = []
                
                if not strategy_outputs:
                    per_stock_strategies[ticker] = "Defensive"
                    per_stock_details[ticker] = {"allowed": ["Defensive"], "scores": {}, "no_strategies": True, "regime": dominant_regime}
                    per_stock_allowed[ticker] = ["Defensive"]
                    continue
                
                # Get strategy names from outputs
                available_strategy_names = list(set(
                    out.strategy_name for out in strategy_outputs if out.ticker == ticker
                ))
                
                if not available_strategy_names:
                    per_stock_strategies[ticker] = "Defensive"
                    per_stock_details[ticker] = {"allowed": ["Defensive"], "scores": {}, "no_strategies": True, "regime": dominant_regime}
                    per_stock_allowed[ticker] = ["Defensive"]
                    continue
                
                # === STEP 4: BANDIT B — Rank strategies in this GLOBAL regime ===
                # Get ALL strategy weights for display, then pick Top 5 for Bandit C
                regime_bandit = self.ensemble_bandits.regime_bandits
                all_bandit_b_weights = regime_bandit.get_bandit(dominant_regime).get_all_weights(available_strategy_names)
                top_5_strategies = regime_bandit.rank_strategies(dominant_regime, available_strategy_names)
                
                # === STEP 5: BANDIT C — Walk-Forward Backtest & Pick Winner ===
                stock_bandit_mgr = self.ensemble_bandits.stock_bandits
                
                # Initialize ALL top 5 strategies at once so they get random unequal weights
                # Per-stock-per-regime model: only ~10 strategies per model
                top_5_names = [s[0] for s in top_5_strategies]
                stock_bandit_mgr.get_bandit(ticker, dominant_regime)._ensure_strategies(top_5_names)
                
                # Build per-strategy timeframe map from strategy specs
                strategy_specs = get_strategies_for_regime(dominant_regime)
                timeframe_map = {spec.name: spec.timeframe for spec in strategy_specs}
                
                # Per-strategy past return using each strategy's own TIMEFRAME
                raw_scores = []
                for strat_name, score_b in top_5_strategies:
                    tf = timeframe_map.get(strat_name, 30)  # fallback 30 days
                    
                    # Calculate risk-adjusted past return (Sharpe-like)
                    risk_adj_return = 0.0
                    if len(df) >= tf and "Close" in df.columns:
                        closes = df["Close"].values[max(0, len(df) - tf):]
                        if len(closes) > 1:
                            daily_rets = np.diff(closes) / closes[:-1]
                            mean_ret = np.mean(daily_rets)
                            std_ret = np.std(daily_rets)
                            if std_ret > 1e-6:
                                # Annualized Sharpe-like ratio
                                risk_adj_return = (mean_ret / std_ret) * np.sqrt(252)
                    
                    # Weight from Stock Bandit (θ_C) — per-stock-per-regime
                    theta_c = stock_bandit_mgr.sample(ticker, strat_name, regime=dominant_regime)
                    
                    raw_scores.append({
                        "name": strat_name,
                        "theta_b": score_b,
                        "theta_c": theta_c,
                        "risk_adj_ret": float(risk_adj_return)
                    })
                
                # Normalize risk-adjusted returns to [0, 1] range for fair linear combination
                rets = [s["risk_adj_ret"] for s in raw_scores]
                min_ret, max_ret = min(rets), max(rets)
                range_ret = max_ret - min_ret if max_ret > min_ret else 1.0
                
                stock_scored = []
                for s in raw_scores:
                    norm_ret = (s["risk_adj_ret"] - min_ret) / range_ret
                    
                    # Final Score = 30% θ_B + 40% Risk-Adj Return + 30% θ_C
                    final_score = (0.3 * s["theta_b"]) + (0.4 * norm_ret) + (0.3 * s["theta_c"])
                    
                    stock_scored.append((
                        s["name"], 
                        s["theta_b"], 
                        s["theta_c"], 
                        final_score, 
                        s["risk_adj_ret"]
                    ))

                # Rank by final score
                stock_scored.sort(key=lambda x: x[3], reverse=True)
                
                if stock_scored:
                    winner_name = stock_scored[0][0]
                    winner_final_score = stock_scored[0][3]
                    winner_theta_c = stock_scored[0][2]
                else:
                    winner_name = "Defensive"
                    winner_final_score = 0.0
                    winner_theta_c = 0.5
                
                per_stock_strategies[ticker] = winner_name
                
                # Build strategy_scores dict for UI (using Bandit B's base scores for the list)
                strategy_scores = {s[0]: s[1] for s in top_5_strategies}
                
                per_stock_details[ticker] = {
                    "allowed": available_strategy_names,
                    "removed": list(set(available_strategy_names) - set([s[0] for s in top_5_strategies])),
                    "scores": strategy_scores,
                    "stability": stability,
                    "hmm_confidence": hmm_confidence,
                    "regime": dominant_regime,
                    "winner_theta_c": winner_theta_c,
                    "all_bandit_b_weights": {k: round(v, 4) for k, v in all_bandit_b_weights.items()},
                    "candidates": [
                        {
                            "Strategy": s[0],
                            "θ_B": round(s[1], 4),
                            "Score": round(s[3], 4),
                            "Past_Return": round(s[4], 4),
                            "θ_C": round(s[2], 4),
                        }
                        for s in stock_scored
                    ],
                    "stock_filter": [
                        {"Strategy": s[0], "Final": round(s[3], 4), "θ_C": round(s[2], 4), "Past_Ret": round(s[4], 4)}
                        for s in stock_scored
                    ],
                }
                per_stock_allowed[ticker] = available_strategy_names
            
            # Track the most recent dominant regime (use most common across stocks)
            if regime_outputs:
                from collections import Counter
                regime_counts = Counter(info["dominant_regime"] for info in regime_outputs.values())
                dominant_regime = regime_counts.most_common(1)[0][0]
            else:
                dominant_regime = "Sideways"
            self.last_regime = dominant_regime
            
            # Save decisions for post-trade feedback (include per-stock regimes)
            self.last_decisions = per_stock_strategies.copy()
            self.last_per_stock_regimes = {t: info["dominant_regime"] for t, info in regime_outputs.items()}
            
            # Persist bandit state after every run
            self.ensemble_bandits.save_all()
            
            # Aggregate: most common strategy for display
            if per_stock_strategies:
                from collections import Counter
                strategy_counts = Counter(per_stock_strategies.values())
                selected_strategy = strategy_counts.most_common(1)[0][0]
            else:
                selected_strategy = "Defensive"
            
            # Aggregate for UI
            bandit_scores = {}
            for details in per_stock_details.values():
                bandit_scores.update(details.get("scores", {}))
            
            allowed_strategies = list(set().union(*[set(v) for v in per_stock_allowed.values()])) if per_stock_allowed else ["Defensive"]
            removed_strategies = list(all_removed)
            
            # Create summary decision (for legacy compatibility)
            top_score = bandit_scores.get(selected_strategy, 0.5)
            strategy_decision = type('StrategyDecision', (), {
                'selected_strategy': selected_strategy,
                'scores': bandit_scores,
                'bandit_score': top_score,
                'selection_reason': f'3-Factor Ensemble: {selected_strategy} (score: {top_score:.3f})',
                'expected_return': top_score * 0.1,
                'alternatives': [k for k in bandit_scores.keys() if k != selected_strategy][:3],
                'rationale': '3-Factor Ensemble (θ_B + HMM + Stability)'
            })()
        
        # === 7. POSITION SIZING ===
        vol_series = pd.Series({
            ticker: (
                (data["GARCH_Vol"].iloc[-1] if "GARCH_Vol" in data.columns else data["Realized_Vol"].iloc[-1])
                if "Realized_Vol" in data.columns else 0.15
            ) if hasattr(data, "columns") else 0.15
            for ticker, data in enriched_data.items()
        })
        forecast_vol = vol_series # Simplification
        
        # Stability Scores for Sizing
        stability_series = pd.Series({
            t: info.get("stability_score", 1.0) 
            for t, info in regime_outputs.items()
        })
        
        # Inject Volatility Scalar into regime_outputs for UI
        target_vol_val = 0.15
        for t, vol in forecast_vol.items():
            if t in regime_outputs:
                # Avoid division by zero
                safe_vol = max(vol, 0.01)
                scalar = target_vol_val / safe_vol
                regime_outputs[t]["volatility_scalar"] = scalar
                regime_outputs[t]["forecast_vol"] = vol
        
        # Prepare weights for sizing
        # IMPORTANT: Zero out weights for stocks in "Defensive" mode to ensure liquidation.
        sizing_weights = user_weights.copy()
        for ticker, strategy in per_stock_strategies.items():
            if "Defensive" in strategy or "Cash" in strategy:
                if ticker in sizing_weights.index:
                    sizing_weights[ticker] = 0.0

        position_sizes = compute_position_sizes(
            user_weights=sizing_weights,
            forecast_vol=forecast_vol,
            total_capital=self.portfolio_state.cash,
            stability_scores=stability_series,
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
            new_probability=getattr(strategy_decision, 'bandit_score', bandit_scores.get(selected_strategy, 0.5)),
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
        POST-MARKET: Update all 3 bandits with differentiated rewards.
        
        Architecture:
        - Bandit A (Global Trust): Global update (aggregated from all stocks)
        - Bandit B (Global Ranking): Global update (aggregated from all stocks)
        - Bandit C (Stock Preference): Per-Stock update
        
        Flow:
            1. Decay ALL bandits (A, B) — ONCE per cycle
            2. For each stock:
               a. Compute reward
               b. Update Global Bandit A (trust in its regime)
               c. Update Global Bandit B (strategy in its regime)
               d. Update Local Bandit C (strategy per stock)
            3. Save all
        """
        update_count = 0
        per_stock_regimes = getattr(self, 'last_per_stock_regimes', {})
        is_ambiguous = False
        
        # ===== STEP 1: Decay EVERYONE ONCE =====
        # Decays all global params
        self.ensemble_bandits.decay_all_bandits()
        print(f"  ⏳ Decayed all bandits (Global A & B)")
        
        # ===== STEP 2: Update per-stock =====
        for ticker, strategy in self.last_decisions.items():
            if ticker not in enriched_data:
                continue
            
            df = enriched_data[ticker]
            if df.empty:
                continue
            
            # Use this stock's regime (fallback to global)
            regime = per_stock_regimes.get(ticker, self.last_regime or "Sideways")
            
            qty = self.portfolio_state.positions.get(ticker, 0)
            
            if "Returns" in df.columns:
                daily_return = df["Returns"].iloc[-1] if abs(qty) > 0 else 0.0
            elif "Return_1D" in df.columns:
                daily_return = df["Return_1D"].iloc[-1] if abs(qty) > 0 else 0.0
            else:
                daily_return = 0.0
            
            if "Realized_Vol" in df.columns:
                vol_60d = df["Realized_Vol"].iloc[-1]
            else:
                vol_60d = 0.15
            
            # Using actual raw return for new feedback setup
            rewards = {"A": daily_return, "B": daily_return, "C": daily_return}
            
            # Update arms for THIS ticker
            # Note: persistence.update_arm handles the global/local routing
            self.ensemble_bandits.update_arm(
                ticker=ticker,
                regime=regime,
                strategy_name=strategy,
                rewards=rewards,
            )
            update_count += 1
            
        print(f"  🧠 Learning complete: {update_count} per-stock feedback updates aggregated")
        
        # ===== STEP 3: Save =====
        if update_count > 0:
            self.ensemble_bandits.save_all()
            print(f"🧠 Learning complete: {update_count} per-stock updates")

    def _compute_avg_drawdown(self, enriched_data: dict) -> float:
        dds = []
        for df in enriched_data.values():
            if isinstance(df, pd.DataFrame) and "Max_Drawdown" in df.columns:
                dds.append(df["Max_Drawdown"].iloc[-1])
        return np.mean(dds) if dds else 0.0

    def get_performance(self): return self.monitor.compute_metrics()
    def get_decision_history(self, n: int = 10): return self.monitor.get_recent_decisions(n)
    def get_bandit_stats(self): return self.ensemble_bandits.get_stats()


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
