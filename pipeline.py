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
from layers.L3_strategy_universe.registry import run_single_strategy
from layers.L7_position_sizing import compute_position_sizes, signal_participation
from layers.L8_signal_generation import generate_portfolio_signals, log_signals
from layers.L9_execution_scheduler import StrategySwitchManager, SwitchDecision
from layers.L10_trade_execution import run_execution_cycle, log_transactions_from_fills, snapshot_prices
from layers.L11_rebalancing import PortfolioState, log_cycle_summary, get_latest_cycle_number
from layers.L12_performance_benchmark import PerformanceMonitor, DecisionExplanation

# Hierarchical Bandit System (persistent across restarts)
from layers.L5_bandit import BanditPersistenceManager
from layers.L5_bandit.reward import (
    compute_reward,
    attributable_return,
    differentiated_exp3_rewards,
)


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
    emergency_triggered: bool = False  # L0 kill switch fired this cycle


class StrategyEngine:
    """
    Main Strategy Engine orchestrating all 13 layers (0-12).
    """
    
    def __init__(self, policy: Optional[UserPolicy] = None, bandit_dir=None,
                 use_garch: bool = True, logging_enabled: bool = True):
        self.policy = policy
        # GARCH dominates per-cycle cost; backtests disable it and fall back
        # to realized volatility for the L7 forecast.
        self.use_garch = use_garch
        # CSV trade/cycle logging does a read-modify-write per call, which is
        # both slow and polluting across thousands of backtest cycles. The UI
        # keeps it on; the harness turns it off and records its own artifacts.
        self.logging_enabled = logging_enabled

        # --- Ablation switches -------------------------------------------
        # Defaults reproduce the full engine exactly. backtest.ablations flips
        # these to isolate the contribution of individual components; keeping
        # them as explicit options avoids monkeypatching core decision logic,
        # which would make ablation results hard to trust.
        #
        # blend_weights: (w_global, w_hmm) for Bandit A regime blending.
        #   (0.0, 1.0) disables Bandit A — raw HMM posteriors are used.
        self.blend_weights = (0.60, 0.40)
        # strategy_pool: "regime" restricts candidates to the detected
        #   regime's pod (10 strategies). "all" pools all 40 regardless of
        #   regime — the flat-bandit comparison for the pod hypothesis.
        self.strategy_pool = "regime"
        # use_signal_participation: False reverts to the pre-fix behavior
        #   where the winning strategy's signal did not affect position size.
        self.use_signal_participation = True
        # score_weights: (theta_B, normalized past return, theta_C) blend used
        #   to pick the winning strategy. Hardcoded as 0.3/0.4/0.3 before;
        #   exposed so it can be tuned on the validation window only.
        self.score_weights = (0.3, 0.4, 0.3)
        # confidence_gate: blended posterior below which a regime call is
        #   flagged ambiguous.
        self.confidence_gate = 0.55
        # past_return_mode: how the middle term of the selection score is
        #   computed. See _strategy_past_sharpe for why "stock" is broken.
        #     "strategy" — replay the strategy's OWN signals over its
        #                  TIMEFRAME and score the returns they produced.
        #     "stock"    — legacy behavior: the STOCK's trailing Sharpe over
        #                  the strategy's TIMEFRAME. Identical for any two
        #                  strategies sharing a timeframe, so it cannot rank
        #                  them. Kept only as an ablation arm.
        #     "off"      — drop the term; theta_B and theta_C are renormalized
        #                  to carry its weight.
        self.past_return_mode = "strategy"
        # Replay cost control. Signals are cached per (strategy, ticker, date)
        # so consecutive cycles reuse overlapping windows.
        self.past_return_max_evals = 60
        self._signal_replay_cache: Dict[tuple, tuple] = {}
        # fully_invested: True renormalizes target weights to sum to 1.0, i.e.
        #   the book is always 100% deployed. False lets conviction and the
        #   risk caps actually reduce exposure, holding the remainder in cash.
        #
        #   Default True. NOTE the tradeoff: renormalizing restores full
        #   deployment but also rescales away part of the volatility and
        #   leverage caps, since a book cut to 0.6 gross by those caps is
        #   pushed back to 1.0. Relative weights still reflect conviction, but
        #   the portfolio can no longer de-risk into cash. Treat this as a
        #   tuned hyperparameter (it is in backtest.tuning's grid), not a
        #   setting to fix by inspecting test-period results.
        self.fully_invested = True
        
        # State-persistent layers
        self.regime_manager = RegimeManager()                  # L2
        self.switch_manager = StrategySwitchManager()          # L9
        # Initialize with policy capital, or default to 10K if no policy
        initial_capital = policy.total_capital if policy else 10_000.0
        self.portfolio_state = PortfolioState(cash=initial_capital, initial_capital=initial_capital)  # L11 State
        self.monitor = PerformanceMonitor()                     # L12
        
        # Hierarchical Bandit System (persistent across restarts).
        # bandit_dir lets a backtest isolate learned state per run.
        self.ensemble_bandits = BanditPersistenceManager.load(base_dir=bandit_dir)  # L5
        
        self.current_strategy: Optional[str] = None

        # Tracking for post-trade feedback
        self.last_decisions: Dict[str, str] = {}    # Ticker → Strategy Name
        self.last_regime: Optional[str] = None       # For transition detection
        # Captured at decision time so the NEXT cycle can attribute the
        # realized move to the strategy that chose the exposure.
        self.last_participation: Dict[str, float] = {}   # Ticker → [0, 1]
        self.last_prices: Dict[str, float] = {}          # Ticker → close at T
        self.last_ambiguous: Dict[str, bool] = {}        # Ticker → regime uncertain
    
    def set_policy(self, policy: UserPolicy) -> None:
        self.policy = policy
        # Initialize portfolio state with policy capital
        self.portfolio_state = PortfolioState(cash=policy.total_capital)

    # ------------------------------------------------------------------
    # Per-strategy historical performance (middle term of the L6 score)
    # ------------------------------------------------------------------

    def _replayed_signal(self, spec, ticker: str, df: pd.DataFrame) -> tuple:
        """
        Signal/confidence the strategy would have emitted on the LAST bar of df.

        Cached on (strategy, ticker, last date) because consecutive cycles
        replay heavily overlapping windows — without this the replay would
        recompute the same signals every week.
        """
        last_date = df["Date"].iloc[-1] if "Date" in df.columns else df.index[-1]
        key = (spec.name, ticker, pd.Timestamp(last_date))

        cached = self._signal_replay_cache.get(key)
        if cached is not None:
            return cached

        try:
            outputs = run_single_strategy(spec, {ticker: df})
        except Exception:
            outputs = []

        result = (0, 0.0)
        for out in outputs:
            if out.ticker == ticker:
                result = (out.signal, out.confidence)
                break

        self._signal_replay_cache[key] = result
        return result

    def _strategy_past_sharpe(self, spec, ticker: str, df: pd.DataFrame,
                              timeframe: int) -> float:
        """
        Risk-adjusted return the STRATEGY itself produced over its TIMEFRAME.

        Why this replaces the old computation
        -------------------------------------
        The previous version computed the Sharpe ratio of the STOCK over the
        last `timeframe` bars. That value depends only on (ticker, timeframe)
        — never on the strategy — so any two candidates sharing a timeframe
        received identical scores and the term could not rank them. Bull-
        Volatile has six of ten strategies at timeframe=20, so when the top-5
        candidates tied, min == max, the normalizer fell back to a range of
        1.0, and every candidate scored exactly 0.0. Forty percent of the
        selection score was then a constant.

        This version replays the strategy's own decisions: for each bar in the
        lookback it recomputes the signal from data up to that bar only, then
        scores the next bar's return by how far the signal deviated from a
        neutral stance — the same attribution the bandits are rewarded on, so
        the scorer and the reward measure the same quantity.

        Look-ahead: bar i uses df[:i+1] to decide and (i -> i+1) to score, and
        i+1 never exceeds the last row of df. Since df is already the decision
        slice (dates <= T), nothing after T is touched.
        """
        if "Close" not in df.columns or len(df) < 3:
            return 0.0

        # Need one extra bar to realize the final decision's return.
        n_eval = min(int(timeframe), len(df) - 1, int(self.past_return_max_evals))
        if n_eval < 2:
            return 0.0

        closes = df["Close"].to_numpy(dtype=float)
        start = len(df) - 1 - n_eval

        attributed = []
        for i in range(start, len(df) - 1):
            if closes[i] <= 0:
                continue
            signal, confidence = self._replayed_signal(spec, ticker, df.iloc[:i + 1])
            participation = signal_participation(signal, confidence)
            fwd_return = (closes[i + 1] - closes[i]) / closes[i]
            attributed.append(attributable_return(participation, fwd_return))

        if len(attributed) < 2:
            return 0.0

        arr = np.asarray(attributed, dtype=float)
        std = float(arr.std())
        if std < 1e-9:
            # A strategy that never took a position has no risk-adjusted
            # record to speak of; score it neutral rather than infinite.
            return 0.0

        return float(arr.mean() / std) * float(np.sqrt(252))

    def _stock_past_sharpe(self, df: pd.DataFrame, timeframe: int) -> float:
        """Legacy term: the STOCK's trailing Sharpe. Ablation arm only."""
        if "Close" not in df.columns or len(df) < timeframe:
            return 0.0
        closes = df["Close"].to_numpy(dtype=float)[-int(timeframe):]
        if len(closes) <= 1:
            return 0.0
        daily = np.diff(closes) / closes[:-1]
        std = float(np.std(daily))
        if std <= 1e-6:
            return 0.0
        return float(np.mean(daily) / std) * float(np.sqrt(252))

    def run(
        self,
        stock_data_dict: Dict[str, pd.DataFrame],
        user_weights: Optional[pd.Series] = None,
        current_date: Optional[str] = None,
        execution_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        commission_per_trade: float = 1.0,
        cost_bps: float = 0.0,
        execution_date: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run one full decision + execution cycle.

        Args:
            stock_data_dict: DECISION data. Must contain no bars after
                `current_date` — everything the model sees for regime
                detection, strategy evaluation and sizing comes from here.
            current_date: signal date T (decisions use closes through T).
            execution_data_dict: EXECUTION data, which must additionally
                contain the T+1 bar so fills can occur at the next session's
                open. Kept separate from decision data so a backtest can
                enforce point-in-time discipline: withholding T+1 from the
                decision path is what prevents the engine from seeing its own
                execution bar. Defaults to stock_data_dict (live/UI use,
                where the frame ends at the last available bar anyway).
            commission_per_trade: flat per-fill commission in currency units.
            cost_bps: proportional cost in basis points of traded notional
                (bid-ask/slippage proxy). Total fee = commission + bps.
        """
        start_time = time.time()
        current_date_str = current_date or datetime.now().strftime("%Y-%m-%d")
        
        if self.policy is None:
            raise ValueError("Policy not set. Call set_policy() first.")

        # Execution may see one extra bar (T+1) that the decision path must not.
        if execution_data_dict is None:
            execution_data_dict = stock_data_dict
        
        # === 0. USER & POLICY ===
        if user_weights is None:
            user_weights = pd.Series(self.policy.weights)
        risk_limits = self.policy.get_strictest_limits()
        risk_tolerance = self._infer_tolerance(risk_limits)
        
        # === 1. MARKET DATA & FEATURES ===
        enriched_data = {}
        for ticker, df in stock_data_dict.items():
            if df is not None and not df.empty:
                enriched_data[ticker] = compute_all_features(
                    df, use_garch=getattr(self, "use_garch", True)
                )
                
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
            dominant_regime = "Crisis"
            regime_outputs = {}
            allowed_strategies = ["Defensive"]
            removed_strategies = ["ALL_OTHERS"]
            bandit_scores = {}
            per_stock_strategies = {t: "Defensive" for t in stock_data_dict.keys()}
            per_stock_details = {}
            per_stock_participation = {t: 0.0 for t in stock_data_dict.keys()}

            # Every ticker is marked "Defensive", so the sizing block below
            # zeroes its target weight, which makes L8 emit SELL for every
            # open position. No explicit position_sizes assignment is needed
            # here — one used to exist and was silently overwritten by the
            # unconditional sizing block further down.

            switch_decision = SwitchDecision(
                should_switch=True,
                reason="Emergency Protocol",
                current_strategy=self.current_strategy or "None",
                new_strategy="EMERGENCY_EXIT",
                new_probability=1.0,
            )

            # L12 reads .selection_reason off this object unconditionally, so
            # it must be a real object rather than None.
            strategy_decision = type('StrategyDecision', (), {
                'selected_strategy': selected_strategy,
                'scores': {},
                'bandit_score': 1.0,
                'selection_reason': (
                    f'Kill switch: drawdown {current_drawdown:.2%} exceeded '
                    f'limit {self.policy.emergency_drawdown_threshold:.2%}'
                ),
                'expected_return': 0.0,
                'alternatives': [],
                'rationale': 'L0 emergency drawdown protocol (overrides L5)',
            })()

            # Suppress next-cycle bandit feedback. The liquidation was forced
            # by L0 policy, not chosen by the bandits, so crediting or
            # blaming any strategy arm for its outcome would corrupt the
            # learned weights.
            self.last_decisions = {}
            self.last_per_stock_regimes = {}
            self.last_regime = dominant_regime
            
        else:
            # ==== PRE-MARKET: PER-STOCK FLOW ====
            # Each stock gets its own HMM regime detection from its own data,
            # blended with its own Bandit A trust weights.
            
            per_stock_strategies = {}
            per_stock_details = {}
            per_stock_allowed = {}
            per_stock_participation = {}   # Ticker → capital participation [0, 1]
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
                w_global, w_hmm = getattr(self, "blend_weights", (0.60, 0.40))
                blended = blend_regime(
                    hmm_posteriors, bandit_a_weights,
                    w_global=w_global, w_hmm=w_hmm,
                )
                
                dominant_regime = max(blended, key=blended.get)
                hmm_confidence = max(blended.values())
                stability = regime_output.stability_score if regime_output else 0.5
                
                # Confidence gate
                gate = getattr(self, "confidence_gate", 0.55)
                is_ambiguous = hmm_confidence < gate
                if is_ambiguous:
                    print(f"  ⚠ {ticker}: Ambiguous regime (confidence={hmm_confidence:.2f} < {gate})")
                
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
                
                # "regime" = this regime's 10-strategy pod (default).
                # "all"    = all 40 strategies, ignoring the regime gate.
                if getattr(self, "strategy_pool", "regime") == "all":
                    pool_regimes = ["Bull-Quiet", "Bull-Volatile", "Sideways", "Crisis"]
                else:
                    pool_regimes = [dominant_regime]

                strategy_outputs = []
                for pool_regime in pool_regimes:
                    try:
                        strategy_outputs.extend(run_strategies_for_regime(
                            regime=pool_regime,
                            stock_data_dict={ticker: df},
                        ))
                    except Exception as e:
                        print(f"⚠️ Strategy execution failed for {ticker} (regime={pool_regime}): {e}")
                
                if not strategy_outputs:
                    per_stock_strategies[ticker] = "Defensive"
                    per_stock_details[ticker] = {"allowed": ["Defensive"], "scores": {}, "no_strategies": True, "regime": dominant_regime}
                    per_stock_allowed[ticker] = ["Defensive"]
                    per_stock_participation[ticker] = 0.0
                    continue

                # Index this ticker's outputs by strategy name so the winner's
                # signal/confidence can be recovered after ranking. These were
                # previously discarded, which left strategy choice with no
                # causal path to the portfolio.
                outputs_by_name = {
                    out.strategy_name: out
                    for out in strategy_outputs
                    if out.ticker == ticker
                }

                available_strategy_names = list(outputs_by_name.keys())

                if not available_strategy_names:
                    per_stock_strategies[ticker] = "Defensive"
                    per_stock_details[ticker] = {"allowed": ["Defensive"], "scores": {}, "no_strategies": True, "regime": dominant_regime}
                    per_stock_allowed[ticker] = ["Defensive"]
                    per_stock_participation[ticker] = 0.0
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
                spec_map = {spec.name: spec for spec in strategy_specs}

                # Per-strategy past return using each strategy's own TIMEFRAME
                past_mode = getattr(self, "past_return_mode", "strategy")
                raw_scores = []
                for strat_name, score_b in top_5_strategies:
                    tf = timeframe_map.get(strat_name, 30)  # fallback 30 days
                    spec = spec_map.get(strat_name)

                    # Risk-adjusted past return. "strategy" replays the
                    # strategy's own signals; "stock" is the legacy term that
                    # cannot separate strategies sharing a timeframe.
                    if past_mode == "off" or spec is None:
                        risk_adj_return = 0.0
                    elif past_mode == "stock":
                        risk_adj_return = self._stock_past_sharpe(df, tf)
                    else:
                        risk_adj_return = self._strategy_past_sharpe(spec, ticker, df, tf)

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
                spread_ret = max_ret - min_ret
                # All candidates tied: the term carries no ranking information,
                # so make it NEUTRAL. Mapping every candidate to 0.0 (the old
                # behavior) silently zeroed out w_r of the score instead.
                all_tied = spread_ret <= 1e-12

                w_b, w_r, w_c = getattr(self, "score_weights", (0.3, 0.4, 0.3))
                if past_mode == "off":
                    # Redistribute the middle term's weight proportionally
                    # rather than letting it shrink every candidate equally.
                    bc_total = w_b + w_c
                    if bc_total > 0:
                        w_b, w_r, w_c = w_b / bc_total, 0.0, w_c / bc_total

                stock_scored = []
                for s in raw_scores:
                    norm_ret = 0.5 if all_tied else (s["risk_adj_ret"] - min_ret) / spread_ret

                    # Final Score = w_b*θ_B + w_r*Risk-Adj Return + w_c*θ_C
                    final_score = (w_b * s["theta_b"]) + (w_r * norm_ret) + (w_c * s["theta_c"])
                    
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

                # === STEP 6: Winner's directional view → capital participation ===
                # This is where strategy selection becomes consequential: the
                # chosen strategy's own signal and confidence scale the target
                # weight in L7. A bearish winner exits the name to cash.
                winner_output = outputs_by_name.get(winner_name)
                if winner_output is not None:
                    winner_signal = winner_output.signal
                    winner_confidence = winner_output.confidence
                else:
                    # Winner not present in this ticker's outputs (e.g. the
                    # "Defensive" fallback). Stay flat rather than guess.
                    winner_signal = -1
                    winner_confidence = 0.0

                if getattr(self, "use_signal_participation", True):
                    participation = signal_participation(winner_signal, winner_confidence)
                else:
                    # Ablation: strategy choice does not affect position size,
                    # reproducing the engine's behavior before signals were wired in.
                    participation = 1.0
                per_stock_participation[ticker] = participation

                print(
                    f"     ↳ {ticker}: {winner_name} "
                    f"signal={winner_signal:+d} conf={winner_confidence:.2f} "
                    f"→ participation={participation:.2f}"
                )
                
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
                    "winner_signal": winner_signal,
                    "winner_confidence": winner_confidence,
                    "participation": participation,
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
            # Capture what the strategy actually decided and the price it
            # decided at, so next cycle can compute a reward attributable to
            # THIS strategy rather than to the asset's raw drift.
            self.last_participation = dict(per_stock_participation)
            self.last_ambiguous = {
                t: bool(info.get("is_ambiguous", False))
                for t, info in regime_outputs.items()
            }
            try:
                self.last_prices = snapshot_prices(stock_data_dict, current_date_str)
            except Exception:
                self.last_prices = {}
            
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

        # Winner's signal x confidence, per ticker. Missing entries default to
        # 1.0 (no conviction adjustment) rather than 0, so a ticker that never
        # reached strategy selection is not silently liquidated.
        participation_series = pd.Series({
            t: per_stock_participation.get(t, 1.0)
            for t in sizing_weights.index
        })

        # Size against total equity (cash + holdings), not cash alone. Using
        # cash made Capital_Allocation collapse to ~0 after the first cycle,
        # since the book is close to fully invested by then.
        equity_snapshot = snapshot_prices(stock_data_dict, current_date_str)
        sizing_capital = self.portfolio_state.current_equity(equity_snapshot)

        position_sizes = compute_position_sizes(
            user_weights=sizing_weights,
            forecast_vol=forecast_vol,
            total_capital=sizing_capital,
            stability_scores=stability_series,
            participation=participation_series,
            target_vol=0.15,
            max_vol=risk_limits["max_volatility"],
            max_dd=risk_limits["max_drawdown"],
            max_leverage=risk_limits["max_leverage"],
            fully_invested=getattr(self, "fully_invested", False),
        )
        
        # === 8. SIGNAL GENERATION ===
        # Convert position sizes to portfolio weights for signal generation
        if isinstance(position_sizes, pd.Series):
            new_portfolio_df = pd.DataFrame({
                "Ticker": position_sizes.index.tolist(),
                "Weight": position_sizes.values.tolist()
            })
        elif position_sizes is None or position_sizes.empty:
            new_portfolio_df = pd.DataFrame(columns=["Ticker", "Weight"])
        else:
            # Select columns by NAME. The previous positional slice took the
            # last column — Capital_Allocation, denominated in dollars — as
            # "Weight". That only produced sane numbers because L8 then
            # renormalized the column back into fractions, which also silently
            # cancelled the volatility and leverage caps.
            new_portfolio_df = pd.DataFrame({
                "Ticker": position_sizes["Ticker"].tolist(),
                "Weight": position_sizes["Adjusted_Weight"].tolist(),
            })
        
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
        if getattr(self, "logging_enabled", True):
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
        # Prefer an explicitly supplied execution date. A rule-based holiday
        # calendar cannot know about ad-hoc closures (Hurricane Sandy, funeral
        # closures) or observed-holiday shifts where the NYSE traded anyway
        # (2010-12-31, Juneteenth 2021) — nine such days between 2005 and
        # 2025. When the caller knows which bars actually exist, it should say
        # so; guessing wrong meant asking for a bar that was not in the
        # execution slice and filling at a stale price.
        execution_date_str = execution_date or get_next_trading_day(current_date_str)
        
        execution_report = {}
        if not signals.empty:
            try:
                execution_report = run_execution_cycle(
                    state=self.portfolio_state,
                    price_data_dict=execution_data_dict,  # includes the T+1 bar
                    signals_df=signals,
                    new_portfolio_weights=new_portfolio_df,
                    date=execution_date_str,  # T+1 execution date
                    commission_per_trade=commission_per_trade,
                    cost_bps=cost_bps,
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
                if getattr(self, "logging_enabled", True):
                    if fills_df is not None and not fills_df.empty:
                        log_transactions_from_fills(
                            fills_df=fills_df,
                            execution_date=execution_date_str,  # T+1 execution date
                        )

                # Log cycle summary with actual P/L values
                cycle_num = get_latest_cycle_number() + 1 if getattr(self, "logging_enabled", True) else 0

                # Count positions with non-zero qty
                num_positions = len(self.portfolio_state.positions)

                # Extract marginal fees from this execution only
                fees_this_cycle = 0.0
                if execution_report and "fees_paid" in execution_report:
                    fees_this_cycle = execution_report["fees_paid"]

                if getattr(self, "logging_enabled", True):
                    log_cycle_summary(
                        execution_date=execution_date_str,  # T+1 execution date
                        rebalance_frequency=self.policy.rebalance_frequency,
                        portfolio_value=portfolio_value,
                        cash=self.portfolio_state.cash,
                        initial_capital=self.policy.total_capital,
                        pnl=pnl,
                        return_pct=return_pct,
                        cycle_number=cycle_num,
                        realized_pnl=realized_pnl,
                        unrealized_pnl=unrealized_pnl,
                        cumulative_realized_pnl=realized_pnl,
                        transaction_costs=fees_this_cycle,  # MARGINAL, not cumulative
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
            emergency_triggered=emergency_triggered,
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

            # === 1. Realized asset move over the ACTUAL holding period ===
            # Previously this was a single day's return regardless of the
            # rebalance interval, so a week-long position was graded on its
            # last day.
            entry_price = self.last_prices.get(ticker)
            current_price = None
            if "Close" in df.columns and len(df):
                current_price = float(df["Close"].iloc[-1])

            if entry_price and current_price and entry_price > 0:
                asset_return = (current_price - entry_price) / entry_price
            else:
                # No usable entry reference — fall back to one day.
                for col in ("Returns", "Return_1D"):
                    if col in df.columns:
                        asset_return = float(df[col].iloc[-1])
                        break
                else:
                    asset_return = 0.0

            # === 2. Attribute the move to the STRATEGY, not the asset ===
            # Credit the deviation from a neutral stance. Feeding the raw
            # asset return gave every candidate strategy the same reward,
            # so the bandits could not learn which one was responsible.
            participation = self.last_participation.get(ticker, 0.5)
            r_attr = attributable_return(participation, asset_return)

            # === 3. Risk-adjust (paper Eq. 7) ===
            # compute_reward was imported but never called; raw returns went
            # straight to the bandits with no outlier dampening.
            vol_60d = 0.15
            if "Realized_Vol" in df.columns:
                v = df["Realized_Vol"].iloc[-1]
                if np.isfinite(v) and v > 0:
                    vol_60d = float(v)

            r_final = compute_reward(r_attr, vol_60d)

            # === 4. Differentiate per bandit level (paper Sec VI-E) ===
            # Signed rewards — EXP3 needs the sign so losing arms decay.
            rewards = differentiated_exp3_rewards(
                r_final,
                is_ambiguous=self.last_ambiguous.get(ticker, False),
            )

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
