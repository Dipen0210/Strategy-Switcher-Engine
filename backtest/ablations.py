# backtest/ablations.py
"""
Component-removal ladder.

Each ablation disables exactly ONE mechanism and leaves everything else
untouched, so the difference between it and the full engine attributes
performance to that mechanism. Each entry maps to a specific claim in the
paper, listed in CLAIM_MAP below.

Ablations are applied as hooks that receive the constructed engine. Where
possible they flip an explicit engine switch rather than patching decision
logic; where a switch does not exist (the bandits) they replace bound methods
on the specific engine INSTANCE, never on the class, so runs stay isolated.
"""

from __future__ import annotations

import random

import numpy as np


def _noop(*args, **kwargs) -> None:
    """
    Module-level no-op used to disable a bound method on an instance.

    MUST NOT be a lambda. Bandit objects are pickled to disk every cycle, and
    a lambda attached as an instance attribute is unpicklable — which made
    save_all() raise on every cycle, so the affected arms silently produced a
    flat 0.00% equity curve instead of a result.
    """
    return None


def _constant_half(*args, **kwargs) -> float:
    """Fixed per-stock preference for the Bandit C ablation (picklable)."""
    return 0.5


# ---------------------------------------------------------------- hooks

def _ablate_bandit_a(engine, config) -> None:
    """Use raw HMM posteriors; the learned regime-trust prior is removed."""
    engine.blend_weights = (0.0, 1.0)
    engine.ensemble_bandits.global_bandit.update_arm = _noop


def _ablate_bandit_b(engine, config) -> None:
    """Strategy ranking becomes a uniform random draw with no learning."""
    manager = engine.ensemble_bandits.regime_bandits

    def rank_uniform(regime, strategy_names):
        bandit = manager.get_bandit(regime)
        bandit._ensure_strategies(strategy_names)
        for key in bandit.strategies:
            bandit.strategies[key] = 1.0 / len(bandit.strategies)
        k = min(5, len(strategy_names))
        chosen = random.sample(list(strategy_names), k)
        return [(name, 1.0 / len(strategy_names)) for name in chosen]

    manager.rank_strategies = rank_uniform
    manager.update_strategy = _noop


def _ablate_bandit_c(engine, config) -> None:
    """Per-stock preference is constant; only theta_B and past return rank."""
    manager = engine.ensemble_bandits.stock_bandits
    manager.sample = _constant_half
    manager.update_strategy = _noop


def _ablate_hmm(engine, config) -> None:
    """
    Remove regime detection entirely — one fixed regime for all dates.

    Tests whether HMM regime identification contributes anything beyond
    running a single fixed pod.
    """
    from layers.L2_regime_intelligence.regime_selection import RegimeOutput

    fixed = "Sideways"

    def constant_regime(ticker, df):
        return RegimeOutput(
            probabilities={fixed: 1.0},
            dominant_regime=fixed,
            allowed_strategies=[],
            stability_score=1.0,
            hmm_confidence=1.0,
            is_ambiguous=False,
            transition_flag=False,
        )

    engine.regime_manager.predict_regime = constant_regime


def _ablate_pods(engine, config) -> None:
    """
    Flat 40-arm bandit: candidates are all 40 strategies regardless of regime.

    This is the direct test of the paper's Contribution #4 (that clustering
    into regime pods beats an unrestricted universe).
    """
    engine.strategy_pool = "all"


def _ablate_decay(engine, config) -> None:
    """Freeze the forgetting curve (delta = 1.0): weights never revert."""
    bandits = engine.ensemble_bandits
    bandits.global_bandit.decay_all = _noop
    bandits.regime_bandits.decay_all = _noop
    bandits.stock_bandits.decay_all = _noop


def _ablate_exploration(engine, config) -> None:
    """Pure exploitation: epsilon = 0, no forced probing of weak arms."""
    manager = engine.ensemble_bandits.regime_bandits
    original = manager.get_bandit

    def rank_greedy(regime, strategy_names):
        bandit = original(regime)
        bandit._ensure_strategies(strategy_names)
        ranked = sorted(
            ((n, bandit.strategies[n]) for n in strategy_names),
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:5]

    manager.rank_strategies = rank_greedy


def _ablate_signal_participation(engine, config) -> None:
    """
    Strategy choice no longer affects position size.

    Reproduces the engine's behavior before signals were wired into L7, and
    therefore measures whether that link matters at all.
    """
    engine.use_signal_participation = False


def _ablate_to_hmm_only(engine, config) -> None:
    """
    HMM-only: regime gating stays, every preference inside the pod is removed.

    Interpretation note. The architecture commits to ONE strategy per stock per
    cycle, so "equal weight within the active pod" is realized as a uniform
    draw over pod members: every member is equally likely, no member accrues
    preference, and nothing is learned. That isolates the contribution of
    regime detection itself — if the HMM pods carry the performance, this arm
    should hold up; if it matches the full engine, the entire learning stack is
    decoration.

    Distinct from `no_learning`, where the bandits are frozen but still hold
    their random initial weights and therefore express an arbitrary fixed
    preference. Here the preference is uniform at every cycle.
    """
    manager = engine.ensemble_bandits.regime_bandits

    def rank_uniform(regime, strategy_names):
        bandit = manager.get_bandit(regime)
        bandit._ensure_strategies(strategy_names)
        names = list(strategy_names)
        for key in bandit.strategies:
            bandit.strategies[key] = 1.0 / max(len(bandit.strategies), 1)
        k = min(5, len(names))
        chosen = random.sample(names, k)
        return [(name, 1.0 / len(names)) for name in chosen]

    manager.rank_strategies = rank_uniform
    manager.update_strategy = _noop

    engine.ensemble_bandits.global_bandit.update_arm = _noop
    stock = engine.ensemble_bandits.stock_bandits
    stock.sample = _constant_half
    stock.update_strategy = _noop

    # With no learned preference, the past-return term would silently become
    # the sole tie-breaker and this would stop being an HMM-only arm.
    engine.past_return_mode = "off"


def _ablate_strategy_past_return(engine, config) -> None:
    """
    Revert the middle score term to the STOCK's trailing Sharpe.

    This is the pre-fix behavior: the term depended only on (ticker,
    timeframe), so candidates sharing a timeframe were indistinguishable.
    Comparing against the full engine measures whether replaying each
    strategy's own signals actually improves selection.
    """
    engine.past_return_mode = "stock"


def _ablate_past_return(engine, config) -> None:
    """Drop the middle term entirely; only theta_B and theta_C rank."""
    engine.past_return_mode = "off"


def _no_op(engine, config) -> None:
    return None


# ---------------------------------------------------------------- registry

ABLATIONS: dict[str, dict] = {
    "full": {
        "hooks": [_no_op],
        "description": "Complete engine, all components active",
        "tests_claim": "(reference configuration)",
    },
    "no_bandit_a": {
        "hooks": [_ablate_bandit_a],
        "description": "Raw HMM posteriors; no learned regime trust",
        "tests_claim": "Bandit A adds value over the HMM alone (Sec VI-B)",
    },
    "no_bandit_b": {
        "hooks": [_ablate_bandit_b],
        "description": "Random strategy selection within the pod",
        "tests_claim": "Bandit B learns which strategies work (Sec VI-C)",
    },
    "no_bandit_c": {
        "hooks": [_ablate_bandit_c],
        "description": "Constant per-stock preference",
        "tests_claim": "Per-stock-per-regime models add value (Sec VI-D)",
    },
    "no_hmm": {
        "hooks": [_ablate_hmm],
        "description": "One fixed regime for all dates",
        "tests_claim": "Regime detection drives allocation (Sec IV)",
    },
    "no_pods": {
        "hooks": [_ablate_pods],
        "description": "Flat 40-strategy universe, no regime gating",
        "tests_claim": "Pods beat an unrestricted universe (Contribution #4)",
    },
    "no_decay": {
        "hooks": [_ablate_decay],
        "description": "delta = 1.0, weights never revert to uniform",
        "tests_claim": "The 0.99 forgetting curve aids adaptation (Sec VI-F)",
    },
    "no_exploration": {
        "hooks": [_ablate_exploration],
        "description": "epsilon = 0, pure exploitation",
        "tests_claim": "epsilon-greedy exploration pays for itself (Sec VI-C)",
    },
    "no_signal_participation": {
        "hooks": [_ablate_signal_participation],
        "description": "Strategy choice does not affect position size",
        "tests_claim": "Strategy signals are consequential at all",
    },
    "hmm_only": {
        "hooks": [_ablate_to_hmm_only],
        "description": "Regime pods active, uniform choice within pod, no learning",
        "tests_claim": "Regime detection alone drives the result; the bandit "
                       "stack adds an increment on top (Sec VI)",
    },
    "stock_past_return": {
        "hooks": [_ablate_strategy_past_return],
        "description": "Middle score term = stock trailing Sharpe (pre-fix)",
        "tests_claim": "Per-strategy historical performance ranks candidates "
                       "better than the stock's own trailing Sharpe (Sec V)",
    },
    "no_past_return": {
        "hooks": [_ablate_past_return],
        "description": "Middle score term removed; theta_B and theta_C only",
        "tests_claim": "The historical-performance term contributes at all",
    },
    "no_learning": {
        "hooks": [_ablate_bandit_a, _ablate_bandit_b, _ablate_bandit_c],
        "description": "All three bandits frozen; regime gating only",
        "tests_claim": "The RL layer as a whole contributes",
    },
}


def get_hooks(ablation: str) -> dict:
    """Resolve an ablation name to {hook_name: fn} for the runner."""
    if ablation not in ABLATIONS:
        raise KeyError(
            f"unknown ablation '{ablation}'. Available: {sorted(ABLATIONS)}"
        )
    hooks = ABLATIONS[ablation]["hooks"]
    return {f"{ablation}_{i}": fn for i, fn in enumerate(hooks)}


def describe_ablations() -> str:
    lines = [f"{'name':<26} {'tests':<52} description", "-" * 120]
    for name, spec in ABLATIONS.items():
        lines.append(f"{name:<26} {spec['tests_claim']:<52} {spec['description']}")
    return "\n".join(lines)
