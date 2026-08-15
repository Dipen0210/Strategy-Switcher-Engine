# layers/L2_regime_intelligence/regime_labels.py
"""
Deterministic HMM state → regime label assignment.

The label-switching problem
---------------------------
Baum-Welch is unsupervised, so the integer labels it assigns to hidden states
are arbitrary and permute freely between fits. Nothing guarantees that state 3
is "Crisis" on the next refit — it might be Bull-Quiet.

Previously both the trainer and the detector hard-coded

    {0: "Bull-Quiet", 1: "Bull-Volatile", 2: "Sideways", 3: "Crisis"}

which silently assumes an ordering EM never promised. That is tolerable for a
single frozen model but breaks the moment you refit — which a walk-forward
backtest does on every fold. Worse, because the bandits are keyed by regime
NAME, a permuted relabel would scramble every learned weight: what the system
had learned about "Crisis" would suddenly be applied to Bull-Quiet.

This module derives the mapping from each fitted state's own return
distribution, so the same economic state always receives the same name:

    Crisis        most negative mean return   (tie-break: highest variance)
    Bull-Quiet    best return/volatility ratio among the remainder
    Bull-Volatile higher variance of the final two
    Sideways      whatever remains

All comparisons carry an explicit index tie-break so the result is fully
deterministic even under exact numerical ties.

Note the moments are read in the model's own (standardized) feature space.
Standardization is a positive affine map applied per column, so it preserves
the ordering of both means and variances — the ranking is unaffected.
"""

from __future__ import annotations

import numpy as np

from layers.L2_regime_intelligence.features import RETURN_FEATURE_IDX


# Canonical regime vocabulary. These strings key the Bandit A/B/C models on
# disk, so they must remain stable across retrains.
CANONICAL_REGIMES = ("Bull-Quiet", "Bull-Volatile", "Sideways", "Crisis")


def _state_moments(model, return_idx: int = RETURN_FEATURE_IDX):
    """
    Extract (mean_return, return_variance) per hidden state.

    Handles hmmlearn's covariance_type variants: "full"/"tied" give a matrix
    per state, "diag"/"spherical" give a vector.
    """
    means = np.asarray(model.means_, dtype=float)
    covars = np.asarray(model.covars_, dtype=float)

    mu = means[:, return_idx]

    if covars.ndim == 3:            # full / tied -> (n_states, k, k)
        var = covars[:, return_idx, return_idx]
    elif covars.ndim == 2:          # diag       -> (n_states, k)
        var = covars[:, return_idx]
    else:                           # spherical  -> (n_states,)
        var = np.asarray(covars, dtype=float).reshape(-1)

    return mu, np.abs(var)


def derive_state_labels(model, return_idx: int = RETURN_FEATURE_IDX) -> dict[int, str]:
    """
    Map raw HMM state indices to canonical regime names.

    Args:
        model: a fitted hmmlearn GaussianHMM
        return_idx: column of the return feature in the training matrix

    Returns:
        {state_index: regime_name}, deterministic for a given fitted model.
    """
    mu, var = _state_moments(model, return_idx)
    n_states = len(mu)

    if n_states != 4:
        # Generic fallback: rank by mean return, best-to-worst, and assign the
        # canonical names in that order.
        order = sorted(range(n_states), key=lambda i: (-mu[i], i))
        return {
            state: CANONICAL_REGIMES[min(rank, len(CANONICAL_REGIMES) - 1)]
            for rank, state in enumerate(order)
        }

    sigma = np.sqrt(np.maximum(var, 1e-12))
    remaining = set(range(4))
    labels: dict[int, str] = {}

    # 1. Crisis — most negative drift; ties broken by higher variance.
    crisis = min(remaining, key=lambda i: (mu[i], -var[i], i))
    labels[crisis] = "Crisis"
    remaining.discard(crisis)

    # 2. Bull-Quiet — best risk-adjusted drift among what is left.
    bull_quiet = max(remaining, key=lambda i: (mu[i] / sigma[i], -var[i], -i))
    labels[bull_quiet] = "Bull-Quiet"
    remaining.discard(bull_quiet)

    # 3. Bull-Volatile — the more variable of the final pair.
    bull_volatile = max(remaining, key=lambda i: (var[i], -i))
    labels[bull_volatile] = "Bull-Volatile"
    remaining.discard(bull_volatile)

    # 4. Sideways — the remainder.
    labels[remaining.pop()] = "Sideways"

    return labels


def describe_state_labels(
    model,
    labels: dict[int, str],
    return_idx: int = RETURN_FEATURE_IDX,
) -> str:
    """Render the derived mapping as a readable table for training logs."""
    mu, var = _state_moments(model, return_idx)
    sigma = np.sqrt(np.maximum(var, 1e-12))

    lines = [f"  {'state':>5}  {'mean_ret':>10}  {'std':>10}  {'ret/std':>9}  label"]
    for state in sorted(labels, key=lambda s: CANONICAL_REGIMES.index(labels[s])):
        lines.append(
            f"  {state:>5}  {mu[state]:>10.4f}  {sigma[state]:>10.4f}  "
            f"{mu[state] / sigma[state]:>9.4f}  {labels[state]}"
        )
    return "\n".join(lines)
