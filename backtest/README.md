# Experimental Harness

Walk-forward evaluation for the Strategy Switcher Engine, built to address five
methodological requirements: true out-of-sample design, prevention of HMM
look-ahead, documented parameter selection, ablation testing, and
robustness/statistical analysis.

## Quick start

```bash
# 1. Prove the harness cannot see the future. Run this FIRST, and again after
#    any change to data handling or scheduling. If it fails, nothing else counts.
python -m backtest.cli leakage

# 2. Component-removal ladder + baselines, across seeds
python -m backtest.cli ablate --seeds 20

# 3. Tune on the validation window, freeze, evaluate the holdout once
python -m backtest.cli protocol --seeds 20

# 4. Transaction-cost sensitivity
python -m backtest.cli costs --seeds 10
```

A smaller end-to-end demonstration lives in `backtest/demo_run.py`.

## Design

### Point-in-time discipline

Every slice handed to the engine is truncated at an explicit cutoff, and the
truncation is asserted at runtime rather than assumed.

| Slice | Contains | Used for |
|---|---|---|
| decision | bars ≤ T | regime detection, strategy selection, sizing |
| execution | bars ≤ T+1 | fills at the T+1 open |
| training | bars < fold start | HMM fitting |

Keeping decision and execution data separate is what stops the engine from
seeing the bar it is about to trade into. `StrategyEngine.run()` takes them as
two arguments for exactly this reason.

### Rolling HMM refits

The shipped `models/macro_hmm.pkl` is trained on 2014–2024. Using it to decide
anything inside that span is in-sample. The harness instead fits one model per
refit period, each trained strictly before the period it governs:

```
fold k:  train on [fold_start − train_years, fold_start)
         govern  [fold_start, fold_{k+1}_start)
```

`select_fold()` refuses to return a model whose training window covers the
decision date, even if the governing ranges were misconfigured.

Because refits happen repeatedly, **deterministic label sorting is mandatory**
(`layers/L2_regime_intelligence/regime_labels.py`). EM numbers its states
arbitrarily; without deriving names from each state's return moments, a refit
could permute labels and apply everything the bandits learned about Crisis to
Bull-Quiet.

### Causal features only

`GARCH_Vol` is excluded from the HMM feature set. `arch_model.fit()` estimates
parameters over the whole sample, so its conditional volatility at time *t*
embeds information from *t+1…T*. The feature set is `Return_1D`,
`Realized_Vol`, `MA_Slope` — all computable at *t* from data up to *t*.

Normalization statistics are fitted on the training window and serialized with
the model, so inference never standardizes using moments of the data it scores.

## Leakage tests

`python -m backtest.cli leakage` runs four tests:

| Test | Checks |
|---|---|
| `decision_slices_truncated` | no bar after T reaches the decision path |
| `folds_trained_before_use` | no HMM governs a date inside its training window |
| `execution_uses_next_open` | fills occur at the T+1 open, not the T close |
| `future_perturbation` | **decisive** — corrupt all data after a cutoff; prior decisions and equity must be bit-identical |

The perturbation test is the one that matters. Structural assertions can be
defeated by a refactor; that test cannot, because if any future information
reaches a decision, scrambling the future must change that decision. It also
verifies the perturbation actually changed *post*-cutoff results, so a
no-op corruption cannot produce a false pass.

## Ablation ladder

Each arm disables exactly one mechanism, mapping to a specific claim:

| Arm | Tests |
|---|---|
| `no_bandit_a` | Bandit A adds value over the HMM alone (Sec VI-B) |
| `no_bandit_b` | Bandit B learns which strategies work (Sec VI-C) |
| `no_bandit_c` | Per-stock-per-regime models add value (Sec VI-D) |
| `no_hmm` | Regime detection drives allocation (Sec IV) |
| `no_pods` | Pods beat an unrestricted universe (**Contribution #4**) |
| `no_decay` | The 0.99 forgetting curve aids adaptation (Sec VI-F) |
| `no_exploration` | ε-greedy exploration pays for itself (Sec VI-C) |
| `no_signal_participation` | Strategy signals are consequential at all |
| `no_learning` | The RL layer as a whole contributes |

Baselines run through the **same** execution machinery and cost model, so
differences reflect decision logic rather than simulation fidelity:

- `buy_and_hold` — equal-weight the universe once, hold
- `static_40` — the paper's "Static 40": every strategy funded at all times,
  no regime gating, no learning. This is the "Strategy Soup" comparison, and
  it had no implementation before.

## Statistics

**Seeds are not optional.** Bandit weights initialize from `np.random.rand`, so
each run is a draw from a distribution. Report mean ± std across ≥20 seeds; a
single-seed Sharpe is one sample, not an estimate.

- `bootstrap_ci` — stationary **block** bootstrap (daily returns are
  autocorrelated and volatility-clustered; an i.i.d. bootstrap understates
  uncertainty)
- `paired_bootstrap_test` — pairs arms on date so the common market component
  is differenced out
- `probabilistic_sharpe_ratio` — accounts for skew and kurtosis
- `deflated_sharpe_ratio` — corrects for how many configurations were searched
- `subperiod_breakdown` — performance inside named episodes (GFC, COVID, 2022)
- `cost_sensitivity_table` — an edge that vanishes between 5 and 25 bps is a
  cost assumption, not an edge

## Parameter-selection protocol

```
1. Search the grid ONLY on [tune_start, tune_end]
2. Select by mean Sharpe across seeds
3. Freeze the winner; record the trial count
4. Run the holdout exactly once
5. Deflate the holdout Sharpe by that trial count
```

`save_protocol_record()` writes the full trail — what was searched, what was
frozen, when the holdout ran — so a reader can verify the protocol rather than
take it on trust. Re-running the holdout after seeing the result and adjusting
anything converts it into a second tuning set and invalidates it.

## Known limitations (disclose these)

1. **Survivorship bias — not corrected.** The universe is tickers that exist
   today, run backward through history. Delisted and bankrupt names are
   silently excluded, which biases returns upward. Fixing this requires
   point-in-time index constituents, which yfinance cannot supply.
2. **Static slippage.** Costs are `commission + notional × bps`. Real slippage
   expands non-linearly when liquidity thins — precisely during the Crisis
   regimes the architecture claims to handle. Report a cost sweep, not a point
   estimate.
3. **Single macro model across stocks.** One HMM scored on each stock's data,
   blended 60/40 toward a trust vector identical across tickers. Cross-sectional
   regime variation comes only from the 40% HMM term.
4. **Compounding scalars.** Position size is `weight × vol_scalar × stability ×
   participation`. `compute_stability` returns 0.5 without history, so the first
   cycles run at half size, and neutral signals halve again. Observed time-in-
   market is well below 100%, which flatters risk-adjusted metrics relative to
   deployed capital — report `time_in_market` alongside Sharpe.
5. **Long/flat only.** L10 rejects selling more than held, so bearish signals
   map to flat, never short.
