# backtest/hmm_schedule.py
"""
Rolling HMM refit schedule for walk-forward evaluation.

The shipped macro model is a single HMM trained on SPY 2014-2024. Using it to
decide anything inside that span is in-sample: the model has already seen the
outcomes it is being asked to anticipate. This module replaces it with a fold
schedule where each model is fit ONLY on data strictly before the fold it
governs.

    fold k:  train on [fold_start - train_years, fold_start)
             govern  [fold_start, fold_{k+1}_start)

The strict inequality is enforced in two places — PointInTimeData.training_slice
asserts it on the data, and assert_no_lookahead() re-checks it per fold — so a
future refactor cannot quietly reintroduce the leak.

Label switching is handled by RegimeDetector.fit(), which derives state names
from each state's own return moments. Without that, every refit could permute
the labels and scramble the bandits, which are keyed by regime NAME.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from layers.L2_regime_intelligence.regime_selection import RegimeDetector


@dataclass
class HMMFold:
    """One trained model and the date range it is allowed to govern."""

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # EXCLUSIVE — model never sees this date
    govern_start: pd.Timestamp
    govern_end: pd.Timestamp     # inclusive
    detector: RegimeDetector
    n_train_samples: int
    state_labels: dict

    def governs(self, date) -> bool:
        d = pd.Timestamp(date).normalize()
        return self.govern_start <= d <= self.govern_end


def _macro_training_frame(
    pit,
    macro_ticker: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame | None:
    """
    Assemble the macro training frame from data strictly before `cutoff`.

    Prefers a dedicated index proxy (SPY). If it is absent from the universe,
    falls back to an equal-weight composite of the available names so the
    regime model still describes "the market" rather than one stock.
    """
    # Dedicated macro proxy, carried outside the traded universe.
    macro = pit.macro_training_slice(cutoff) if hasattr(pit, "macro_training_slice") else None
    if macro is not None:
        return macro

    training = pit.training_slice(cutoff)
    if not training:
        return None

    if macro_ticker in training:
        return training[macro_ticker]

    # Equal-weight composite fallback
    closes = {t: df.set_index("Date")["Close"] for t, df in training.items()}
    matrix = pd.DataFrame(closes).sort_index().dropna(how="all")
    if matrix.empty:
        return None

    normalized = matrix / matrix.iloc[0]
    composite = normalized.mean(axis=1)

    return pd.DataFrame({
        "Date": composite.index,
        "Open": composite.values,
        "High": composite.values,
        "Low": composite.values,
        "Close": composite.values,
        "Volume": 0.0,
    }).reset_index(drop=True)


def build_fold_schedule(
    pit,
    config,
    macro_ticker: str = "SPY",
    verbose: bool = True,
) -> list[HMMFold]:
    """
    Fit one HMM per refit period across the evaluation span.

    Returns folds ordered by date. Folds whose training window is too short
    are skipped; dates they would have governed fall back to the preceding
    fold, or to rule-based detection if none exists yet.
    """
    eval_start = pd.Timestamp(config.start).normalize()
    eval_end = pd.Timestamp(config.end).normalize()

    boundaries: list[pd.Timestamp] = []
    if getattr(config, "hmm_refit_at_rebalance", False):
        # Refit at EVERY rebalance date, each on a trailing window ending
        # strictly before that date. This is the strongest form of the
        # walk-forward requirement — the model deciding at t has seen nothing
        # at or after t. Cost is ~0.12s per fit and folds are built once and
        # shared across all arms and seeds, so the whole schedule is ~1 min.
        boundaries = [
            pd.Timestamp(d).normalize()
            for d in pit.rebalance_schedule(
                config.start, config.end, config.rebalance_frequency
            )
        ]
    else:
        cursor = eval_start
        while cursor <= eval_end:
            boundaries.append(cursor)
            cursor = cursor + pd.DateOffset(months=config.hmm_refit_months)

    # Long schedules print one line per fold otherwise, burying everything else.
    quiet = verbose and len(boundaries) > 24
    if quiet:
        print(f"  fitting {len(boundaries)} HMM folds "
              f"({config.hmm_train_years}y trailing window)...")

    folds: list[HMMFold] = []
    for idx, fold_start in enumerate(boundaries):
        fold_end = (
            boundaries[idx + 1] - pd.Timedelta(days=1)
            if idx + 1 < len(boundaries) else eval_end
        )
        train_start = fold_start - pd.DateOffset(years=config.hmm_train_years)

        frame = _macro_training_frame(pit, macro_ticker, cutoff=fold_start)
        if frame is None:
            continue

        frame = frame[frame["Date"] >= train_start].reset_index(drop=True)
        if len(frame) < config.hmm_min_train_samples:
            if verbose and not quiet:
                print(f"  fold {idx}: only {len(frame)} samples before "
                      f"{fold_start.date()}, skipped")
            continue

        # Fixed random_state across folds (not config.seed) so that seed
        # variation measures BANDIT stochasticity only. Letting the HMM
        # re-fit differently per seed would confound the two sources.
        detector = RegimeDetector(random_state=0)
        try:
            detector.fit(frame)
        except Exception as exc:
            if verbose and not quiet:
                print(f"  fold {idx}: HMM fit failed ({exc}), skipped")
            continue

        fold = HMMFold(
            fold_index=idx,
            train_start=pd.Timestamp(frame["Date"].min()),
            train_end=fold_start,
            govern_start=fold_start,
            govern_end=fold_end,
            detector=detector,
            n_train_samples=len(frame),
            state_labels=dict(detector.state_labels or {}),
        )
        folds.append(fold)

        if verbose and not quiet:
            print(f"  fold {idx}: train {fold.train_start.date()}..<{fold_start.date()} "
                  f"(n={len(frame)}) governs {fold.govern_start.date()}..{fold_end.date()}")

    if quiet and folds:
        print(f"  {len(folds)} folds fitted: "
              f"{folds[0].govern_start.date()}..{folds[-1].govern_end.date()}, "
              f"train n={folds[0].n_train_samples}..{folds[-1].n_train_samples}")

    return folds


def select_fold(folds: list[HMMFold], date) -> HMMFold | None:
    """
    Most recent fold whose training window closed before `date`.

    Never returns a fold trained on data at or after `date`, even if the
    governing ranges were somehow misconfigured — this is the last line of
    defense before a model reaches the decision path.
    """
    d = pd.Timestamp(date).normalize()
    eligible = [f for f in folds if f.train_end <= d]
    if not eligible:
        return None
    return max(eligible, key=lambda f: f.train_end)


def assert_no_lookahead(folds: list[HMMFold]) -> None:
    """Fail loudly if any fold could govern a date its training window covers."""
    for fold in folds:
        if fold.train_end > fold.govern_start:
            raise AssertionError(
                f"fold {fold.fold_index} trains through {fold.train_end.date()} "
                f"but governs from {fold.govern_start.date()} — look-ahead"
            )
        if fold.train_start >= fold.train_end:
            raise AssertionError(
                f"fold {fold.fold_index} has an empty training window"
            )
