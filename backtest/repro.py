# backtest/repro.py
"""
Reproducibility guards.

PYTHONHASHSEED
--------------
Python randomizes string hashing per process. Somewhere in the decision path
the engine's behavior depends on the iteration order of a hash-ordered
container, so two processes with different hash seeds produce different
equity curves from the same seed and the same data.

This was invisible while every run happened in one interpreter: results were
self-consistent and looked deterministic. It surfaced the moment runs were
distributed across worker processes — two identical parallel experiments
disagreed, diverging within a couple of rebalance cycles.

That makes it a REPRODUCIBILITY BUG IN THE ENGINE, not a harness artifact. A
reader who reruns the published script in a fresh interpreter gets different
numbers unless the hash seed is pinned. Pinning it is therefore part of the
experimental protocol and must be stated in the paper.

The env var is read once at interpreter startup, so it cannot be set from
inside a running process — hence the re-exec below.
"""

from __future__ import annotations

import os
import sys

HASH_SEED = "0"


def ensure_hash_seed() -> None:
    """
    Guarantee PYTHONHASHSEED is pinned, re-executing this process if needed.

    Call at the very top of an entry point, before heavy imports. Child
    processes spawned later inherit the pinned value from the environment.
    """
    if os.environ.get("PYTHONHASHSEED") == HASH_SEED:
        return

    os.environ["PYTHONHASHSEED"] = HASH_SEED
    os.execv(sys.executable, [sys.executable] + sys.argv)


def hash_seed_is_pinned() -> bool:
    return os.environ.get("PYTHONHASHSEED") is not None


def warn_if_unpinned() -> None:
    if not hash_seed_is_pinned():
        print(
            "WARNING: PYTHONHASHSEED is not set. Results will not reproduce "
            "across processes. Call backtest.repro.ensure_hash_seed() from "
            "your entry point, or run with PYTHONHASHSEED=0.",
            file=sys.stderr, flush=True,
        )
