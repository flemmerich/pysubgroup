"""Shared checks for the target and quality-function tests."""

import itertools

import numpy as np


def subsets(mask):
    """Every non-empty subset of a Boolean row mask (for small masks only)."""
    rows = np.flatnonzero(mask)
    for size in range(1, len(rows) + 1):
        for chosen in itertools.combinations(rows, size):
            subset = np.zeros(len(mask), dtype=bool)
            subset[list(chosen)] = True
            yield subset


def assert_optimistic(qf, target, mask, **kwargs):
    """The optimistic estimate bounds the quality of every subset of ``mask``."""
    estimate = qf.optimistic_estimate(mask, target)
    best = max(
        (q for q in (qf.evaluate(s, target, **kwargs) for s in subsets(mask)) if not np.isnan(q)),
        default=float("-inf"),
    )
    assert best <= estimate + 1e-9, (type(qf).__name__, best, estimate)
