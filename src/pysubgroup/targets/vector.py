"""Vector-valued target (several numeric attributes per row) and its quality function.

``MeanVectorL1QF`` is the quality function ``q_dst`` (Eq. 2) of H. Grosskreutz,
M. Boley, and M. Krause-Traudes, "Subgroup discovery for election analysis: a
case study in descriptive data mining" (DS 2010): the subgroup's share of the
dataset times the 1-norm distance between the mean target vectors of the
subgroup and of the dataset, for example the vote shares of all parties per
polling district. On one-hot class indicators it equals
``multiclass.ClassDistributionL1QF``.

The paper gives no optimistic estimate; the tight bound used here is derived in
the docstring of ``MeanVectorL1QF``. Written from the paper without reusing code.
Deviation from the paper: empty subgroups score ``nan`` and bound at ``-inf``.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Hashable

import numpy as np
import pandas as pd

from .base import BaseTarget, BoundedQualityFunction

_BLOCK_ROWS = 1 << 16


class VectorTarget(BaseTarget):
    """Mean vector of several numeric target attributes."""

    def fit(self, data: pd.DataFrame, *attributes: Hashable) -> VectorTarget:
        """Store the values of the numeric features ``attributes``, one column each."""
        if not attributes:
            raise ValueError("vector targets require at least one attribute")
        super().fit(data, *attributes)
        for name in attributes:
            dtype = data[name].dtype
            if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                raise TypeError("vector targets require numeric columns")
        values = data[list(attributes)].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError("vector targets must contain only finite, non-missing values")
        self.values_ = values
        self.mean_ = values.mean(axis=0)
        # Deviations from the dataset mean, for the optimistic estimate.
        self.deviations_ = values - self.mean_
        self._row_arrays = {}
        return self

    def _row_array(self, key: tuple, compute) -> np.ndarray:
        """Per-row arrays derived from the deviations, cached for batched statistics."""
        arrays = self.__dict__.setdefault("_row_arrays", {})
        if key not in arrays:
            arrays[key] = compute()
        return arrays[key]


@functools.lru_cache(maxsize=8)
def _sign_vectors(n_dimensions: int) -> np.ndarray:
    """Sign vectors with a positive first entry, one per column; their negations complete them."""
    rows = [(1.0, *rest) for rest in itertools.product((1.0, -1.0), repeat=n_dimensions - 1)]
    return np.array(rows).T


class MeanVectorL1QF(BoundedQualityFunction):
    """``q_dst = n / N * ||m_sg - m||_1`` for mean target vectors (Grosskreutz et al. 2010).

    Optimistic estimate (derived here; the paper gives none). With deviations
    ``d_r = t_r - m`` from the dataset mean, a refinement ``s'`` scores
    ``1 / N * ||sum_{r in s'} d_r||_1 = 1 / N * max_sigma sum_{r in s'} sigma . d_r``
    over sign vectors ``sigma``, which is at most
    ``1 / N * max_sigma sum_{r in s} max(0, sigma . d_r)``. The refinement keeping
    the rows with ``sigma . d_r > 0`` for the maximizing ``sigma`` attains it, so
    the bound is tight.

    The maximum enumerates the ``2^k`` sign vectors for
    ``k <= max_exact_dimensions`` target attributes, at a cost of ``n * 2^(k-1)``
    per subgroup. Beyond that it uses the weaker per-attribute bound
    ``1 / N * sum_i max(sum_r max(0, d_ri), sum_r max(0, -d_ri))``.
    """

    target_type = VectorTarget
    batchable = True

    def __init__(self, max_exact_dimensions: int = 8):
        if isinstance(max_exact_dimensions, bool) or not isinstance(max_exact_dimensions, int):
            raise ValueError("max_exact_dimensions must be an integer")
        self.max_exact_dimensions = max_exact_dimensions

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        size = int(np.count_nonzero(mask))
        if not size:
            return float("nan")
        if size == target.n_samples_:
            return 0.0
        mean = target.values_[mask].mean(axis=0)
        return size / target.n_samples_ * float(np.sum(np.abs(mean - target.mean_)))

    def optimistic_estimate(self, coverage, target) -> float:
        deviations = target.deviations_[self._mask(coverage, target)]
        if not len(deviations):
            return float("-inf")
        n_dimensions = deviations.shape[1]
        if n_dimensions > self.max_exact_dimensions:
            above = np.sum(np.maximum(deviations, 0.0), axis=0)
            below = above - np.sum(deviations, axis=0)
            return float(np.sum(np.maximum(above, below))) / target.n_samples_
        signs = _sign_vectors(n_dimensions)
        positive = np.zeros(signs.shape[1])
        for start in range(0, len(deviations), _BLOCK_ROWS):
            projected = deviations[start:start + _BLOCK_ROWS] @ signs
            positive += np.sum(np.maximum(projected, 0.0), axis=0)
        # For the negated sign vector, sum max(0, -v) = sum max(0, v) - sum v.
        negative = positive - np.sum(deviations, axis=0) @ signs
        return float(max(positive.max(), negative.max())) / target.n_samples_

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        size = statistics.sizes
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = statistics.sum(target.values_) / size[:, None]
            quality = size / target.n_samples_ * np.sum(np.abs(mean - target.mean_), axis=1)
        quality = np.where(size == target.n_samples_, 0.0, quality)
        return np.where(size == 0, np.nan, quality)

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        """The bound of ``optimistic_estimate`` from per-row projections cached on the target.

        The exact bound caches ``max(0, d_r . sigma)`` for all ``2^(k-1)`` sign
        vectors, ``n * 2^(k-1)`` floats.
        """
        self._check_target(target)
        deviations = target.deviations_
        n_dimensions = deviations.shape[1]
        totals = statistics.sum(deviations)
        if n_dimensions > self.max_exact_dimensions:
            positive_part = target._row_array(("above",), lambda: np.maximum(deviations, 0.0))
            above = statistics.sum(positive_part)
            estimate = np.sum(np.maximum(above, above - totals), axis=1) / target.n_samples_
        else:
            signs = _sign_vectors(n_dimensions)
            projected = target._row_array(
                ("projected", n_dimensions), lambda: np.maximum(deviations @ signs, 0.0)
            )
            positive = statistics.sum(projected)
            negative = positive - totals @ signs
            estimate = np.maximum(positive.max(axis=1), negative.max(axis=1)) / target.n_samples_
        return np.where(statistics.sizes == 0, -np.inf, estimate)
