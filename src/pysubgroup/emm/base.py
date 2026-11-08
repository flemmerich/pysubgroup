"""Shared parts of the Exceptional Model Mining (EMM) targets.

EMM (D. Leman, A. Feelders, and A. Knobbe, "Exceptional model mining", ECML
PKDD 2008; W. Duivesteijn, A. Feelders, and A. Knobbe, "Exceptional Model
Mining", Data Mining and Knowledge Discovery 30 (2016) 47-98) fits a model on
several target attributes in a subgroup and scores how much it differs from the
model on the complement or on the whole dataset (DMKD Sect. 3.2.2). This module
holds the entropy weight ``phi_ef`` (DMKD Sect. 3.2.1), the score scales of the
test-based measures, and ``NumericModelTarget``, whose models (correlation and
linear regression) depend only on the size, the sums, and the cross products of
the target values in a group. These moments add up over rows, so the measures on
them evaluate all children of a search node in one batch, and the complement's
moments are the dataset's minus the subgroup's.

Written from the papers without reusing code.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy import special

from ..targets.base import BaseTarget, QualityFunction

REFERENCES = ("complement", "dataset")
SCORES = ("1-p", "statistic")
# Groups whose variance of a model attribute is at most this fraction of the
# dataset's count as constant: their moments differ from 0 only by rounding.
_RELATIVE_VARIANCE = 1e-10


def check_reference(reference: str) -> str:
    """Validate the group a subgroup's model is compared with (DMKD Sect. 3.2.2)."""
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}")
    return reference


def check_score(score: str) -> str:
    """Validate the scale of a test-based measure.

    ``"1-p"`` is the papers' one minus the p-value; ``"statistic"`` is the
    absolute test statistic, which ranks subgroups the same way (for the
    t-test only when the degrees of freedom agree) but does not round to 1.0
    for strong effects.
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    return score


def entropy_weight(sizes, n_samples: int):
    """``phi_ef = -n/N lg(n/N) - n_C/N lg(n_C/N)``, the entropy of the split (DMKD Sect. 3.2.1)."""
    share = np.asarray(sizes, dtype=float) / n_samples
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -share * np.log2(share) - (1 - share) * np.log2(1 - share)
    return np.where((share <= 0) | (share >= 1), 0.0, terms)


def two_sided_score(statistic, p_value, score: str):
    """``1 - p`` or the absolute statistic, ``nan`` where the statistic is."""
    if score == "statistic":
        return np.abs(statistic)
    return np.where(np.isnan(statistic), np.nan, 1.0 - p_value)


def normal_score(z, score: str):
    """Score of a two-sided test of a standard normal statistic ``z``."""
    with np.errstate(invalid="ignore"):
        return two_sided_score(z, 2.0 * special.ndtr(-np.abs(z)), score)


class NumericModelTarget(BaseTarget):
    """Numeric model attributes, with the per-row moments their models are fitted from.

    ``fit`` stores ``values_`` (rows x attributes), the dataset ``mean_``, the
    values centered at that mean (``centered_``) and their pairwise products
    (``products_``, one column per pair ``i <= j``), which keep the moment
    sums accurate. ``scatter_`` is the dataset's scatter matrix
    ``sum (t - mean)(t - mean)^T``.
    """

    n_model_attributes: int | None = None

    def fit(self, data: pd.DataFrame, *attributes: Hashable) -> NumericModelTarget:
        """Store the values of the numeric model attributes ``attributes``."""
        expected = self.n_model_attributes
        if expected is not None and len(attributes) != expected:
            raise ValueError(
                f"{type(self).__name__} requires {expected} attributes, got {len(attributes)}"
            )
        if not attributes:
            raise ValueError(f"{type(self).__name__} requires at least one attribute")
        super().fit(data, *attributes)
        for name in attributes:
            dtype = data[name].dtype
            if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                raise TypeError("model attributes must be numeric columns")
        values = data[list(attributes)].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError("model attributes must contain only finite, non-missing values")
        self.values_ = values
        self.mean_ = values.mean(axis=0)
        self.centered_ = values - self.mean_
        rows, columns = np.triu_indices(values.shape[1])
        self._pairs = (rows, columns)
        self.products_ = self.centered_[:, rows] * self.centered_[:, columns]
        self.totals_ = (
            np.array(self.n_samples_, dtype=float),
            self.centered_.sum(axis=0),
            self.products_.sum(axis=0),
        )
        self.scatter_ = self.scatter(*self.totals_)
        if np.any(np.diagonal(self.scatter_) <= 0):
            raise ValueError("model attributes must not be constant on the dataset")
        return self

    def moments(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Size, sums of centered values, and sums of their products in the rows ``mask``.

        Each has a leading axis of length one, as ``moments_many`` for one child.
        """
        size = np.array([np.count_nonzero(mask)], dtype=float)
        return size, self.centered_[mask].sum(axis=0)[None], self.products_[mask].sum(axis=0)[None]

    def moments_many(self, statistics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The moments of every child summarized by ``statistics``."""
        sizes = np.asarray(statistics.sizes, dtype=float)
        return sizes, statistics.sum(self.centered_), statistics.sum(self.products_)

    def complement(self, sizes, sums, products):
        """Moments of the complements of the groups with the given moments."""
        total_size, total_sums, total_products = self.totals_
        return total_size - sizes, total_sums - sums, total_products - products

    def scatter(self, sizes, sums, products) -> np.ndarray:
        """Scatter matrices ``sum (t - mean_G)(t - mean_G)^T``, one per group."""
        sizes, sums, products = np.asarray(sizes), np.asarray(sums), np.asarray(products)
        k = len(self.attributes_)
        rows, columns = self._pairs
        upper = np.zeros((*sizes.shape, k, k))
        upper[..., rows, columns] = products
        full = upper + np.swapaxes(upper, -1, -2)
        full[..., np.arange(k), np.arange(k)] /= 2
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = sums[..., :, None] * sums[..., None, :] / sizes[..., None, None]
        return full - np.where(sizes[..., None, None] > 0, shift, 0.0)

    def means(self, sizes, sums) -> np.ndarray:
        """Mean target vectors of groups with the given sizes and sums of centered values."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.mean_ + np.asarray(sums) / np.asarray(sizes)[..., None]

    def constant(self, sizes, scatter: np.ndarray) -> np.ndarray:
        """Whether each model attribute is constant in each group (groups x attributes)."""
        variance = np.diagonal(scatter, axis1=-2, axis2=-1)
        dataset = np.diagonal(self.scatter_) / self.n_samples_
        with np.errstate(divide="ignore", invalid="ignore"):
            return variance <= _RELATIVE_VARIANCE * dataset * np.asarray(sizes)[..., None]


class MomentQF(QualityFunction):
    """Quality function computed from the moments of a ``NumericModelTarget``.

    Subclasses implement ``_scores(sizes, sums, products, target)`` for arrays
    with one entry per group; scalar and batched evaluation share it.
    """

    target_type = NumericModelTarget
    batchable = True

    def _scores(self, sizes, sums, products, target) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        return float(self._scores(*target.moments(mask), target)[0])

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        return self._scores(*target.moments_many(statistics), target)
