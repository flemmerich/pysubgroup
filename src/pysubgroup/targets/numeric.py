"""Numeric target and its quality functions.

Provenance: ``StandardQFNumeric`` and ``StandardQFNumericTscore``, their
parameters, and the ``"sum"``/``"max"``/``"average"`` estimators are adapted from
pysubgroup 0.x's ``numeric_target.py`` (https://github.com/flemmerich/pysubgroup,
Apache-2.0, Florian Lemmerich et al.), rewritten for fitted targets and Boolean
row masks. Deliberate changes: a valid bound for the median ``max`` estimator,
``nan`` for empty subgroups, and a t-score of exactly 0 for constant subgroups.
With the optional numba extra, the mean-based measures use the kernels in
``pysubgroup._numba``.

Written from the paper without reusing code: the ordering-based bounds, the
``"combined"`` estimator, and ``VarianceReductionQF``, ``InterclassVarianceQF``,
``StandardDeviationQF``, ``MannWhitneyQF``, ``RankAUCQF`` and
``KolmogorovSmirnovQF`` with their bounds follow

    Lemmerich, F., Atzmueller, M., Puppe, F.: Fast Exhaustive Subgroup Discovery
    with Numerical Target Concepts. Data Mining and Knowledge Discovery 30(3):
    711-762, 2016 (Sections 4 and 5.2).

**Ordering-based bounds** (Definition 1): sorted by decreasing target value, the
best subset of a subgroup among those of each size is the prefix of that size
(*one-pass* measures) or the prefix or suffix of that size (*two-pass*
measures). The best quality among these ``i_P`` (or ``2 i_P``) candidates is a
tight optimistic estimate. ``NumericTarget.order_`` sorts the rows once, so the
subgroup's rows in this order are a gather (``mask[order_]``) instead of a sort
per evaluation; every measure computes its quality and bound from them in
``_ordered``, which ``NumBSD`` calls on its sorted bitsets.

Deviations from the paper, found while implementing it:

- Theorem 4 calls the interclass variance one-pass estimable. It equals
  ``i0 * q_vr`` exactly (``i_P (mu_P - mu0) = -i_notP (mu_notP - mu0)``), and
  ``q_vr`` is two-pass (Theorem 6); we use two passes.
- Ranks are ascending mid-ranks (rank 1 is the lowest value, ties share the mean
  rank), as in Example 9 and Theorems 17-18, not as in the text of Section 4.
- The paper's ``q_auc`` uses ``R_notP`` and thus rewards a *decrease*.
  ``RankAUCQF(direction="positive")`` is the probability that a subgroup row
  exceeds a complement row (ties count 1/2); the paper's ``q_auc`` is
  ``direction="negative"``.
- Theorem 18 fails with ties; ``RankAUCQF`` uses ``(L - 1) / (n - 1)`` with
  ``L`` the number of dataset rows at most the subgroup's maximum, which equals
  it without ties and holds with them (see the class docstring).
- The first bound of Theorem 14 divides by zero for the whole dataset; it is
  taken over subsets of at most ``i0 - 1`` rows (the whole dataset scores 0).
  Its second bound squares the sum of the deviations, as in the theorem and
  its proof; Example 6 sums squared deviations instead, which is no bound.
- ``StandardDeviationQF``: the paper proves that no ordering-based bound
  exists (Theorem 7) and gives none; pysubgroup derives one from Popoviciu's
  inequality.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .. import _numba
from .base import (
    BaseTarget,
    BoundedQualityFunction,
    Direction,
    check_direction,
    check_exponent,
    orient,
)

_ORIENTATIONS = {"positive": (False,), "negative": (True,), "both": (False, True)}


class NumericTarget(BaseTarget):
    """Numeric target distribution inside and outside a subgroup.

    Besides ``values_``, ``mean_`` and ``median_``, the target derives cached
    per-row arrays on first use: ``order_`` (rows by decreasing value, ties in
    row order), ``sorted_values_``, ``ranks_`` (ascending mid-ranks) and
    ``std_`` (population standard deviation).
    """

    def fit(self, data: pd.DataFrame, attribute: Hashable) -> NumericTarget:
        """Store the values of the numeric feature ``attribute``."""
        super().fit(data, attribute)
        column = data[attribute]
        if not pd.api.types.is_numeric_dtype(column.dtype) or pd.api.types.is_bool_dtype(
            column.dtype
        ):
            raise TypeError("numeric targets require a numeric column")
        values = column.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError("numeric targets must contain only finite, non-missing values")
        self.values_ = values
        # Dataset centroids, computed once instead of in every quality evaluation.
        self.mean_ = float(np.mean(values))
        self.median_ = float(np.median(values))
        self._negated_values = None
        self._row_arrays = {}
        return self

    def _row_array(self, key: tuple, compute) -> np.ndarray:
        """Per-row arrays derived from the values, cached for batched statistics."""
        arrays = self.__dict__.setdefault("_row_arrays", {})
        if key not in arrays:
            arrays[key] = compute()
        return arrays[key]

    def _oriented(self, negative: bool) -> tuple[np.ndarray, float, float]:
        """Values, mean, and median; negated for searches for decreases.

        Negation is exact, so a decrease of ``y`` is an increase of ``-y``.
        """
        if not negative:
            return self.values_, self.mean_, self.median_
        if getattr(self, "_negated_values", None) is None:
            self._negated_values = -self.values_
        return self._negated_values, -self.mean_, -self.median_

    @property
    def order_(self) -> np.ndarray:
        """Row positions by decreasing value; ties keep the row order."""
        return self._row_array(("order",), lambda: np.argsort(-self.values_, kind="stable"))

    @property
    def sorted_values_(self) -> np.ndarray:
        return self._row_array(("sorted",), lambda: self.values_[self.order_])

    @property
    def ranks_(self) -> np.ndarray:
        """Ascending mid-ranks in ``[1, n]``; tied values share their mean rank."""
        return self._row_array(("ranks",), lambda: rankdata(self.values_))

    @property
    def std_(self) -> float:
        return float(self._row_array(("std",), lambda: np.std(self.values_)))

    def _deviations(self, negative: bool = False) -> np.ndarray:
        """``value - mean`` per row (negated for decreases)."""
        return self._row_array(
            ("deviations", negative),
            lambda: (-1.0 if negative else 1.0) * (self.values_ - self.mean_),
        )

    def _rank_deviations(self, negative: bool = False) -> np.ndarray:
        """``rank - (n + 1) / 2`` per row (negated for decreases): half-integers, summed exactly."""
        return self._row_array(
            ("rank deviations", negative),
            lambda: (-1.0 if negative else 1.0) * (self.ranks_ - (self.n_samples_ + 1) / 2),
        )

    def _sorted(self, array: np.ndarray, key: tuple) -> np.ndarray:
        """A per-row array in the order of ``order_``."""
        return self._row_array(("sorted",) + key, lambda: array[self.order_])

    def _group_ends(self) -> np.ndarray:
        """Positions in ``order_`` of the last row of every distinct value."""
        def compute():
            values = self.sorted_values_
            return np.flatnonzero(np.append(values[1:] != values[:-1], True))
        return self._row_array(("group ends",), compute)

    def _rows_at_most(self, negative: bool) -> np.ndarray:
        """Number of rows whose value is at most this row's value (at least, if ``negative``)."""
        def compute():
            values, _, _ = self._oriented(negative)
            return np.searchsorted(np.sort(values), values, side="right").astype(float)
        return self._row_array(("at most", negative), compute)


def _use_kernel(mask: np.ndarray) -> bool:
    """Use numba for masks covering at least 1/32 of the rows.

    The kernels scan all rows, while copying ``values_[mask]`` costs time
    proportional to the subgroup; the cutoff is measured at 100k rows.
    """
    return _numba.enabled() and np.count_nonzero(mask) * 32 >= len(mask)


def _prefix_sizes(m: int) -> np.ndarray:
    return np.arange(1, m + 1, dtype=float)


class _OrderedMixin:
    """Quality and optimistic estimate from the subgroup's rows in ``order_``.

    ``_ordered(sorted_mask, target)`` receives ``mask[target.order_]``, the rows
    of the subgroup by decreasing target value, and returns ``(quality,
    estimate)``: ``(nan, -inf)`` for an empty subgroup. The estimate is the
    ordering-based bound where the measure has one, otherwise its closed form.
    """

    target_type = NumericTarget

    def _ordered(self, sorted_mask: np.ndarray, target: NumericTarget) -> tuple[float, float]:
        raise NotImplementedError

    def _evaluate_ordered(self, coverage, target) -> tuple[float, float]:
        mask = self._mask(coverage, target)
        return self._ordered(mask[target.order_], target)

    def _scan(self, parent, condition, target) -> tuple[int, float, float] | None:
        """``(size, quality, estimate)`` of ``parent & condition`` with numba, or None.

        ``parent`` and ``condition`` are bitsets in ``order_`` packed with
        ``_numba.pack64``. Measures with a prefix score in
        ``_numba.ordered_prefix_scan`` return what ``_ordered`` returns, without
        unpacking the rows; callers check ``_numba.enabled()``.
        """
        return None

    @staticmethod
    def _prefix_weights(target: NumericTarget, key: tuple, weights) -> np.ndarray:
        """Size factors ``weights(sizes, i0)`` of a prefix score for sizes ``0..i0``, cached."""
        n = target.n_samples_
        return target._row_array(
            ("prefix weights",) + key, lambda: weights(np.arange(n + 1, dtype=float), n)
        )

    @staticmethod
    def _prefix_scans(parent, condition, values, kind, weights, target, orientations,
                      centroid=0.0) -> dict[bool, tuple[int, float, float]]:
        """``ordered_prefix_scan`` per orientation: forward, or reversed on the negated values."""
        return {
            negative: _numba.ordered_prefix_scan(
                parent, condition, values, kind, weights, -centroid if negative else centroid,
                target.n_samples_, negative,
            )
            for negative in orientations
        }


# Adapted from pysubgroup 0.x's StandardQFNumeric and its Summation/Max/MeanOrdering
# estimators (numeric_target.py, Apache-2.0); "combined", the median "order"
# estimator, and the target-ordered evaluation are written from Lemmerich et al. (2016).
class StandardQFNumeric(_OrderedMixin, BoundedQualityFunction):
    """``n_sg ** a * (centroid_sg - centroid_dataset)`` for the mean or median.

    Optimistic estimators (Lemmerich, dissertation, section 4.2.2.1, and
    Lemmerich et al. 2016, Section 5.2):

    - ``"sum"``: sum of ``value - centroid`` over subgroup values above the
      dataset centroid (mean only; default for the mean; Theorems 10-11).
    - ``"max"`` / ``"average"``: ``n ** a * (max - centroid)`` over the ``n_above``
      subgroup values above the centroid, with ``n = n_above`` for the mean
      (Theorem 12) and ``n = min(2 * n_above, n_sg)`` for the median (default
      for the median; tighter than Theorem 15's ``n_sg ** a * (max - median)``).
      Without values above the centroid, ``"max"`` bounds by ``max - centroid``.
    - ``"combined"``: the smaller of ``"sum"`` and ``"max"`` (mean only), as the
      paper suggests after Example 5.
    - ``"order"``: best quality among the prefixes of the subgroup values sorted
      in descending order, a tight bound for the mean and the median
      (Theorems 1-2). Not batched.

    ``a`` must lie in ``[0, 1]``, where all these bounds hold.

    ``direction="negative"`` searches for a decrease of the centroid,
    ``n_sg ** a * (centroid_dataset - centroid_sg)``, which is the measure on the
    negated target values; its estimators therefore use the values below the
    dataset centroid. ``"both"`` scores the absolute deviation, bounded by the
    larger of both estimates (Theorem 13; the ``"order"`` estimate then uses
    the prefixes and suffixes, two passes).
    """

    ESTIMATORS = ("sum", "max", "average", "combined", "order")

    def __init__(
        self,
        a: float,
        estimator: Literal["default", "sum", "max", "average", "combined", "order"] = "default",
        centroid: Literal["mean", "median"] = "mean",
        direction: Direction = "positive",
    ):
        check_exponent(a)
        if centroid not in ("mean", "median"):
            raise ValueError(f"centroid must be 'mean' or 'median', got {centroid!r}")
        allowed = self.ESTIMATORS if centroid == "mean" else ("max", "order")
        if estimator == "default":
            estimator = allowed[0]
        if estimator not in allowed:
            raise ValueError(f"estimator for centroid={centroid!r} must be one of {allowed}")
        self.a = a
        self.estimator = estimator
        self.centroid = centroid
        self.direction = check_direction(direction)

    @property
    def batchable(self) -> bool:
        """Mean-based measures with an estimator other than ``"order"``."""
        return self.centroid == "mean" and self.estimator != "order"

    @staticmethod
    def standard_qf_numeric(
        a: float, centroid_dataset: float, size_sg: int, centroid_sg: float
    ) -> float:
        return size_sg**a * (centroid_sg - centroid_dataset)

    def _aggregate(self, values: np.ndarray) -> float:
        return float(np.mean(values) if self.centroid == "mean" else np.median(values))

    def evaluate(self, coverage, target) -> float:
        # The negated values give exactly the negated quality.
        return orient(self._increase(self._mask(coverage, target), target), self.direction)

    def _increase(self, mask: np.ndarray, target: NumericTarget) -> float:
        if self.centroid == "mean" and _use_kernel(mask):
            size, total = _numba.masked_sum(target.values_, mask)
            if not size:
                return float("nan")
            mean = target.mean_ if size == target.n_samples_ else total / size
            return self.standard_qf_numeric(self.a, target.mean_, size, mean)
        values = target.values_[mask]
        if not len(values):
            return float("nan")
        centroid_dataset = target.mean_ if self.centroid == "mean" else target.median_
        return self.standard_qf_numeric(
            self.a, centroid_dataset, len(values), self._aggregate(values)
        )

    def optimistic_estimate(self, coverage, target) -> float:
        if self.estimator == "order":
            return self._evaluate_ordered(coverage, target)[1]
        mask = self._mask(coverage, target)
        return max(
            self._estimate(mask, *target._oriented(negative))
            for negative in _ORIENTATIONS[self.direction]
        )

    def _estimate(self, mask: np.ndarray, all_values: np.ndarray, mean: float, median: float):
        """Bound for an increase of the centroid of ``all_values``."""
        centroid_dataset = mean if self.centroid == "mean" else median
        if self.estimator == "sum" and _use_kernel(mask):
            size, _, excess = _numba.masked_excess(all_values, mask, centroid_dataset)
            return float(excess) if size else float("-inf")
        values = all_values[mask]
        if not len(values):
            return float("-inf")
        above = values[values > centroid_dataset]
        excess = float(np.sum(above - centroid_dataset))
        if self.estimator == "sum":
            return excess
        if not len(above):
            # Every subset has a centroid of at most the largest value.
            largest = float(np.max(values)) - centroid_dataset
            return min(excess, largest) if self.estimator == "combined" else largest
        # A median above the centroid needs only half of the subset above it.
        size = len(above) if self.centroid == "mean" else min(2 * len(above), len(values))
        largest = size**self.a * (float(np.max(above)) - centroid_dataset)
        return min(excess, largest) if self.estimator == "combined" else largest

    def _prefix_qualities(self, values: np.ndarray, centroid: float, n_samples: int):
        """Qualities of the prefixes of ``values`` (descending) as subgroups."""
        sizes = _prefix_sizes(len(values))
        if self.centroid == "mean":
            centroids = np.cumsum(values) / sizes
        else:
            index = np.arange(len(values))
            centroids = (values[index // 2] + values[(index + 1) // 2]) / 2
        qualities = sizes**self.a * (centroids - centroid)
        if len(values) == n_samples and self.centroid == "mean":
            qualities[-1] = 0.0  # the dataset mean, without rounding
        return qualities

    def _ordered(self, sorted_mask, target):
        values = target.sorted_values_[sorted_mask]
        if not len(values):
            return float("nan"), float("-inf")
        prefixes = {}
        for negative in _ORIENTATIONS[self.direction]:
            _, mean, median = target._oriented(negative)
            oriented = -values[::-1] if negative else values
            prefixes[negative] = self._prefix_qualities(
                oriented, mean if self.centroid == "mean" else median, target.n_samples_
            )
        best = max(float(np.max(qualities)) for qualities in prefixes.values())
        # Every pass ends with the whole subgroup.
        quality = float(prefixes[self.direction == "negative"][-1])
        if self.direction == "both":
            quality = abs(quality)
        return quality, best

    def _power_weights(self, target: NumericTarget) -> np.ndarray:
        """``k ** a`` for every size ``k`` from 0 to the dataset size."""
        return self._prefix_weights(target, ("power", self.a), lambda k, n: k**self.a)

    def _scan(self, parent, condition, target):
        if self.centroid != "mean":
            return None
        weights = self._power_weights(target)
        scans = self._prefix_scans(
            parent, condition, target.sorted_values_, _numba.PREFIX_MEAN, weights, target,
            _ORIENTATIONS[self.direction], target.mean_,
        )
        size, _, quality = scans[self.direction == "negative"]
        if self.direction == "both":
            quality = abs(quality)
        return size, quality, max(scan[1] for scan in scans.values())

    def fast_bounds_apply(self) -> bool:
        """Whether Theorem 9's bounds from partial subgroups hold (see ``fast_bound``)."""
        return self.centroid == "mean" and self.direction != "both"

    def fast_bound(self, size: int, total: float, theta: float, max_size: int,
                   centroid: float) -> float:
        """Theorem 9: bound on the qualities of all larger prefixes.

        ``size`` rows with the largest (oriented) values, summing to ``total``,
        have been read; ``theta`` is the last of them and ``max_size`` bounds the
        subgroup size. Every remaining row is at most ``theta``, and for
        ``0 <= a <= 1`` the bound is attained when none or all of the
        ``max_size - size`` remaining rows have value ``theta``.
        """
        return max_size**self.a * ((total + (max_size - size) * theta) / max_size - centroid)

    def _check_batch(self, target) -> None:
        self._check_target(target)
        if not self.batchable:
            raise NotImplementedError(
                "batched evaluation requires centroid='mean' and an estimator other than 'order'"
            )

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_batch(target)
        size = statistics.sizes
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(
                size == target.n_samples_, target.mean_, statistics.sum(target.values_) / size
            )
            quality = self.standard_qf_numeric(self.a, target.mean_, size, mean)
        quality = np.where(size == 0, np.nan, quality)
        if self.direction == "negative":
            return -quality
        return np.abs(quality) if self.direction == "both" else quality

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        self._check_batch(target)
        estimates = [
            self._estimate_many(statistics, target, negative)
            for negative in _ORIENTATIONS[self.direction]
        ]
        return np.where(statistics.sizes == 0, -np.inf, np.maximum.reduce(estimates))

    def _estimate_many(self, statistics, target: NumericTarget, negative: bool) -> np.ndarray:
        values, mean, _ = target._oriented(negative)
        excess = None
        if self.estimator in ("sum", "combined"):
            excess = statistics.sum(target._row_array(
                ("excess", negative), lambda: np.maximum(values - mean, 0.0)
            ))
            if self.estimator == "sum":
                return excess
        above = target._row_array(("above", negative), lambda: values > mean)
        n_above = statistics.count(above)
        largest = statistics.maximum(values) - mean
        with np.errstate(invalid="ignore"):
            bound = np.where(n_above > 0, np.power(n_above, self.a) * largest, largest)
        return bound if excess is None else np.minimum(excess, bound)


# Adapted from pysubgroup 0.x's StandardQFNumericTscore (numeric_target.py, Apache-2.0).
class StandardQFNumericTscore(BoundedQualityFunction):
    """T-score ``sqrt(n_sg) * (mean_sg - mean_dataset) / std_sg``; 0 if ``std_sg`` is 0.

    ``direction="negative"`` negates the score to search for a decrease of the
    mean; ``"both"`` takes its absolute value. Not estimable by ordering
    (Lemmerich et al. 2016, Theorem 8); no finite bound is known. Batched.

    A subgroup whose values are all equal scores exactly 0. pysubgroup 0.x tests the
    rounded ``np.std``, which is about ``1e-17`` for e.g. three values 0.1 and
    turns the score into ``-3e16``.
    """

    target_type = NumericTarget
    batchable = True

    def __init__(self, direction: Direction = "positive"):
        self.direction = check_direction(direction)

    @staticmethod
    def t_score(mean_dataset: float, size_sg: int, mean_sg: float, std_sg: float) -> float:
        if std_sg == 0:
            return 0.0
        return size_sg**0.5 * (mean_sg - mean_dataset) / std_sg

    def evaluate(self, coverage, target) -> float:
        return orient(self._increase(self._mask(coverage, target), target), self.direction)

    def _increase(self, mask: np.ndarray, target: NumericTarget) -> float:
        if _use_kernel(mask):
            size, total = _numba.masked_sum(target.values_, mask)
            if not size:
                return float("nan")
            # The dataset mean keeps the whole dataset at quality 0 despite rounding.
            # Squared deviations from one of its values vanish only for a constant subgroup.
            first = target.values_[np.argmax(mask)]
            if not _numba.masked_squared_deviation(target.values_, mask, first):
                return 0.0
            mean = target.mean_ if size == target.n_samples_ else total / size
            deviation = _numba.masked_squared_deviation(target.values_, mask, mean)
            return self.t_score(target.mean_, size, mean, (deviation / size) ** 0.5)
        values = target.values_[mask]
        if not len(values):
            return float("nan")
        if np.min(values) == np.max(values):
            return 0.0
        return self.t_score(
            target.mean_,
            len(values),
            float(np.mean(values)),
            float(np.std(values)),
        )

    def optimistic_estimate(self, coverage, target) -> float:
        """No finite bound is known: ``inf``, or ``-inf`` for an empty subgroup."""
        return float("inf") if np.any(self._mask(coverage, target)) else float("-inf")

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        deviations = target._deviations()
        squares = target._row_array(("squared deviations",), lambda: np.square(deviations))
        size = statistics.sizes
        constant = statistics.maximum(target.values_) == -statistics.maximum(
            target._oriented(True)[0]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = statistics.sum(deviations) / size
            variance = np.maximum(statistics.sum(squares) / size - np.square(shift), 0.0)
            increase = np.sqrt(size) * shift / np.sqrt(variance)
        increase = np.where(constant | (size >= target.n_samples_), 0.0, increase)
        quality = np.where(size == 0, np.nan, increase)
        if self.direction == "negative":
            return -quality
        return np.abs(quality) if self.direction == "both" else quality

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        return np.where(statistics.sizes > 0, np.inf, -np.inf)


def _check_estimator(estimator: str, allowed: tuple[str, ...]) -> str:
    if estimator not in allowed:
        raise ValueError(f"estimator must be one of {allowed}, got {estimator!r}")
    return estimator


def _two_pass_maximum(deviations: np.ndarray, score) -> float:
    """Best ``score(prefix sums, sizes)`` over the prefixes and suffixes of ``deviations``."""
    sizes = _prefix_sizes(len(deviations))
    return max(
        float(np.max(score(np.cumsum(deviations), sizes))),
        float(np.max(score(np.cumsum(deviations[::-1]), sizes))),
    )


class VarianceReductionQF(_OrderedMixin, BoundedQualityFunction):
    """Variance reduction ``i_P / (i0 - i_P) * (mu_P - mu0) ** 2``.

    Breiman et al. (1984), Kloesgen (1996), in the survey of Lemmerich,
    Atzmueller and Puppe (2016, Section 4, 1c). Symmetric: increases and
    decreases of the mean score alike. The whole dataset scores 0.

    Estimators (Lemmerich et al. 2016):

    - ``"closed"`` (default, batched): the smaller of Theorem 14's bounds,
      ``k / (i0 - k) * max((T_max - mu0)**2, (T_min - mu0)**2)`` with
      ``k = min(i_P, i0 - 1)``, and ``max(S+**2, S-**2) / (i0 - 1)`` with ``S+``
      (``S-``) the sum of the deviations above (below) ``mu0``.
    - ``"order"``: two-pass ordering-based bound (Theorem 6), tight.
    """

    target_type = NumericTarget

    def __init__(self, estimator: Literal["closed", "order"] = "closed"):
        self.estimator = _check_estimator(estimator, ("closed", "order"))

    @property
    def batchable(self) -> bool:
        return self.estimator == "closed"

    def _scale(self, target: NumericTarget) -> float:
        return 1.0

    def _score(self, total, size, n_samples: int):
        """From the sum of deviations ``total``: ``total**2 / (size * (i0 - size))``."""
        size = np.asarray(size, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.square(total) / (size * (n_samples - size))
        return np.where(size >= n_samples, 0.0, score)

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        size = int(np.count_nonzero(mask))
        if not size:
            return float("nan")
        total = float(np.sum(target._deviations()[mask]))
        return float(self._score(total, size, target.n_samples_)) * self._scale(target)

    def optimistic_estimate(self, coverage, target) -> float:
        if self.estimator == "order":
            return self._evaluate_ordered(coverage, target)[1]
        mask = self._mask(coverage, target)
        deviations = target._deviations()[mask]
        if not len(deviations):
            return float("-inf")
        return float(self._closed(
            len(deviations), float(np.max(deviations)), float(np.min(deviations)),
            float(np.sum(np.maximum(deviations, 0))), float(np.sum(np.minimum(deviations, 0))),
            target,
        ))

    def _closed(self, size, largest, smallest, above, below, target):
        n = target.n_samples_
        if n == 1:
            return np.zeros(np.shape(size))
        k = np.minimum(size, n - 1)
        with np.errstate(invalid="ignore"):  # empty children: 0 * inf, masked by callers
            first = k / (n - k) * np.maximum(np.square(largest), np.square(smallest))
        second = np.maximum(np.square(above), np.square(below)) / (n - 1)
        return np.minimum(first, second) * self._scale(target)

    def _ordered(self, sorted_mask, target):
        deviations = target._sorted(target._deviations(), ("deviations",))[sorted_mask]
        if not len(deviations):
            return float("nan"), float("-inf")
        n = target.n_samples_
        quality = float(self._score(np.sum(deviations), len(deviations), n))
        best = _two_pass_maximum(deviations, lambda sums, sizes: self._score(sums, sizes, n))
        scale = self._scale(target)
        return quality * scale, best * scale

    def _scan(self, parent, condition, target):
        # The score squares the sum, so negating the reversed rows changes nothing.
        deviations = target._sorted(target._deviations(), ("deviations",))
        weights = self._prefix_weights(target, ("split",), lambda k, n: k * (n - k))
        scans = self._prefix_scans(parent, condition, deviations,
                                   _numba.PREFIX_VARIANCE_REDUCTION, weights, target,
                                   (False, True))
        size, _, quality = scans[False]
        scale = self._scale(target)
        return size, quality * scale, max(scan[1] for scan in scans.values()) * scale

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        size = statistics.sizes
        quality = self._score(statistics.sum(target._deviations()), size, target.n_samples_)
        return np.where(size == 0, np.nan, quality * self._scale(target))

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        deviations = target._deviations()
        positive = target._row_array(("positive deviations",), lambda: np.maximum(deviations, 0))
        negative = target._row_array(("negative deviations",), lambda: np.minimum(deviations, 0))
        estimate = self._closed(
            statistics.sizes, statistics.maximum(deviations),
            -statistics.maximum(target._deviations(True)),
            statistics.sum(positive), statistics.sum(negative), target,
        )
        return np.where(statistics.sizes == 0, -np.inf, estimate)


class InterclassVarianceQF(VarianceReductionQF):
    """Interclass variance ``i_P (mu_P - mu0)**2 + i_notP (mu_notP - mu0)**2``.

    Morishita (1998), Morishita and Sese (2000), in the survey of Lemmerich et
    al. (2016, Section 4, 1d). Since ``i_P (mu_P - mu0) = -i_notP (mu_notP -
    mu0)``, it equals ``i0`` times the variance reduction, so it shares that
    measure's bounds, scaled by ``i0``. Deviation: Theorem 4 calls it one-pass
    estimable; like the variance reduction it is two-pass (Theorem 6).
    """

    def _scale(self, target: NumericTarget) -> float:
        return float(target.n_samples_)


class StandardDeviationQF(_OrderedMixin, BoundedQualityFunction):
    """Generic variance-based measure ``i_P ** a * (sigma_P - sigma_0)``.

    Aumann and Lindell (2003), in the survey of Lemmerich et al. (2016,
    Section 4, 2a). ``sigma`` is the population standard deviation (``ddof=0``).
    ``direction="negative"`` rewards a decrease, ``i_P ** a * (sigma_0 -
    sigma_P)``, and ``"both"`` the absolute difference.

    The paper proves that the measure is not estimable by ordering
    (Theorem 7) and gives no bound. pysubgroup derives one, for ``a >= 0``:
    by Popoviciu's inequality, every subset ``r`` has ``sigma_r <= (T_max -
    T_min) / 2 =: s`` over the subgroup's values, so with ``c = s - sigma_0``,
    ``|r| ** a * (sigma_r - sigma_0) <= i_P ** a * c`` if ``c >= 0`` and ``<= c``
    otherwise (``|r| >= 1``). For decreases, ``sigma_r >= 0`` gives
    ``i_P ** a * sigma_0``. Batched.
    """

    target_type = NumericTarget
    batchable = True

    def __init__(self, a: float, direction: Direction = "positive"):
        if isinstance(a, bool) or not isinstance(a, int | float) or a < 0:
            raise ValueError(f"a must be a non-negative number. Received a={a}")
        self.a = a
        self.direction = check_direction(direction)

    def _increase(self, size, total, squares, target):
        """``size ** a * (sigma - sigma_0)`` from the sums of deviations and of their squares."""
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = np.maximum(squares / size - np.square(total / size), 0.0)
            quality = np.power(size, self.a) * (np.sqrt(variance) - target.std_)
        return np.where(size >= target.n_samples_, 0.0, quality)

    def _oriented_quality(self, increase):
        if self.direction == "negative":
            return -increase
        return np.abs(increase) if self.direction == "both" else increase

    def _bound(self, size, spread, target):
        sizes = np.power(np.asarray(size, dtype=float), self.a)
        excess = spread / 2 - target.std_
        increase = np.where(excess >= 0, sizes * excess, excess)
        decrease = sizes * target.std_
        if self.direction == "positive":
            return increase
        return decrease if self.direction == "negative" else np.maximum(increase, decrease)

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        deviations = target._deviations()[mask]
        if not len(deviations):
            return float("nan")
        increase = self._increase(len(deviations), np.sum(deviations),
                                  np.sum(np.square(deviations)), target)
        return float(self._oriented_quality(increase))

    def optimistic_estimate(self, coverage, target) -> float:
        values = target.values_[self._mask(coverage, target)]
        if not len(values):
            return float("-inf")
        return float(self._bound(len(values), np.max(values) - np.min(values), target))

    def _ordered(self, sorted_mask, target):
        deviations = target._sorted(target._deviations(), ("deviations",))[sorted_mask]
        if not len(deviations):
            return float("nan"), float("-inf")
        increase = self._increase(len(deviations), np.sum(deviations),
                                  np.sum(np.square(deviations)), target)
        bound = self._bound(len(deviations), deviations[0] - deviations[-1], target)
        return float(self._oriented_quality(increase)), float(bound)

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        deviations = target._deviations()
        squares = target._row_array(("squared deviations",), lambda: np.square(deviations))
        size = statistics.sizes
        increase = self._increase(size, statistics.sum(deviations), statistics.sum(squares),
                                  target)
        return np.where(size == 0, np.nan, self._oriented_quality(increase))

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        spread = statistics.maximum(target.values_) + statistics.maximum(target._oriented(True)[0])
        size = statistics.sizes
        with np.errstate(invalid="ignore"):
            bound = self._bound(size, spread, target)
        return np.where(size == 0, -np.inf, bound)


class _RankQF(_OrderedMixin, BoundedQualityFunction):
    """Measures of the rank sum ``D = sum_P (rank - (i0 + 1) / 2)``, odd under negation."""

    target_type = NumericTarget

    def __init__(self, estimator: Literal["closed", "order"] = "closed",
                 direction: Direction = "positive"):
        self.estimator = _check_estimator(estimator, ("closed", "order"))
        self.direction = check_direction(direction)

    @property
    def batchable(self) -> bool:
        return self.estimator == "closed"

    def _score(self, total, size, n_samples: int):
        """The increase measure from the rank-deviation sum ``total`` of ``size`` rows."""
        raise NotImplementedError

    def _oriented_quality(self, total, size, n_samples):
        if self.direction == "negative":
            return self._score(-total, size, n_samples)
        if self.direction == "both":
            return self._score(np.abs(total), size, n_samples)
        return self._score(total, size, n_samples)

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        size = int(np.count_nonzero(mask))
        if not size:
            return float("nan")
        total = float(np.sum(target._rank_deviations()[mask]))
        return float(self._oriented_quality(total, size, target.n_samples_))

    def optimistic_estimate(self, coverage, target) -> float:
        if self.estimator == "order":
            return self._evaluate_ordered(coverage, target)[1]
        mask = self._mask(coverage, target)
        if not mask.any():
            return float("-inf")
        return max(
            float(self._closed_scalar(mask, target, negative))
            for negative in _ORIENTATIONS[self.direction]
        )

    def _ordered(self, sorted_mask, target):
        deviations = target._sorted(target._rank_deviations(), ("rank deviations",))[sorted_mask]
        if not len(deviations):
            return float("nan"), float("-inf")
        n = target.n_samples_
        quality = float(self._oriented_quality(np.sum(deviations), len(deviations), n))
        sizes = _prefix_sizes(len(deviations))
        best = float("-inf")
        for negative in _ORIENTATIONS[self.direction]:
            # Descending ranks for increases, ascending (negated) ranks for decreases.
            oriented = -deviations[::-1] if negative else deviations
            best = max(best, float(np.max(self._score(np.cumsum(oriented), sizes, n))))
        return quality, best

    def _scan(self, parent, condition, target):
        deviations = target._sorted(target._rank_deviations(), ("rank deviations",))
        weights = self._prefix_weights(target, (self._prefix_kind,), self._prefix_weight)
        scans = self._prefix_scans(parent, condition, deviations, self._prefix_kind, weights,
                                   target, _ORIENTATIONS[self.direction])
        # The score increases with the sum, so for "both" the larger pass scores |D|.
        size = next(iter(scans.values()))[0]
        return (size, max(scan[2] for scan in scans.values()),
                max(scan[1] for scan in scans.values()))

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        size = statistics.sizes
        with np.errstate(divide="ignore", invalid="ignore"):
            quality = self._oriented_quality(
                statistics.sum(target._rank_deviations()), size, target.n_samples_
            )
        return np.where(size == 0, np.nan, quality)

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        estimates = [
            self._closed_many(statistics, target, negative)
            for negative in _ORIENTATIONS[self.direction]
        ]
        return np.where(statistics.sizes == 0, -np.inf, np.maximum.reduce(estimates))


class MannWhitneyQF(_RankQF):
    """Mann-Whitney z-statistic ``i_P (R_P / i_P - (i0 + 1) / 2) / sqrt(i_P i_notP (i0 + 1) / 12)``.

    Kloesgen (1996), Pieters et al. (2010), in the survey of Lemmerich et al.
    (2016, Section 4, 4a): ``R_P`` is the sum of the subgroup's ascending
    mid-ranks (no tie correction of the variance). It equals ``sqrt(12 / (i0 +
    1)) * q_mw'`` with the paper's ``q_mw'``. The whole dataset scores 0.
    ``direction="negative"`` rewards lower ranks (the statistic on the negated
    values), ``"both"`` takes the absolute value.

    Estimators:

    - ``"closed"`` (default, batched): the smaller of Theorem 17's bounds,
      ``oe1 = sum of (rank - (i0 + 1) / 2)`` over the subgroup's rows above the
      mean rank and ``sqrt(i+) * (rank_max - (i0 + 1) / 2)`` with ``i+`` such
      rows, scaled to the z-statistic. Without rows above the mean rank, every
      subset scores at most ``(rank_max - (i0 + 1) / 2) / sqrt(i0 - 1)`` (scaled).
    - ``"order"``: one-pass ordering-based bound (Theorem 3), tight.
    """

    _prefix_kind = _numba.PREFIX_MANN_WHITNEY

    @staticmethod
    def _prefix_weight(size, n_samples):
        return np.sqrt(size * (n_samples - size) * (n_samples + 1) / 12)

    def _score(self, total, size, n_samples):
        size = np.asarray(size, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = total / np.sqrt(size * (n_samples - size) * (n_samples + 1) / 12)
        return np.where(size >= n_samples, 0.0, score)

    def _bound(self, above, n_above, largest, target):
        n = target.n_samples_
        if n == 1:
            return np.zeros_like(np.asarray(largest, dtype=float))
        scale = np.sqrt(12 / (n + 1))
        with np.errstate(invalid="ignore"):
            bound = np.minimum(above, np.sqrt(n_above) * largest)
            fallback = largest / np.sqrt(n - 1)
        return scale * np.where(n_above > 0, bound, fallback)

    def _closed_scalar(self, mask, target, negative):
        deviations = target._rank_deviations(negative)[mask]
        positive = deviations[deviations > 0]
        return self._bound(np.sum(positive), len(positive), np.max(deviations), target)

    def _closed_many(self, statistics, target, negative):
        deviations = target._rank_deviations(negative)
        above = target._row_array(("rank excess", negative), lambda: np.maximum(deviations, 0))
        positive = target._row_array(("rank above", negative), lambda: deviations > 0)
        return self._bound(statistics.sum(above), statistics.count(positive),
                           statistics.maximum(deviations), target)


class RankAUCQF(_RankQF):
    """Area under the ROC curve of the subgroup against its complement.

    Pieters et al. (2010), in the survey of Lemmerich et al. (2016, Section 4,
    4b). With ascending mid-ranks, ``direction="positive"`` scores ``(R_P -
    i_P (i_P + 1) / 2) / (i_P i_notP)``, the probability that a subgroup row has
    a larger value than a complement row (ties count 1/2). The paper's
    ``q_auc``, with ``R_notP``, is the probability of a smaller value:
    ``direction="negative"``. ``"both"`` scores the larger of the two. The whole
    dataset scores 0.5. Bounds:

    - ``"closed"`` (default, batched): ``(L - 1) / (i0 - 1)`` with ``L`` the
      number of dataset rows with a value at most the subgroup's maximum.
      Without ties it is Theorem 18 (one subgroup row, the largest). Unlike
      Theorem 18 it holds with ties: each pair of a row of a subset ``r`` and a
      complement row scores at most 1 if the complement row is at most the
      maximum, else 0, so the subset scores at most ``(L - |r|) / (i0 - |r|) <=
      (L - 1) / (i0 - 1)``.
    - ``"order"``: one-pass ordering-based bound (Theorem 3), tight.
    """

    _prefix_kind = _numba.PREFIX_RANK_AUC

    @staticmethod
    def _prefix_weight(size, n_samples):
        return size * (n_samples - size)

    def _score(self, total, size, n_samples):
        size = np.asarray(size, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = 0.5 + total / (size * (n_samples - size))
        return np.where(size >= n_samples, 0.5, score)

    def _bound(self, at_most, target):
        n = target.n_samples_
        if n == 1:
            return np.full_like(np.asarray(at_most, dtype=float), 0.5)
        return (at_most - 1) / (n - 1)

    def _closed_scalar(self, mask, target, negative):
        return self._bound(np.max(target._rows_at_most(negative)[mask]), target)

    def _closed_many(self, statistics, target, negative):
        return self._bound(statistics.maximum(target._rows_at_most(negative)), target)


class KolmogorovSmirnovQF(_OrderedMixin, BoundedQualityFunction):
    """Kolmogorov-Smirnov measure ``sqrt(i_P i_notP / i0) * Delta``.

    Jorge et al. (2006), Lucas et al. (2007), in the survey of Lemmerich et al.
    (2016, Section 4, 5): ``Delta`` is the two-sided supremum of the difference
    between the empirical distribution functions of the subgroup and its
    complement, so the measure rewards any change of the distribution. The
    whole dataset scores 0.

    Optimistic estimate (Theorem 16): ``Delta <= 1``, and ``k (i0 - k)`` grows
    up to ``k = i0 / 2``, so a subgroup of ``i_P`` rows is bounded by
    ``sqrt(k (i0 - k) / i0)`` with ``k = min(i_P, floor(i0 / 2))``. Not batched:
    ``Delta`` needs the subgroup's distribution.
    """

    target_type = NumericTarget

    def evaluate(self, coverage, target) -> float:
        return self._evaluate_ordered(coverage, target)[0]

    @staticmethod
    def _bound(size: int, n: int) -> float:
        k = min(size, n // 2)
        return float(np.sqrt(k * (n - k) / n))

    def optimistic_estimate(self, coverage, target) -> float:
        size = int(np.count_nonzero(self._mask(coverage, target)))
        return self._bound(size, target.n_samples_) if size else float("-inf")

    def _ordered(self, sorted_mask, target):
        n, size = target.n_samples_, int(np.count_nonzero(sorted_mask))
        if not size:
            return float("nan"), float("-inf")
        if size == n:
            return 0.0, self._bound(size, n)
        # Rows with a value of at least each distinct value, in and outside the subgroup.
        ends = target._group_ends()
        inside = np.cumsum(sorted_mask)[ends]
        outside = ends + 1 - inside
        delta = float(np.max(np.abs(inside / size - outside / (n - size))))
        return float(np.sqrt(size * (n - size) / n)) * delta, self._bound(size, n)
