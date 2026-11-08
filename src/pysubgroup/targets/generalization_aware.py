"""Generalization-aware quality functions.

A generalization of a subgroup ``P`` is a subgroup described by a proper subset
of ``P``'s conditions; the empty description (the whole dataset) is a
generalization of every subgroup. Generalization-aware measures compare a
subgroup with all its generalizations, so that adding a condition that does not
change the target scores no better than leaving it out:

- ``GeneralizationAwareQF(qf)``: the increase in quality,
  ``q(P) - max_{H ⊂ P} q(H)``.
- ``GeneralizationAwareStandardQF(a)``: the increase of the target share,
  ``(n_P / n) ** a * (p_P - max_{H ⊂ P} p_H)`` (``r_bin`` of the reference).
- ``GeneralizationAwareStandardQFNumeric(a)``: the increase of the mean,
  ``n_P ** a * (mean_P - max_{H ⊂ P} mean_H)`` (``r_num``).

Lemmerich, F.: Novel Techniques for Efficient and Effective Subgroup Discovery.
Dissertation, University of Würzburg, 2014, Sections 2.5.4 and 6.2 (the measures
follow Batal and Hauskrecht and Grosskreutz et al.'s election analysis, its
references [29] and [102]). Implemented from the dissertation without reusing
code.

Each measure scores a subgroup from a value ``v`` of the subgroup (its quality,
target share, or mean; ``value``) and the largest value of a generalization,
which the search passes as ``reference``:

- ``evaluate(coverage, target, reference=max_{H ⊂ P} v(H))``;
- ``optimistic_estimate(coverage, target, reference=max_{H ⊆ P} v(H),
  differences=...)`` bounds the quality of every proper refinement of ``P``,
  all of which have ``P`` and its generalizations among their
  generalizations. It need not bound the quality of ``P`` itself.

Difference-based estimates (Section 6.2.2-6.2.4) also use the rows that a
generalization ``P''`` covers beyond a subgroup ``P' ⊆ P`` it generalizes: by
Lemma 2, every refinement ``S`` of ``P`` has a generalization that covers at
most these rows beyond ``S``. ``difference(generalization, coverage, target)``
summarizes them for a direct generalization; ``merge_differences`` combines
the summaries of all direct generalizations of ``P`` and of their own
``differences`` into those of ``P``, which ``optimistic_estimate`` accepts as
``differences``. Pairs whose ``P''`` is not a direct generalization of ``P'``
are never needed: their difference contains that of a direct pair. Measures
without such estimates return None.

Only ``GeneralizationAwareApriori`` and ``GeneralizationAwareDFS`` supply the
reference during a search; ``utils.generalization_reference`` computes it for a
given rule. Batchable measures also take an array of references in
``evaluate_many`` and ``optimistic_estimate_many`` and give the values of many
subgroups with ``value_many`` (without differences).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import BoundedQualityFunction, QualityFunction, check_exponent
from .binary import BinaryTarget
from .frequency import AreaQF
from .numeric import NumericTarget


class GeneralizationAwareMeasure(BoundedQualityFunction):
    """Base class of quality functions that compare a subgroup with its generalizations."""

    def value(self, coverage, target) -> float:
        """The value ``v`` of the subgroup that is compared with its generalizations."""
        raise NotImplementedError

    def value_many(self, statistics, target) -> np.ndarray:
        """``value`` of the subgroups summarized by ``statistics`` (if ``batchable``)."""
        raise NotImplementedError

    def difference(self, generalization, coverage, target):
        """Summary of the rows of a direct ``generalization`` outside the subgroup, or None."""
        return None

    def merge_differences(self, differences: list):
        """The summaries of a subgroup from those of its pairs (None entries are ignored)."""
        return None

    #: Whether differences are also computed in batches, from one statistic per
    #: subgroup (``difference_state``): ``difference_states_many``,
    #: ``differences_many``, ``merge_differences_many``, and the ``differences``
    #: of ``optimistic_estimate_many``, with ``inf`` for no summary.
    batched_differences: bool = False

    def difference_state(self, coverage, target) -> float:
        """The statistic of a subgroup from which the differences of its pairs follow."""
        raise NotImplementedError

    def difference_states_many(self, statistics, target) -> np.ndarray:
        """``difference_state`` of the subgroups summarized by ``statistics``."""
        raise NotImplementedError

    def differences_many(self, generalization_states, states) -> np.ndarray:
        """Summaries of pairs (direct generalization, subgroup) from their states."""
        raise NotImplementedError

    def merge_differences_many(self, differences: list) -> np.ndarray:
        """``merge_differences`` for arrays of summaries (``inf``: no summary)."""
        raise NotImplementedError

    @staticmethod
    def _checked(reference):
        if reference is None:
            raise ValueError(
                "generalization-aware measures need the largest value of a generalization "
                "(reference); search with GeneralizationAwareApriori or GeneralizationAwareDFS, "
                "or compute it with "
                "utils.generalization_reference"
            )
        return reference


class GeneralizationAwareQF(GeneralizationAwareMeasure):
    """Increase in quality over all generalizations: ``q(P) - max_{H ⊂ P} q(H)``.

    ``qf`` must give the whole dataset a finite quality, the reference of every
    single condition. With a bounded ``qf``, ``oe(P) - max_{H ⊆ P} q(H)``
    bounds every refinement ``S`` of ``P``: ``q(S) <= oe(P)``, and ``P`` and its
    generalizations are generalizations of ``S``. Otherwise the estimate is
    infinite. There are no difference-based estimates.
    """

    def __init__(self, qf: QualityFunction):
        if isinstance(qf, GeneralizationAwareMeasure | AreaQF):
            raise TypeError(f"{type(qf).__name__} cannot be made generalization-aware")
        self.qf = qf
        self.target_type = qf.target_type

    @property
    def batchable(self) -> bool:
        return self.qf.batchable

    def value(self, coverage, target) -> float:
        return self.qf.evaluate(coverage, target)

    def value_many(self, statistics, target) -> np.ndarray:
        return np.asarray(self.qf.evaluate_many(statistics, target), dtype=float)

    def evaluate(self, coverage, target, *, reference: float | None = None) -> float:
        return self.value(coverage, target) - self._checked(reference)

    def evaluate_many(self, statistics, target, *, reference=None) -> np.ndarray:
        reference = self._checked(reference)
        return self.value_many(statistics, target) - reference

    def optimistic_estimate(
        self, coverage, target, *, reference: float | None = None, differences=None
    ) -> float:
        reference = self._checked(reference)
        if not isinstance(self.qf, BoundedQualityFunction):
            return float("inf")
        return self.qf.optimistic_estimate(coverage, target) - reference

    def optimistic_estimate_many(self, statistics, target, *, reference=None) -> np.ndarray:
        reference = self._checked(reference)
        if not isinstance(self.qf, BoundedQualityFunction):
            return np.full(statistics.n_children, float("inf"))
        return np.asarray(self.qf.optimistic_estimate_many(statistics, target)) - reference


def _difference_factor(a: float, size: int, removed: int) -> float:
    """``max_{1 <= s <= size} s ** a * removed / (s + removed)`` (Theorems 3 and 4).

    The maximum is at ``size`` for ``a = 1``, at 1 for ``a = 0``, and otherwise
    at ``min(a * removed / (1 - a), size)``, the root of the derivative. Below 1
    that root gives a larger value than ``s = 1``, a valid but looser bound, as
    in the dissertation.
    """
    if not removed:
        return 0.0
    if a == 1:
        best = size
    elif a == 0:
        best = 1
    else:
        best = min(a * removed / (1 - a), size)
    return best**a * removed / (best + removed)


def _difference_factor_many(a: float, size: np.ndarray, removed: np.ndarray) -> np.ndarray:
    """``_difference_factor`` per subgroup; ``inf`` where nothing was removed from a pair."""
    if a == 1:
        best = size
    elif a == 0:
        best = np.ones_like(size)
    else:
        best = np.minimum(a * removed / (1 - a), size)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = best**a * removed / (best + removed)
    return np.where(np.isinf(removed), np.inf, np.where(removed == 0, 0.0, factor))


class GeneralizationAwareStandardQF(GeneralizationAwareMeasure):
    """``(n_P / n) ** a * (p_P - max_{H ⊂ P} p_H)`` for target shares ``p``.

    The dissertation's ``r_bin`` weights by the size ``n_P ** a``; the relative
    size of ``StandardQF`` differs by the constant ``n ** a``, and so do the
    estimates below. Searches for an increase of the target share,
    ``0 <= a <= 1``. Optimistic estimates of proper refinements (the smaller of
    both; 0 without positives, as every refinement has share 0):

    - Theorem 2: ``(tp_P / n) ** a * (1 - max_{H ⊆ P} p_H)``, the quality of a
      refinement keeping only positives.
    - Theorem 3 (difference-based): with ``d`` the fewest negatives that a
      direct generalization ``P''`` of some ``P' ⊆ P`` covers beyond ``P'``,
      ``max_{1 <= s <= tp_P} s ** a * d / (s + d) / n ** a``. The summary of a
      subgroup's differences is that ``d``.
    """

    target_type = BinaryTarget
    batchable = True
    batched_differences = True

    def __init__(self, a: float):
        self.a = check_exponent(a)

    def _counts(self, coverage, target: BinaryTarget) -> tuple[int, int]:
        mask = self._mask(coverage, target)
        return int(np.count_nonzero(mask)), int(np.count_nonzero(mask & target.positives_))

    def _counts_many(self, statistics, target: BinaryTarget) -> tuple[np.ndarray, np.ndarray]:
        self._check_target(target)
        return statistics.sizes, statistics.count(target.positives_)

    def value(self, coverage, target) -> float:
        size, positives = self._counts(coverage, target)
        return positives / size if size else float("nan")

    def value_many(self, statistics, target) -> np.ndarray:
        size, positives = self._counts_many(statistics, target)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(size == 0, np.nan, positives / size)

    def evaluate(self, coverage, target, *, reference: float | None = None) -> float:
        reference = self._checked(reference)
        size, positives = self._counts(coverage, target)
        if not size:
            return float("nan")
        return (size / target.n_samples_) ** self.a * (positives / size - reference)

    def evaluate_many(self, statistics, target, *, reference=None) -> np.ndarray:
        reference = self._checked(reference)
        size, positives = self._counts_many(statistics, target)
        with np.errstate(divide="ignore", invalid="ignore"):
            quality = (size / target.n_samples_) ** self.a * (positives / size - reference)
        return np.where(size == 0, np.nan, quality)

    def difference(self, generalization, coverage, target) -> int:
        mask = self._mask(coverage, target)
        outside = self._mask(generalization, target) & ~mask
        return int(np.count_nonzero(outside & ~target.positives_))

    def merge_differences(self, differences: list) -> int | None:
        return min((d for d in differences if d is not None), default=None)

    def difference_state(self, coverage, target) -> float:
        size, positives = self._counts(coverage, target)
        return float(size - positives)

    def difference_states_many(self, statistics, target) -> np.ndarray:
        size, positives = self._counts_many(statistics, target)
        return (size - positives).astype(float)

    def differences_many(self, generalization_states, states) -> np.ndarray:
        # The negatives of a generalization outside a subgroup it contains.
        return generalization_states - states

    def merge_differences_many(self, differences: list) -> np.ndarray:
        return np.minimum.reduce(differences)

    def optimistic_estimate(
        self, coverage, target, *, reference: float | None = None,
        differences: int | None = None,
    ) -> float:
        reference = self._checked(reference)
        size, positives = self._counts(coverage, target)
        if not size:
            return float("-inf")
        if not positives:
            return 0.0
        estimate = (positives / target.n_samples_) ** self.a * (1 - reference)
        if differences is not None:
            estimate = min(estimate, _difference_factor(self.a, positives, differences)
                           / target.n_samples_**self.a)
        return estimate

    def optimistic_estimate_many(self, statistics, target, *, reference=None,
                                 differences=None) -> np.ndarray:
        reference = self._checked(reference)
        size, positives = self._counts_many(statistics, target)
        estimate = (positives / target.n_samples_) ** self.a * (1 - reference)
        if differences is not None:
            estimate = np.minimum(estimate, _difference_factor_many(self.a, positives, differences)
                                  / target.n_samples_**self.a)
        return np.where(size == 0, -np.inf, np.where(positives == 0, 0.0, estimate))


def _pareto(pairs) -> tuple[tuple[int, float], ...]:
    """The pairs ``(removed rows, smallest removed value)`` that no other pair dominates.

    A pair with fewer removed rows and a larger smallest value gives a smaller
    bound for every subgroup.
    """
    kept = []
    for removed, smallest in sorted(pairs, key=lambda pair: (pair[0], -pair[1])):
        if not kept or smallest > kept[-1][1]:
            kept.append((removed, smallest))
    return tuple(kept)


class GeneralizationAwareStandardQFNumeric(GeneralizationAwareMeasure):
    """``n_P ** a * (mean_P - max_{H ⊂ P} mean_H)``, the dissertation's ``r_num``.

    Searches for an increase of the mean, ``0 <= a <= 1``. Optimistic estimates
    of proper refinements (the smaller of both):

    - The standard bounds of ``StandardQFNumeric`` with the largest mean
      ``m = max_{H ⊆ P} mean_H`` in place of the dataset mean. Every
      refinement ``S`` has ``P`` and its generalizations as generalizations, so
      ``r_num(S) <= n_S ** a * (mean_S - mean_G)`` for each of them; these
      bounds hold for any constant centroid and decrease with it, so the
      largest mean gives the smallest. ``estimator``: ``"sum"`` (default),
      ``sum(max(y - m, 0))`` over the subgroup's values ``y``; ``"max"``,
      ``k ** a * (max(y) - m)`` for the ``k`` values above ``m``; ``"combined"``,
      the smaller of both; ``"order"``, the best prefix of the values sorted in
      decreasing order (tight; ``n log n``).
    - Theorem 4 (difference-based): for a direct generalization ``P''`` of some
      ``P' ⊆ P`` that covers ``d`` rows beyond ``P'``, the smallest of them
      with value ``y_min``,
      ``max(0, max_{1 <= s <= n_P} s ** a * d / (s + d) * (max_P(y) - y_min))``.
      The summary of a subgroup's differences is the Pareto set of the pairs
      ``(d, y_min)``, the smallest bound over them is taken.
    """

    target_type = NumericTarget
    ESTIMATORS = ("sum", "max", "combined", "order")

    def __init__(self, a: float, estimator: Literal["sum", "max", "combined", "order"] = "sum"):
        self.a = check_exponent(a)
        if estimator not in self.ESTIMATORS:
            raise ValueError(f"estimator must be one of {self.ESTIMATORS}, got {estimator!r}")
        self.estimator = estimator

    def value(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        size = int(np.count_nonzero(mask))
        if not size:
            return float("nan")
        return target.mean_ if size == target.n_samples_ else float(np.mean(target.values_[mask]))

    def evaluate(self, coverage, target, *, reference: float | None = None) -> float:
        reference = self._checked(reference)
        mask = self._mask(coverage, target)
        size = int(np.count_nonzero(mask))
        if not size:
            return float("nan")
        return size**self.a * (self.value(mask, target) - reference)

    def difference(self, generalization, coverage, target) -> tuple[tuple[int, float], ...]:
        mask = self._mask(coverage, target)
        outside = target.values_[self._mask(generalization, target) & ~mask]
        return ((len(outside), float(np.min(outside)) if len(outside) else float("inf")),)

    def merge_differences(self, differences: list) -> tuple[tuple[int, float], ...] | None:
        pairs = [pair for summary in differences if summary is not None for pair in summary]
        return _pareto(pairs) if pairs else None

    def _standard_estimate(self, values: np.ndarray, reference: float) -> float:
        if self.estimator == "order":
            ordered = np.sort(values)[::-1]
            sizes = np.arange(1, len(ordered) + 1)
            return float(np.max(sizes**self.a * (np.cumsum(ordered) / sizes - reference)))
        above = values[values > reference]
        if not len(above):
            # Every refinement has a mean of at most the largest value.
            return float(np.max(values)) - reference
        excess = float(np.sum(above - reference))
        largest = len(above) ** self.a * (float(np.max(above)) - reference)
        return {"sum": excess, "max": largest, "combined": min(excess, largest)}[self.estimator]

    def optimistic_estimate(
        self, coverage, target, *, reference: float | None = None,
        differences: tuple[tuple[int, float], ...] | None = None,
    ) -> float:
        reference = self._checked(reference)
        values = target.values_[self._mask(coverage, target)]
        if not len(values):
            return float("-inf")
        estimate = self._standard_estimate(values, reference)
        largest = float(np.max(values))
        for removed, smallest in differences or ():
            if not removed or largest <= smallest:
                return min(estimate, 0.0)
            estimate = min(estimate, _difference_factor(self.a, len(values), removed)
                           * (largest - smallest))
        return estimate
