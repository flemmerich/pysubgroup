"""Filters (constraints) on the subgroups a search may report.

A filter decides from a subgroup's positional Boolean row mask, the fitted
target, and the subgroup's conditions whether the subgroup is acceptable.

Anti-monotone filters (``anti_monotone = True``) hold for all generalizations of
a subgroup that satisfies them; equivalently, once a subgroup fails, all its
refinements fail too. Searches use them to prune: a failing subgroup is neither
reported nor refined. Other filters only decide whether an evaluated subgroup
enters the result set; its refinements are still searched.

Filters with ``batchable = True`` also decide for all children of a search node
at once: ``is_satisfied_many(statistics, target)`` receives their
``ChildStatistics`` and returns one Boolean per child. Searches evaluate batches
only if every pruning filter is batchable.

Post-hoc filters on result lists are in ``utils``: ``overlap_filter``,
``unique_attributes``, and ``statistic_filter``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from .rules import Condition
from .targets import BaseTarget, LabelledTarget, MulticlassTarget


class Filter(ABC):
    """Constraint on reported subgroups."""

    anti_monotone: bool = False
    batchable: bool = False

    @abstractmethod
    def is_satisfied(
        self, coverage: np.ndarray, target: BaseTarget, conditions: tuple[Condition, ...]
    ) -> bool:
        """Whether the subgroup covering ``coverage`` and described by ``conditions`` passes."""

    def is_satisfied_many(self, statistics, target: BaseTarget) -> np.ndarray:
        """Whether each subgroup summarized by ``statistics`` passes (if ``batchable``)."""
        raise NotImplementedError(f"{type(self).__name__} does not evaluate batches")


def _row_threshold(value: int | float, n_samples: int, name: str) -> int | float:
    """Absolute row count for an int, or a fraction of the dataset for a float."""
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{name} must be a non-negative int or a fraction in [0, 1]")
    if isinstance(value, float):
        if value > 1:
            raise ValueError(f"a fractional {name} must be in [0, 1]")
        return value * n_samples
    return value


class MinSizeFilter(Filter):
    """At least ``min_size`` covered rows; a float is a fraction of the dataset."""

    anti_monotone = True
    batchable = True

    def __init__(self, min_size: int | float):
        _row_threshold(min_size, 1, "min_size")
        self.min_size = min_size

    def is_satisfied(self, coverage, target, conditions) -> bool:
        threshold = _row_threshold(self.min_size, target.n_samples_, "min_size")
        return int(np.count_nonzero(coverage)) >= math.ceil(threshold)

    def is_satisfied_many(self, statistics, target) -> np.ndarray:
        threshold = _row_threshold(self.min_size, target.n_samples_, "min_size")
        return statistics.sizes >= math.ceil(threshold)


class MaxSizeFilter(Filter):
    """At most ``max_size`` covered rows; a float is a fraction of the dataset."""

    batchable = True

    def __init__(self, max_size: int | float):
        _row_threshold(max_size, 1, "max_size")
        self.max_size = max_size

    def is_satisfied(self, coverage, target, conditions) -> bool:
        threshold = _row_threshold(self.max_size, target.n_samples_, "max_size")
        return int(np.count_nonzero(coverage)) <= math.floor(threshold)

    def is_satisfied_many(self, statistics, target) -> np.ndarray:
        threshold = _row_threshold(self.max_size, target.n_samples_, "max_size")
        return statistics.sizes <= math.floor(threshold)


class MinPositivesFilter(Filter):
    """At least ``min_positives`` covered positive rows (binary or soft-classifier targets)."""

    anti_monotone = True
    batchable = True

    def __init__(self, min_positives: int):
        if isinstance(min_positives, bool) or not isinstance(min_positives, int):
            raise ValueError("min_positives must be an int")
        self.min_positives = min_positives

    def is_satisfied(self, coverage, target, conditions) -> bool:
        if not isinstance(target, LabelledTarget):
            raise TypeError("MinPositivesFilter requires a target with positive labels")
        return int(np.count_nonzero(coverage & target.positives_)) >= self.min_positives

    def is_satisfied_many(self, statistics, target) -> np.ndarray:
        if not isinstance(target, LabelledTarget):
            raise TypeError("MinPositivesFilter requires a target with positive labels")
        return statistics.count(target.positives_) >= self.min_positives


class MinTargetShareFilter(Filter):
    """Share of positive rows in the subgroup of at least ``min_share``."""

    batchable = True

    def __init__(self, min_share: float):
        if not 0 <= min_share <= 1:
            raise ValueError("min_share must be in [0, 1]")
        self.min_share = min_share

    def is_satisfied(self, coverage, target, conditions) -> bool:
        if not isinstance(target, LabelledTarget):
            raise TypeError("MinTargetShareFilter requires a target with positive labels")
        size = int(np.count_nonzero(coverage))
        positives = np.count_nonzero(coverage & target.positives_)
        return bool(size) and positives >= self.min_share * size

    def is_satisfied_many(self, statistics, target) -> np.ndarray:
        if not isinstance(target, LabelledTarget):
            raise TypeError("MinTargetShareFilter requires a target with positive labels")
        sizes = statistics.sizes
        return (sizes > 0) & (statistics.count(target.positives_) >= self.min_share * sizes)


class MinMajorityCountFilter(Filter):
    """At least ``min_count`` covered rows of one class (multiclass targets).

    The rows of the subgroup's most frequent class are the support of its best
    class association rule. No refinement has more rows of any class, so the
    filter is anti-monotone; it is CMAR's class-distribution pruning (Li, Han
    and Pei, ICDM 2001, Section 3.1) and the frequent ruleitems of CBA. A float
    is a fraction of the dataset.
    """

    anti_monotone = True
    batchable = True

    def __init__(self, min_count: int | float):
        _row_threshold(min_count, 1, "min_count")
        self.min_count = min_count

    def _threshold(self, target) -> int:
        if not isinstance(target, MulticlassTarget):
            raise TypeError("MinMajorityCountFilter requires a MulticlassTarget")
        return math.ceil(_row_threshold(self.min_count, target.n_samples_, "min_count"))

    def is_satisfied(self, coverage, target, conditions) -> bool:
        threshold = self._threshold(target)
        counts = np.bincount(target.codes_[coverage], minlength=len(target.classes_))
        return int(counts.max()) >= threshold

    def is_satisfied_many(self, statistics, target) -> np.ndarray:
        threshold = self._threshold(target)
        counts = np.stack([statistics.count(mask) for mask in target.class_masks()], axis=1)
        return counts.max(axis=1) >= threshold


class CustomFilter(Filter):
    """Filter from a function ``(coverage, target, conditions) -> bool``.

    Declare ``anti_monotone=True`` only if refinements of a failing subgroup
    always fail too; otherwise the search may miss results.
    """

    def __init__(
        self,
        function: Callable[[np.ndarray, BaseTarget, tuple[Condition, ...]], bool],
        anti_monotone: bool = False,
    ):
        self.function = function
        self.anti_monotone = anti_monotone

    def is_satisfied(self, coverage, target, conditions) -> bool:
        return bool(self.function(coverage, target, conditions))
