"""Base classes of targets and quality functions.

A target is fitted on the dataset and stores per-row target information as
positional arrays. A quality function keeps only its configuration;
``evaluate(coverage, target)`` scores the subgroup given by a positional Boolean
row mask against a fitted target, and bounded measures add
``optimistic_estimate(coverage, target)``, an upper bound on the quality of
every subset of the subgroup. Each quality function accepts the target class in
its ``target_type``.

Quality functions with ``batchable = True`` also score many subgroups at once:
``evaluate_many(statistics, target)`` and ``optimistic_estimate_many`` receive
the ``ChildStatistics`` of the children of a search node (their sizes, and
counts, sums, and maxima of target arrays over their rows) and return one value
per child, equal to the scalar results (float sums up to rounding). Searches
then evaluate all children of a node in one pass, also on a GPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Literal

import numpy as np
import pandas as pd

DIRECTIONS = ("positive", "negative", "both")
Direction = Literal["positive", "negative", "both"]


def check_direction(direction: str) -> str:
    """Validate the deviation direction a quality function searches for.

    ``"positive"`` rewards an increase of the target (share or centroid),
    ``"negative"`` a decrease, and ``"both"`` the absolute deviation.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return direction


def check_exponent(a: float) -> float:
    """Validate the size exponent ``a`` of the standard measures.

    Their optimistic estimates hold only for ``0 <= a <= 1``, so other values
    are rejected.
    """
    if isinstance(a, bool) or not isinstance(a, int | float) or not 0 <= a <= 1:
        raise ValueError(f"a must be a number in [0, 1], got {a!r}")
    return a


def orient(quality: float, direction: str) -> float:
    """A positive-direction quality for ``direction``, for measures odd under negation."""
    if direction == "negative":
        return -quality
    if direction == "both":
        return abs(quality)
    return quality


class BaseTarget:
    """Base class for target concepts.

    ``fit`` receives the dataset together with the features that define the
    target and stores per-instance target information as positional arrays, so
    quality functions can evaluate any subgroup given as a Boolean row mask.
    Private attributes (``_name``) hold caches derived from the fitted data;
    ``fit`` drops them, so fitting again never reuses them.
    """

    def fit(self, data: pd.DataFrame, *attributes: Hashable) -> BaseTarget:
        """Validate ``data`` and record its size and the target attributes."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("targets require a pandas DataFrame")
        if not len(data):
            raise ValueError("targets require at least one row")
        absent = [name for name in attributes if name not in data.columns]
        if absent:
            raise ValueError(f"target columns are missing: {absent}")
        for name in [name for name in vars(self) if name.startswith("_")]:
            del self.__dict__[name]
        self.attributes_ = tuple(attributes)
        self.n_samples_ = len(data)
        return self

    def _fit_arguments(self) -> tuple:
        """The arguments after ``data`` that ``fit`` was called with."""
        return self.attributes_

    def refit(self, data: pd.DataFrame) -> BaseTarget:
        """A new target of this type fitted on ``data`` with the same arguments."""
        if not hasattr(self, "n_samples_"):
            raise ValueError("only fitted targets can be refitted")
        return type(self)().fit(data, *self._fit_arguments())


class LabelledTarget(BaseTarget):
    """Base class for targets with a positive label per row.

    Subclasses store the labels with ``_set_positives``: ``positives_`` is the
    positional Boolean mask of positive rows and ``n_positives_`` its count.
    Filters and plots that only need positive labels accept any labelled target.
    """

    def _set_positives(self, positives: np.ndarray) -> None:
        self.positives_ = positives
        self.n_positives_ = int(np.count_nonzero(positives))


class QualityFunction(ABC):
    """Base class for quality functions over fitted targets."""

    target_type: type[BaseTarget] = BaseTarget
    batchable: bool = False

    @abstractmethod
    def evaluate(self, coverage: np.ndarray, target: BaseTarget) -> float:
        """Return the quality of the subgroup covering ``coverage``."""

    def evaluate_many(self, statistics, target: BaseTarget) -> np.ndarray:
        """Qualities of the subgroups summarized by ``statistics`` (if ``batchable``)."""
        raise NotImplementedError(f"{type(self).__name__} does not evaluate batches")

    def _check_target(self, target: BaseTarget) -> None:
        if not isinstance(target, self.target_type):
            raise TypeError(
                f"{type(self).__name__} requires a {self.target_type.__name__}, "
                f"got {type(target).__name__}"
            )
        if not hasattr(target, "n_samples_"):
            raise ValueError("target must be fitted before computing qualities")

    def _mask(self, coverage: np.ndarray, target: BaseTarget) -> np.ndarray:
        self._check_target(target)
        mask = np.asarray(coverage)
        if mask.dtype != bool or mask.shape != (target.n_samples_,):
            raise ValueError("coverage must be a Boolean mask with one entry per row")
        return mask


class BoundedQualityFunction(QualityFunction):
    """Quality function with an optimistic estimate for pruning."""

    @abstractmethod
    def optimistic_estimate(self, coverage: np.ndarray, target: BaseTarget) -> float:
        """Return an upper bound on the quality of any subset of the subgroup."""

    def optimistic_estimate_many(self, statistics, target: BaseTarget) -> np.ndarray:
        """Optimistic estimates of the subgroups summarized by ``statistics``."""
        raise NotImplementedError(f"{type(self).__name__} does not evaluate batches")
