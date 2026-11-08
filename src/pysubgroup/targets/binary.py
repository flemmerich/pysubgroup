"""Binary target (membership in one target class) and its quality functions.

Provenance: the measures, their parameters, and the optimistic estimators are
adapted from pysubgroup 0.x's ``binary_target.py``
(https://github.com/flemmerich/pysubgroup, Apache-2.0, Florian Lemmerich et al.),
rewritten for fitted targets and Boolean row masks. Deliberate changes: valid
bounds for subgroups without positives, a closed-form chi-squared test, and
``nan`` for empty subgroups.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from ..encoding import equals
from .base import (
    BoundedQualityFunction,
    Direction,
    LabelledTarget,
    QualityFunction,
    check_direction,
    check_exponent,
)


class BinaryTarget(LabelledTarget):
    """Membership in one target class; the target column is excluded from rules."""

    def fit(self, data: pd.DataFrame, attribute: Hashable, value) -> BinaryTarget:
        """Store which rows have ``data[attribute] == value``.

        A missing ``value`` matches missing entries; otherwise missing entries
        are negatives.
        """
        super().fit(data, attribute)
        column = data[attribute]
        matches = column.isna() if pd.isna(value) else equals(column, value)
        self.value_ = value
        self._set_positives(matches.fillna(False).to_numpy(dtype=bool))
        return self

    def _fit_arguments(self) -> tuple:
        return (*self.attributes_, self.value_)


# Adapted from pysubgroup 0.x's StandardQF (binary_target.py, Apache-2.0).
class StandardQF(BoundedQualityFunction):
    """``(n_sg / n) ** a * (p_sg - p)`` for target shares ``p_sg`` and ``p``.

    Different values of ``a`` are order-equivalent to many popular measures:
    ``a=0`` (lift difference), ``a=0.5`` (simple binomial), ``a=1`` (WRAcc).

    ``direction="negative"`` searches for a decrease of the target share,
    ``(n_sg / n) ** a * (p - p_sg)``, which is the measure on the complementary
    label; ``"both"`` scores the absolute deviation. The optimistic estimates
    follow by swapping positives and negatives.
    """

    target_type = BinaryTarget
    batchable = True

    def __init__(self, a: float, direction: Direction = "positive"):
        self.a = check_exponent(a)
        self.direction = check_direction(direction)

    @staticmethod
    def standard_qf(
        a: float,
        size_dataset: int,
        positives_dataset: int,
        size_sg: int,
        positives_sg: int,
    ) -> float:
        if size_sg == 0:
            return float("nan")
        share_difference = positives_sg / size_sg - positives_dataset / size_dataset
        return (size_sg / size_dataset) ** a * share_difference

    def _counts(self, coverage, target: BinaryTarget) -> tuple[int, int]:
        mask = self._mask(coverage, target)
        # ``mask & positives_`` avoids the copy of ``positives_[mask]`` (~50x faster).
        return int(np.count_nonzero(mask)), int(np.count_nonzero(mask & target.positives_))

    @staticmethod
    def _oriented(
        target: BinaryTarget, size_sg: int, positives_sg: int, negative: bool
    ) -> tuple[int, int]:
        """Positives of the dataset and the subgroup; negatives for a decrease."""
        if negative:
            return target.n_samples_ - target.n_positives_, size_sg - positives_sg
        return target.n_positives_, positives_sg

    def _directions(self) -> tuple[bool, ...]:
        return {"positive": (False,), "negative": (True,), "both": (False, True)}[self.direction]

    def evaluate(self, coverage, target) -> float:
        size_sg, positives_sg = self._counts(coverage, target)
        positives_dataset, positives = self._oriented(
            target, size_sg, positives_sg, self.direction == "negative"
        )
        quality = self.standard_qf(self.a, target.n_samples_, positives_dataset, size_sg, positives)
        return abs(quality) if self.direction == "both" else quality

    def optimistic_estimate(self, coverage, target) -> float:
        """Quality of the subset that keeps only the subgroup's positives.

        (Its negatives for a decrease; the larger of both for ``"both"``.)
        Without positives, every subset scores at most a single negative row.
        """
        size_sg, positives_sg = self._counts(coverage, target)
        if not size_sg:
            return float("-inf")
        estimates = []
        for negative in self._directions():
            positives_dataset, positives = self._oriented(target, size_sg, positives_sg, negative)
            estimates.append(self.standard_qf(
                self.a, target.n_samples_, positives_dataset, positives or 1, positives
            ))
        return max(estimates)

    @staticmethod
    def _standard_qf_many(a, size_dataset, positives_dataset, size_sg, positives_sg):
        """``standard_qf`` per subgroup (``nan`` for empty ones), with the same rounding."""
        with np.errstate(divide="ignore", invalid="ignore"):
            share_difference = positives_sg / size_sg - positives_dataset / size_dataset
            quality = np.power(size_sg / size_dataset, a) * share_difference
        return np.where(size_sg == 0, np.nan, quality)

    def _counts_many(self, statistics, target: BinaryTarget) -> tuple[np.ndarray, np.ndarray]:
        self._check_target(target)
        return statistics.sizes, statistics.count(target.positives_)

    def evaluate_many(self, statistics, target) -> np.ndarray:
        size_sg, positives_sg = self._counts_many(statistics, target)
        positives_dataset, positives = self._oriented(
            target, size_sg, positives_sg, self.direction == "negative"
        )
        quality = self._standard_qf_many(
            self.a, target.n_samples_, positives_dataset, size_sg, positives
        )
        return np.abs(quality) if self.direction == "both" else quality

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        size_sg, positives_sg = self._counts_many(statistics, target)
        estimates = []
        for negative in self._directions():
            positives_dataset, positives = self._oriented(target, size_sg, positives_sg, negative)
            estimates.append(self._standard_qf_many(
                self.a, target.n_samples_, positives_dataset, np.maximum(positives, 1), positives
            ))
        return np.where(size_sg == 0, -np.inf, np.maximum.reduce(estimates))

    def optimistic_generalisation(self, coverage, target) -> float:
        """Quality of the superset that adds all positives outside the subgroup.

        (All negatives for a decrease; the larger of both for ``"both"``.)
        """
        size_sg, positives_sg = self._counts(coverage, target)
        estimates = []
        for negative in self._directions():
            positives_dataset, positives = self._oriented(target, size_sg, positives_sg, negative)
            remaining = positives_dataset - positives
            estimates.append(self.standard_qf(
                self.a, target.n_samples_, positives_dataset, size_sg + remaining,
                positives_dataset,
            ))
        return max(estimates)


class LiftQF(StandardQF):
    """StandardQF with ``a=0``: the target-share difference, ignoring size."""

    def __init__(self, direction: Direction = "positive"):
        super().__init__(0.0, direction)


class SimpleBinomialQF(StandardQF):
    """StandardQF with ``a=0.5``; approximates the binomial test for small subgroups."""

    def __init__(self, direction: Direction = "positive"):
        super().__init__(0.5, direction)


class WRAccQF(StandardQF):
    """StandardQF with ``a=1`` (weighted relative accuracy)."""

    def __init__(self, direction: Direction = "positive"):
        super().__init__(1.0, direction)


# Parameters and direction handling adapted from pysubgroup 0.x's ChiSquaredQF
# (binary_target.py, Apache-2.0); the statistic is computed in closed form.
class ChiSquaredQF(QualityFunction):
    """Chi-squared test of independence between a subgroup and its complement.

    ``stat="chi2"`` returns the test statistic and ``stat="p"`` the p-value of the
    uncorrected 2x2 test. With ``direction="positive"`` or ``"negative"`` the value
    is negated when the target share deviates in the other direction. Subgroups or
    complements smaller than ``min_instances`` score ``-inf``.
    """

    target_type = BinaryTarget
    batchable = True

    def __init__(
        self,
        direction: Literal["both", "positive", "negative"] = "both",
        min_instances: int = 5,
        stat: Literal["chi2", "p"] = "chi2",
    ):
        check_direction(direction)
        if stat not in ("chi2", "p"):
            raise ValueError("stat must be 'chi2' or 'p'")
        self.direction = direction
        self.min_instances = min_instances
        self.stat = stat

    @staticmethod
    def chi_squared_qf(
        size_dataset: int,
        positives_dataset: int,
        size_sg: int,
        positives_sg: int,
        min_instances: int = 5,
        direction: str = "both",
        stat: str = "chi2",
    ) -> float:
        size_complement = size_dataset - size_sg
        if size_sg < min_instances or size_complement < min_instances:
            return float("-inf")
        negatives_dataset = size_dataset - positives_dataset
        positives_complement = positives_dataset - positives_sg
        negatives_sg = size_sg - positives_sg
        negatives_complement = negatives_dataset - negatives_sg
        denominator = size_sg * size_complement * positives_dataset * negatives_dataset
        if denominator:
            determinant = positives_sg * negatives_complement - positives_complement * negatives_sg
            chi2 = size_dataset * determinant**2 / denominator
        else:
            chi2 = 0.0
        value = chi2 if stat == "chi2" else float(stats.chi2.sf(chi2, 1))
        if direction == "both":
            return value
        share_sg = positives_sg / size_sg
        share_dataset = positives_dataset / size_dataset
        if (direction == "positive" and share_sg > share_dataset) or (
            direction == "negative" and share_sg < share_dataset
        ):
            return value
        return -value

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        return self.chi_squared_qf(
            target.n_samples_,
            target.n_positives_,
            int(np.count_nonzero(mask)),
            int(np.count_nonzero(mask & target.positives_)),
            self.min_instances,
            self.direction,
            self.stat,
        )

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        # Python integers (object arrays) keep the arithmetic of ``chi_squared_qf``
        # exact, so both paths round identically and cannot overflow.
        size_dataset, positives_dataset = target.n_samples_, target.n_positives_
        size_sg = statistics.sizes.astype(object)
        positives_sg = statistics.count(target.positives_).astype(object)
        size_complement = size_dataset - size_sg
        negatives_dataset = size_dataset - positives_dataset
        positives_complement = positives_dataset - positives_sg
        negatives_sg = size_sg - positives_sg
        negatives_complement = negatives_dataset - negatives_sg
        denominator = size_sg * size_complement * positives_dataset * negatives_dataset
        determinant = positives_sg * negatives_complement - positives_complement * negatives_sg
        defined = (denominator != 0).astype(bool)
        chi2 = np.zeros(len(size_sg))
        chi2[defined] = (
            size_dataset * determinant[defined] ** 2 / denominator[defined]
        ).astype(float)
        value = chi2 if self.stat == "chi2" else stats.chi2.sf(chi2, 1)
        if self.direction != "both":
            share_sg = (positives_sg / np.where(size_sg == 0, 1, size_sg)).astype(float)
            share_dataset = positives_dataset / size_dataset
            agrees = (
                share_sg > share_dataset
                if self.direction == "positive"
                else share_sg < share_dataset
            )
            value = np.where(agrees, value, -value)
        too_small = (
            (size_sg < self.min_instances) | (size_complement < self.min_instances)
        ).astype(bool)
        return np.where(too_small, -np.inf, value)
