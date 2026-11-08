"""Soft-classifier target (labels and prediction scores) and the SubROC measures.

Implements the SubROC measures (Siegl et al., https://doi.org/10.48550/arXiv.2505.11283).

Provenance: the measures, their bounds, and ``BaseSoftClassifierPerformanceQF``
are adapted from pysubgroup 0.x's ``model_predictions_target.py``
(https://github.com/flemmerich/pysubgroup, Apache-2.0, Tom Siegl), rewritten for
fitted targets and Boolean row masks. Deliberate changes: ARL is vectorized and
counts ties as one half, and the ARL and weighted bounds are corrected (they
could fall below a subgroup's quality with tied scores or negative deviations).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Literal

import numpy as np
import pandas as pd

from ..encoding import equals
from .base import BoundedQualityFunction, LabelledTarget


class SoftClassifierTarget(LabelledTarget):
    """Binary labels and continuous prediction scores for later model-based QFs.

    ``order_`` lists the rows by decreasing score (ties in row order);
    ``sorted_positives_`` and ``sorted_scores_`` are labels and scores in that order.
    """

    def fit(
        self,
        data: pd.DataFrame,
        label_attribute: Hashable,
        score_attribute: Hashable,
        positive_value=1,
    ) -> SoftClassifierTarget:
        """Store which rows are labelled ``positive_value`` and their scores."""
        super().fit(data, label_attribute, score_attribute)
        labels = data[label_attribute]
        if labels.isna().any() or labels.nunique() > 2:
            raise ValueError("soft classifier targets require binary, non-missing labels")
        scores = data[score_attribute].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(scores).all():
            raise ValueError("soft classifier targets require finite scores")
        self.positive_value_ = positive_value
        self._set_positives(equals(labels, positive_value).to_numpy(dtype=bool))
        self.scores_ = scores
        self.order_ = np.argsort(-scores, kind="stable")
        self.sorted_positives_ = self.positives_[self.order_]
        self.sorted_scores_ = scores[self.order_]
        self._dataset_performance = {}
        return self

    def _fit_arguments(self) -> tuple:
        return (*self.attributes_, self.positive_value_)

    def _subgroup(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Labels and scores of the subgroup's rows by decreasing score."""
        in_subgroup = mask[self.order_]
        return self.sorted_positives_[in_subgroup], self.sorted_scores_[in_subgroup]


# Soft-classifier performance measures (SubROC)
# Measures, bounds, and BaseSoftClassifierPerformanceQF are adapted from pysubgroup 0.x's
# model_predictions_target.py (Tom Siegl, Apache-2.0); ARL is vectorized, and the
# ARL and weighted bounds are corrected.


def _threshold_counts(positives: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True and false positives scoring at least each distinct score, by decreasing score."""
    if not len(scores):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if len(scores) > 1 and (scores[1:] > scores[:-1]).any():
        order = np.argsort(-scores, kind="stable")
        positives, scores = positives[order], scores[order]
    ends = np.flatnonzero(np.append(scores[1:] != scores[:-1], True))
    true_positives = np.cumsum(positives, dtype=np.int64)[ends]
    return true_positives, ends + 1 - true_positives


def _increments(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Counts before each distinct score and the increase at it."""
    before = np.concatenate(([0], counts[:-1]))
    return before, counts - before


def average_ranking_loss(positives: np.ndarray, scores: np.ndarray) -> float:
    """Average ranking loss (ARL): mean number of negatives ranked above a positive.

    Negatives with equal score count one half (Duivesteijn and Thaele, ICDM 2014).
    """
    true_positives, false_positives = _threshold_counts(positives, scores)
    if not len(true_positives) or not true_positives[-1]:
        raise ValueError("the average ranking loss requires a positive")
    _, new_positives = _increments(true_positives)
    above, tied = _increments(false_positives)
    return float(np.sum(new_positives * (2 * above + tied)) / (2 * true_positives[-1]))


def roc_auc_score(positives: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve: the share of positive-negative pairs ranked correctly.

    Ties count one half (the Mann-Whitney statistic).
    """
    true_positives, false_positives = _threshold_counts(positives, scores)
    if not len(true_positives) or not true_positives[-1] or not false_positives[-1]:
        raise ValueError("the ROC AUC requires positives and negatives")
    below, new_positives = _increments(true_positives)
    _, new_negatives = _increments(false_positives)
    pairs = 2 * true_positives[-1] * false_positives[-1]
    return float(np.sum(new_negatives * (2 * below + new_positives)) / pairs)


def pr_auc_score(positives: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (trapezoidal, starting at precision 1)."""
    true_positives, false_positives = _threshold_counts(positives, scores)
    if not len(true_positives) or not true_positives[-1]:
        raise ValueError("the PR AUC requires a positive")
    recall = np.concatenate(([0.0], true_positives / true_positives[-1]))
    precision = np.concatenate(([1.0], true_positives / (true_positives + false_positives)))
    return float(np.sum(np.diff(recall) * (precision[1:] + precision[:-1])) / 2)


def _contains_error(positives: np.ndarray, scores: np.ndarray) -> bool:
    """Whether some negative scores strictly higher than some positive."""
    return bool(
        positives.any()
        and (~positives).any()
        and scores[positives].min() < scores[~positives].max()
    )


def _contains_tie(positives: np.ndarray, scores: np.ndarray) -> bool:
    """Whether some positive and negative share a score."""
    return bool(np.intersect1d(scores[positives], scores[~positives]).size)


def _arl_upper_bound(positives: np.ndarray, scores: np.ndarray) -> float:
    """ARL of the lowest-scored positive together with all negatives."""
    if not positives.any():
        return 0.0
    lowest = scores[positives].min()
    negatives = scores[~positives]
    return float(np.count_nonzero(negatives > lowest) + 0.5 * np.count_nonzero(negatives == lowest))


def _roc_auc_lower_bound(positives: np.ndarray, scores: np.ndarray) -> float:
    if not len(scores) or _contains_error(positives, scores):
        return 0.0
    return 0.5 if _contains_tie(positives, scores) else 1.0


def _pr_auc_lower_bound(positives: np.ndarray, scores: np.ndarray) -> float:
    """PR AUC of the lowest-scored positive with all negatives scored at least as high."""
    if not len(scores):
        return 0.0
    if not _contains_error(positives, scores) and not _contains_tie(positives, scores):
        return 1.0
    lowest = scores[positives].min()
    negatives = scores[~positives]
    negatives = negatives[negatives >= lowest]
    subset_positives = np.zeros(len(negatives) + 1, dtype=bool)
    subset_positives[0] = True
    return pr_auc_score(subset_positives, np.concatenate(([lowest], negatives)))


class BaseSoftClassifierPerformanceQF(BoundedQualityFunction):
    """Performance deviation of a soft classifier in the subgroup from the dataset.

    The quality is ``(performance_sg - performance_dataset)``, negated for score-type
    measures so that worse subgroup performance is always positive, multiplied by
    ``class_balance_sg ** subgroup_class_balance_weight * n_sg ** subgroup_size_weight``.
    The class balance is the minority/majority label ratio, or 0 unless both labels
    occur. Subgroups on which the performance measure is undefined score ``-inf``.

    Implements the SubROC measures (Siegl et al., https://doi.org/10.48550/arXiv.2505.11283).
    """

    target_type = SoftClassifierTarget

    def __init__(
        self,
        performance_measure: Callable[[np.ndarray, np.ndarray], float],
        performance_measure_type: Literal["score", "loss"],
        performance_measure_bound: Callable[[np.ndarray, np.ndarray], float] | None = None,
        is_defined: Callable[[np.ndarray], bool] = lambda positives: True,
        subgroup_class_balance_weight: float = 0,
        subgroup_size_weight: float = 0,
    ):
        """
        ``performance_measure_bound`` returns the worst performance on any subset
        (a lower bound for scores, an upper bound for losses). ``is_defined``
        receives the subgroup's positive labels. Measures and bounds receive the
        subgroup's labels and scores by decreasing score.
        """
        self.performance_measure = performance_measure
        self.performance_measure_type = performance_measure_type
        self.performance_measure_bound = performance_measure_bound
        self.is_defined = is_defined
        self.subgroup_class_balance_weight = subgroup_class_balance_weight
        self.subgroup_size_weight = subgroup_size_weight

    def _deviation(self, performance: float, target: SoftClassifierTarget) -> float:
        cache = target.__dict__.setdefault("_dataset_performance", {})
        if self.performance_measure not in cache:
            cache[self.performance_measure] = self.performance_measure(
                target.positives_, target.scores_
            )
        deviation = performance - cache[self.performance_measure]
        return -deviation if self.performance_measure_type == "score" else deviation

    def evaluate(self, coverage, target) -> float:
        positives, scores = target._subgroup(self._mask(coverage, target))
        if not self.is_defined(positives):
            return float("-inf")
        n_positives = int(np.count_nonzero(positives))
        n_negatives = len(positives) - n_positives
        balance = (
            min(n_positives, n_negatives) / max(n_positives, n_negatives)
            if n_positives and n_negatives
            else 0.0
        )
        weight = (
            balance**self.subgroup_class_balance_weight
            * len(positives) ** self.subgroup_size_weight
        )
        return self._deviation(self.performance_measure(positives, scores), target) * weight

    def optimistic_estimate(self, coverage, target) -> float:
        """Bound from the worst-case performance; ``inf`` if no bound is known.

        A bound exists without weighting or for
        ``0 < subgroup_size_weight <= subgroup_class_balance_weight``.
        """
        mask = self._mask(coverage, target)
        if (
            self.performance_measure_bound is None
            or self.subgroup_size_weight < 0
            or self.subgroup_size_weight > self.subgroup_class_balance_weight
        ):
            return float("inf")
        positives, scores = target._subgroup(mask)
        estimate = self._deviation(self.performance_measure_bound(positives, scores), target)
        if estimate < 0 and (self.subgroup_class_balance_weight or self.subgroup_size_weight):
            return 0.0  # weights are non-negative, so no subset scores above 0
        if self.subgroup_size_weight > 0:
            n_positives = int(np.count_nonzero(positives))
            minority = min(n_positives, len(positives) - n_positives)
            if not minority:
                return 0.0
            estimate *= (2 * minority) ** self.subgroup_size_weight
        return estimate


def _has_positive(positives: np.ndarray) -> bool:
    return bool(positives.any())


def _has_both_labels(positives: np.ndarray) -> bool:
    return bool(positives.any() and not positives.all())


class ARLQF(BaseSoftClassifierPerformanceQF):
    """Increase of the average ranking loss in the subgroup (SubROC's rasl measure)."""

    def __init__(self, subgroup_class_balance_weight: float = 0, subgroup_size_weight: float = 0):
        super().__init__(
            average_ranking_loss,
            "loss",
            _arl_upper_bound,
            _has_positive,
            subgroup_class_balance_weight,
            subgroup_size_weight,
        )


class ROCAUCQF(BaseSoftClassifierPerformanceQF):
    """Decrease of the ROC AUC in the subgroup (SubROC's rROCAUC measure)."""

    def __init__(self, subgroup_class_balance_weight: float = 0, subgroup_size_weight: float = 0):
        super().__init__(
            roc_auc_score,
            "score",
            _roc_auc_lower_bound,
            _has_both_labels,
            subgroup_class_balance_weight,
            subgroup_size_weight,
        )


class PRAUCQF(BaseSoftClassifierPerformanceQF):
    """Decrease of the PR AUC in the subgroup (SubROC's rPRAUC measure)."""

    def __init__(self, subgroup_class_balance_weight: float = 0, subgroup_size_weight: float = 0):
        super().__init__(
            pr_auc_score,
            "score",
            _pr_auc_lower_bound,
            _has_positive,
            subgroup_class_balance_weight,
            subgroup_size_weight,
        )
