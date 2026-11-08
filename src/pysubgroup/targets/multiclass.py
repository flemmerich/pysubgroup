"""Multiclass target (one nominal label per row) and its quality functions.

The measures compare the class distribution ``p`` of a subgroup of size ``n``
with the class distribution ``p0`` of the dataset of size ``N``:

- ``SplitQF``: ``n * sum_i (p_i - p0_i) ** 2``;
- ``MulticlassChiSquaredQF``: ``n * sum_i (p_i - p0_i) ** 2 / p0_i`` (Pearson);
- ``GiniQF``: ``n / (N - n) * sum_i (p_i - p0_i) ** 2``;
- ``ClassDistributionL1QF``: ``n / N * sum_i |p_i - p0_i|``;
- ``MaxConfidenceQF``: ``max_i p_i``, the confidence of the class association
  rule from the subgroup to its majority class (CBA, CMAR).

The first three measures and their optimistic estimates are from H. Grosskreutz,
S. Rueping, N. Shaabani, and S. Wrobel, "Optimistic estimate pruning strategies
for fast exhaustive subgroup discovery", technical report, Fraunhofer IAIS,
2008 (Table 1, Section 4, Appendix A). The last is ``q_dst`` (Eq. 2) of
H. Grosskreutz, M. Boley, and M. Krause-Traudes, "Subgroup discovery for
election analysis: a case study in descriptive data mining" (DS 2010), applied
to one-hot class indicators (``vector.MeanVectorL1QF`` is the general form);
that paper gives no optimistic estimate, so the tight bound used here is derived
in the docstring of ``ClassDistributionL1QF``. Written from the papers without
reusing code.

``MaxConfidenceQF`` follows from the definition of confidence (Liu, Hsu and Ma,
KDD 1998); its optimistic estimate is the trivial tight bound 1.

Deviations from the papers: empty subgroups score ``nan`` and bound at ``-inf``;
the Gini quality of the whole dataset (``n = N``, where ``n / (N - n)`` is
undefined) is 0, the limit of its zero deviation, and so are the Gini bound
terms whose refinement size equals ``N``.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Hashable

import numpy as np
import pandas as pd

from .base import BaseTarget, BoundedQualityFunction


class MulticlassTarget(BaseTarget):
    """Class distribution of a nominal target attribute."""

    def fit(self, data: pd.DataFrame, attribute: Hashable) -> MulticlassTarget:
        """Store the class of every row of ``data[attribute]``.

        ``classes_`` lists the observed classes (sorted where possible),
        ``codes_`` holds each row's class index, and ``class_counts_`` and
        ``priors_`` the dataset class distribution.
        """
        super().fit(data, attribute)
        column = data[attribute]
        if column.isna().any():
            raise ValueError("multiclass targets require non-missing labels")
        try:
            codes, classes = pd.factorize(column, sort=True)
        except TypeError:
            codes, classes = pd.factorize(column)
        self.classes_ = np.asarray(classes, dtype=object)
        self.codes_ = codes.astype(np.intp)
        self.class_counts_ = np.bincount(self.codes_, minlength=len(self.classes_))
        self.priors_ = self.class_counts_ / self.n_samples_
        self._class_masks = None
        return self

    def class_masks(self) -> list[np.ndarray]:
        """Boolean row mask of every class, cached for batched statistics."""
        if getattr(self, "_class_masks", None) is None:
            self._class_masks = [self.codes_ == k for k in range(len(self.classes_))]
        return self._class_masks


@functools.lru_cache(maxsize=8)
def _class_sets(n_classes: int) -> np.ndarray:
    """Membership matrix of all non-empty class sets, one row per set."""
    rows = list(itertools.product((0.0, 1.0), repeat=n_classes))[1:]
    return np.array(rows)


class MulticlassQF(BoundedQualityFunction):
    """Base class of quality functions on the subgroup's class counts.

    ``quality`` and ``estimate`` take the class counts along the last axis, so
    they score one subgroup (a vector) or many (one row per subgroup).
    """

    target_type = MulticlassTarget
    batchable = True

    def _class_counts(self, coverage, target: MulticlassTarget) -> np.ndarray:
        mask = self._mask(coverage, target)
        return np.bincount(target.codes_[mask], minlength=len(target.classes_))

    def evaluate(self, coverage, target) -> float:
        counts = self._class_counts(coverage, target)
        if not counts.sum():
            return float("nan")
        return float(self.quality(counts, target.class_counts_))

    def optimistic_estimate(self, coverage, target) -> float:
        counts = self._class_counts(coverage, target)
        if not counts.sum():
            return float("-inf")
        return float(self.estimate(counts, target.class_counts_))

    def _class_counts_many(self, statistics, target: MulticlassTarget) -> np.ndarray:
        self._check_target(target)
        columns = [statistics.count(mask) for mask in target.class_masks()]
        return np.stack(columns, axis=1)

    def evaluate_many(self, statistics, target) -> np.ndarray:
        counts = self._class_counts_many(statistics, target)
        with np.errstate(divide="ignore", invalid="ignore"):
            quality = self.quality(counts, target.class_counts_)
        return np.where(statistics.sizes == 0, np.nan, quality)

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        counts = self._class_counts_many(statistics, target)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimate = self._estimate(counts, target.class_counts_)
        return np.where(statistics.sizes == 0, -np.inf, estimate)

    def _estimate(self, counts_sg: np.ndarray, counts_dataset: np.ndarray):
        return self.estimate(counts_sg, counts_dataset)

    @staticmethod
    def quality(counts_sg: np.ndarray, counts_dataset: np.ndarray) -> float:
        """Quality of a subgroup with class counts ``counts_sg``."""
        raise NotImplementedError

    @staticmethod
    def estimate(counts_sg: np.ndarray, counts_dataset: np.ndarray) -> float:
        """Upper bound on the quality of every non-empty sub-multiset of ``counts_sg``."""
        raise NotImplementedError


def _linear_estimate(counts_sg, counts_dataset, weights) -> float:
    """``OE_Lin`` (Section 4.2): ``sum_i c_i max(n_i (1 - p0_i)^2, (n - n_i) p0_i^2)``.

    Per class, the best refinement keeps either only that class (``n_i`` rows)
    or all other classes (``n - n_i`` rows) (Appendix A.1).
    """
    priors = counts_dataset / counts_dataset.sum()
    size = counts_sg.sum(axis=-1, keepdims=True)
    terms = np.maximum(counts_sg * (1 - priors) ** 2, (size - counts_sg) * priors**2)
    return np.sum(weights * terms, axis=-1)


class SplitQF(MulticlassQF):
    """Split quality ``n * sum_i (p_i - p0_i) ** 2`` (Grosskreutz et al. 2008).

    The optimistic estimate is ``OE_Lin`` with ``c_i = 1`` (Section 4.2).
    """

    @staticmethod
    def quality(counts_sg, counts_dataset) -> float:
        size = counts_sg.sum(axis=-1, keepdims=True)
        priors = counts_dataset / counts_dataset.sum()
        return size[..., 0] * np.sum((counts_sg / size - priors) ** 2, axis=-1)

    @staticmethod
    def estimate(counts_sg, counts_dataset) -> float:
        return _linear_estimate(counts_sg, counts_dataset, 1.0)


class MulticlassChiSquaredQF(MulticlassQF):
    """Pearson's chi-squared ``n * sum_i (p_i - p0_i) ** 2 / p0_i`` (Grosskreutz et al. 2008).

    This is the goodness-of-fit statistic of the subgroup's class counts against
    the dataset distribution, not the independence test of subgroup and
    complement (``binary.ChiSquaredQF``), which is larger by ``N / (N - n)``.
    The optimistic estimate is ``OE_Lin`` with ``c_i = 1 / p0_i`` (Section 4.2).
    """

    @staticmethod
    def quality(counts_sg, counts_dataset) -> float:
        size = counts_sg.sum(axis=-1, keepdims=True)
        priors = counts_dataset / counts_dataset.sum()
        return size[..., 0] * np.sum((counts_sg / size - priors) ** 2 / priors, axis=-1)

    @staticmethod
    def estimate(counts_sg, counts_dataset) -> float:
        priors = counts_dataset / counts_dataset.sum()
        return _linear_estimate(counts_sg, counts_dataset, 1 / priors)


def _odds(size, size_dataset: int):
    """``size / (N - size)``, and 0 where ``size == N``."""
    size = np.asarray(size, dtype=float)
    complement = size_dataset - size
    return np.divide(size, complement, out=np.zeros_like(size), where=complement > 0)


class GiniQF(MulticlassQF):
    """Gini quality ``n / (N - n) * sum_i (p_i - p0_i) ** 2`` (Grosskreutz et al. 2008).

    The optimistic estimate is ``OE_Gini = sum_i max(E_i-, E_i^g, E_i+)``
    (Section 4.3, Appendix A.2) with ``E_i+ = n_i / (N - n_i) * (1 - p0_i)^2``
    (keep only class ``i``), ``E_i- = (n - n_i) / (N - n + n_i) * p0_i^2`` (drop
    class ``i``), and ``E_i^g = n / (N - n) * (p_i - p0_i)^2`` (keep everything).
    """

    @staticmethod
    def quality(counts_sg, counts_dataset) -> float:
        size, size_dataset = counts_sg.sum(axis=-1, keepdims=True), counts_dataset.sum()
        deviation = np.sum((counts_sg / size - counts_dataset / size_dataset) ** 2, axis=-1)
        return _odds(size[..., 0], size_dataset) * deviation

    @staticmethod
    def estimate(counts_sg, counts_dataset) -> float:
        size, size_dataset = counts_sg.sum(axis=-1, keepdims=True), counts_dataset.sum()
        priors = counts_dataset / size_dataset
        keep = _odds(counts_sg, size_dataset) * (1 - priors) ** 2
        drop = _odds(size - counts_sg, size_dataset) * priors**2
        unchanged = _odds(size, size_dataset) * (counts_sg / size - priors) ** 2
        return np.sum(np.maximum.reduce([keep, drop, unchanged]), axis=-1)


class ClassDistributionL1QF(MulticlassQF):
    """``q_dst = n / N * sum_i |p_i - p0_i|`` of class indicators (Grosskreutz et al. 2010).

    Optimistic estimate (derived here; the paper gives none). Let ``S`` be the
    classes with ``p'_i > p0_i`` in a refinement of size ``n'``. Since both
    distributions sum to 1, ``n' * sum_i |p'_i - p0_i| = 2 n' (p'_S - p0_S)``, and
    as for Piatetsky-Shapiro (Section 4.1) ``n' (p'_S - p0_S) <= n_S (1 - p0_S)``,
    where ``n_S`` counts the subgroup's rows in ``S``. Hence every refinement
    scores at most ``2 / N * max_S n_S (1 - p0_S)``, which the refinement keeping
    exactly the rows in the maximizing ``S`` attains, so the bound is tight.

    The maximum enumerates the ``2^k - 1`` class sets for ``k <= max_exact_classes``.
    Beyond that it uses the weaker per-class bound
    ``1 / N * sum_i max(n_i (1 - p0_i), (n - n_i) p0_i)``, capped by
    ``2 / N * (n - sum_i n_i p0_i)``, which never binds.
    """

    def __init__(self, max_exact_classes: int = 12):
        if isinstance(max_exact_classes, bool) or not isinstance(max_exact_classes, int):
            raise ValueError("max_exact_classes must be an integer")
        self.max_exact_classes = max_exact_classes

    @staticmethod
    def quality(counts_sg, counts_dataset) -> float:
        size, size_dataset = counts_sg.sum(axis=-1, keepdims=True), counts_dataset.sum()
        deviation = np.sum(np.abs(counts_sg / size - counts_dataset / size_dataset), axis=-1)
        return size[..., 0] / size_dataset * deviation

    def optimistic_estimate(self, coverage, target) -> float:
        counts = self._class_counts(coverage, target)
        if not counts.sum():
            return float("-inf")
        return float(self.estimate(counts, target.class_counts_, self.max_exact_classes))

    def _estimate(self, counts_sg, counts_dataset):
        return self.estimate(counts_sg, counts_dataset, self.max_exact_classes)

    @staticmethod
    def estimate(counts_sg, counts_dataset, max_exact_classes: int = 12) -> float:
        size_dataset = counts_dataset.sum()
        priors = counts_dataset / size_dataset
        n_classes = counts_sg.shape[-1]
        if n_classes <= max_exact_classes:
            sets = _class_sets(n_classes)
            # Class counts are integers, so the set sums are exact in either order.
            set_counts = sets @ counts_sg if counts_sg.ndim == 1 else counts_sg @ sets.T
            return 2 / size_dataset * np.max(set_counts * (1 - sets @ priors), axis=-1)
        size = counts_sg.sum(axis=-1, keepdims=True)
        per_class = np.sum(
            np.maximum(counts_sg * (1 - priors), (size - counts_sg) * priors), axis=-1
        )
        return np.minimum(per_class, 2 * (size[..., 0] - counts_sg @ priors)) / size_dataset


class MaxConfidenceQF(MulticlassQF):
    """Confidence ``max_i n_i / n`` of the rule from the subgroup to its majority class.

    This is the confidence of the "possible rule" of a condition set in CBA
    (Liu, Hsu and Ma, KDD 1998). Every non-empty subgroup has a one-row
    refinement with confidence 1, so the optimistic estimate is 1 (tight).
    """

    @staticmethod
    def quality(counts_sg, counts_dataset) -> float:
        return counts_sg.max(axis=-1) / counts_sg.sum(axis=-1)

    @staticmethod
    def estimate(counts_sg, counts_dataset) -> float:
        return np.ones(counts_sg.shape[:-1])
