"""Nominal model attributes (one or more discrete columns) and the weighted KL quality.

``NominalTarget`` holds several discrete model attributes, as in exceptional
model mining (EMM). ``WeightedKLQF`` is the Weighted Kullback-Leibler quality
(Definition 4) of M. van Leeuwen and A. Knobbe, "Diverse subgroup set
discovery", Data Mining and Knowledge Discovery 25 (2012) 208-242:

    phi_WKL(G) = |G| / |S| * sum_i KL(P_G(M_i) || P_S(M_i)),

with base-2 logarithms and the empirical distributions ``P_G`` and ``P_S`` of
every model attribute ``M_i`` in the subgroup and in the dataset, treating the
attributes as independent. Its models also drive the compression-based subgroup
selection of ``dssd.DSSD``. Written from the paper without reusing code.

``discrete_codes`` gives the model attributes of every target with discrete
model data: a ``BinaryTarget`` (one binary attribute), a ``MulticlassTarget``
(one nominal attribute), and a ``NominalTarget``.

Deviations from the paper: missing values of a model attribute count as a value
of their own, and empty subgroups score ``nan``.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

from .base import BaseTarget, QualityFunction
from .binary import BinaryTarget
from .multiclass import MulticlassTarget


class NominalTarget(BaseTarget):
    """Value distributions of one or more discrete model attributes."""

    def fit(self, data: pd.DataFrame, *attributes: Hashable) -> NominalTarget:
        """Store the value of every row in each column ``attributes``.

        ``values_`` lists each attribute's observed values (sorted where
        possible, missing values last), ``codes_`` (rows x attributes) the value
        index of every row, and ``value_counts_`` each attribute's value counts.
        """
        if not attributes:
            raise ValueError("nominal targets require at least one attribute")
        super().fit(data, *attributes)
        codes, values = [], []
        for name in attributes:
            try:
                column_codes, uniques = pd.factorize(data[name], sort=True, use_na_sentinel=False)
            except TypeError:
                column_codes, uniques = pd.factorize(data[name], use_na_sentinel=False)
            codes.append(column_codes.astype(np.intp))
            values.append(np.asarray(uniques, dtype=object))
        self.values_ = values
        self.codes_ = np.stack(codes, axis=1)
        self.value_counts_ = [
            np.bincount(self.codes_[:, i], minlength=len(v)) for i, v in enumerate(values)
        ]
        return self


def discrete_codes(target: BaseTarget) -> list[tuple[np.ndarray, int]]:
    """The model attributes of ``target`` as ``(codes per row, number of values)`` pairs."""
    if not hasattr(target, "n_samples_"):
        raise ValueError("target must be fitted before computing qualities")
    if isinstance(target, NominalTarget):
        return [
            (target.codes_[:, i], len(values)) for i, values in enumerate(target.values_)
        ]
    if isinstance(target, MulticlassTarget):
        return [(target.codes_, len(target.classes_))]
    if isinstance(target, BinaryTarget):
        return [(target.positives_.astype(np.intp), 2)]
    raise TypeError(
        "discrete model attributes require a BinaryTarget, MulticlassTarget, or "
        f"NominalTarget, got {type(target).__name__}"
    )


def value_masks(target: BaseTarget) -> list[list[np.ndarray]]:
    """Boolean row mask of every value of every model attribute, cached for batches.

    The same mask objects are returned until the target is fitted again, so
    batched backends and GP-growth's recorded requests recognize them.
    """
    masks = target.__dict__.get("_value_masks")
    if masks is None:
        masks = target.__dict__["_value_masks"] = [
            [codes == v for v in range(n)] for codes, n in discrete_codes(target)
        ]
    return masks


def _kl_terms(counts: np.ndarray, sizes: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """``sum_x p(x) log2(p(x) / q(x))`` per row of ``counts`` (subgroups x values)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = counts / sizes[..., None]
        terms = np.where(counts > 0, shares * np.log2(shares / priors), 0.0)
    return terms.sum(axis=-1)


class WeightedKLQF(QualityFunction):
    """Weighted Kullback-Leibler quality of van Leeuwen and Knobbe (2012, Def. 4).

    Accepts every target with discrete model attributes (see ``discrete_codes``).
    """

    batchable = True

    def _check_target(self, target: BaseTarget) -> None:
        discrete_codes(target)

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        size = int(np.count_nonzero(mask))
        if not size:
            return float("nan")
        total = 0.0
        for codes, n_values in discrete_codes(target):
            counts = np.bincount(codes[mask], minlength=n_values)
            priors = np.bincount(codes, minlength=n_values) / target.n_samples_
            total += float(_kl_terms(counts, np.array(size), priors))
        return size / target.n_samples_ * total

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        sizes = statistics.sizes
        total = np.zeros(len(sizes))
        for (codes, n_values), masks in zip(
            discrete_codes(target), value_masks(target), strict=True
        ):
            counts = np.stack([statistics.count(mask) for mask in masks], axis=1)
            priors = np.bincount(codes, minlength=n_values) / target.n_samples_
            total += _kl_terms(counts, sizes, priors)
        with np.errstate(invalid="ignore"):
            return np.where(sizes == 0, np.nan, sizes / target.n_samples_ * total)
