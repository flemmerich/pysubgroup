"""Decision Table Majority (DTM) classifier: nominal inputs and a nominal output.

From W. Duivesteijn, A. Feelders, and A. Knobbe, "Exceptional Model Mining",
Data Mining and Knowledge Discovery 30 (2016), Sect. 5.4.1 (first in Leman,
Feelders, and Knobbe, ECML PKDD 2008, Sect. 3.3). A DTM classifier estimates
``P(y | x_1, ..., x_k)`` by the relative frequencies of ``y`` in each
combination (cell) of input values; cells without rows get the distribution of
``y`` on the whole dataset. With ``I`` input combinations and ``J`` output
values:

- ``phi_Hel`` (``HellingerQF``): ``sum_cells sum_y (sqrt P_G(y | cell) -
  sqrt P_C(y | cell))^2`` between the estimates on the subgroup and its
  complement; with ``entropy=True`` it is weighted with the entropy of the
  split, ``phi_ef * phi_Hel``;
- ``phi_BDeu`` (``BDeuQF``): the BDeu score of the classifier on the subgroup
  alone, ``prod_cells Gamma(a / I) / Gamma(a / I + n_cell) prod_y
  Gamma(a / (I J) + n_cell,y) / Gamma(a / (I J))`` with the equivalent sample
  size ``a``, here as its natural logarithm.

Both are computed from counts per (cell, output value), so they are batched.
``I`` is the product of the inputs' numbers of values; cells that occur in no
row add nothing to either measure and are skipped. Written from the papers
without reusing code. Deviations: missing values count as a value of their own
(as in ``NominalTarget``); the empty subgroup (and for ``phi_Hel`` with
``reference="complement"`` the empty complement) scores ``nan``. The logarithm
agrees with the paper's reported ``phi_BDeu = -1.075`` (DMKD Sect. 6.7.2) for
``a = 1``, the default. The entropy weight of ``phi_Hel`` is an extension in
the spirit of DMKD Sect. 3.2.1, which recommends a size term in quality
measures but defines ``phi_Hel`` without one (Leman et al. use a minimum
support of 25% instead).
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy.special import gammaln

from ..targets.base import QualityFunction
from ..targets.nominal import NominalTarget
from .base import check_reference, entropy_weight


class DecisionTableTarget(NominalTarget):
    """Nominal inputs and an output (``fit(data, *inputs, output)``) for a DTM classifier.

    ``fit`` also stores ``cells_`` (the index of every row's input combination
    among the observed ones), ``n_combinations_`` (``I``, the product of the
    inputs' numbers of values), ``labels_`` (output value codes),
    ``n_labels_`` (``J``), ``table_`` (counts per observed cell and output
    value), and ``prior_`` (the output distribution on the whole dataset).
    """

    def fit(self, data: pd.DataFrame, *attributes: Hashable) -> DecisionTableTarget:
        if len(attributes) < 2:
            raise ValueError("decision tables require at least one input and an output")
        super().fit(data, *attributes)
        inputs = self.codes_[:, :-1]
        combinations, self.cells_ = np.unique(inputs, axis=0, return_inverse=True)
        self.cells_ = self.cells_.reshape(-1)
        self.combinations_ = combinations
        self.n_combinations_ = int(np.prod([len(values) for values in self.values_[:-1]]))
        self.labels_ = self.codes_[:, -1]
        self.n_labels_ = len(self.values_[-1])
        self.table_ = self._table(np.ones(self.n_samples_, dtype=bool))
        self.prior_ = self.table_.sum(axis=0) / self.n_samples_
        return self

    def _table(self, mask: np.ndarray) -> np.ndarray:
        index = self.cells_[mask] * self.n_labels_ + self.labels_[mask]
        size = len(self.combinations_) * self.n_labels_
        return np.bincount(index, minlength=size).reshape(len(self.combinations_), self.n_labels_)

    def tables(self, mask: np.ndarray) -> np.ndarray:
        """Counts per observed cell and output value in the rows ``mask`` (1 x cells x J)."""
        return self._table(mask)[None]

    def tables_many(self, statistics) -> np.ndarray:
        """Counts per observed cell and output value of every child (children x cells x J)."""
        masks = self.__dict__.get("_cell_masks")
        if masks is None:
            codes = self.cells_ * self.n_labels_ + self.labels_
            masks = self.__dict__["_cell_masks"] = [
                codes == code for code in range(self.table_.size)
            ]
        counts = np.stack([statistics.count(mask) for mask in masks], axis=-1)
        return counts.reshape(-1, *self.table_.shape)


def conditional_estimates(tables: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """``P(y | cell)`` per table, with the dataset distribution for cells without rows."""
    totals = tables.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(totals > 0, tables / totals, prior)


class _DecisionTableQF(QualityFunction):
    target_type = DecisionTableTarget
    batchable = True

    def _scores(self, tables: np.ndarray, target: DecisionTableTarget) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        return float(self._scores(target.tables(mask), target)[0])

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        return self._scores(target.tables_many(statistics), target)


class HellingerQF(_DecisionTableQF):
    """``phi_Hel``: Hellinger distance of the DTM estimates (DMKD Sect. 5.4.1).

    ``reference="complement"`` (default, as in the papers) compares with the
    complement, ``"dataset"`` with the whole dataset (DMKD Sect. 3.2.2).
    ``entropy=True`` weights the distance with ``phi_ef``, the entropy of the
    split (DMKD Sect. 3.2.1), so that small subgroups score lower; this weight
    is not in the papers.
    """

    def __init__(self, reference: str = "complement", entropy: bool = False):
        self.reference = check_reference(reference)
        self.entropy = bool(entropy)

    def _scores(self, tables, target) -> np.ndarray:
        reference = target.table_ - tables if self.reference == "complement" else target.table_
        estimates = conditional_estimates(tables, target.prior_)
        reference_estimates = conditional_estimates(reference, target.prior_)
        distance = np.square(np.sqrt(estimates) - np.sqrt(reference_estimates)).sum(axis=(-1, -2))
        sizes = tables.sum(axis=(-1, -2))
        valid = sizes > 0
        if self.reference == "complement":
            valid &= sizes < target.n_samples_
        else:
            distance = np.where(sizes == target.n_samples_, 0.0, distance)
        if self.entropy:
            distance = entropy_weight(sizes, target.n_samples_) * distance
        return np.where(valid, distance, np.nan)


class BDeuQF(_DecisionTableQF):
    """``phi_BDeu``: natural log of the BDeu score of the DTM on the subgroup (DMKD Sect. 5.4.1).

    Measures how predictable the output is from the inputs in the subgroup,
    ignoring the complement; ``equivalent_sample_size`` is the papers' ``theta``.
    """

    def __init__(self, equivalent_sample_size: float = 1.0):
        if not equivalent_sample_size > 0:
            raise ValueError("equivalent_sample_size must be positive")
        self.equivalent_sample_size = float(equivalent_sample_size)

    def _scores(self, tables, target) -> np.ndarray:
        cell_prior = self.equivalent_sample_size / target.n_combinations_
        value_prior = cell_prior / target.n_labels_
        totals = tables.sum(axis=-1)
        score = (gammaln(cell_prior) - gammaln(cell_prior + totals)).sum(axis=-1) + (
            gammaln(value_prior + tables) - gammaln(value_prior)
        ).sum(axis=(-1, -2))
        return np.where(totals.sum(axis=-1) > 0, score, np.nan)
