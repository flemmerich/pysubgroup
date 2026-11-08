"""Association model: the association between two nominal targets.

From W. Duivesteijn, A. Feelders, and A. Knobbe, "Exceptional Model Mining",
Data Mining and Knowledge Discovery 30 (2016), Sect. 5.2. ``phi_dev``
(``AssociationDevianceQF``) is the deviance

    phi_dev(D) = 2 sum_cells observed * log(observed / fitted)

of the homogeneous association log-linear model on the three-way table of the
targets ``x``, ``y`` and the subgroup indicator ``D`` (all two-way interactions,
no three-way interaction) against the saturated model; it is large when the
association between ``x`` and ``y`` differs between subgroup and complement.
The maximum-likelihood fit has no closed form; it is computed by iterative
proportional fitting of the margins ``(x, y)``, ``(x, D)``, and ``(y, D)``.

Tables are built from the counts of every ``(x, y)`` cell, so the measure is
batched (the fit iterates on all children of a node at once). Written from
the paper without reusing code. Deviations: missing values count as a value of
their own (as in ``NominalTarget``); the empty subgroup and complement score
``nan``. The papers suggest a minimum fitted count of 5 per cell for the
chi-squared approximation, which is not enforced.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

from ..targets.base import QualityFunction
from ..targets.nominal import NominalTarget


class AssociationTarget(NominalTarget):
    """Two nominal model attributes ``x`` and ``y`` (``fit(data, x, y)``)."""

    def fit(self, data: pd.DataFrame, *attributes: Hashable) -> AssociationTarget:
        if len(attributes) != 2:
            raise ValueError(f"AssociationTarget requires 2 attributes, got {len(attributes)}")
        super().fit(data, *attributes)
        shape = tuple(len(values) for values in self.values_)
        self.cells_ = np.ravel_multi_index((self.codes_[:, 0], self.codes_[:, 1]), shape)
        self.table_ = np.bincount(self.cells_, minlength=shape[0] * shape[1]).reshape(shape)
        return self

    def cell_masks(self) -> list[np.ndarray]:
        """Row mask of every ``(x, y)`` cell (row-major), cached for batches."""
        masks = self.__dict__.get("_cell_masks")
        if masks is None:
            masks = self.__dict__["_cell_masks"] = [
                self.cells_ == cell for cell in range(self.table_.size)
            ]
        return masks


def homogeneous_association(table: np.ndarray, max_iter: int = 1000, tol: float = 1e-10):
    """Fitted counts of the homogeneous association model for tables (... x dx x dy x 2)."""
    fitted = np.ones_like(table, dtype=float)
    margins = [(-1,), (-2,), (-3,)]
    observed = [table.sum(axis=axis, keepdims=True) for axis in margins]
    for _ in range(max_iter):
        for axis, target in zip(margins, observed, strict=True):
            current = fitted.sum(axis=axis, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                fitted = fitted * np.where(current > 0, target / current, 0.0)
        # After the last scaling the (y, D) margins fit; check the other two.
        gap = max(
            np.max(np.abs(fitted.sum(axis=axis, keepdims=True) - target), initial=0.0)
            for axis, target in zip(margins[:2], observed[:2], strict=True)
        )
        if gap <= tol * max(1.0, float(table.sum(axis=(-1, -2, -3)).max(initial=0.0))):
            break
    return fitted


def deviance(table: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """``2 sum observed log(observed / fitted)`` over the last three axes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(table > 0, table * np.log(table / fitted), 0.0)
    return 2.0 * terms.sum(axis=(-1, -2, -3))


class AssociationDevianceQF(QualityFunction):
    """``phi_dev``: deviance of the homogeneous association model (DMKD Sect. 5.2)."""

    target_type = AssociationTarget
    batchable = True

    def __init__(self, max_iter: int = 1000, tol: float = 1e-10):
        self.max_iter = max_iter
        self.tol = tol

    def _scores(self, subgroup_tables: np.ndarray, target: AssociationTarget) -> np.ndarray:
        tables = np.stack([subgroup_tables, target.table_ - subgroup_tables], axis=-1)
        fitted = homogeneous_association(tables.astype(float), self.max_iter, self.tol)
        sizes = subgroup_tables.sum(axis=(-1, -2))
        quality = np.maximum(deviance(tables, fitted), 0.0)
        return np.where((sizes > 0) & (sizes < target.n_samples_), quality, np.nan)

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        counts = np.bincount(target.cells_[mask], minlength=target.table_.size)
        return float(self._scores(counts.reshape(target.table_.shape)[None], target)[0])

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        counts = np.stack([statistics.count(mask) for mask in target.cell_masks()], axis=-1)
        return self._scores(counts.reshape(-1, *target.table_.shape), target)
