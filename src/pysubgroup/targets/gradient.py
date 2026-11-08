"""Gradient target: the per-rule objective of second-order gradient boosting.

Mario Boley, Simon Teshuva, Pierre Le Bodic and Geoffrey I. Webb: Better Short
than Greedy: Interpretable Models through Optimal Rule Boosting. Proceedings of
the 2021 SIAM International Conference on Data Mining (SDM), 2021
(arXiv:2101.08380). Implemented from the paper (Eq. 2.3 and Theorem 3.1); no
code from its realkd implementation is reused.

A ``GradientTarget`` stores the first and second derivatives g_i and h_i of a
loss at the current predictions of a boosted model. ``GradientBoostingQF``
scores a subgroup I by the loss reduction of adding the rule "if I then w" with
the optimal weight w = -G / (lam + H) (Eq. 2.2):

    obj(I) = G^2 / (2 n (lam + H)),   G = sum_{i in I} g_i,  H = sum_{i in I} h_i  (Eq. 2.3).

Its optimistic estimate is the tight bound bnd(I) = max{obj(J) : J subset of I}
of Theorem 3.1: with the rows sorted by decreasing g_i / h_i, an optimal J is a
prefix or a suffix of I in that order, so the bound costs O(|I|) after sorting
once.

Additions: rows with h_i = 0 sort by the sign of g_i (ratio +-inf, or 0 if g_i
= 0 as well). If lam + H = 0, obj is 0. The theorem assumes h_i > 0; with h_i =
0 it still holds by continuity when lam > 0.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

from .base import BaseTarget, BoundedQualityFunction

_CHUNK_CELLS = 1 << 22  # subgroup x row cells per cumulative-sum block


def gradient_order(gradients: np.ndarray, hessians: np.ndarray) -> np.ndarray:
    """Row order by decreasing g_i / h_i (Theorem 3.1); ties keep the row order."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(hessians > 0, gradients / hessians, gradients * np.inf)
    ratios[(hessians <= 0) & (gradients == 0)] = 0.0
    return np.argsort(-ratios, kind="stable")


def boosting_objective(
    gradient_sums, hessian_sums, reg_lambda: float, n_samples: int
) -> np.ndarray:
    """obj = G^2 / (2 n (lam + H)) (Eq. 2.3), 0 where lam + H is 0."""
    gradient_sums = np.asarray(gradient_sums, dtype=float)
    denominator = 2.0 * n_samples * (reg_lambda + np.asarray(hessian_sums, dtype=float))
    return np.divide(
        gradient_sums**2, denominator,
        out=np.zeros(np.broadcast(gradient_sums, denominator).shape), where=denominator > 0,
    )


def objective_bounds(
    masks: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, reg_lambda: float
) -> np.ndarray:
    """bnd(I) of Theorem 3.1 for each row mask in ``masks`` (subgroups x rows).

    The columns of ``masks`` and the arrays must already be in ``gradient_order``.
    The best prefix and the best suffix are taken over every position, which
    includes the empty set (objective 0).
    """
    masks = np.atleast_2d(masks)
    n_samples = masks.shape[1]
    bounds = np.zeros(len(masks))
    step = max(1, _CHUNK_CELLS // max(1, n_samples))
    for start in range(0, len(masks), step):
        block = masks[start:start + step]
        g = np.where(block, gradients, 0.0)
        h = np.where(block, hessians, 0.0)
        prefix = boosting_objective(
            np.cumsum(g, axis=1), np.cumsum(h, axis=1), reg_lambda, n_samples
        )
        # Reversed cumulative sums keep suffix sums as accurate as prefix sums.
        suffix = boosting_objective(
            np.cumsum(g[:, ::-1], axis=1), np.cumsum(h[:, ::-1], axis=1), reg_lambda, n_samples
        )
        if n_samples:
            bounds[start:start + step] = np.maximum(prefix.max(axis=1), suffix.max(axis=1))
    return bounds


class GradientTarget(BaseTarget):
    """Gradients g_i and hessians h_i >= 0 of a loss at the current predictions."""

    def fit(self, data: pd.DataFrame, gradient: Hashable, hessian: Hashable) -> GradientTarget:
        """Store the numeric columns ``gradient`` and ``hessian``."""
        super().fit(data, gradient, hessian)
        arrays = []
        for name in (gradient, hessian):
            column = data[name]
            if not pd.api.types.is_numeric_dtype(column.dtype) or pd.api.types.is_bool_dtype(
                column.dtype
            ):
                raise TypeError("gradient targets require numeric columns")
            values = column.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(values).all():
                raise ValueError("gradients and hessians must be finite and non-missing")
            arrays.append(values)
        if (arrays[1] < 0).any():
            raise ValueError("hessians must be non-negative")
        self.gradients_, self.hessians_ = arrays
        self.order_ = gradient_order(*arrays)
        self._sorted = (arrays[0][self.order_], arrays[1][self.order_])
        return self


class GradientBoostingQF(BoundedQualityFunction):
    """Boosting objective obj(I) = G^2 / (2 n (lam + H)) with the tight bound of Theorem 3.1.

    Parameters
    ----------
    reg_lambda : float
        Regularization lam >= 0 on the squared rule weight.
    """

    target_type = GradientTarget

    def __init__(self, reg_lambda: float = 1.0):
        if isinstance(reg_lambda, bool) or not isinstance(reg_lambda, int | float) or not (
            np.isfinite(reg_lambda) and reg_lambda >= 0
        ):
            raise ValueError("reg_lambda must be a finite non-negative number")
        self.reg_lambda = float(reg_lambda)

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        return float(boosting_objective(
            target.gradients_[mask].sum(), target.hessians_[mask].sum(),
            self.reg_lambda, target.n_samples_,
        ))

    def optimistic_estimate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        if not mask.any():
            return float("-inf")
        return float(objective_bounds(mask[target.order_], *target._sorted, self.reg_lambda)[0])
