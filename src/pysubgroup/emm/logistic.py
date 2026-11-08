"""Logistic regression model: a numeric predictor of a binary label.

From W. Duivesteijn, A. Feelders, and A. Knobbe, "Exceptional Model Mining",
Data Mining and Knowledge Discovery 30 (2016), Sect. 5.4 (first in Leman,
Feelders, and Knobbe, ECML PKDD 2008, Sect. 3.3). ``phi_sed``
(``LogisticInteractionQF``) fits

    logit P(y = 1 | x) = b0 + b1 D + b2 x + b3 D x

on the whole dataset, with ``D`` the subgroup indicator, so that intercept and
slope may both differ between subgroup and complement, and scores one minus the
two-sided p-value of the test of ``b3 = 0``. The papers call it "a standard
test"; this module uses the Wald test, ``z = b3 / se(b3)`` with the standard
error from the inverse Fisher information at the maximum-likelihood estimate.

The fit is Newton-Raphson (iteratively reweighted least squares) with step
halving, on ``x`` standardized by its dataset mean and deviation, which leaves
``z`` unchanged. It is iterative, so the measure evaluates one subgroup at a
time. Written from the papers without reusing code. Deviations: groups in which
the label or ``x`` is constant (in the subgroup or the complement), and fits
that do not converge (such as perfectly separated groups, where the estimate
does not exist), score ``nan``.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy.special import expit

from ..encoding import equals
from ..targets.base import BaseTarget, QualityFunction
from .base import check_score, normal_score


class LogisticRegressionTarget(BaseTarget):
    """A numeric predictor ``x`` and a binary label ``y == value`` (``fit(data, x, y, value)``)."""

    def fit(
        self, data: pd.DataFrame, x: Hashable, y: Hashable, value=True
    ) -> LogisticRegressionTarget:
        """Store ``x`` (``values_``, finite) and the labels ``labels_ = data[y] == value``.

        A missing ``value`` matches missing entries; otherwise missing entries
        are negatives, as for ``BinaryTarget``.
        """
        super().fit(data, x, y)
        dtype = data[x].dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            raise TypeError("the predictor must be a numeric column")
        values = data[x].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError("the predictor must contain only finite, non-missing values")
        column = data[y]
        matches = column.isna() if pd.isna(value) else equals(column, value)
        self.value_ = value
        self.values_ = values
        self.labels_ = matches.fillna(False).to_numpy(dtype=bool)
        deviation = values.std()
        if deviation == 0:
            raise ValueError("the predictor must not be constant on the dataset")
        self.standardized_ = (values - values.mean()) / deviation
        return self

    def _fit_arguments(self) -> tuple:
        return (*self.attributes_, self.value_)


def fit_logistic(design: np.ndarray, labels: np.ndarray, max_iter: int = 50, tol: float = 1e-10):
    """Maximum-likelihood coefficients and their covariance, or ``None`` without convergence.

    Newton-Raphson on the log-likelihood with step halving; converged once
    the largest coefficient change is below ``tol * (1 + |coefficient|)``.
    """
    y = labels.astype(float)

    def log_likelihood(beta):
        eta = design @ beta
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    beta = np.zeros(design.shape[1])
    current = log_likelihood(beta)
    for _ in range(max_iter):
        p = expit(design @ beta)
        information = design.T @ (design * (p * (1 - p))[:, None])
        try:
            step = np.linalg.solve(information, design.T @ (y - p))
        except np.linalg.LinAlgError:
            return None
        for _ in range(30):
            candidate = beta + step
            value = log_likelihood(candidate)
            if value >= current - 1e-12 * abs(current):
                break
            step = step / 2
        else:
            return None
        beta, current = candidate, value
        if np.all(np.abs(step) <= tol * (1 + np.abs(beta))):
            p = expit(design @ beta)
            information = design.T @ (design * (p * (1 - p))[:, None])
            try:
                return beta, np.linalg.inv(information)
            except np.linalg.LinAlgError:
                return None
    return None


class LogisticInteractionQF(QualityFunction):
    """``phi_sed``: Wald test of the interaction ``b3`` of subgroup and predictor (DMKD Sect. 5.4).

    ``score="1-p"`` (default) gives one minus the two-sided p-value,
    ``score="statistic"`` the order-equivalent ``|z|``.
    """

    target_type = LogisticRegressionTarget

    def __init__(self, score: str = "1-p", max_iter: int = 50):
        self.score = check_score(score)
        self.max_iter = max_iter

    def coefficients(self, coverage, target):
        """``(b0, b1, b2, b3)`` for the standardized predictor and their covariance, or ``None``."""
        mask = self._mask(coverage, target)
        labels, x = target.labels_, target.standardized_
        for group in (mask, ~mask):
            if not group.any() or labels[group].all() or not labels[group].any():
                return None
            if np.ptp(x[group]) == 0:
                return None
        indicator = mask.astype(float)
        design = np.column_stack([np.ones_like(x), indicator, x, indicator * x])
        return fit_logistic(design, labels, self.max_iter)

    def evaluate(self, coverage, target) -> float:
        fitted = self.coefficients(coverage, target)
        if fitted is None:
            return float("nan")
        beta, covariance = fitted
        z = beta[3] / np.sqrt(covariance[3, 3])
        return float(normal_score(np.array(z), self.score))
