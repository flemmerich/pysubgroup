"""Linear regression models: simple regression and general linear regression.

From W. Duivesteijn, A. Feelders, and A. Knobbe, "Exceptional Model Mining",
Data Mining and Knowledge Discovery 30 (2016):

- Sect. 5.3 (first in Leman, Feelders, and Knobbe, ECML PKDD 2008, Sect. 3.2),
  ``phi_ssd`` (``SlopeSignificanceQF``): for the least-squares slopes ``b_G``
  and ``b_C`` of ``y = b0 + b1 x`` in the subgroup and its complement, with
  ``s^2 = sum e^2 / ((n - 2) sum (x - mean x)^2)``, one minus the two-sided
  p-value of ``t' = (b_G - b_C) / sqrt(s_G^2 + s_C^2)`` under a t distribution
  with ``df = (s_G^2 + s_C^2)^2 / (s_G^4 / (n - 2) + s_C^4 / (n_C - 2))``.
- Sect. 5.6, ``phi_Cook`` (``CooksDistanceQF``, Definition 6): Cook's distance
  between the coefficients ``beta_G`` fitted on the subgroup and ``beta``
  fitted on the whole dataset, for the model ``Y = X beta + e`` with an
  intercept, ``(beta_G - beta)^T Z^T (Z (X^T X)^-1 Z^T)^-1 Z (beta_G - beta) /
  (m' s^2)``, where ``X`` and ``s^2 = e^T e / (N - m)`` belong to the dataset
  fit and ``Z`` selects ``m'`` of the ``m`` coefficients.

All quantities follow from the moments of ``base.NumericModelTarget``, so both
measures are batched. Written from the papers without reusing code.
Deviations: the empty subgroup and complement, groups with a constant
predictor, and groups too small for the fit (at most two rows for ``phi_ssd``,
fewer rows than coefficients or collinear predictors for ``phi_Cook``) score
``nan``; the whole dataset has ``phi_Cook = 0``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from scipy import special

from .base import MomentQF, NumericModelTarget, check_score, two_sided_score

INTERCEPT = "intercept"
# Groups whose predictors' correlation matrix has a smaller determinant count as collinear.
_MIN_DETERMINANT = 1e-10


class RegressionTarget(NumericModelTarget):
    """A predictor ``x`` and a response ``y`` (``fit(data, x, y)``) for simple linear regression."""

    n_model_attributes = 2


def simple_regression(target: NumericModelTarget, sizes, sums, products):
    """Slopes, intercepts, and slope variances ``s^2`` of ``y = b0 + b1 x`` per group."""
    scatter = target.scatter(sizes, sums, products)
    sxx, sxy, syy = scatter[..., 0, 0], scatter[..., 0, 1], scatter[..., 1, 1]
    valid = ~target.constant(sizes, scatter)[..., 0] & (np.asarray(sizes) > 2)
    means = target.means(sizes, sums)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = sxy / sxx
        residuals = np.maximum(syy - sxy * slope, 0.0)
        variance = residuals / ((sizes - 2) * sxx)
        intercept = means[..., 1] - slope * means[..., 0]
    return (
        np.where(valid, slope, np.nan),
        np.where(valid, intercept, np.nan),
        np.where(valid, variance, np.nan),
    )


class SlopeSignificanceQF(MomentQF):
    """``phi_ssd``: significance of the slope difference to the complement (DMKD Sect. 5.3).

    ``score="1-p"`` (default) gives the papers' one minus the two-sided p-value,
    ``score="statistic"`` gives ``|t'|`` (order-equivalent only for equal
    degrees of freedom). The papers consider the approximation accurate for
    ``n + n_C >= 40``.
    """

    target_type = RegressionTarget

    def __init__(self, score: str = "1-p"):
        self.score = check_score(score)

    def _scores(self, sizes, sums, products, target) -> np.ndarray:
        complement = target.complement(sizes, sums, products)
        slope, _, variance = simple_regression(target, sizes, sums, products)
        slope_c, _, variance_c = simple_regression(target, *complement)
        spread = variance + variance_c
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (slope - slope_c) / np.sqrt(spread)
            df = np.square(spread) / (
                np.square(variance) / (sizes - 2) + np.square(variance_c) / (complement[0] - 2)
            )
        # Exact fits in both groups leave df undefined; an infinite |t'| has p = 0 anyway.
        df = np.where(np.isfinite(df), df, 1.0)
        with np.errstate(invalid="ignore"):
            p_value = 2.0 * special.stdtr(df, -np.abs(t))
        return two_sided_score(t, p_value, self.score)


class GeneralRegressionTarget(NumericModelTarget):
    """Predictors and a response (``fit(data, *predictors, response)``) for linear regression.

    ``fit`` also fits ``Y = X beta + e`` with an intercept on the whole dataset:
    ``coefficient_names_`` (``"intercept"`` first, then the predictors),
    ``coefficients_``, ``residual_variance_`` (``s^2 = e^T e / (N - m)``), and
    ``inverse_gram_`` (``(X^T X)^-1``).
    """

    def fit(self, data: pd.DataFrame, *attributes: Hashable) -> GeneralRegressionTarget:
        if len(attributes) < 2:
            raise ValueError("general regression requires at least one predictor and a response")
        if INTERCEPT in attributes[:-1]:
            raise ValueError(f"a predictor must not be named {INTERCEPT!r}")
        super().fit(data, *attributes)
        n_coefficients = len(attributes)
        if self.n_samples_ <= n_coefficients:
            raise ValueError("general regression requires more rows than coefficients")
        coefficients, valid, residuals = linear_regression(self, *self.totals_)
        if not valid:
            raise ValueError("the predictors are collinear on the dataset")
        self.coefficient_names_ = (INTERCEPT, *attributes[:-1])
        self.coefficients_ = coefficients
        self.residual_variance_ = float(residuals) / (self.n_samples_ - n_coefficients)
        # With predictors centered at the dataset mean, X^T X is block diagonal;
        # (X^T X)^-1 = B diag(1/N, S^-1) B^T with B = [[1, -mean^T], [0, I]].
        k = n_coefficients - 1
        centered_inverse = np.zeros((n_coefficients, n_coefficients))
        centered_inverse[0, 0] = 1 / self.n_samples_
        centered_inverse[1:, 1:] = np.linalg.inv(self.scatter_[:k, :k])
        shift = np.eye(n_coefficients)
        shift[0, 1:] = -self.mean_[:k]
        self.inverse_gram_ = shift @ centered_inverse @ shift.T
        return self


def linear_regression(target: NumericModelTarget, sizes, sums, products):
    """Least-squares coefficients (intercept first), validity, and residual sums of squares.

    The last model attribute is the response, the others are the predictors.
    """
    sizes = np.asarray(sizes, dtype=float)
    scatter = target.scatter(sizes, sums, products)
    k = scatter.shape[-1] - 1
    sxx, sxy, syy = scatter[..., :k, :k], scatter[..., :k, k], scatter[..., k, k]
    constant = target.constant(sizes, scatter)[..., :k].any(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = 1 / np.sqrt(np.diagonal(sxx, axis1=-2, axis2=-1))
        correlation = sxx * scale[..., :, None] * scale[..., None, :]
    valid = (sizes > k) & ~constant
    determinant = np.linalg.det(np.where(valid[..., None, None], correlation, np.eye(k)))
    valid &= determinant > _MIN_DETERMINANT
    safe = np.where(valid[..., None, None], sxx, np.eye(k))
    slopes = np.linalg.solve(safe, np.where(valid[..., None], sxy, 0.0)[..., None])[..., 0]
    means = target.means(sizes, sums)
    intercept = means[..., k] - np.sum(means[..., :k] * slopes, axis=-1)
    residuals = np.maximum(syy - np.sum(sxy * slopes, axis=-1), 0.0)
    coefficients = np.concatenate([intercept[..., None], slopes], axis=-1)
    return np.where(valid[..., None], coefficients, np.nan), valid, residuals


class CooksDistanceQF(MomentQF):
    """``phi_Cook``: Cook's distance of the subgroup's coefficients to the dataset's (Def. 6).

    ``coefficients`` names the coefficients of interest (the rows of ``Z``):
    ``"intercept"`` and predictor names; ``None`` (default) takes all of them.
    For example, the Giffen experiment of DMKD Sect. 6.6.1 restricts the
    measure to the price coefficient.
    """

    target_type = GeneralRegressionTarget

    def __init__(self, coefficients: Sequence[Hashable] | None = None):
        if coefficients is not None:
            coefficients = tuple(coefficients)
            if not coefficients or len(set(coefficients)) != len(coefficients):
                raise ValueError("coefficients must be distinct and non-empty")
        self.coefficients = coefficients

    def _selection(self, target: GeneralRegressionTarget) -> np.ndarray:
        names = target.coefficient_names_
        if self.coefficients is None:
            return np.arange(len(names))
        unknown = [name for name in self.coefficients if name not in names]
        if unknown:
            raise ValueError(f"unknown coefficients {unknown}; the target has {list(names)}")
        return np.array([names.index(name) for name in self.coefficients])

    def _scores(self, sizes, sums, products, target) -> np.ndarray:
        selected = self._selection(target)
        coefficients, _, _ = linear_regression(target, sizes, sums, products)
        difference = (coefficients - target.coefficients_)[..., selected]
        metric = np.linalg.inv(target.inverse_gram_[np.ix_(selected, selected)])
        distance = np.einsum("...i,ij,...j->...", difference, metric, difference) / (
            len(selected) * target.residual_variance_
        )
        return np.where(sizes == target.n_samples_, 0.0, distance)
