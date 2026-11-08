"""Likelihood-based measures for any model class with a maximum-likelihood fit.

A ``LikelihoodModel`` fits its parameters on a group of rows and gives each row
its log-likelihood under fitted parameters; ``LikelihoodTarget`` holds such a
model fitted on the dataset, and ``LikelihoodQF`` scores a subgroup with one of
two statistics:

- ``"ratio"`` (default): the likelihood-ratio statistic of separate models on
  the subgroup ``G`` and its complement ``C`` against one model on the dataset
  ``D``,

      LR = 2 [l_G(theta_G) + l_C(theta_C) - l_D(theta_D)],

  where ``l_S(theta)`` sums the rows' log-likelihoods in ``S`` and ``theta_S``
  is the maximum-likelihood fit on ``S``. Under the null hypothesis that both
  groups share the parameters, ``LR`` is asymptotically chi-squared with as many
  degrees of freedom as the model has parameters (S. S. Wilks, "The large-sample
  distribution of the likelihood ratio for testing composite hypotheses", Ann.
  Math. Statist. 9 (1938) 60-62). For linear regression with a common error
  variance this is the likelihood-ratio form of the Chow test. ``score="1-p"``
  (default) gives one minus the p-value, as the test-based measures of DMKD
  Sect. 5; ``score="statistic"`` gives ``LR``.
- ``"density"``: the mean likelihood (the density, not its logarithm) of the
  subgroup's model on the subgroup's rows minus its mean on the complement's
  rows, the measure of pysubgroup 0.x's ``EMM_Likelihood``. It has no publication;
  pysubgroup reproduces its results, compared as a black box. With the
  pysubgroup 0.x model ``PolyRegression_ModelClass`` it corresponds to
  ``PolynomialRegressionModel(degree, variance=1.0)``; it depends on the scale
  of the response.

``PolynomialRegressionModel`` is the built-in model class, ``y = b_0 + b_1 x +
... + b_d x^d + e`` with normal errors. Other model classes subclass
``LikelihoodModel``. The measures fit a model per group, so they evaluate one
subgroup at a time (not batched).

Written without reusing code; pysubgroup 0.x's ``EMM_Likelihood`` and
``PolyRegression_ModelClass`` served only as the specification of the
``"density"`` statistic and as a black box in the tests. Deviations: the empty
subgroup and complement, and groups the model cannot be fitted on (fewer rows
than parameters, fewer distinct ``x`` values than coefficients, or an exact
fit with an estimated variance), score ``nan``. With ``variance=1.0``,
pysubgroup 0.x also gives ``nan`` for ``degree + 1`` rows; pysubgroup fits them.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy import special

from ..targets.base import BaseTarget, QualityFunction
from .base import check_score

STATISTICS = ("ratio", "density")
# Estimated error variances below this fraction of the response's variance on the
# dataset count as exact fits, whose likelihood is unbounded.
_RELATIVE_VARIANCE = 1e-10


class LikelihoodModel(ABC):
    """A model class with maximum-likelihood fits on groups of rows.

    The instance passed to ``LikelihoodTarget`` keeps only the configuration;
    the target prepares a copy on the dataset. Subclasses implement:

    - ``prepare(data, attributes)``: validate the model attributes and store
      the arrays the fits need (positional, one entry per row of ``data``);
    - ``n_parameters``: the number of free parameters (the degrees of freedom
      of the likelihood-ratio test);
    - ``fit(mask)``: the maximum-likelihood parameters on the rows ``mask``, or
      ``None`` if they are not defined;
    - ``log_likelihood(parameters, mask)``: each row's log-likelihood under
      ``parameters``, for the rows ``mask`` (all rows if ``None``).

    ``describe(parameters)`` names the parameters for ``to_dataframe``.
    """

    @abstractmethod
    def prepare(self, data: pd.DataFrame, attributes: tuple[Hashable, ...]) -> None:
        """Store the model attributes of ``data``."""

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of free parameters of the model."""

    @abstractmethod
    def fit(self, mask: np.ndarray) -> Any | None:
        """Maximum-likelihood parameters on the rows ``mask``, ``None`` if undefined."""

    @abstractmethod
    def log_likelihood(self, parameters: Any, mask: np.ndarray | None = None) -> np.ndarray:
        """Log-likelihood of each row in ``mask`` (all rows if ``None``) under ``parameters``."""

    def describe(self, parameters: Any | None) -> dict[str, float]:
        """The parameters by name (empty unless a subclass names them)."""
        return {}


@dataclass(frozen=True)
class PolynomialFit:
    """A fitted polynomial regression: ``coefficients`` of ``x^0, ..., x^d`` and ``variance``."""

    coefficients: np.ndarray
    variance: float
    # Coefficients for x standardized by the dataset mean and deviation, used for predictions.
    standardized: np.ndarray


class PolynomialRegressionModel(LikelihoodModel):
    """``y = b_0 + b_1 x + ... + b_d x^d + e`` with ``e ~ N(0, sigma^2)`` (attributes ``x, y``).

    ``variance=None`` (default) estimates ``sigma^2`` by maximum likelihood,
    ``RSS / n``, which counts as a parameter; a number fixes it (``1.0`` gives
    pysubgroup 0.x's ``PolyRegression_ModelClass``). The coefficients are least
    squares in both cases.
    """

    def __init__(self, degree: int = 1, variance: float | None = None):
        if int(degree) != degree or degree < 0:
            raise ValueError("degree must be a non-negative integer")
        if variance is not None and not variance > 0:
            raise ValueError("variance must be positive or None")
        self.degree = int(degree)
        self.variance = variance

    def prepare(self, data: pd.DataFrame, attributes: tuple[Hashable, ...]) -> None:
        if len(attributes) != 2:
            raise ValueError(f"polynomial regression requires x and y, got {len(attributes)}")
        for name in attributes:
            dtype = data[name].dtype
            if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                raise TypeError("model attributes must be numeric columns")
        x, y = (data[name].to_numpy(dtype=float, na_value=np.nan) for name in attributes)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("model attributes must contain only finite, non-missing values")
        if x.std() == 0:
            raise ValueError("x must not be constant on the dataset")
        # Powers of the standardized x keep the least-squares problem well conditioned.
        self._shift, self._scale = x.mean(), x.std()
        self.design_ = np.vander((x - self._shift) / self._scale, self.degree + 1, increasing=True)
        self.response_ = y
        self._min_variance = _RELATIVE_VARIANCE * y.var()

    @property
    def n_parameters(self) -> int:
        return self.degree + 1 + (self.variance is None)

    def fit(self, mask: np.ndarray) -> PolynomialFit | None:
        size = np.count_nonzero(mask)
        if size < self.n_parameters:
            return None
        design, response = self.design_[mask], self.response_[mask]
        standardized, _, rank, _ = np.linalg.lstsq(design, response)
        if rank < self.degree + 1:
            return None
        variance = self.variance
        if variance is None:
            variance = float(np.sum(np.square(response - design @ standardized))) / size
            if variance <= self._min_variance:
                return None
        # Substitute z = (x - shift) / scale to get the coefficients of x.
        coefficients = Polynomial(standardized)(Polynomial([-self._shift, 1]) / self._scale).coef
        coefficients = np.pad(coefficients, (0, self.degree + 1 - len(coefficients)))
        return PolynomialFit(coefficients, variance, standardized)

    def log_likelihood(self, parameters: PolynomialFit, mask: np.ndarray | None = None):
        design, response = self.design_, self.response_
        if mask is not None:
            design, response = design[mask], response[mask]
        residuals = response - design @ parameters.standardized
        variance = parameters.variance
        return -0.5 * (np.log(2 * np.pi * variance) + np.square(residuals) / variance)

    def describe(self, parameters: PolynomialFit | None) -> dict[str, float]:
        names = [f"b{power}" for power in range(self.degree + 1)] + ["variance"]
        if parameters is None:
            return dict.fromkeys(names, float("nan"))
        values = [*parameters.coefficients, parameters.variance]
        return {name: float(value) for name, value in zip(names, values, strict=True)}


class LikelihoodTarget(BaseTarget):
    """The attributes of a ``LikelihoodModel`` (``fit(data, *attributes)``).

    ``fit`` stores ``model_``, a copy of ``model`` prepared on the dataset,
    ``parameters_``, the maximum-likelihood fit on the dataset, and
    ``log_likelihood_``, its log-likelihood ``l_D(theta_D)``.

    What the likelihood-ratio statistic of ``LikelihoodQF`` finds: each fit
    maximizes the likelihood on its own rows, so

        LR = 2 [l_G(theta_G) - l_G(theta_D)] + 2 [l_C(theta_C) - l_C(theta_D)],

    with both terms non-negative: how much less likely the subgroup's (and the
    complement's) rows are under the dataset's model than under their own fit
    in the same model class. For a normal distribution, each term is
    ``2 n KL(N(mu_G, sigma_G^2) || N(mu_D, sigma_D^2))``, so subgroups score
    high when their mean or spread differs from the dataset's, weighted by
    size. The measure does not reward subgroups the model class fits poorly: a
    misfit (say, a bimodal subgroup under a normal model) lowers ``l_G(theta_G)``
    and ``l_G(theta_D)`` alike and cancels; only differences between the
    fitted parameters count. Finding such subgroups needs a goodness-of-fit
    measure instead, such as a comparison with a more flexible model class.
    """

    def __init__(self, model: LikelihoodModel):
        if not isinstance(model, LikelihoodModel):
            raise TypeError("model must be a LikelihoodModel")
        self.model = model

    def fit(self, data: pd.DataFrame, *attributes: Hashable) -> LikelihoodTarget:
        super().fit(data, *attributes)
        model = copy.deepcopy(self.model)
        model.prepare(data, self.attributes_)
        parameters = model.fit(np.ones(self.n_samples_, dtype=bool))
        if parameters is None:
            raise ValueError("the model cannot be fitted on the dataset")
        self.model_ = model
        self.parameters_ = parameters
        self.log_likelihood_ = float(model.log_likelihood(parameters).sum())
        return self

    def refit(self, data: pd.DataFrame) -> LikelihoodTarget:
        if not hasattr(self, "n_samples_"):
            raise ValueError("only fitted targets can be refitted")
        return type(self)(self.model).fit(data, *self.attributes_)


class LikelihoodQF(QualityFunction):
    """Likelihood-based measure of a ``LikelihoodTarget`` (see the module docstring).

    ``statistic="ratio"`` (default) is the likelihood-ratio test of separate
    models on subgroup and complement, scored as one minus its chi-squared
    p-value (``score="1-p"``) or as the statistic itself
    (``score="statistic"``). ``statistic="density"`` is pysubgroup 0.x's mean
    likelihood of the subgroup's model on the subgroup minus on the complement;
    it takes no ``score``.
    """

    target_type = LikelihoodTarget

    def __init__(self, statistic: str = "ratio", score: str = "1-p"):
        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}")
        self.score = check_score(score)
        if statistic == "density" and score != "1-p":
            raise ValueError('score applies only to statistic="ratio"')
        self.statistic = statistic

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        size = np.count_nonzero(mask)
        if size == 0 or size == target.n_samples_:
            return float("nan")
        model = target.model_
        parameters = model.fit(mask)
        if parameters is None:
            return float("nan")
        if self.statistic == "density":
            density = np.exp(model.log_likelihood(parameters))
            return float(density[mask].mean() - density[~mask].mean())
        complement = model.fit(~mask)
        if complement is None:
            return float("nan")
        separate = model.log_likelihood(parameters, mask).sum() + model.log_likelihood(
            complement, ~mask
        ).sum()
        ratio = max(2.0 * (separate - target.log_likelihood_), 0.0)
        if self.score == "statistic":
            return float(ratio)
        return float(special.chdtr(model.n_parameters, ratio))
