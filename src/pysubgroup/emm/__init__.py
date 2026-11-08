"""Exceptional Model Mining (EMM): model classes over several targets and their quality functions.

Following W. Duivesteijn, A. Feelders, and A. Knobbe, "Exceptional Model
Mining", Data Mining and Knowledge Discovery 30 (2016) 47-98, and D. Leman,
A. Feelders, and A. Knobbe, "Exceptional model mining", ECML PKDD 2008. Each
target fits its model attributes on the dataset; the searches exclude them from
the subgroup descriptions. One module per model class:

- ``base``: ``NumericModelTarget`` (moments of numeric model attributes),
  ``entropy_weight`` (``phi_ef``), the ``reference`` and ``score`` options
- ``correlation``: ``CorrelationTarget``, ``CorrelationSignificanceQF``
  (``phi_scd``), ``CorrelationDifferenceQF`` (``phi_abs``, ``phi_ent``)
- ``regression``: ``RegressionTarget`` with ``SlopeSignificanceQF``
  (``phi_ssd``); ``GeneralRegressionTarget`` with ``CooksDistanceQF``
  (``phi_Cook``)
- ``logistic``: ``LogisticRegressionTarget``, ``LogisticInteractionQF``
  (``phi_sed``, Wald test; not batched)
- ``likelihood``: ``LikelihoodTarget`` with any ``LikelihoodModel`` (built in:
  ``PolynomialRegressionModel``), ``LikelihoodQF`` (likelihood-ratio test or
  pysubgroup 0.x's mean-density difference; not batched)
- ``association``: ``AssociationTarget``, ``AssociationDevianceQF`` (``phi_dev``)
- ``decision_table``: ``DecisionTableTarget``, ``HellingerQF`` (``phi_Hel``),
  ``BDeuQF`` (``phi_BDeu``)
- ``gp_growth``: ``GPGrowth``, exhaustive search on generic pattern trees
  (Lemmerich, Becker, and Atzmueller, ECML PKDD 2012) for any target, with
  ``ValuationDomain``, ``ValuationQF`` and the paper's domains
  (``VarianceDomain``, ``CovarianceDomain``, ``CorrelationDomain``,
  ``SlopeDomain``, ``DecisionTableDomain``)

Result statistics are in ``utils.to_dataframe``, the scatter plot with
regression lines is ``visualization.plot_model_scatter``.

The Weighted Kullback-Leibler measure of DSSD on several nominal attributes
stays in ``targets.nominal`` (``NominalTarget``, ``WeightedKLQF``). The
Bayesian network model (DMKD Sect. 5.5) is not implemented. See the book's
chapter on exceptional model mining (user_guide/09-exceptional-model-mining.md).
"""

from .association import AssociationDevianceQF, AssociationTarget
from .base import MomentQF, NumericModelTarget, entropy_weight
from .correlation import CorrelationDifferenceQF, CorrelationSignificanceQF, CorrelationTarget
from .decision_table import BDeuQF, DecisionTableTarget, HellingerQF
from .gp_growth import (
    CorrelationDomain,
    CovarianceDomain,
    DecisionTableDomain,
    GPGrowth,
    SlopeDomain,
    ValuationDomain,
    ValuationQF,
    VarianceDomain,
)
from .likelihood import (
    LikelihoodModel,
    LikelihoodQF,
    LikelihoodTarget,
    PolynomialRegressionModel,
)
from .logistic import LogisticInteractionQF, LogisticRegressionTarget
from .regression import (
    CooksDistanceQF,
    GeneralRegressionTarget,
    RegressionTarget,
    SlopeSignificanceQF,
)

__all__ = [
    "AssociationDevianceQF",
    "AssociationTarget",
    "BDeuQF",
    "CooksDistanceQF",
    "CorrelationDifferenceQF",
    "CorrelationDomain",
    "CorrelationSignificanceQF",
    "CorrelationTarget",
    "CovarianceDomain",
    "DecisionTableDomain",
    "DecisionTableTarget",
    "GPGrowth",
    "GeneralRegressionTarget",
    "HellingerQF",
    "LikelihoodModel",
    "LikelihoodQF",
    "LikelihoodTarget",
    "LogisticInteractionQF",
    "LogisticRegressionTarget",
    "MomentQF",
    "NumericModelTarget",
    "PolynomialRegressionModel",
    "RegressionTarget",
    "SlopeDomain",
    "SlopeSignificanceQF",
    "ValuationDomain",
    "ValuationQF",
    "VarianceDomain",
    "entropy_weight",
]
