"""Targets and their quality functions, one module per target type.

- ``base``: ``BaseTarget``, ``LabelledTarget``, ``QualityFunction``, ``BoundedQualityFunction``
- ``frequency``: ``FITarget``, ``CountQF``, ``AreaQF``
- ``combined``: ``CombinedQF`` (weighted sum of quality functions)
- ``generalization_aware``: ``GeneralizationAwareQF`` (increase in quality over
  all generalizations), ``GeneralizationAwareStandardQF`` and
  ``GeneralizationAwareStandardQFNumeric`` (increase of the target share or mean)
- ``binary``: ``BinaryTarget``, ``StandardQF`` (``LiftQF``, ``SimpleBinomialQF``,
  ``WRAccQF``), ``ChiSquaredQF``
- ``multiclass``: ``MulticlassTarget``, ``SplitQF``, ``MulticlassChiSquaredQF``,
  ``GiniQF``, ``ClassDistributionL1QF``, ``MaxConfidenceQF``
- ``gradient``: ``GradientTarget``, ``GradientBoostingQF`` (the rule objective of
  second-order gradient boosting)
- ``numeric``: ``NumericTarget``, ``StandardQFNumeric``, ``StandardQFNumericTscore``,
  ``VarianceReductionQF``, ``InterclassVarianceQF``, ``StandardDeviationQF``,
  ``MannWhitneyQF``, ``RankAUCQF``, ``KolmogorovSmirnovQF``
- ``soft_classifier``: ``SoftClassifierTarget`` and the SubROC measures ``ARLQF``,
  ``ROCAUCQF``, ``PRAUCQF``
- ``nominal``: ``NominalTarget`` (several discrete model attributes), ``WeightedKLQF``
- ``vector``: ``VectorTarget``, ``MeanVectorL1QF``
"""

from .base import BaseTarget, BoundedQualityFunction, LabelledTarget, QualityFunction
from .binary import BinaryTarget, ChiSquaredQF, LiftQF, SimpleBinomialQF, StandardQF, WRAccQF
from .combined import CombinedQF
from .frequency import AreaQF, CountQF, FITarget
from .generalization_aware import (
    GeneralizationAwareMeasure,
    GeneralizationAwareQF,
    GeneralizationAwareStandardQF,
    GeneralizationAwareStandardQFNumeric,
)
from .gradient import GradientBoostingQF, GradientTarget
from .multiclass import (
    ClassDistributionL1QF,
    GiniQF,
    MaxConfidenceQF,
    MulticlassChiSquaredQF,
    MulticlassQF,
    MulticlassTarget,
    SplitQF,
)
from .nominal import NominalTarget, WeightedKLQF
from .numeric import (
    InterclassVarianceQF,
    KolmogorovSmirnovQF,
    MannWhitneyQF,
    NumericTarget,
    RankAUCQF,
    StandardDeviationQF,
    StandardQFNumeric,
    StandardQFNumericTscore,
    VarianceReductionQF,
)
from .soft_classifier import (
    ARLQF,
    PRAUCQF,
    ROCAUCQF,
    BaseSoftClassifierPerformanceQF,
    SoftClassifierTarget,
)
from .vector import MeanVectorL1QF, VectorTarget

__all__ = [
    "ARLQF",
    "AreaQF",
    "BaseSoftClassifierPerformanceQF",
    "BaseTarget",
    "BinaryTarget",
    "BoundedQualityFunction",
    "ChiSquaredQF",
    "ClassDistributionL1QF",
    "CombinedQF",
    "CountQF",
    "FITarget",
    "GeneralizationAwareMeasure",
    "GeneralizationAwareQF",
    "GeneralizationAwareStandardQF",
    "GeneralizationAwareStandardQFNumeric",
    "GiniQF",
    "GradientBoostingQF",
    "GradientTarget",
    "InterclassVarianceQF",
    "KolmogorovSmirnovQF",
    "LabelledTarget",
    "LiftQF",
    "MannWhitneyQF",
    "MaxConfidenceQF",
    "MeanVectorL1QF",
    "MulticlassChiSquaredQF",
    "MulticlassQF",
    "MulticlassTarget",
    "NominalTarget",
    "NumericTarget",
    "PRAUCQF",
    "QualityFunction",
    "ROCAUCQF",
    "RankAUCQF",
    "SimpleBinomialQF",
    "SoftClassifierTarget",
    "SplitQF",
    "StandardDeviationQF",
    "StandardQF",
    "StandardQFNumeric",
    "StandardQFNumericTscore",
    "VarianceReductionQF",
    "VectorTarget",
    "WRAccQF",
    "WeightedKLQF",
]
