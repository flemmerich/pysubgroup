"""Weighted sums of quality functions.

``CombinedQF(measures, weights)`` scores a subgroup by ``sum_i w_i q_i`` with
non-negative weights. There is no paper behind it; pysubgroup 0.x's unfinished
``CombinedInterestingnessMeasure`` suggested the feature, and no code is reused.

Optimistic estimate: every subset ``S`` of a subgroup ``P`` has
``w_i q_i(S) <= w_i oe_i(P)`` for ``w_i >= 0``, so ``sum_i w_i oe_i(P)`` bounds
the combination. It is tight if one subset attains every ``oe_i``; otherwise
the parts' best subsets differ and the sum overestimates. A measure without an
optimistic estimate makes the sum infinite. Negative weights are rejected: they
would need lower bounds of the parts, which the measures do not provide. An
estimate of ``-inf`` (no subset scores above ``-inf``, e.g. an empty subgroup)
makes the sum ``-inf``, also next to infinite terms.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import BaseTarget, BoundedQualityFunction, QualityFunction
from .frequency import AreaQF
from .generalization_aware import GeneralizationAwareMeasure


def _common_target_type(measures: Sequence[QualityFunction]) -> type[BaseTarget]:
    """The target class that every measure accepts: the most specific of their types."""
    types = {qf.target_type for qf in measures}
    for candidate in types:
        if all(issubclass(candidate, other) for other in types):
            return candidate
    names = sorted(t.__name__ for t in types)
    raise TypeError(f"the measures require incompatible targets: {names}")


class CombinedQF(BoundedQualityFunction):
    """Weighted sum ``sum_i w_i q_i`` of quality functions on one target.

    ``weights`` default to 1 and must be non-negative and finite, at least one
    positive; terms with weight 0 are ignored. All measures must accept one
    target class (e.g. ``WRAccQF`` and ``CountQF`` on a ``BinaryTarget``). The
    optimistic estimate is ``sum_i w_i oe_i`` (see the module docstring),
    infinite if a weighted measure is unbounded. Batchable if every weighted
    measure is. ``AreaQF`` and generalization-aware measures cannot be combined.
    """

    def __init__(self, measures: Sequence[QualityFunction], weights: Sequence[float] | None = None):
        measures = tuple(measures)
        if not measures:
            raise ValueError("CombinedQF needs at least one measure")
        for qf in measures:
            if not isinstance(qf, QualityFunction):
                raise TypeError(f"measures must be quality functions, got {type(qf).__name__}")
            if isinstance(qf, AreaQF | GeneralizationAwareMeasure):
                raise TypeError(f"{type(qf).__name__} cannot be combined")
        weights = (1.0,) * len(measures) if weights is None else tuple(weights)
        if len(weights) != len(measures):
            raise ValueError(f"got {len(weights)} weights for {len(measures)} measures")
        if not all(
            isinstance(w, int | float) and not isinstance(w, bool) and np.isfinite(w) and w >= 0
            for w in weights
        ):
            raise ValueError(f"weights must be non-negative finite numbers, got {weights}")
        if not any(weights):
            raise ValueError("at least one weight must be positive")
        self.measures = measures
        self.weights = tuple(float(w) for w in weights)
        self.target_type = _common_target_type(measures)
        self._terms = [(w, qf) for w, qf in zip(self.weights, measures, strict=True) if w]

    @property
    def batchable(self) -> bool:
        return all(qf.batchable for _, qf in self._terms)

    def _check_target(self, target: BaseTarget) -> None:
        super()._check_target(target)
        for qf in self.measures:
            qf._check_target(target)

    def evaluate(self, coverage, target) -> float:
        self._check_target(target)
        return float(sum(w * qf.evaluate(coverage, target) for w, qf in self._terms))

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        return np.sum(
            [w * np.asarray(qf.evaluate_many(statistics, target), dtype=float)
             for w, qf in self._terms],
            axis=0,
        )

    @staticmethod
    def _sum(terms: list) -> np.ndarray:
        """Sum of the weighted estimates; ``-inf`` wherever one of them is ``-inf``."""
        terms = np.asarray(terms, dtype=float)
        with np.errstate(invalid="ignore"):
            total = terms.sum(axis=0)
        return np.where((terms == -np.inf).any(axis=0), -np.inf, total)

    def optimistic_estimate(self, coverage, target) -> float:
        """``sum_i w_i oe_i``: an upper bound on the quality of every subset."""
        self._check_target(target)
        return float(self._sum([
            w * (qf.optimistic_estimate(coverage, target)
                 if isinstance(qf, BoundedQualityFunction) else np.inf)
            for w, qf in self._terms
        ]))

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        return self._sum([
            w * (np.asarray(qf.optimistic_estimate_many(statistics, target), dtype=float)
                 if isinstance(qf, BoundedQualityFunction)
                 else np.full(statistics.n_children, np.inf))
            for w, qf in self._terms
        ])

    def __repr__(self) -> str:
        return f"CombinedQF({list(self.measures)!r}, weights={list(self.weights)!r})"
