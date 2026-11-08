"""Frequency target: target-free support of subgroups (frequent itemset mining).

Its measures, the subgroup size and the size times the description length, use
no target information and therefore accept any target.

Provenance: ``CountQF`` and ``AreaQF`` are adapted from pysubgroup 0.x's
``fi_target.py`` (https://github.com/flemmerich/pysubgroup, Apache-2.0, Florian
Lemmerich et al.), rewritten for fitted targets and Boolean row masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseTarget, BoundedQualityFunction, QualityFunction


class FITarget(BaseTarget):
    """Target-free frequent-itemset/subgroup support."""

    def fit(self, data: pd.DataFrame) -> FITarget:
        return super().fit(data)


# Adapted from pysubgroup 0.x's CountQF and AreaQF (fi_target.py, Apache-2.0).
class CountQF(BoundedQualityFunction):
    """Subgroup size; usable with any target, e.g. for frequent itemset mining."""

    batchable = True

    def evaluate(self, coverage, target) -> float:
        return int(np.count_nonzero(self._mask(coverage, target)))

    def optimistic_estimate(self, coverage, target) -> float:
        return self.evaluate(coverage, target)

    def evaluate_many(self, statistics, target) -> np.ndarray:
        self._check_target(target)
        return statistics.sizes

    def optimistic_estimate_many(self, statistics, target) -> np.ndarray:
        return self.evaluate_many(statistics, target)


class AreaQF(QualityFunction):
    """Subgroup size times the number of conditions (``depth``) describing it."""

    batchable = True

    def evaluate(self, coverage, target, *, depth: int) -> float:
        return int(np.count_nonzero(self._mask(coverage, target))) * depth

    def evaluate_many(self, statistics, target, *, depth: int) -> np.ndarray:
        self._check_target(target)
        return statistics.sizes * depth
