"""NumBSD: exhaustive subgroup discovery for numeric targets on sorted bitsets.

Lemmerich, F., Atzmueller, M., Puppe, F.: Fast Exhaustive Subgroup Discovery
with Numerical Target Concepts. Data Mining and Knowledge Discovery 30(3):
711-762, 2016. Implemented from Algorithm 1 and Sections 5.1.2, 5.2.2-5.2.3,
without reusing code.
"""

from __future__ import annotations

import numpy as np

from .. import _numba
from ..rules import Rule
from ..targets import NumericTarget, StandardQFNumeric
from .base_algorithms import SubgroupDiscoveryTask, _Search


class NumBSD:
    """NumBSD: depth-first search with look-ahead on bitsets sorted by the target.

    The rows are sorted once by decreasing target value, and every condition
    becomes a bitset in this order (Section 5.1.2). The rows of a refinement
    ``parent & condition`` are then read in target order without sorting, and
    one pass yields its quality and its tight ordering-based optimistic
    estimate, the best quality of a prefix (or of a prefix or suffix for
    two-pass measures; Definition 1, Theorems 1-6). Measures without an
    ordering-based bound (``KolmogorovSmirnovQF``, ``StandardDeviationQF``) use
    their closed-form bound, computed from the same rows.

    ``recurse`` (Algorithm 1) works as follows:

    - Look-ahead: it computes every refinement of the current subgroup by a
      remaining condition and offers it to the top-k results. Refinements whose
      estimate exceeds the minimum required quality become the next
      candidates.
    - It then visits the candidates by decreasing estimate. Each one is
      refined by the candidates after it.

    With ``fast_bounds`` (Section 5.2.3, Theorem 9) and a mean-based
    ``StandardQFNumeric`` (direction ``"positive"`` or ``"negative"``), the scan
    of a refinement checks after every 64-row word whether no prefix read so
    far, nor any larger one, can reach the results, and then stops early. The
    upper bound on the size is ``min(|parent|, |condition|)``. Fast bounds need
    the numba extra; the NumPy path reads every row, and both take the same
    decisions.

    With numba, the other mean-based measures, the variance reduction, the
    interclass variance, ``MannWhitneyQF`` and ``RankAUCQF`` also scan the
    packed bitsets directly (``_numba.ordered_prefix_scan``, one pass per
    direction); the median, ``StandardDeviationQF`` and
    ``KolmogorovSmirnovQF`` unpack each refinement's rows in NumPy.

    Adaptations:

    - The pseudo-code calls ``RECURSE(..., nextSelectors \\ sel)`` for every
      candidate, which would visit each subgroup once per order of its
      conditions. Here each candidate is removed before its subtree is
      searched, so it is refined only by the candidates after it.
    - A candidate is skipped if the minimum required quality has risen above
      its estimate since the look-ahead (as in DpSubgroup).
    - Like the other exhaustive algorithms in pysubgroup: a subgroup is pruned
      when its estimate does not exceed the minimum required quality, uses at
      most one condition per feature, and must pass the anti-monotone filters;
      ties are broken by condition index.

    After ``execute``, ``n_refinements_`` counts the refinements computed and
    ``n_stopped_early_`` those that fast bounds stopped before their last word.
    """

    def __init__(self, fast_bounds: bool = True):
        self.fast_bounds = fast_bounds

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task, batch_support=False)
        target, qf = task.target, task.qf
        if not isinstance(target, NumericTarget):
            raise TypeError(f"NumBSD requires a NumericTarget, got {type(target).__name__}")
        self._search = search
        self._ordered = hasattr(qf, "_ordered")
        self._kernel = _numba.enabled() and hasattr(qf, "_scan")
        self._order = target.order_
        n = target.n_samples_
        sorted_masks = search.masks[:, self._order]
        self._words = _numba.pack64(sorted_masks)
        self._sizes = sorted_masks.sum(axis=1)
        self._mean_scan = (
            self.fast_bounds and _numba.enabled() and isinstance(qf, StandardQFNumeric)
            and qf.fast_bounds_apply()
        )
        if self._mean_scan:
            self._values = np.ascontiguousarray(target.sorted_values_)
            self._weights = qf._power_weights(target)
            self._reverse = qf.direction == "negative"
            self._centroid = -target.mean_ if self._reverse else target.mean_
        self.n_refinements_ = 0
        self.n_stopped_early_ = 0
        everything = _numba.pack64(np.ones(n, dtype=bool))
        self._recurse((), everything, n, list(range(len(search.conditions))))
        return search.results()

    def _unsorted(self, sorted_mask: np.ndarray) -> np.ndarray:
        """The row mask in the dataset's row order."""
        mask = np.zeros(len(sorted_mask), dtype=bool)
        mask[self._order] = sorted_mask
        return mask

    def _sorted_mask(self, words: np.ndarray) -> np.ndarray:
        bits = np.unpackbits(words.view(np.uint8), bitorder="little")
        return bits[: self._search.task.target.n_samples_].astype(bool)

    def _refinement(self, parent_words, parent_size: int, indices: tuple[int, ...],
                    threshold: float):
        """Algorithm 1, ``computeRefinement``: ``(words, size, sorted mask, quality, estimate)``.

        Returns None if the refinement is empty or fast bounds pruned it.
        """
        self.n_refinements_ += 1
        index = indices[-1]
        task = self._search.task
        qf, target = task.qf, task.target
        words = parent_words & self._words[index]
        sorted_mask = None
        if self._mean_scan:
            max_size = min(parent_size, int(self._sizes[index]))
            size, total, estimate, stopped = _numba.ordered_mean_scan(
                parent_words, self._words[index], self._values, self._weights, self._centroid,
                max_size, threshold, self._reverse, True,
            )
            if stopped:
                self.n_stopped_early_ += 1
                return None
            if not size:
                return None
            quality = 0.0 if size == target.n_samples_ else (
                size**qf.a * (total / size - self._centroid)
            )
        elif self._kernel and (scanned := qf._scan(parent_words, self._words[index], target)):
            size, quality, estimate = scanned
            if not size:
                return None
        else:
            sorted_mask = self._sorted_mask(words)
            size = int(np.count_nonzero(sorted_mask))
            if not size:
                return None
            if self._ordered:
                quality, estimate = qf._ordered(sorted_mask, target)
            else:
                mask = self._unsorted(sorted_mask)
                quality = self._search.evaluate(mask, indices)
                estimate = self._search.optimistic_estimate(mask)
        return words, size, sorted_mask, float(quality), float(estimate)

    def _recurse(self, prefix: tuple[int, ...], parent_words, parent_size: int,
                 remaining: list[int]) -> None:
        search = self._search
        used = {search.groups[i] for i in prefix}
        candidates = []
        for index in remaining:
            if search.groups[index] in used:
                continue
            threshold = search.result.min_required_quality()
            indices = (*prefix, index)
            refinement = self._refinement(parent_words, parent_size, indices, threshold)
            if refinement is None:
                continue
            words, size, sorted_mask, quality, estimate = refinement
            mask = None
            if search.pruning_filters or search.result_filters:
                if sorted_mask is None:
                    sorted_mask = self._sorted_mask(words)
                mask = self._unsorted(sorted_mask)
                if not search._passes(search.pruning_filters, mask, indices):
                    continue
            if not estimate > search.result.min_required_quality():
                continue
            search.add(quality, mask, indices)
            candidates.append((estimate, index, words, size))
        if len(prefix) + 1 >= search.task.depth:
            return
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        for position, (estimate, index, words, size) in enumerate(candidates):
            if not estimate > search.result.min_required_quality():
                continue
            later = [candidate[1] for candidate in candidates[position + 1:]]
            if later:
                self._recurse((*prefix, index), words, size, later)
