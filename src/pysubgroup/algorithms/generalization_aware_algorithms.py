"""Searches with generalization-aware quality functions.

Lemmerich, F.: Novel Techniques for Efficient and Effective Subgroup Discovery.
Dissertation, University of Würzburg, 2014: ``GeneralizationAwareApriori``
follows Section 6.3 (an Apriori-based algorithm, as Batal and Hauskrecht's);
``GeneralizationAwareDFS`` uses the reverse depth-first search of Section 3.2.1
(Algorithm 4), which the dissertation proposes for these measures but does not
evaluate. Implemented from the dissertation without reusing code.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..rules import Condition, Rule
from ..targets import GeneralizationAwareMeasure
from .base_algorithms import Apriori, SubgroupDiscoveryTask, _Search

DIFFERENCE_BASED = ("auto", "on", "off")


def _interval(condition: Condition) -> tuple[float, float] | None:
    """The values ``(lower, upper]`` a numeric condition accepts; None for other operators."""
    if condition.operator == "le":
        return float("-inf"), condition.value
    if condition.operator == "gt":
        return condition.value, float("inf")
    if condition.operator == "interval":
        lower = float("-inf") if condition.lower is None else condition.lower
        upper = float("inf") if condition.upper is None else condition.upper
        return lower, upper
    return None


def has_overlapping_or_negated_conditions(conditions: list[Condition]) -> bool:
    """Whether a condition is negated (``ne``) or two intervals on one feature overlap.

    ``le``, ``gt``, and ``interval`` conditions are intervals ``(lower, upper]``;
    two of them overlap if they accept a common value, as ``x <= 3`` and
    ``x <= 5`` do, but not the disjoint bins of ``generate_conditions``.
    """
    intervals = defaultdict(list)
    for condition in conditions:
        if condition.operator == "ne":
            return True
        interval = _interval(condition)
        if interval is not None:
            intervals[condition.feature].append(interval)
    for feature_intervals in intervals.values():
        highest = None
        for lower, upper in sorted(feature_intervals):
            if highest is not None and lower < highest:
                return True
            highest = upper if highest is None else max(highest, upper)
    return False


class GeneralizationAwareApriori:
    """Exhaustive level-wise search for generalization-aware quality functions.

    Apriori evaluates all subgroups with ``k`` conditions before those with
    ``k + 1``, so the generalizations of a candidate are known when it is
    evaluated. Every subgroup ``P`` of the previous level keeps its rows and
    ``B(P) = max_{H ⊆ P} v(H)``, the largest value (``qf.value``) of ``P`` and
    its generalizations. A candidate's reference, the largest value of a
    generalization, is the maximum of ``B`` over its direct generalizations
    (one condition removed); for single conditions it is the value of the whole
    dataset. ``B`` of the candidate is the larger of its value and its
    reference, and gives its optimistic estimate.

    With the difference-based estimates (Theorems 3 and 4), every subgroup also
    keeps the summary of the rows its direct generalizations cover beyond it
    (``qf.difference``), merged with the summaries kept by these
    generalizations (``qf.merge_differences``). The dissertation's algorithm
    propagates the same statistics by minima over the direct generalizations.
    ``difference_based`` is ``"on"``, ``"off"``, or ``"auto"`` (default): use
    them only if the search space has negated conditions or overlapping
    intervals on one feature (``has_overlapping_or_negated_conditions``).
    Only there do refinements remove few rows from a generalization, so that
    the estimates prune; otherwise they cost more time than they save. After
    ``execute``, ``difference_based_`` says whether they were used.

    As in ``Apriori``, a candidate is generated only if all its direct
    generalizations were promising (passed the anti-monotone filters with an
    optimistic estimate above the minimum required quality). The estimate of a
    generalization bounds the quality of all its refinements, so the other
    candidates cannot reach the results. Evaluates one subgroup at a time on the
    CPU; after ``execute``, ``n_evaluated_`` counts the subgroups evaluated.
    """

    def __init__(self, difference_based: str = "auto"):
        if difference_based not in DIFFERENCE_BASED:
            raise ValueError(
                f"difference_based must be one of {DIFFERENCE_BASED}, got {difference_based!r}"
            )
        self.difference_based = difference_based

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        qf, target = task.qf, task.target
        if not isinstance(qf, GeneralizationAwareMeasure):
            raise TypeError(
                f"GeneralizationAwareApriori requires a generalization-aware quality function, "
                f"got {type(qf).__name__}"
            )
        search = _Search(task, batch_support=False)
        self.difference_based_ = self.difference_based == "on" or (
            self.difference_based == "auto"
            and has_overlapping_or_negated_conditions(search.conditions)
        )
        everything = np.ones(target.n_samples_, dtype=bool)
        root = qf.value(everything, target)
        if not np.isfinite(root):
            raise ValueError(f"the whole dataset must have a finite value, got {root}")
        # Subgroups of the previous level: (B, merged differences, row mask).
        kept: dict[tuple[int, ...], tuple[float, object, np.ndarray]] = {
            (): (root, None, everything)
        }
        self.n_evaluated_ = 0
        level = [((), None, list(range(len(search.conditions))))]
        for depth in range(1, task.depth + 1):
            promising = []
            for parent, parent_mask, candidates in level:
                children = search.expand(parent, parent_mask, candidates)
                for i in range(len(candidates)):
                    if not children.admissible(i):
                        continue
                    self.n_evaluated_ += 1
                    indices, mask = children.indices(i), children.mask(i)
                    generalizations = [
                        kept[indices[:j] + indices[j + 1:]] for j in range(len(indices))
                    ]
                    reference = max(best for best, _, _ in generalizations)
                    search.add(qf.evaluate(mask, target, reference=reference), mask, indices)
                    if depth == task.depth:
                        continue
                    best = max(qf.value(mask, target), reference)
                    differences = None
                    if self.difference_based_:
                        differences = qf.merge_differences(
                            [qf.difference(rows, mask, target) for _, _, rows in generalizations]
                            + [merged for _, merged, _ in generalizations]
                        )
                    estimate = qf.optimistic_estimate(
                        mask, target, reference=best, differences=differences
                    )
                    if estimate >= search.result.min_required_quality():
                        promising.append((estimate, indices, (best, differences, mask)))
            if depth == task.depth:
                break
            minimum = search.result.min_required_quality()
            kept = {indices: state for estimate, indices, state in promising if estimate > minimum}
            level = Apriori._next_level(
                search, {indices: mask for indices, (_, _, mask) in kept.items()}
            )
        return search.results()


class _Table:
    """The promising children of an expanded subgroup ``G``, sorted by condition.

    For each child ``G + c``: ``B(G + c) = max_{H ⊆ G + c} v(H)``, its optimistic
    estimate and, with difference-based estimates, its merged differences and
    (batched) its difference state.
    """

    __slots__ = ("candidates", "best", "estimates", "differences", "states")

    def __init__(self, candidates, best, estimates, differences, states):
        self.candidates = candidates
        self.best = best
        self.estimates = estimates
        self.differences = differences
        self.states = states

    def lookup(self, candidates: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """Positions of ``candidates`` in the (non-empty) table and whether they are promising."""
        positions = np.minimum(np.searchsorted(self.candidates, candidates),
                               len(self.candidates) - 1)
        found = self.candidates[positions] == candidates
        return positions, found & (self.estimates[positions] > threshold)


class GeneralizationAwareDFS:
    """Exhaustive reverse depth-first search for generalization-aware quality functions.

    The search tree is DFS's (a subgroup is refined by conditions on later
    features), but siblings are visited in decreasing order of their conditions,
    each subtree before the next sibling (the dissertation's Algorithm 4). Then
    every subgroup is expanded after all its generalizations. Expanding a
    subgroup ``G`` evaluates its children ``G + c`` together and keeps the
    promising ones in a table: ``B(G + c) = max_{H ⊆ G + c} v(H)`` and the
    optimistic estimate. The direct generalizations of ``G + c`` are ``G`` and
    ``(G - g) + c`` for the conditions ``g`` of ``G``, children of the expanded
    ``G - g``. So the reference of ``G + c`` is the maximum of ``B(G)`` and of
    ``B((G - g) + c)`` from the tables of the ``G - g``, looked up for all
    children of ``G`` at once before any child is evaluated.

    As in ``GeneralizationAwareApriori``, a child is evaluated only if all its
    direct generalizations are promising: they passed the anti-monotone filters,
    and their estimates are above the current minimum required quality (the
    estimate of a generalization bounds all its refinements). The tables of all
    expanded subgroups with fewer than ``depth - 1`` conditions are kept until
    the search ends. With a batchable quality function (``evaluate_many`` with
    references, ``value_many``, ``optimistic_estimate_many``) and batchable
    pruning filters, the children of a subgroup are evaluated in one pass, and
    so are the difference-based estimates of measures with
    ``batched_differences``; otherwise these are computed one child at a time
    on host masks. ``difference_based`` is ``"on"``, ``"off"``, or ``"auto"`` as
    for ``GeneralizationAwareApriori``. After ``execute``, ``difference_based_``
    says whether they were used, ``n_evaluated_`` counts the evaluated
    subgroups, and ``n_tables_`` the kept tables.
    """

    def __init__(self, difference_based: str = "auto"):
        if difference_based not in DIFFERENCE_BASED:
            raise ValueError(
                f"difference_based must be one of {DIFFERENCE_BASED}, got {difference_based!r}"
            )
        self.difference_based = difference_based

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        qf, target = task.qf, task.target
        if not isinstance(qf, GeneralizationAwareMeasure):
            raise TypeError(
                f"GeneralizationAwareDFS requires a generalization-aware quality function, "
                f"got {type(qf).__name__}"
            )
        search = _Search(task)
        self.difference_based_ = self.difference_based == "on" or (
            self.difference_based == "auto"
            and has_overlapping_or_negated_conditions(search.conditions)
        )
        self._batched_differences = (
            self.difference_based_ and search.batched and qf.batched_differences
        )
        everything = np.ones(target.n_samples_, dtype=bool)
        root = qf.value(everything, target)
        if not np.isfinite(root):
            raise ValueError(f"the whole dataset must have a finite value, got {root}")
        self.n_evaluated_ = 0
        self._tables: dict[tuple[int, ...], _Table] = {}
        if self._batched_differences:
            self._expand(search, (), None, root, np.inf, qf.difference_state(everything, target))
        else:
            self._expand(search, (), None, root, None, None)
        self.n_tables_ = len(self._tables)
        del self._tables
        return search.results()

    def _host_mask(self, search: _Search, indices: tuple[int, ...]) -> np.ndarray:
        if not indices:
            return np.ones(search.task.target.n_samples_, dtype=bool)
        return search.mask(indices)

    def _expand(self, search: _Search, node: tuple[int, ...], node_mask, best: float,
                differences, state) -> None:
        """Evaluate the children of ``node`` (``B(node) = best``) and descend into them.

        ``differences`` are the merged differences of ``node`` and ``state`` its
        difference state (batched), or None.
        """
        qf, target = search.task.qf, search.task.target
        candidates = np.asarray(search.refinements(node), dtype=np.intp)
        threshold = search.result.min_required_quality()
        # The tables of the G - g decide which children are evaluated at all.
        allowed = np.ones(len(candidates), dtype=bool)
        lookups = []
        for j in range(len(node)):
            general = node[:j] + node[j + 1:]
            table = self._tables.get(general)
            if table is None or not len(table.candidates):
                # No promising child, or the estimate of G - g fell to the threshold
                # before it was expanded.
                return
            positions, promising = table.lookup(candidates, threshold)
            allowed &= promising
            lookups.append((general, table, positions))
        candidates = candidates[allowed]
        if not len(candidates):
            return
        reference = np.full(len(candidates), best)
        generalizations = []
        for general, table, positions in lookups:
            positions = positions[allowed]
            reference = np.maximum(reference, table.best[positions])
            generalizations.append((general, table, positions))
        children = search.expand(node, node_mask, candidates)
        selected = np.flatnonzero([children.admissible(i) for i in range(len(candidates))])
        if not len(selected):
            return
        self.n_evaluated_ += len(selected)
        statistics = children.statistics if search.batched else None
        if search.batched:
            with np.errstate(divide="ignore", invalid="ignore"):
                quality = qf.evaluate_many(statistics, target, reference=reference)
        else:
            quality = np.full(len(candidates), np.nan)
            for i in selected:
                quality[i] = qf.evaluate(children.mask(i), target, reference=reference[i])
        for i in selected:
            indices = (*node, int(candidates[i]))
            if search.result.accepts(quality[i], indices) and children.passes_result_filters(
                i, indices
            ):
                search.result.add(float(quality[i]), indices)
        if len(node) + 1 >= search.task.depth:
            return
        if search.batched:
            values = qf.value_many(statistics, target)
        else:
            values = np.full(len(candidates), np.nan)
            for i in selected:
                values[i] = qf.value(children.mask(i), target)
        bests = np.fmax(values, reference)
        summaries = states = None
        if self._batched_differences:
            states = qf.difference_states_many(statistics, target)
            parts = [qf.differences_many(state, states), np.full(len(candidates), differences)]
            for _, table, positions in generalizations:
                parts.append(qf.differences_many(table.states[positions], states))
                parts.append(table.differences[positions])
            summaries = qf.merge_differences_many(parts)
            with np.errstate(divide="ignore", invalid="ignore"):
                estimates = qf.optimistic_estimate_many(statistics, target, reference=bests,
                                                        differences=summaries)
        elif self.difference_based_:
            # Differences need the rows of the generalizations: host masks, one child at a time.
            host = self._host_mask(search, node)
            pairs = [(self._host_mask(search, general), table, positions)
                     for general, table, positions in generalizations]
            summaries = [None] * len(candidates)
            estimates = np.full(len(candidates), -np.inf)
            for i in selected:
                condition = search.masks[candidates[i]]
                mask = host & condition
                parts = [qf.difference(host, mask, target), differences]
                for general_mask, table, positions in pairs:
                    parts.append(qf.difference(general_mask & condition, mask, target))
                    parts.append(table.differences[positions[i]])
                summaries[i] = qf.merge_differences(parts)
                estimates[i] = qf.optimistic_estimate(mask, target, reference=bests[i],
                                                      differences=summaries[i])
        elif search.batched:
            with np.errstate(divide="ignore", invalid="ignore"):
                estimates = qf.optimistic_estimate_many(statistics, target, reference=bests)
        else:
            estimates = np.full(len(candidates), -np.inf)
            for i in selected:
                estimates[i] = qf.optimistic_estimate(children.mask(i), target,
                                                      reference=bests[i])
        estimates = np.asarray(estimates, dtype=float)
        threshold = search.result.min_required_quality()
        promising = selected[estimates[selected] > threshold]
        if isinstance(summaries, list):
            summaries = [summaries[i] for i in promising]
        elif summaries is not None:
            summaries = summaries[promising]
        self._tables[node] = table = _Table(
            candidates[promising], bests[promising], estimates[promising], summaries,
            None if states is None else states[promising],
        )
        for position in range(len(promising) - 1, -1, -1):
            if not table.estimates[position] > search.result.min_required_quality():
                continue
            self._expand(
                search, (*node, int(table.candidates[position])),
                children.mask(promising[position]), table.best[position],
                None if summaries is None else summaries[position],
                None if states is None else table.states[position],
            )
