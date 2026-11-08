"""Subgroup discovery search algorithms: DFS, best-first, OPUS, DpSubgroup, Apriori, SD-Map, VLSD,
beam search.

A ``SubgroupDiscoveryTask`` combines a dataset, a target fitted on it, candidate
``Condition``s, and a quality function. ``algorithm.execute(task)`` returns the
best subgroups as ``(quality, Rule)`` pairs, sorted by decreasing quality.

Subgroups are conjunctions with at most one condition per feature, never empty
and never on target attributes. Only subgroups that pass all ``filters`` and
have a quality above ``min_quality`` are reported. Anti-monotone filters and the
optimistic estimates of bounded quality functions prune the search. DFS,
best-first search, OPUS, DpSubgroup, Apriori, SD-Map, SD-Map*, and VLSD are
exhaustive and find the same qualities; beam search is a heuristic.

Evaluation: DFS, best-first search, OPUS, DpSubgroup, Apriori, and beam search
evaluate all refinements of a node together (``_Search.expand``) and then take
their decisions child by child in the order of the scalar algorithm. With a
batchable quality function and batchable pruning filters (see
``targets.base``), the statistics of all children come from one pass over the
condition masks, on the CPU or, with ``device="cuda"``, on a GPU (CuPy);
otherwise each child is evaluated on its own row mask. SD-Map and VLSD always
evaluate one subgroup at a time on the CPU.

Provenance: the control flow of ``DFS``, ``BestFirstSearch``, ``Apriori``, and
``BeamSearch``, the task parameters, and the top-k result handling
(``minimum_required_quality``/``add_if_required``) are adapted from pysubgroup 0.x's
``algorithms.py`` and ``utils.py`` (https://github.com/flemmerich/pysubgroup,
Apache-2.0, Florian Lemmerich et al.), rewritten for condition masks and filters.
``OPUS``, ``DpSubgroup``, ``VLSD``, and ``SDMap`` are written from their papers without
reusing code.
"""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
import pandas as pd

from .._evaluation import DEVICES, Backend, BatchedChildren, ScalarChildren
from ..encoding import Encoder, _column_index
from ..filters import Filter
from ..rules import Condition, Rule
from ..targets import AreaQF, BaseTarget, BoundedQualityFunction, QualityFunction


@dataclass
class SubgroupDiscoveryTask:
    """Parameters of a top-k subgroup discovery search.

    ``batched`` selects how the children of a search node are evaluated:
    ``None`` (default) batches whenever the algorithm, the quality function, and
    all pruning filters support it, ``True`` requires batching, and ``False``
    evaluates one subgroup at a time. ``device="cuda"`` keeps the condition masks
    on a CUDA GPU and requires batching (``pip install pysubgroup[gpu]``).
    """

    data: pd.DataFrame
    target: BaseTarget
    search_space: list[Condition]
    qf: QualityFunction
    result_set_size: int = 10
    depth: int = 3
    min_quality: float = float("-inf")
    filters: list[Filter] = field(default_factory=list)
    batched: bool | None = None
    device: str = "cpu"


class _Entry:
    """Result entry ordered by quality; ties prefer lexicographically smaller indices."""

    __slots__ = ("quality", "indices")

    def __init__(self, quality: float, indices: tuple[int, ...]):
        self.quality = quality
        self.indices = indices

    def __lt__(self, other: _Entry) -> bool:
        return (self.quality, other.indices) < (other.quality, self.indices)


class _TopK:
    """Bounded min-heap of the best entries seen so far."""

    def __init__(self, size: int, min_quality: float):
        self.size = size
        self.min_quality = min_quality
        self.heap: list[_Entry] = []
        self.members: set[tuple[int, ...]] = set()

    def min_required_quality(self) -> float:
        return self.heap[0].quality if len(self.heap) >= self.size else self.min_quality

    def accepts(self, quality: float, indices: tuple[int, ...]) -> bool:
        return (
            quality >= self.min_quality
            and indices not in self.members
            and (len(self.heap) < self.size or self.heap[0] < _Entry(quality, indices))
        )

    def add(self, quality: float, indices: tuple[int, ...]) -> None:
        if not self.accepts(quality, indices):
            return
        entry = _Entry(quality, indices)
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, entry)
        else:
            self.members.discard(heapq.heapreplace(self.heap, entry).indices)
        self.members.add(indices)

    def best(self, size: int) -> list[_Entry]:
        kept = [entry for entry in self.heap if entry.quality > self.min_quality]
        return sorted(kept, reverse=True)[:size]


class _Search:
    """Shared state of one search: condition masks, filters, quality evaluation, and results."""

    def __init__(self, task: SubgroupDiscoveryTask, batch_support: bool = True):
        """``batch_support`` is False for algorithms that never evaluate batches."""
        target, data = task.target, task.data
        if not hasattr(target, "n_samples_"):
            raise ValueError("target must be fitted on the task data")
        if not isinstance(data, pd.DataFrame) or len(data) != target.n_samples_:
            raise ValueError("task data must be the DataFrame the target was fitted on")
        if task.depth < 1 or task.result_set_size < 1:
            raise ValueError("depth and result_set_size must be positive")
        excluded = set(target.attributes_)
        columns = list(data.columns)
        # A column's name and its position are one feature: conditions use the
        # spelling seen first for their column.
        spelling: dict[int, str | int] = {}
        conditions = []
        for condition in task.search_space:
            position = _column_index(condition.feature, columns)
            if columns[position] in excluded:
                continue
            feature = spelling.setdefault(position, condition.feature)
            if feature != condition.feature:
                condition = replace(condition, feature=feature)
            conditions.append(condition)
        # Group conditions by feature (first-seen order), so that refinements
        # only add conditions on later features.
        group_of_feature: dict = {}
        for condition in conditions:
            group_of_feature.setdefault(condition.feature, len(group_of_feature))
        conditions.sort(key=lambda condition: group_of_feature[condition.feature])
        self.conditions = conditions
        groups = np.array([group_of_feature[c.feature] for c in conditions], dtype=int)
        self.groups = groups
        # Refinements of a subgroup ending in condition i start at next_start[i].
        self.next_start = np.searchsorted(groups, groups, side="right")
        if conditions:
            coverage = Encoder(conditions).fit_transform(data)
            self.masks = np.array([coverage.condition_mask(i) for i in range(len(conditions))])
        else:
            self.masks = np.empty((0, len(data)), dtype=bool)
        self.task = task
        self.pruning_filters = [f for f in task.filters if f.anti_monotone]
        self.result_filters = [f for f in task.filters if not f.anti_monotone]
        self.result = _TopK(task.result_set_size, task.min_quality)
        self.is_area = isinstance(task.qf, AreaQF)
        self.is_bounded = isinstance(task.qf, BoundedQualityFunction)
        self.batched = self._batching(task, batch_support)
        self.backend = Backend(self.masks, task.device) if self.batched else None

    def _batching(self, task: SubgroupDiscoveryTask, batch_support: bool) -> bool:
        if task.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {task.device!r}")
        if task.batched not in (None, True, False):
            raise ValueError("batched must be None, True, or False")
        if not batch_support:
            reason = "this algorithm evaluates one subgroup at a time"
        elif not task.qf.batchable:
            reason = f"{type(task.qf).__name__} does not evaluate batches"
        else:
            unbatchable = [type(f).__name__ for f in self.pruning_filters if not f.batchable]
            reason = f"filters {unbatchable} do not evaluate batches" if unbatchable else None
        if task.device != "cpu" and (task.batched is False or reason):
            raise ValueError(f"device={task.device!r} requires batched evaluation: "
                             f"{reason or 'batched=False'}")
        if task.batched and reason:
            raise ValueError(f"batched=True is not possible: {reason}")
        return task.batched is not False and reason is None

    def expand(self, parent: tuple[int, ...], parent_mask, candidates):
        """The refinements of ``parent`` (rows ``parent_mask``, None for all) by ``candidates``."""
        children = BatchedChildren if self.batched else ScalarChildren
        return children(self, parent, parent_mask, candidates)

    def node_mask(self, indices: tuple[int, ...]):
        """Row mask of a subgroup on the evaluation device; None for the empty description."""
        if not indices:
            return None
        return self.backend.mask(indices) if self.batched else self.mask(indices)

    def add_child(self, children, i: int) -> float:
        """Offer child ``i`` of an expansion to the result set; returns its quality."""
        indices = tuple(sorted(children.indices(i)))
        quality = children.quality(i)
        if self.result.accepts(quality, indices) and children.passes_result_filters(i, indices):
            self.result.add(quality, indices)
        return quality

    def refinements(self, indices: tuple[int, ...]) -> range:
        start = self.next_start[indices[-1]] if indices else 0
        return range(start, len(self.conditions))

    def mask(self, indices: tuple[int, ...]) -> np.ndarray:
        return np.logical_and.reduce(self.masks[list(indices)], axis=0)

    def _passes(self, filters: list[Filter], mask: np.ndarray, indices: tuple[int, ...]) -> bool:
        if not filters:
            return True
        conditions = tuple(self.conditions[i] for i in sorted(indices))
        return all(f.is_satisfied(mask, self.task.target, conditions) for f in filters)

    def admissible(self, mask: np.ndarray, indices: tuple[int, ...]) -> bool:
        """Non-empty and passing all anti-monotone filters; otherwise prune."""
        return bool(mask.any()) and self._passes(self.pruning_filters, mask, indices)

    def add(self, quality: float, mask: np.ndarray, indices: tuple[int, ...]) -> None:
        """Offer a subgroup to the result set, checking the other filters."""
        indices = tuple(sorted(indices))
        if self.result.accepts(quality, indices) and self._passes(
            self.result_filters, mask, indices
        ):
            self.result.add(quality, indices)

    def evaluate(self, mask: np.ndarray, indices: tuple[int, ...]) -> float:
        if isinstance(self.task.qf, AreaQF):
            return self.task.qf.evaluate(mask, self.task.target, depth=len(indices))
        return self.task.qf.evaluate(mask, self.task.target)

    def optimistic_estimate(self, mask: np.ndarray) -> float:
        if isinstance(self.task.qf, BoundedQualityFunction):
            return self.task.qf.optimistic_estimate(mask, self.task.target)
        return float("inf")

    def results(self) -> list[tuple[float, Rule]]:
        return [
            (entry.quality, Rule(tuple(self.conditions[i] for i in entry.indices)))
            for entry in self.result.best(self.task.result_set_size)
        ]


# Adapted from pysubgroup 0.x's DFS (algorithms.py, Apache-2.0).
class DFS:
    """Exhaustive depth-first search with optimistic-estimate pruning."""

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task)
        self._expand(search, (), None)
        return search.results()

    def _expand(self, search: _Search, parent: tuple[int, ...], parent_mask) -> None:
        # Children are visited in order, each subtree before the next sibling.
        children = search.expand(parent, parent_mask, search.refinements(parent))
        for i in range(len(children.candidates)):
            if not children.admissible(i):
                continue
            if not children.estimate(i) > search.result.min_required_quality():
                continue
            search.add_child(children, i)
            indices = children.indices(i)
            if len(indices) < search.task.depth:
                self._expand(search, indices, children.mask(i))


# Adapted from pysubgroup 0.x's BestFirstSearch (algorithms.py, Apache-2.0).
class BestFirstSearch:
    """Exhaustive search expanding the subgroup with the highest optimistic estimate first."""

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task)
        queue: list[tuple[float, tuple[int, ...]]] = [(float("-inf"), ())]
        while queue:
            negative_estimate, parent = heapq.heappop(queue)
            if not -negative_estimate > search.result.min_required_quality():
                break
            children = search.expand(
                parent, search.node_mask(parent), search.refinements(parent)
            )
            for i in range(len(children.candidates)):
                if not children.admissible(i):
                    continue
                search.add_child(children, i)
                indices = children.indices(i)
                if len(indices) < task.depth:
                    estimate = children.estimate(i)
                    if estimate >= search.result.min_required_quality():
                        heapq.heappush(queue, (-estimate, indices))
        return search.results()


class OPUS:
    """OPUS_o: Optimized Pruning for Unordered Search, optimization variant.

    Webb, G. I.: OPUS: An Efficient Admissible Algorithm for Unordered Search.
    Journal of Artificial Intelligence Research 3:431-465, 1995
    (https://doi.org/10.1613/jair.227). Implemented from Figure 10 of the paper.

    Like best-first search, OPUS expands the open subgroup with the highest
    optimistic estimate first. Unlike our fixed-order search (Section 3 of the
    paper), every node keeps its own set of *active* operators (conditions) for
    the subtree below it, which enables two improvements:

    - When a child is pruned, its condition is removed from the active
      operators of all its siblings, as every node containing it refines the
      pruned child. Fixed-order search re-evaluates and prunes that condition
      again below each sibling.
    - The remaining conditions are allocated so that the children with the
      lowest optimistic estimates, which are most likely to be pruned before
      expansion, receive the largest subtrees.

    Adaptations: the pruning threshold value(BEST) of the paper is the top-k
    minimum required quality, which also implements the paper's "lower limit on
    an acceptable solution" (Section 5.2.1); a child's active operators exclude
    conditions on its features (one condition per feature); anti-monotone
    filters prune like optimistic estimates; and ``depth`` limits the
    description size. Each queued subgroup stores its active conditions, so
    memory can exceed that of ``BestFirstSearch``.
    """

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task)
        tiebreak = itertools.count()
        # Step 1: OPEN holds the start node (the empty description) with all
        # operators active. Entries are (-optimistic estimate, tiebreak,
        # description, active operators); equal estimates are expanded
        # last-in-first-out, as in the paper's experiments (Section 7).
        open_nodes = [(float("-inf"), 0, (), tuple(range(len(search.conditions))))]
        while open_nodes:
            # Steps 3-4: expand the open node with the highest optimistic estimate.
            # Step 7(b)ii prunes OPEN whenever the threshold rises; with a priority
            # queue this is done lazily: once the best open estimate no longer
            # exceeds the threshold, no open node can contribute any more.
            negative_estimate, _, node, active = heapq.heappop(open_nodes)
            if not -negative_estimate > search.result.min_required_quality():
                break
            remaining = list(active)  # step 5: RemainingOperators
            children = []  # step 6: NewNodes
            # Step 7: generate and evaluate every child; better children update
            # the top-k results (step 7b).
            expansion = search.expand(node, search.node_mask(node), active)
            for i, operator in enumerate(active):
                if not expansion.admissible(i):
                    remaining.remove(operator)  # empty or anti-monotone filter fails
                    continue
                search.add_child(expansion, i)
                children.append((expansion.estimate(i), operator, expansion.indices(i)))
            # Step 8: prune children that cannot reach the results and remove their
            # operators from RemainingOperators, i.e. from all siblings' subtrees.
            threshold = search.result.min_required_quality()
            kept = []
            for estimate, operator, child in children:
                if estimate > threshold:
                    kept.append((estimate, operator, child))
                else:
                    remaining.remove(operator)
            # Step 9: allocate operators, lowest optimistic estimate first, so the
            # least promising children get the largest subtrees.
            kept.sort(key=lambda item: item[0])
            for estimate, operator, child in kept:
                remaining.remove(operator)
                used = {search.groups[i] for i in child}
                child_active = tuple(r for r in remaining if search.groups[r] not in used)
                # Step 10: queue children that can still be refined (optimistic
                # pruning against the threshold happened in step 8).
                if len(child) < task.depth and child_active:
                    heapq.heappush(
                        open_nodes, (-estimate, -next(tiebreak), child, child_active)
                    )
        return search.results()


class DpSubgroup:
    """DpSubgroup: optimistic-estimate pruning with look-ahead and branch reordering.

    Grosskreutz, H., Rueping, S., Shaabani, N., Wrobel, S.: Optimistic Estimate
    Pruning Strategies for Fast Exhaustive Subgroup Discovery. Technical report,
    Fraunhofer IAIS, 2008. Implemented from Algorithm 1 and Section 3.

    Conditions (the paper's terms) are ordered by decreasing support. Each call
    of the recursive procedure works on a set of candidate conditions, the
    analogue of the paper's conditional FP-tree, and runs two loops:

    - Look-ahead: every refinement of the current subgroup by a candidate is
      evaluated and offered to the top-k results, which raises the minimum
      required quality as early as possible, and its optimistic estimate is
      stored.
    - Branch reordering: only then does the search descend, in the order of the
      ``exploration`` strategy, into the refinements whose optimistic estimate
      exceeds the current minimum required quality. Below a condition ``t``, the
      candidates are the conditions more frequent than ``t`` whose stored
      estimate still exceeds it, so a pruned refinement removes its condition
      from all sibling branches.

    Exploration strategies (Section 3.3):

    - ``"frequency"``: decreasing support;
    - ``"optimistic_estimate"``: highest optimistic estimate first;
    - ``"term_rank"`` (default; best in the paper's experiments): highest sum of
      the qualities of the current top-k subgroups that contain the condition
      first.

    Ties keep the frequency order. With ``iterative_deepening`` the search is
    repeated with depth limits 1, 2, ..., ``depth``, each starting from the
    results so far; the paper found that this never helps.

    Deviations and adaptations:

    - Boolean condition masks replace FP-trees; the paper notes that its pruning
      and exploration strategies do not depend on FP-trees (Section 6). The
      frequency order is the support on the whole dataset, with ties broken by
      condition index; the paper does not say in which direction
      ``"frequency"`` traverses, and the most frequent condition comes first
      here.
    - ``optEstim`` is the task's quality function: for ``StandardQF(a=1)``
      (Piatetsky-Shapiro, up to the constant ``1/N``) its optimistic estimate is
      the paper's ``n * p * (1 - p0)`` (Section 4.1). Its multi-class
      estimates are those of ``SplitQF``, ``MulticlassChiSquaredQF`` and
      ``GiniQF`` (``OE_Lin`` and ``OE_Gini``, Sections 4.2 and 4.3).
    - The minimum generality ``minG`` is generalized to anti-monotone filters,
      which prune like it; other filters only filter results.
    - Like the other exhaustive algorithms, a branch is pruned when its
      estimate does not exceed the minimum required quality, and subgroups use
      at most one condition per feature.
    - ``"term_rank"`` uses the current top-k results, not the optional longer
      queue of footnote 3.
    """

    STRATEGIES = ("frequency", "optimistic_estimate", "term_rank")

    def __init__(self, exploration: str = "term_rank", iterative_deepening: bool = False):
        if exploration not in self.STRATEGIES:
            raise ValueError(f"exploration must be one of {self.STRATEGIES}, got {exploration!r}")
        self.exploration = exploration
        self.iterative_deepening = iterative_deepening

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task)
        supports = search.masks.sum(axis=1)
        terms = sorted(range(len(search.conditions)), key=lambda i: (-supports[i], i))
        for depth in range(1, task.depth + 1) if self.iterative_deepening else (task.depth,):
            self._dp_subgroup(search, terms, (), None, depth)
        return search.results()

    def _dp_subgroup(
        self,
        search: _Search,
        terms: list[int],
        prefix: tuple[int, ...],
        prefix_mask,
        depth: int,
    ) -> None:
        # First loop (look-ahead): evaluate every refinement before descending.
        estimates: dict[int, float] = {}
        children = search.expand(prefix, prefix_mask, terms)
        child_of: dict[int, int] = {}
        for i, term in enumerate(terms):
            if not children.admissible(i):
                continue  # too small or failing an anti-monotone filter
            search.add_child(children, i)
            estimates[term] = children.estimate(i)
            child_of[term] = i
        if len(prefix) + 1 >= depth:
            return
        # Second loop: descend in the order of the exploration strategy. The
        # threshold (the paper's minQ) is re-read before every branch, as
        # recursive calls raise it.
        candidates = list(estimates)  # admissible terms in frequency order
        position = {term: i for i, term in enumerate(candidates)}
        for term in self._recurse_order(search, candidates, estimates, position):
            threshold = search.result.min_required_quality()
            if not estimates[term] > threshold:
                continue
            group = search.groups[term]
            conditional = [
                other
                for other in candidates[: position[term]]
                if estimates[other] > threshold and search.groups[other] != group
            ]
            if conditional:
                self._dp_subgroup(
                    search, conditional, (*prefix, term), children.mask(child_of[term]), depth
                )

    def _recurse_order(
        self,
        search: _Search,
        candidates: list[int],
        estimates: dict[int, float],
        position: dict[int, int],
    ) -> list[int]:
        if self.exploration == "frequency":
            return candidates
        if self.exploration == "optimistic_estimate":
            return sorted(candidates, key=lambda t: (-estimates[t], position[t]))
        rank: dict[int, float] = defaultdict(float)
        for entry in search.result.heap:
            for term in entry.indices:
                rank[term] += entry.quality
        return sorted(candidates, key=lambda t: (-rank[t], position[t]))


# Adapted from pysubgroup 0.x's Apriori (algorithms.py, Apache-2.0).
class Apriori:
    """Exhaustive level-wise search.

    A candidate of length ``k + 1`` is generated only if all its subsets of
    length ``k`` were promising, i.e. passed the anti-monotone filters with an
    optimistic estimate above the current minimum required quality.
    """

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task)
        # A level is a list of (parent, parent mask, candidate conditions); its
        # candidates are the parent's refinements, in the order of generation.
        level = [((), None, list(range(len(search.conditions))))]
        depth = 1
        while level:
            promising = []
            for parent, parent_mask, candidates in level:
                children = search.expand(parent, parent_mask, candidates)
                for i in range(len(candidates)):
                    if not children.admissible(i):
                        continue
                    search.add_child(children, i)
                    estimate = children.estimate(i)
                    if estimate >= search.result.min_required_quality():
                        promising.append((estimate, children, i))
            if depth == task.depth:
                break
            minimum = search.result.min_required_quality()
            promising_masks = {
                children.indices(i): children.mask(i)
                for estimate, children, i in promising
                if estimate > minimum
            }
            level = self._next_level(search, promising_masks)
            depth += 1
        return search.results()

    @staticmethod
    def _next_level(search: _Search, promising: dict[tuple[int, ...], object]) -> list[tuple]:
        by_prefix = defaultdict(list)
        for indices in promising:
            by_prefix[indices[:-1]].append(indices[-1])
        level = []
        for prefix, suffixes in by_prefix.items():
            suffixes = sorted(suffixes)
            for position, first in enumerate(suffixes):
                parent = (*prefix, first)
                candidates = [
                    second
                    for second in suffixes[position + 1:]
                    if second >= search.next_start[first]  # not the same feature
                    and all(
                        subset in promising
                        for subset in combinations((*parent, second), len(parent))
                    )
                ]
                if candidates:
                    level.append((parent, promising[parent], candidates))
        return level


class _FPNode:
    """FP-tree node; ``chunks`` hold the row positions of all paths through it."""

    __slots__ = ("item", "parent", "children", "chunks")

    def __init__(self, item: int | None, parent: _FPNode | None):
        self.item = item
        self.parent = parent
        self.children: dict[int, _FPNode] = {}
        self.chunks: list[np.ndarray] = []


class SDMap:
    """SD-Map: exhaustive FP-growth search (Atzmueller and Puppe, PKDD 2006).

    Rows are inserted into an FP-tree as paths of the conditions they satisfy,
    ordered by decreasing support. Each subgroup is mined from the conditional
    tree of its prefix, whose nodes keep the covered rows, so any quality
    function can score the subgroup. Conditions failing an anti-monotone
    filter never enter a tree.

    With ``use_optimistic_estimates`` (SD-Map*, Atzmueller and Lemmerich,
    PKDD 2009; Lemmerich, Atzmueller and Puppe, DMKD 2016, Section 5.3.1):

    - Selector pruning: a conditional tree is mined only if the optimistic
      estimate of its prefix exceeds the current minimum required quality.
    - Header pruning: when a conditional tree is built, items whose refinement
      of the prefix has an optimistic estimate that does not exceed it are left
      out of the tree.
    - Look-ahead and reordering: all header items of a tree are evaluated
      first, then mined by decreasing optimistic estimate (ties in header
      order), so that good subgroups raise the threshold early.
    - Refinements at the depth limit are evaluated directly from the prefix
      paths, without a conditional tree or optimistic estimates.

    Since nodes keep their rows rather than aggregated target statistics, any
    bound works here, including the ordering-based ones that the paper's
    FP-trees cannot compute.
    """

    def __init__(self, use_optimistic_estimates: bool = False):
        self.use_optimistic_estimates = use_optimistic_estimates

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task, batch_support=False)
        supports = search.masks.sum(axis=1)
        items = [
            index
            for index in range(len(search.conditions))
            if search.admissible(search.masks[index], (index,))
        ]
        items.sort(key=lambda index: -supports[index])
        root = _FPNode(None, None)
        header: dict[int, list[_FPNode]] = defaultdict(list)
        rows: dict[_FPNode, list[int]] = defaultdict(list)
        if items:
            order = np.array(items)
            transactions = search.masks[order].T
            for row, covered in enumerate(transactions):
                node = root
                for item in order[covered]:
                    child = node.children.get(item)
                    if child is None:
                        child = node.children[item] = _FPNode(int(item), node)
                        header[int(item)].append(child)
                    rows[child].append(row)
                    node = child
        for node, node_rows in rows.items():
            node.chunks.append(np.array(node_rows, dtype=np.intp))
        self._mine(search, header, ())
        return search.results()

    def _mine(
        self, search: _Search, header: dict[int, list[_FPNode]], prefix: tuple[int, ...],
        known: dict[int, tuple[np.ndarray, float]] | None = None,
    ) -> None:
        if self.use_optimistic_estimates:
            self._mine_with_estimates(search, header, prefix, known or {})
            return
        task = search.task
        for item, nodes in header.items():
            mask = self._item_mask(search, nodes)
            indices = tuple(sorted((*prefix, item)))
            if not search.admissible(mask, indices):
                continue
            search.add(search.evaluate(mask, indices), mask, indices)
            if len(indices) >= task.depth:
                continue
            conditional = self._conditional_tree(search, nodes, {search.groups[i] for i in indices})
            if conditional:
                self._mine(search, conditional, (*prefix, item))

    @staticmethod
    def _item_mask(search: _Search, nodes: list[_FPNode]) -> np.ndarray:
        covered = np.concatenate([chunk for node in nodes for chunk in node.chunks])
        mask = np.zeros(len(search.task.data), dtype=bool)
        mask[covered] = True
        return mask

    def _mine_with_estimates(
        self, search: _Search, header: dict[int, list[_FPNode]], prefix: tuple[int, ...],
        known: dict[int, tuple[np.ndarray, float]],
    ) -> None:
        """SD-Map*: evaluate all header items, then mine them by decreasing estimate.

        ``known`` holds the masks and estimates that header pruning computed.
        """
        task = search.task
        items = []
        for position, (item, nodes) in enumerate(header.items()):
            indices = tuple(sorted((*prefix, item)))
            if item in known:
                mask, estimate = known[item]
            else:
                mask = self._item_mask(search, nodes)
                if not search.admissible(mask, indices):
                    continue
                estimate = search.optimistic_estimate(mask)
            search.add(search.evaluate(mask, indices), mask, indices)
            items.append((estimate, position, item, nodes, indices))
        items.sort(key=lambda entry: (-entry[0], entry[1]))
        for estimate, _, item, nodes, indices in items:
            if len(indices) >= task.depth:
                continue
            if not estimate > search.result.min_required_quality():
                continue
            excluded = {search.groups[i] for i in indices}
            if len(indices) + 1 == task.depth:
                # The refinements are leaves: evaluate them without a tree or estimates.
                for other, mask in self._prefix_rows(search, nodes, excluded).items():
                    leaf = tuple(sorted((*indices, other)))
                    if search.admissible(mask, leaf):
                        search.add(search.evaluate(mask, leaf), mask, leaf)
                continue
            conditional, estimates = self._pruned_conditional_tree(
                search, nodes, indices, excluded
            )
            if conditional:
                self._mine(search, conditional, (*prefix, item), estimates)

    @staticmethod
    def _conditional_tree(
        search: _Search, nodes: list[_FPNode], excluded_groups: set[int],
        kept: set[int] | None = None,
    ) -> dict[int, list[_FPNode]]:
        """Tree of the prefix paths above ``nodes``, without features already used.

        If ``kept`` is given, only those items enter the tree.
        """
        root = _FPNode(None, None)
        header: dict[int, list[_FPNode]] = defaultdict(list)
        for node in nodes:
            path = []
            ancestor = node.parent
            while ancestor.item is not None:
                if search.groups[ancestor.item] not in excluded_groups and (
                    kept is None or ancestor.item in kept
                ):
                    path.append(ancestor.item)
                ancestor = ancestor.parent
            current = root
            for item in reversed(path):
                child = current.children.get(item)
                if child is None:
                    child = current.children[item] = _FPNode(item, current)
                    header[item].append(child)
                child.chunks.extend(node.chunks)
                current = child
        return header

    @staticmethod
    def _prefix_rows(
        search: _Search, nodes: list[_FPNode], excluded_groups: set[int]
    ) -> dict[int, np.ndarray]:
        """Row masks of ``prefix & item`` for every item on the prefix paths above ``nodes``."""
        rows: dict[int, list[np.ndarray]] = defaultdict(list)
        for node in nodes:
            ancestor = node.parent
            while ancestor.item is not None:
                if search.groups[ancestor.item] not in excluded_groups:
                    rows[ancestor.item].extend(node.chunks)
                ancestor = ancestor.parent
        masks = {}
        for item, chunks in rows.items():
            mask = np.zeros(len(search.task.data), dtype=bool)
            mask[np.concatenate(chunks)] = True
            masks[item] = mask
        return masks

    def _pruned_conditional_tree(
        self, search: _Search, nodes: list[_FPNode], indices: tuple[int, ...],
        excluded_groups: set[int],
    ) -> tuple[dict[int, list[_FPNode]], dict[int, tuple[np.ndarray, float]]]:
        """Header pruning: a conditional tree without items that cannot reach the results.

        Rows of ``prefix & item`` are collected for every item on the prefix
        paths; items that fail the anti-monotone filters or whose optimistic
        estimate does not exceed the minimum required quality are left out of
        the tree, also as ancestors on other paths.
        """
        known = {}
        for item, mask in self._prefix_rows(search, nodes, excluded_groups).items():
            if not search.admissible(mask, tuple(sorted((*indices, item)))):
                continue
            estimate = search.optimistic_estimate(mask)
            if estimate > search.result.min_required_quality():
                known[item] = (mask, estimate)
        if not known:
            return {}, known
        return self._conditional_tree(search, nodes, excluded_groups, set(known)), known


class SDMapStar(SDMap):
    """SD-Map* : SD-Map with optimistic-estimate pruning of conditional trees."""

    def __init__(self):
        super().__init__(use_optimistic_estimates=True)


@dataclass
class _VerticalList:
    """A subgroup in vertical representation (Lopez-Martinez-Carrasco et al., Section 4).

    The paper stores the description with the bitsets of the true and false
    positive instance IDs. One bitset of covered rows generalizes this to any
    target; ``refine`` concatenates descriptions and intersects the bitsets.
    """

    description: tuple[int, ...]
    rows: np.ndarray
    quality: float = float("nan")
    estimate: float = float("inf")

    def refine(self, other: _VerticalList) -> _VerticalList:
        """Extend this description by the non-common suffix of ``other`` (Definition 4)."""
        return _VerticalList((*self.description, other.description[-1]), self.rows & other.rows)


class VLSD:
    """VLSD: Vertical List Subgroup Discovery.

    Lopez-Martinez-Carrasco, A., Juarez, J. M., Campos, M., Canovas-Segura, B.:
    VLSD - An Efficient Subgroup Discovery Algorithm Based on Equivalence Classes
    and Optimistic Estimate. Algorithms 16(6):274, 2023 (CC BY 4.0).

    Implemented from the paper's Algorithms 1-3:

    - ``GENERATE_SUBGROUPS_S1``: the subgroups of one selector whose optimistic
      estimate reaches the threshold, sorted by ``sort_criterion_in_S1``.
    - A triangular matrix ``M`` of all two-selector subgroups that cover instances
      and whose optimistic estimate reaches the threshold.
    - For every selector of S1, its equivalence class ``M[selector]`` is evaluated,
      sorted by ``sort_criterion_in_other_sizes``, and explored by ``SEARCH``:
      each member is refined with every later member, skipping pairs whose last
      selectors ``M`` already rules out.

    The paper uses fixed quality and optimistic-estimate thresholds and no depth
    limit. Here both thresholds are the task's current top-k minimum (subgroups
    are evaluated only if their estimate exceeds it), ``depth`` limits the
    description size, patterns use each attribute at most once (as the paper's
    patterns do), and ``M`` keeps the estimates of the pairs, whose bitsets are
    recomputed per equivalence class to save memory. The sort criteria (the
    paper's ``sort_criterion_in_S1`` and ``sort_criterion_in_other_sizes``) order
    by quality: ``"quality-ascending"``, ``"quality-descending"``, or ``"no-order"``.
    """

    SORT_CRITERIA = ("no-order", "quality-ascending", "quality-descending")

    def __init__(
        self,
        sort_criterion_in_s1: str = "no-order",
        sort_criterion_in_other_sizes: str = "no-order",
    ):
        for criterion in (sort_criterion_in_s1, sort_criterion_in_other_sizes):
            if criterion not in self.SORT_CRITERIA:
                raise ValueError(f"sort criteria must be one of {self.SORT_CRITERIA}")
        self.sort_criterion_in_s1 = sort_criterion_in_s1
        self.sort_criterion_in_other_sizes = sort_criterion_in_other_sizes

    @staticmethod
    def _sort(subgroups: list[_VerticalList], criterion: str) -> list[_VerticalList]:
        if criterion == "no-order":
            return subgroups
        return sorted(subgroups, key=lambda s: s.quality,
                      reverse=criterion == "quality-descending")

    @staticmethod
    def _passes(search: _Search, subgroup: _VerticalList) -> bool:
        """Covers instances (and passes the anti-monotone filters), estimate above threshold."""
        if not search.admissible(subgroup.rows, subgroup.description):
            return False
        subgroup.estimate = search.optimistic_estimate(subgroup.rows)
        return subgroup.estimate > search.result.min_required_quality()

    @staticmethod
    def _evaluate(search: _Search, subgroup: _VerticalList) -> None:
        subgroup.quality = search.evaluate(subgroup.rows, subgroup.description)
        search.add(subgroup.quality, subgroup.rows, subgroup.description)

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task, batch_support=False)
        # Algorithm 2 (GENERATE_SUBGROUPS_S1) and Algorithm 1, lines 4-9.
        s1 = []
        for index in range(len(search.conditions)):
            subgroup = _VerticalList((index,), search.masks[index])
            if self._passes(search, subgroup):
                self._evaluate(search, subgroup)
                s1.append(subgroup)
        s1 = self._sort(s1, self.sort_criterion_in_s1)
        if task.depth < 2 or len(s1) < 2:
            return search.results()

        # Algorithm 1, lines 10-17: the triangular matrix M of two-selector subgroups.
        size = len(search.conditions)
        matrix = np.full((size, size), float("-inf"))  # estimates; -inf for NULL
        classes: list[list[_VerticalList]] = []
        for x, sx in enumerate(s1):
            members = []
            for sy in s1[x + 1:]:
                last_x, last_y = sx.description[-1], sy.description[-1]
                if search.groups[last_x] == search.groups[last_y]:
                    continue  # a pattern uses every attribute at most once
                sxy = sx.refine(sy)
                if self._passes(search, sxy):
                    matrix[last_x, last_y] = matrix[last_y, last_x] = sxy.estimate
                    members.append(sxy)
            classes.append(members)

        # Algorithm 1, lines 18-31: evaluate and explore every equivalence class.
        for members in classes:
            for subgroup in members:
                self._evaluate(search, subgroup)
            members = self._sort(members, self.sort_criterion_in_other_sizes)
            if task.depth > 2:
                self._search(search, members, matrix)
        return search.results()

    def _search(self, search: _Search, members: list[_VerticalList], matrix: np.ndarray) -> None:
        """Algorithm 3 (SEARCH) on one equivalence class."""
        members = list(members)
        while len(members) > 1:
            sx = members.pop(0)
            if not sx.estimate > search.result.min_required_quality():
                continue  # the threshold has risen since sx was generated
            refinements = []
            for sy in members:
                pair_estimate = matrix[sx.description[-1], sy.description[-1]]
                if not pair_estimate > search.result.min_required_quality():
                    continue  # M is NULL, or the pair's estimate rules out all supersets
                sxy = sx.refine(sy)
                if self._passes(search, sxy):
                    self._evaluate(search, sxy)
                    refinements.append(sxy)
            if refinements and len(refinements[0].description) < search.task.depth:
                self._search(
                    search, self._sort(refinements, self.sort_criterion_in_other_sizes), matrix
                )


# Adapted from pysubgroup 0.x's BeamSearch (algorithms.py, Apache-2.0); here the beam
# ignores filters that are not anti-monotone.
class BeamSearch:
    """Heuristic level-wise search refining only the ``beam_width`` best subgroups.

    The beam keeps the best subgroups found at any depth. Each round refines
    beam members not refined before, until the beam stops changing or ``depth``
    is reached. With ``beam_width_adaptive`` the width equals the result set size.
    Filters that are not anti-monotone only restrict the reported subgroups, not
    the beam, so their refinements can still be explored.
    """

    def __init__(self, beam_width: int = 20, beam_width_adaptive: bool = False):
        self.beam_width = beam_width
        self.beam_width_adaptive = beam_width_adaptive

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        beam_width = task.result_set_size if self.beam_width_adaptive else self.beam_width
        if beam_width < task.result_set_size:
            raise ValueError("beam width must be at least the result set size")
        search = _Search(task)
        beam = _TopK(beam_width, task.min_quality)
        refined: set[tuple[int, ...]] = set()
        candidates: list[tuple[int, ...]] = [()]
        for _ in range(task.depth):
            previous = set(beam.members)
            for parent in candidates:
                if parent in refined:
                    continue
                refined.add(parent)
                used = {search.conditions[i].feature for i in parent}
                candidates = [
                    index
                    for index in range(len(search.conditions))
                    if search.conditions[index].feature not in used
                ]
                children = search.expand(parent, search.node_mask(parent), candidates)
                for i, index in enumerate(candidates):
                    if children.admissible(i):
                        quality = search.add_child(children, i)
                        beam.add(quality, tuple(sorted((*parent, index))))
            if beam.members == previous:
                break
            candidates = [entry.indices for entry in beam.heap]
        return search.results()
