"""Diverse subgroup set discovery (DSSD) and measures of subgroup set diversity.

M. van Leeuwen and A. Knobbe, "Diverse subgroup set discovery", Data Mining and
Knowledge Discovery 25 (2012) 208-242 (open access), extending their ECML PKDD
2011 paper. Written from the paper without reusing code.

``DSSD`` is Algorithm 1, a beam search in three phases:

1. A level-wise beam search up to ``task.depth`` conditions keeps the
   ``pool_size`` best subgroups of all levels (``j`` in the paper); the beam of
   every level is chosen from the level's candidates by a subgroup selection
   strategy instead of taking the best ``beam_width``.
2. Dominance pruning (Sect. 6.3): the conditions of every pooled subgroup are
   tried in the order they were added, and a condition is dropped for good if
   the quality does not decrease without it. Syntactically equal subgroups are
   then removed.
3. The same selection strategy picks the final ``task.result_set_size``
   subgroups (``k``) from the pool.

Selection strategies (Sect. 5), on candidates ordered by decreasing quality:

- ``"top_k"``: the best ``beam_width`` (standard beam search);
- ``"description"`` (Desc(k)): skip a candidate whose quality equals (exactly)
  that of a selected subgroup whose conditions differ from its own in at most
  one;
- ``"var_description"`` (VarDesc(c, l)): skip a candidate with a condition on a
  feature already used ``c * l`` times (``attribute_uses`` is ``c``, ``l`` the
  longest candidate); stop once every feature is used up;
- ``"cover"`` (Cover(k)): greedily take the candidate maximizing
  ``Omega(G, Sel) * phi(G)`` with ``Omega(G, Sel) = mean_{t in G} alpha **
  c(t, Sel)`` and ``c(t, Sel)`` the number of selected subgroups covering ``t``,
  among candidates with positive quality;
- ``"var_cover"`` (VarCover(f)): the same until no candidate scores at least
  ``min_fraction`` times the best candidate quality;
- ``"compression"`` (Compress(k)) and ``"var_compression"`` (VarCompress):
  every tuple is encoded by the code of the last selected subgroup covering it
  (the dataset's code if none does), and the candidate whose code shortens the
  encoding of the dataset most is taken; the variable version stops when no
  candidate shortens it. A subgroup's code is the WKL model: optimal code
  lengths ``-log2 P_G(M_i = x)`` of the empirical distribution of every model
  attribute in the subgroup, so this needs discrete model attributes
  (``targets.nominal.discrete_codes``).

Refinement operator (Sect. 6.2): a feature may carry several ``ne``
conditions, or several ``le``/``gt`` conditions; any other condition is the
only one on its feature. With ``n_bins`` set (6 in the paper), every numeric
feature with threshold or interval conditions in the search space is refined
on the fly instead: the feature's values inside the subgroup being refined are
split into ``n_bins`` bins of equal frequency, and ``le`` and ``gt`` conditions
at the ``n_bins - 1`` cut points replace its conditions from the search space.
The paper's minimum coverage ``mincov`` is a ``MinSizeFilter`` in the task.

``cover_redundancy`` (Definition 7) and ``joint_entropy`` (Definition 8)
measure how diverse the covers of a subgroup set are; ``subgroup_covers``
computes the covers of rules.

Deviations from and choices beyond the paper:

- Numeric conditions are ``x <= v`` and ``x > v`` (the paper's ``<`` and ``>``),
  so that both sides of a cut point together cover the subgroup; cut points are
  values of the subgroup (``np.quantile`` with ``method="inverted_cdf"``), and
  a cut point at the subgroup's maximum, which would leave ``x > v`` empty, is
  dropped.
- "All but one conditions equal" (Desc) is read as ``max(|X|, |Y|) - |X & Y|
  <= 1`` for descriptions ``X`` and ``Y`` (they may differ in length after
  dominance pruning); qualities must be exactly equal, as in the authors'
  binary (a tolerance would also drop unrelated subgroups whose quality ties,
  which is common for WRAcc on small datasets).
- Cover-based selection never selects a subgroup with quality ``<= 0``, as the
  authors' binary does; the beam or result can then be smaller than its size.
- The compression strategies take the candidate with the largest gain from the
  first step on; for the WKL quality, whose value times ``|S|`` is exactly that
  gain, this is the top-ranked candidate, as the paper states.
- Variable-size strategies select at most ``task.result_set_size`` subgroups in
  the final phase, as the fixed-size ones do.
- Ties in quality go to the candidate generated first (beam order, then the
  order of the search space, then dynamic conditions by feature and cut point).
  Filters that are not anti-monotone decide only which subgroups enter the
  pool. ``task.min_quality`` restricts both, as in ``BeamSearch``: candidates
  with quality ``>= min_quality`` may enter the beam, and only those with
  quality ``> min_quality`` enter the pool. Dominance pruning only drops a
  condition if the generalization passes every filter of the task.

Cover-based selection is lazy: the score of a subgroup with positive
quality never increases while the selection grows, so stale scores are upper
bounds, and only candidates reaching the top of a heap are rescored. The result
equals the plain greedy selection of the paper.
"""

from __future__ import annotations

import heapq
import itertools
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .._evaluation import _BLOCK_ELEMENTS, Backend, ChildStatistics
from ..encoding import Encoder, prepare_frame
from ..rules import Condition, Rule
from ..targets.nominal import discrete_codes
from .base_algorithms import SubgroupDiscoveryTask, _Search

SELECTIONS = (
    "top_k",
    "description",
    "var_description",
    "cover",
    "var_cover",
    "compression",
    "var_compression",
)
_THRESHOLDS = frozenset({"le", "gt"})
_REPEATABLE = (frozenset({"ne"}), _THRESHOLDS)


class _Candidate:
    """A subgroup: condition ids in the order they were added, quality, generation order."""

    __slots__ = ("ids", "key", "quality", "seq")

    def __init__(self, ids: tuple[int, ...], quality: float, seq: int):
        self.ids = ids
        self.key = tuple(sorted(ids))
        self.quality = quality
        self.seq = seq

    def rank(self) -> tuple[float, int]:
        return (-self.quality, self.seq)


class _Space:
    """The conditions of a search: the search space and conditions created on the fly."""

    def __init__(self, search: _Search, data: pd.DataFrame, n_bins: int | None):
        self.search = search
        self.conditions: list[Condition] = list(search.conditions)
        self.ids = {condition: i for i, condition in enumerate(self.conditions)}
        self.n_bins = n_bins
        self.n_rows = len(data)
        self.values: dict = {}
        self.integer: dict = {}
        if n_bins is not None:
            for condition in self.conditions:
                feature = condition.feature
                if condition.operator not in {"le", "gt", "interval"} or feature in self.values:
                    continue
                column = data.iloc[:, feature] if isinstance(feature, int) else data[feature]
                dtype = column.dtype
                if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                    self.values[feature] = column.to_numpy(dtype=float, na_value=np.nan)
                    self.integer[feature] = pd.api.types.is_integer_dtype(dtype)
        self.static = [
            i
            for i, condition in enumerate(self.conditions)
            if not (
                condition.feature in self.values
                and condition.operator in {"le", "gt", "interval"}
            )
        ]
        self.features = list(dict.fromkeys(c.feature for c in self.conditions))

    def register(self, condition: Condition) -> int:
        index = self.ids.get(condition)
        if index is None:
            index = self.ids[condition] = len(self.conditions)
            self.conditions.append(condition)
        return index

    def condition_mask(self, index: int) -> np.ndarray:
        if index < len(self.search.conditions):
            return self.search.masks[index]
        condition = self.conditions[index]
        values = self.values[condition.feature]
        return values <= condition.value if condition.operator == "le" else values > condition.value

    def mask(self, ids: Sequence[int]) -> np.ndarray:
        mask = np.ones(self.n_rows, dtype=bool)
        for index in ids:
            mask &= self.condition_mask(index)
        return mask

    def describe(self, ids: Iterable[int]) -> tuple[Condition, ...]:
        return tuple(self.conditions[i] for i in ids)

    def refinements(self, ids: tuple[int, ...], mask: np.ndarray) -> list[int]:
        """Condition ids that refine the subgroup ``ids`` covering ``mask``."""
        used: dict = {}
        for index in ids:
            condition = self.conditions[index]
            used.setdefault(condition.feature, set()).add(condition.operator)
        present = set(ids)

        def allowed(condition: Condition) -> bool:
            operators = used.get(condition.feature)
            if operators is None:
                return True
            operators = operators | {condition.operator}
            return any(operators <= repeatable for repeatable in _REPEATABLE)

        children = [i for i in self.static if i not in present and allowed(self.conditions[i])]
        for feature, values in self.values.items():
            if not allowed(Condition(feature, "le", 0)):
                continue
            inside = values[mask]
            inside = inside[~np.isnan(inside)]
            if len(inside) < 2:
                continue
            cuts = np.unique(
                np.quantile(inside, np.arange(1, self.n_bins) / self.n_bins, method="inverted_cdf")
            )
            for cut in cuts[cuts < inside.max()]:
                value = int(cut) if self.integer[feature] else float(cut)
                for operator in ("le", "gt"):
                    index = self.register(Condition(feature, operator, value))
                    if index not in present:
                        children.append(index)
        return children


def evaluate_children(search: _Search, space, ids, parent_mask, children):
    """Blocks of children with their admissibility, quality, and host row masks.

    ``space`` gives ``n_rows``, ``condition_mask(index)`` and ``describe(ids)``
    for the condition ids of ``children``; evaluation is batched if the search is.
    """
    task = search.task
    step = max(1, _BLOCK_ELEMENTS // max(1, space.n_rows))
    depth = len(ids) + 1
    for start in range(0, len(children), step):
        block = children[start:start + step]
        masks = np.empty((len(block), space.n_rows), dtype=bool)
        for i, index in enumerate(block):
            np.logical_and(parent_mask, space.condition_mask(index), out=masks[i])
        if search.batched:
            statistics = ChildStatistics(
                Backend(masks, task.device), None, np.arange(len(block))
            )
            admissible = statistics.sizes > 0
            for f in search.pruning_filters:
                admissible &= f.is_satisfied_many(statistics, task.target)
            with np.errstate(divide="ignore", invalid="ignore"):
                if search.is_area:
                    quality = task.qf.evaluate_many(statistics, task.target, depth=depth)
                else:
                    quality = task.qf.evaluate_many(statistics, task.target)
            quality = np.asarray(quality, dtype=float)
        else:
            admissible = np.zeros(len(block), dtype=bool)
            quality = np.full(len(block), np.nan)
            for i, index in enumerate(block):
                child = (*ids, index)
                conditions = space.describe(child)
                if masks[i].any() and all(
                    f.is_satisfied(masks[i], task.target, conditions)
                    for f in search.pruning_filters
                ):
                    admissible[i] = True
                    quality[i] = search.evaluate(masks[i], child)
        yield block, admissible, quality, masks


class _Pool:
    """The ``size`` best subgroups seen (``UpdateTopK``); ties keep the earlier one."""

    def __init__(self, size: int):
        self.size = size
        self.heap: list[tuple[float, int, _Candidate]] = []

    def add(self, candidate: _Candidate) -> None:
        entry = (candidate.quality, -candidate.seq, candidate)
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, entry)
        elif entry[:2] > self.heap[0][:2]:
            heapq.heapreplace(self.heap, entry)

    def candidates(self) -> list[_Candidate]:
        return sorted((entry[2] for entry in self.heap), key=_Candidate.rank)


class DSSD:
    """Diverse subgroup set discovery (van Leeuwen and Knobbe 2012, Algorithm 1).

    Parameters follow the paper: ``selection`` is the subgroup selection
    strategy (see the module docstring), ``beam_width`` the beam width ``w`` of
    the fixed-size strategies, ``pool_size`` the number ``j`` of subgroups kept
    by the beam search, ``alpha`` the weight of the cover strategies,
    ``attribute_uses`` the ``c`` of VarDesc, ``min_fraction`` the ``f`` of
    VarCover, and ``n_bins`` the number of bins of the on-the-fly
    discretization of numeric features (None uses the search space as given).
    The task gives the quality measure, the maximum depth, the final number of
    subgroups ``k`` (``result_set_size``), and the filters (``mincov`` is a
    ``MinSizeFilter``). The paper's experiments use ``w = 100``,
    ``j = 10000``, ``k = 100``, depth 5, and ``mincov = 10``.

    ``execute`` returns ``(quality, Rule)`` pairs sorted by decreasing quality;
    each rule lists its conditions in the order they were added.
    """

    def __init__(
        self,
        selection: str = "cover",
        beam_width: int = 100,
        pool_size: int = 10_000,
        alpha: float = 0.9,
        attribute_uses: int = 2,
        min_fraction: float = 0.5,
        n_bins: int | None = 6,
        dominance_pruning: bool = True,
    ):
        self.selection = selection
        self.beam_width = beam_width
        self.pool_size = pool_size
        self.alpha = alpha
        self.attribute_uses = attribute_uses
        self.min_fraction = min_fraction
        self.n_bins = n_bins
        self.dominance_pruning = dominance_pruning

    def _validate(self, task: SubgroupDiscoveryTask) -> None:
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {self.selection!r}")
        for name in ("beam_width", "pool_size", "attribute_uses"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.pool_size < task.result_set_size:
            raise ValueError("pool_size must be at least the result set size")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.min_fraction <= 1:
            raise ValueError("min_fraction must be in [0, 1]")
        if self.n_bins is not None and (
            isinstance(self.n_bins, bool) or not isinstance(self.n_bins, int) or self.n_bins < 2
        ):
            raise ValueError("n_bins must be at least 2 or None")

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        self._validate(task)
        search = _Search(task)
        if self.selection in {"compression", "var_compression"}:
            discrete_codes(task.target)
        space = _Space(search, task.data, self.n_bins)
        pool = _Pool(self.pool_size)
        seq = itertools.count()
        beam: list[tuple[tuple[int, ...], np.ndarray]] = [((), np.ones(len(task.data), bool))]
        for depth in range(1, task.depth + 1):
            candidates: list[_Candidate] = []
            seen: set[tuple[int, ...]] = set()
            for ids, mask in beam:
                children = space.refinements(ids, mask)
                for block, admissible, quality, masks in evaluate_children(
                    search, space, ids, mask, children
                ):
                    for i, index in enumerate(block):
                        child = (*ids, index)
                        key = tuple(sorted(child))
                        if not admissible[i] or key in seen:
                            continue
                        seen.add(key)
                        candidate = _Candidate(child, float(quality[i]), next(seq))
                        candidates.append(candidate)
                        if self._enters_pool(search, space, candidate, masks[i]):
                            pool.add(candidate)
            candidates = sorted(
                (c for c in candidates if c.quality >= task.min_quality), key=_Candidate.rank
            )
            if not candidates or depth == task.depth:
                break
            selected = self._select(candidates, self.beam_width, space, task, depth)
            beam = [(c.ids, space.mask(c.ids)) for c in selected]
        final = pool.candidates()
        if self.dominance_pruning:
            final = [self._prune(search, space, c) for c in final]
        unique: dict[tuple[int, ...], _Candidate] = {}
        for candidate in sorted(final, key=_Candidate.rank):
            unique.setdefault(candidate.key, candidate)
        final = list(unique.values())
        length = max((len(c.ids) for c in final), default=1)
        selected = self._select(final, task.result_set_size, space, task, length)
        selected = sorted(selected[: task.result_set_size], key=_Candidate.rank)
        return [(c.quality, Rule(space.describe(c.ids))) for c in selected]

    @staticmethod
    def _enters_pool(search: _Search, space: _Space, candidate: _Candidate, mask) -> bool:
        if not candidate.quality > search.task.min_quality:
            return False
        filters = search.result_filters
        if not filters:
            return True
        conditions = space.describe(candidate.ids)
        return all(f.is_satisfied(mask, search.task.target, conditions) for f in filters)

    @staticmethod
    def _passes_all(search: _Search, space: _Space, ids, mask) -> bool:
        conditions = space.describe(ids)
        target = search.task.target
        return all(f.is_satisfied(mask, target, conditions) for f in search.task.filters)

    def _prune(self, search: _Search, space: _Space, candidate: _Candidate) -> _Candidate:
        """Dominance pruning (Sect. 6.3): drop conditions, in the order added, that do not help."""
        ids = list(candidate.ids)
        quality = search.evaluate(space.mask(ids), tuple(ids))
        position = 0
        while position < len(ids) and len(ids) > 1:
            trial = ids[:position] + ids[position + 1:]
            mask = space.mask(trial)
            if self._passes_all(search, space, trial, mask):
                trial_quality = search.evaluate(mask, tuple(trial))
                if trial_quality >= quality:
                    ids, quality = trial, trial_quality
                    continue
            position += 1
        return _Candidate(tuple(ids), float(quality), candidate.seq)

    def _select(
        self,
        candidates: list[_Candidate],
        size: int,
        space: _Space,
        task: SubgroupDiscoveryTask,
        length: int,
    ) -> list[_Candidate]:
        """Subgroup selection (Sect. 5) from candidates ordered by ``_Candidate.rank``."""
        if not candidates:
            return []
        selection = self.selection
        if selection == "top_k":
            return candidates[:size]
        if selection == "description":
            return _description_selection(candidates, size)
        if selection == "var_description":
            return _variable_description_selection(
                candidates, self.attribute_uses * length, space
            )
        masks = _CandidateMasks(candidates, space)
        if selection == "cover":
            return _cover_selection(candidates, masks, self.alpha, size, None)
        if selection == "var_cover":
            best = max(c.quality for c in candidates)
            return _cover_selection(
                candidates, masks, self.alpha, None, self.min_fraction * best
            )
        attributes = discrete_codes(task.target)
        if selection == "compression":
            return _compression_selection(candidates, masks, attributes, size)
        return _compression_selection(candidates, masks, attributes, None)


class _CandidateMasks:
    """Row masks of candidates, computed on demand from their conditions."""

    def __init__(self, candidates: list[_Candidate], space: _Space):
        self.candidates = candidates
        self.space = space

    def __getitem__(self, i: int) -> np.ndarray:
        return self.space.mask(self.candidates[i].ids)


def _similar(a: _Candidate, b: _Candidate) -> bool:
    shared = len(set(a.key) & set(b.key))
    return max(len(a.key), len(b.key)) - shared <= 1


def _description_selection(candidates: list[_Candidate], size: int) -> list[_Candidate]:
    selected: list[_Candidate] = []
    for candidate in candidates:
        if len(selected) >= size:
            break
        redundant = False
        # Candidates come by decreasing quality, so equal qualities are at the end.
        for other in reversed(selected):
            if other.quality != candidate.quality:
                break
            if _similar(candidate, other):
                redundant = True
                break
        if not redundant:
            selected.append(candidate)
    return selected


def _variable_description_selection(
    candidates: list[_Candidate], limit: int, space: _Space
) -> list[_Candidate]:
    uses: Counter = Counter()
    saturated: set = set()
    selected: list[_Candidate] = []
    for candidate in candidates:
        if len(saturated) == len(space.features):
            break
        features = [space.conditions[i].feature for i in candidate.ids]
        if any(feature in saturated for feature in features):
            continue
        selected.append(candidate)
        for feature in features:
            uses[feature] += 1
            if uses[feature] >= limit:
                saturated.add(feature)
    return selected


def _cover_selection(
    candidates: list[_Candidate],
    masks: _CandidateMasks,
    alpha: float,
    size: int | None,
    threshold: float | None,
) -> list[_Candidate]:
    """Greedy cover-based selection: Cover(k) with ``size``, VarCover with ``threshold``.

    Only candidates with positive quality are selected.
    """
    n_rows = masks.space.n_rows
    counts = np.zeros(n_rows, dtype=np.int64)
    weights = np.ones(n_rows)
    chosen: list[int] = []

    def full(count: int) -> bool:
        return size is not None and count >= size

    def score(i: int) -> tuple[float, np.ndarray]:
        mask = masks[i]
        return float(weights[mask].mean()) * candidates[i].quality, mask

    def take(mask: np.ndarray, i: int) -> None:
        chosen.append(i)
        counts[mask] += 1
        weights[mask] = alpha ** counts[mask]

    # Lazy greedy: for positive qualities, stale scores bound the current ones.
    heap = [(-c.quality, i) for i, c in enumerate(candidates) if c.quality > 0]
    heapq.heapify(heap)
    fresh: dict[int, int] = {}
    while heap and not full(len(chosen)):
        stale, i = heap[0]
        if threshold is not None and -stale < threshold:
            return [candidates[j] for j in chosen]
        if fresh.get(i) == len(chosen):
            heapq.heappop(heap)
            take(masks[i], i)
            continue
        value, _ = score(i)
        fresh[i] = len(chosen)
        heapq.heapreplace(heap, (-value, i))
    return [candidates[j] for j in chosen]


def _code_lengths(attributes, mask: np.ndarray | None) -> np.ndarray:
    """Code length in bits of every row of ``mask`` under the WKL model fitted on it."""
    lengths = None
    for codes, n_values in attributes:
        rows = codes if mask is None else codes[mask]
        # Values absent from the rows get -inf but are never looked up.
        with np.errstate(divide="ignore"):
            log_shares = np.log2(np.bincount(rows, minlength=n_values) / len(rows))
        part = -log_shares[rows]
        lengths = part if lengths is None else lengths + part
    return lengths


def _compression_selection(
    candidates: list[_Candidate],
    masks: _CandidateMasks,
    attributes,
    size: int | None,
) -> list[_Candidate]:
    """Compression-based selection: Compress(k) with ``size``, VarCompress without."""
    n_rows = masks.space.n_rows
    current = _code_lengths(attributes, None)
    packed = np.empty((len(candidates), (n_rows + 7) // 8), dtype=np.uint8)
    gains = np.empty(len(candidates))
    for i in range(len(candidates)):
        mask = masks[i]
        packed[i] = np.packbits(mask, bitorder="little")
        gains[i] = current[mask].sum() - _code_lengths(attributes, mask).sum()
    step = max(1, _BLOCK_ELEMENTS // max(1, n_rows))
    chosen: list[int] = []
    while len(chosen) < len(candidates) and (size is None or len(chosen) < size):
        best = int(np.argmax(gains))
        if size is None and gains[best] <= 0:
            break
        chosen.append(best)
        mask = masks[best]
        change = np.zeros(n_rows)
        change[mask] = _code_lengths(attributes, mask) - current[mask]
        current[mask] += change[mask]
        rows = np.flatnonzero(change)
        if len(rows):
            for start in range(0, len(candidates), step):
                block = np.unpackbits(
                    packed[start:start + step], axis=1, count=n_rows, bitorder="little"
                )
                gains[start:start + step] += block[:, rows] @ change[rows]
        gains[chosen] = -np.inf
    return [candidates[i] for i in chosen]


def subgroup_covers(rules: Iterable, data) -> np.ndarray:
    """Row masks (subgroups x rows) of ``rules`` on ``data``.

    ``rules`` holds ``Rule`` objects or ``(quality, Rule)`` pairs, as returned
    by the search algorithms.
    """
    rules = [rule[1] if isinstance(rule, tuple) else rule for rule in rules]
    conditions = list(dict.fromkeys(c for rule in rules for c in rule.conditions))
    frame = prepare_frame(data)
    covers = np.ones((len(rules), len(frame)), dtype=bool)
    if conditions:
        coverage = Encoder(conditions).fit_transform(frame)
        position = {condition: i for i, condition in enumerate(conditions)}
        for row, rule in enumerate(rules):
            covers[row] = coverage.intersect(position[c] for c in rule.conditions)
    return covers


def _covers(covers) -> np.ndarray:
    covers = np.asarray(covers, dtype=bool)
    if covers.ndim != 2:
        raise ValueError("covers must be a Boolean matrix with one row per subgroup")
    return covers


def cover_redundancy(covers) -> float:
    """Cover redundancy CR (Definition 7) of the subgroup row masks ``covers``.

    The mean relative deviation of every row's cover count from the mean cover
    count; lower values mean more diverse covers. ``nan`` if no row is covered.
    """
    counts = _covers(covers).sum(axis=0)
    expected = counts.mean() if counts.size else 0.0
    if not expected:
        return float("nan")
    return float(np.abs(counts - expected).mean() / expected)


def joint_entropy(covers) -> float:
    """Joint entropy H in bits (Definition 8) of the subgroup row masks ``covers``.

    Higher values mean less redundancy; ``H <= len(covers)``.
    """
    covers = _covers(covers)
    if not covers.shape[1]:
        return 0.0
    _, counts = np.unique(covers.T, axis=0, return_counts=True)
    shares = counts / covers.shape[1]
    return float(-(shares * np.log2(shares)).sum() + 0.0)
