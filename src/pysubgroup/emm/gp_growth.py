"""GP-growth: exhaustive exceptional model mining on generic pattern trees.

From F. Lemmerich, M. Becker, and M. Atzmueller, "Generic pattern trees for
exhaustive exceptional model mining", ECML PKDD 2012 (LNCS 7524, 277-292).
GP-growth is FP-growth (Han, Pei, and Yin, SIGMOD 2000) with the frequency
counts of the tree nodes replaced by valuation bases (Sect. 3): condensed
representations of row sets that suffice to compute the model parameters. A
valuation domain is a commutative semigroup ``(V, +)`` (Def. 1); the projector
``phi`` maps a row to its basis (Def. 2), so the basis of a set of rows
aggregates those of its rows; an optional subtraction gives the basis of the
complement from the dataset's (Sect. 3.1); the model projector ``chi`` maps a
basis to model parameters, which a quality function scores (Sect. 3.2).

- ``ValuationDomain``: base class (``project`` is ``phi``, ``aggregate`` the
  semigroup operation, ``subtract`` the subtraction, ``model`` is ``chi``).
- The paper's domains (Sect. 4): ``VarianceDomain`` (``(1, x, x^2)``),
  ``CovarianceDomain`` (``(1, x, y, C)``, aggregated with the pairwise update
  of Bennett et al., CLUSTER 2009), ``CorrelationDomain``, ``SlopeDomain``, and
  ``DecisionTableDomain`` (counts per input combination and output value; the
  Hellinger distance, which equals ``HellingerQF``).
- ``ValuationQF``: ``distance(chi(v_S), chi(v_R))`` between the bases of a
  subgroup and of its complement or the dataset; usable in every search.
- ``GPGrowth``: the search. Its valuation follows from the task's quality
  function: a ``ValuationQF`` brings its domain; for any other batchable
  quality function, a condensed domain holds exactly the statistics its
  batched evaluation reads (sizes, counts and sums, aggregated by addition,
  and maxima, aggregated by ``max``), found by evaluating it once on the whole
  dataset; otherwise the bases are the row sets themselves, the paper's
  trivial valuation basis (Sect. 3.1), with which any quality function works.

The search builds a GP-tree from the rows' conditions in order of decreasing
support and mines it depth first: the basis of a pattern aggregates the bases
of its item's nodes, and the refinements of a pattern are mined from the
conditional tree of its prefix paths. The paper's version is a pre-publication
draft (its Table 1 is incomplete); the algorithm in Sects. 3.1 and 3.2 is
complete. Written from the paper without reusing code.

Deviations and extensions:

- Look-ahead as in FP-growth's counting pass: before a conditional tree is
  built, the bases of all refinements are aggregated from the prefix paths and
  evaluated; refinements failing an anti-monotone filter (generalizing
  FP-growth's minimum support) are scored but left out of the tree.
- Rows with the same conditions are aggregated before they enter the tree,
  and each node aggregates all bases inserted into it at once
  (``aggregate_all``), which equals the pairwise aggregation since the
  operation is associative and commutative.
- ``use_optimistic_estimates`` (not in the paper, which leaves bounds to
  future work, Sect. 3.2) prunes with the quality function's batched
  optimistic estimates as SD-Map* does: refinements whose estimate does not
  exceed the current minimum required quality are left out of the
  conditional trees, and trees are mined by decreasing estimate.
- The automatic condensed domain is not in the paper; Theorem 1 motivates it,
  since batched quality functions are computed in one parallel pass.
- The paper requires but does not give the covariance subtraction; it is the
  pairwise update solved for one part. Aggregating many bases at once uses
  the update's closed form ``C = sum_j C_j + sum_j n_j (mean_j - mean)(...)``,
  and the update adds nothing for an empty part (the paper divides by zero).
- The numeric domains center the values at the dataset mean before
  projection; the models are shift invariant, and centering avoids the
  cancellation in the paper's ``E[X^2] - E[X]^2``.
- The paper leaves the quality ``q`` open; ``ValuationQF`` uses the absolute
  difference of the model parameters (``ValuationDomain.distance``), and the
  paper's Hellinger distance for decision tables. Empty subgroups and empty
  complements score ``nan``.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from ..algorithms.base_algorithms import SubgroupDiscoveryTask, _Search
from ..rules import Rule
from ..targets import AreaQF, BaseTarget, BoundedQualityFunction, QualityFunction
from ..targets.numeric import NumericTarget
from .base import NumericModelTarget, check_reference
from .decision_table import DecisionTableTarget, conditional_estimates

VALUATIONS = ("auto", "rows")


class ValuationDomain(ABC):
    """A valuation domain with projector ``phi`` and model projector ``chi`` (Defs. 1 and 2).

    Valuation bases are float vectors. Methods take arrays with the basis on
    the last axis and broadcast over the leading axes. ``additive`` domains
    aggregate by vector addition and subtract by vector subtraction.
    """

    additive: bool = True

    @abstractmethod
    def project(self, target: BaseTarget) -> np.ndarray:
        """``phi`` of every row: an array of rows x basis width."""

    def aggregate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The semigroup operation: the basis of the union of two disjoint row sets."""
        if self.additive:
            return a + b
        raise NotImplementedError(f"{type(self).__name__} must implement aggregate")

    def aggregate_all(self, bases: np.ndarray) -> np.ndarray:
        """The aggregate of the bases along the first axis (at least one)."""
        if self.additive:
            return bases.sum(axis=0)
        return functools.reduce(self.aggregate, bases)

    def subtract(self, total: np.ndarray, part: np.ndarray) -> np.ndarray:
        """The basis of the rows of ``total`` outside ``part`` (Sect. 3.1)."""
        if self.additive:
            return total - part
        raise NotImplementedError(f"{type(self).__name__} has no subtraction")

    def size(self, bases: np.ndarray) -> np.ndarray:
        """Number of rows each basis represents (the first component by default)."""
        return bases[..., 0]

    @abstractmethod
    def model(self, bases: np.ndarray, target: BaseTarget) -> np.ndarray:
        """``chi``: the model parameters of each basis."""

    def distance(self, model: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Difference of model parameters; the absolute difference (not in the paper)."""
        return np.abs(model - reference)


def _centered(target: BaseTarget, columns: tuple[int, ...]) -> np.ndarray:
    """Numeric model attribute values (rows x columns), centered at the dataset mean."""
    if not isinstance(target, NumericTarget | NumericModelTarget):
        raise TypeError("numeric valuation domains require a NumericTarget or NumericModelTarget")
    values = np.asarray(target.values_, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if max(columns) >= values.shape[1]:
        raise ValueError(f"the target has only {values.shape[1]} model attributes")
    values = values[:, list(columns)]
    if not np.isfinite(values).all():
        raise ValueError("model attributes must contain only finite, non-missing values")
    return values - values.mean(axis=0)


class VarianceDomain(ValuationDomain):
    """Variance of one attribute: ``phi(i) = (1, x, x^2)``, ``chi = v3/v1 - (v2/v1)^2`` (Sect. 4).

    ``attribute`` is the position of the attribute among the target's model
    attributes (a ``NumericTarget`` has one).
    """

    def __init__(self, attribute: int = 0):
        self.attribute = int(attribute)

    def project(self, target):
        x = _centered(target, (self.attribute,))[:, 0]
        return np.column_stack([np.ones_like(x), x, x * x])

    def model(self, bases, target):
        with np.errstate(divide="ignore", invalid="ignore"):
            return bases[..., 2] / bases[..., 0] - np.square(bases[..., 1] / bases[..., 0])


class CovarianceDomain(ValuationDomain):
    """Covariance of attributes ``x``, ``y``: ``phi(i) = (1, x, y, 0)``, ``chi = v4/v1`` (Sect. 4).

    The fourth component is the co-moment ``C = sum (x - mean_x)(y - mean_y)``,
    aggregated with the pairwise update ``C = C1 + C2 + n1 n2 / (n1 + n2)
    (mean_x2 - mean_x1)(mean_y2 - mean_y1)`` (Bennett et al. 2009); further
    components (squares in subclasses) add up.
    """

    additive = False

    def __init__(self, x: int = 0, y: int = 1):
        self.x, self.y = int(x), int(y)

    def _squares(self, values: np.ndarray) -> list[np.ndarray]:
        return []

    def project(self, target):
        values = _centered(target, (self.x, self.y))
        x, y = values[:, 0], values[:, 1]
        return np.column_stack(
            [np.ones_like(x), x, y, np.zeros_like(x), *self._squares(values)]
        )

    @staticmethod
    def _shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``n_a n_b / (n_a + n_b) (mean_x,b - mean_x,a)(mean_y,b - mean_y,a)``, 0 if empty."""
        na, nb = a[..., 0], b[..., 0]
        nonempty = (na > 0) & (nb > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = (
                na * nb / (na + nb)
                * (b[..., 1] / nb - a[..., 1] / na)
                * (b[..., 2] / nb - a[..., 2] / na)
            )
        return np.where(nonempty, shift, 0.0)

    def aggregate(self, a, b):
        result = a + b
        result[..., 3] += self._shift(a, b)
        return result

    def aggregate_all(self, bases):
        result = bases.sum(axis=0)
        sizes = bases[:, 0]
        if result[0] > 0:
            nonempty = sizes > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                deviations = bases[:, 1:3] / sizes[:, None] - result[1:3] / result[0]
            deviations[~nonempty] = 0.0
            result[3] += np.sum(sizes * deviations[:, 0] * deviations[:, 1])
        return result

    def subtract(self, total, part):
        rest = total - part
        rest[..., 3] -= self._shift(part, rest)
        return rest

    def model(self, bases, target):
        with np.errstate(divide="ignore", invalid="ignore"):
            return bases[..., 3] / bases[..., 0]


class CorrelationDomain(CovarianceDomain):
    """Correlation coefficient: ``phi(i) = (1, x, y, 0, x^2, y^2)`` (Sect. 4).

    ``chi = (v4/v1) / (sqrt(v5/v1 - (v2/v1)^2) sqrt(v6/v1 - (v3/v1)^2))``.
    """

    def _squares(self, values):
        return [values[:, 0] ** 2, values[:, 1] ** 2]

    def model(self, bases, target):
        n = bases[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            variance_x = bases[..., 4] / n - np.square(bases[..., 1] / n)
            variance_y = bases[..., 5] / n - np.square(bases[..., 2] / n)
            return (bases[..., 3] / n) / (np.sqrt(variance_x) * np.sqrt(variance_y))


class SlopeDomain(CovarianceDomain):
    """Slope of the regression of ``y`` on ``x``: ``phi(i) = (1, x, y, 0, x^2)`` (Sect. 4).

    ``chi = Cov(X, Y) / Var(X) = v1 v4 / (v1 v5 - v2^2)``.
    """

    def _squares(self, values):
        return [values[:, 0] ** 2]

    def model(self, bases, target):
        with np.errstate(divide="ignore", invalid="ignore"):
            return bases[..., 0] * bases[..., 3] / (
                bases[..., 0] * bases[..., 4] - np.square(bases[..., 1])
            )


class DecisionTableDomain(ValuationDomain):
    """DTM classifier: counts per input combination and output value (Sect. 4).

    ``phi(i)`` has a one for the row's (combination, output value) pair, among
    the combinations observed in the dataset; ``chi`` gives ``P(y | x_1, ...,
    x_k)`` per combination, with the dataset's output distribution for
    combinations without rows; the distance is the Hellinger distance
    ``sum (sqrt P_S(y | x) - sqrt P_R(y | x))^2``. Requires a
    ``DecisionTableTarget``.
    """

    def project(self, target):
        if not isinstance(target, DecisionTableTarget):
            raise TypeError("DecisionTableDomain requires a DecisionTableTarget")
        width = len(target.combinations_) * target.n_labels_
        bases = np.zeros((target.n_samples_, width))
        bases[np.arange(target.n_samples_), target.cells_ * target.n_labels_ + target.labels_] = 1
        return bases

    def size(self, bases):
        return bases.sum(axis=-1)

    def model(self, bases, target):
        tables = bases.reshape(*bases.shape[:-1], len(target.combinations_), target.n_labels_)
        return conditional_estimates(tables, target.prior_)

    def distance(self, model, reference):
        return np.square(np.sqrt(model) - np.sqrt(reference)).sum(axis=(-1, -2))


class ValuationQF(QualityFunction):
    """``distance(chi(v_S), chi(v_R))`` for the bases of a subgroup and of its reference.

    ``reference="complement"`` (default) compares with the complement, whose
    basis is the dataset's minus the subgroup's (Sect. 3.1); ``"dataset"``
    compares with the whole dataset. Empty subgroups and empty complements
    score ``nan``. ``GPGrowth`` aggregates the domain's bases in its tree; the
    other searches project and aggregate the rows of each subgroup.
    """

    target_type = BaseTarget

    def __init__(self, domain: ValuationDomain, reference: str = "complement"):
        if not isinstance(domain, ValuationDomain):
            raise TypeError("domain must be a ValuationDomain")
        self.domain = domain
        self.reference = check_reference(reference)

    def bases(self, target: BaseTarget) -> tuple[np.ndarray, np.ndarray]:
        """The projected rows and the dataset's basis, cached on the target."""
        cache = target.__dict__.setdefault("_valuation_bases", {})
        if self.domain not in cache:
            rows = self.domain.project(target)
            cache[self.domain] = (rows, self.domain.aggregate_all(rows))
        return cache[self.domain]

    def evaluate(self, coverage, target) -> float:
        mask = self._mask(coverage, target)
        if not mask.any():
            return float("nan")
        rows, _ = self.bases(target)
        return float(self.score(self.domain.aggregate_all(rows[mask])[None], target)[0])

    def score(self, bases: np.ndarray, target: BaseTarget) -> np.ndarray:
        """Qualities of the subgroups with the given bases (one per row of ``bases``)."""
        _, total = self.bases(target)
        domain = self.domain
        total = np.broadcast_to(total, bases.shape)
        sizes = domain.size(bases)
        if self.reference == "complement":
            reference = domain.subtract(total, bases)
            valid = (sizes > 0) & (domain.size(reference) > 0)
        else:
            reference, valid = total, sizes > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            quality = domain.distance(domain.model(bases, target), domain.model(reference, target))
        return np.where(valid, quality, np.nan).astype(float)


class _Recorder:
    """Statistics of a single child covering all rows that record what is requested."""

    n_children = 1

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.requests: list[tuple[str, np.ndarray]] = []

    def _record(self, kind: str, array: np.ndarray) -> None:
        if not any(k == kind and a is array for k, a in self.requests):
            self.requests.append((kind, array))

    @property
    def sizes(self) -> np.ndarray:
        return np.array([self.n_samples], dtype=np.int64)

    def count(self, rows: np.ndarray) -> np.ndarray:
        self._record("count", rows)
        return np.array([np.count_nonzero(rows)], dtype=np.int64)

    def sum(self, values: np.ndarray) -> np.ndarray:
        self._record("sum", values)
        return np.asarray(values, dtype=float).sum(axis=0)[None]

    def maximum(self, values: np.ndarray) -> np.ndarray:
        self._record("maximum", values)
        return np.array([np.max(values)], dtype=float)


class _StatisticsDomain(ValuationDomain):
    """The size, counts and sums (added) and maxima (``max``) of target arrays.

    ``chi`` gives the statistics object that batched quality functions and
    filters read (the interface of ``ChildStatistics``).
    """

    additive = False

    def __init__(self, requests: list[tuple[str, np.ndarray]]):
        self.requests = requests
        self.columns: dict[tuple[str, int], tuple[np.ndarray, slice]] = {}
        start, maxima = 1, [False]
        for kind, array in requests:
            width = 1 if kind != "sum" or np.ndim(array) == 1 else array.shape[1]
            self.columns[(kind, id(array))] = (array, slice(start, start + width))
            maxima += [kind == "maximum"] * width
            start += width
        self.width = start
        self.maxima = np.array(maxima)

    def project(self, target):
        columns = [np.ones(target.n_samples_)]
        for _, array in self.requests:
            values = np.asarray(array, dtype=float)
            columns.append(values if values.ndim == 1 else values.reshape(len(values), -1))
        return np.column_stack(columns)

    def aggregate(self, a, b):
        result = a + b
        if self.maxima.any():
            result[..., self.maxima] = np.maximum(a[..., self.maxima], b[..., self.maxima])
        return result

    def aggregate_all(self, bases):
        result = bases.sum(axis=0)
        if self.maxima.any():
            result[self.maxima] = bases[:, self.maxima].max(axis=0)
        return result

    def model(self, bases, target):
        return _BasisStatistics(self, bases)


class _BasisStatistics:
    """``ChildStatistics`` of subgroups read from their condensed bases (one row each)."""

    def __init__(self, domain: _StatisticsDomain, bases: np.ndarray):
        self.domain = domain
        self.bases = bases
        self.n_children = len(bases)

    def _columns(self, kind: str, array: np.ndarray) -> np.ndarray:
        entry = self.domain.columns.get((kind, id(array)))
        if entry is None or entry[0] is not array:
            raise RuntimeError(
                f"the quality function or a filter requested a {kind} of an array that its "
                "evaluation on the whole dataset did not request; use GPGrowth(valuation='rows')"
            )
        return self.bases[:, entry[1]]

    @property
    def sizes(self) -> np.ndarray:
        return np.rint(self.bases[:, 0]).astype(np.int64)

    def count(self, rows: np.ndarray) -> np.ndarray:
        return np.rint(self._columns("count", rows)[:, 0]).astype(np.int64)

    def sum(self, values: np.ndarray) -> np.ndarray:
        columns = self._columns("sum", values)
        return columns[:, 0] if np.ndim(values) == 1 else columns

    def maximum(self, values: np.ndarray) -> np.ndarray:
        return self._columns("maximum", values)[:, 0]


class _Scored:
    """Admissibility, qualities, and optimistic estimates of a batch of patterns."""

    def __init__(self, admissible, quality, estimate, passes):
        self.admissible = admissible
        self.quality = quality
        self.estimate = estimate
        self.passes = passes


class _RowSets:
    """The trivial valuation basis: row positions, aggregated by concatenation (Sect. 3.1)."""

    def __init__(self, search: _Search, use_estimates: bool):
        self.search = search
        self.use_estimates = use_estimates

    def of_rows(self, positions: np.ndarray) -> np.ndarray:
        return positions

    def combine(self, bases: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(bases)

    def score(self, bases: list[np.ndarray], patterns: list[tuple[int, ...]]) -> _Scored:
        search = self.search
        count = len(patterns)
        admissible = np.zeros(count, dtype=bool)
        quality = np.full(count, np.nan)
        estimate = np.full(count, np.inf)
        masks = []
        for i, (positions, pattern) in enumerate(zip(bases, patterns, strict=True)):
            mask = np.zeros(search.task.target.n_samples_, dtype=bool)
            mask[positions] = True
            masks.append(mask)
            if search.admissible(mask, pattern):
                admissible[i] = True
                quality[i] = search.evaluate(mask, pattern)
                if self.use_estimates:
                    estimate[i] = search.optimistic_estimate(mask)

        def passes(i: int) -> bool:
            return search._passes(search.result_filters, masks[i], patterns[i])

        return _Scored(admissible, quality, estimate, passes)


class _Vectors:
    """Condensed bases: a quality domain (or none) followed by a ``_StatisticsDomain``.

    Used only if all pruning filters are batchable.
    """

    def __init__(self, search: _Search, use_estimates: bool):
        task = search.task
        target, qf = task.target, task.qf
        self.search = search
        self.quality_domain = qf.domain if isinstance(qf, ValuationQF) else None
        self.use_estimates = (
            use_estimates and self.quality_domain is None
            and isinstance(qf, BoundedQualityFunction)
        )
        recorder = _Recorder(target.n_samples_)
        with np.errstate(all="ignore"):
            if self.quality_domain is None:
                if isinstance(qf, AreaQF):
                    qf.evaluate_many(recorder, target, depth=1)
                else:
                    qf.evaluate_many(recorder, target)
                if self.use_estimates:
                    qf.optimistic_estimate_many(recorder, target)
            for f in task.filters:
                if f.batchable:
                    f.is_satisfied_many(recorder, target)
        self.statistics = _StatisticsDomain(recorder.requests)
        parts = [self.statistics.project(target)]
        if self.quality_domain is not None:
            parts.insert(0, qf.bases(target)[0])
        self.rows = np.column_stack(parts)
        self.split = self.rows.shape[1] - self.statistics.width

    def of_rows(self, positions: np.ndarray) -> np.ndarray:
        return self._aggregate_all(self.rows[positions])

    def _aggregate_all(self, bases: np.ndarray) -> np.ndarray:
        k = self.split
        statistics = self.statistics.aggregate_all(bases[:, k:])
        if self.quality_domain is None:
            return statistics
        return np.concatenate([self.quality_domain.aggregate_all(bases[:, :k]), statistics])

    def combine(self, bases: list[np.ndarray]) -> np.ndarray:
        return self._aggregate_all(np.stack(bases))

    def score(self, bases: list[np.ndarray], patterns: list[tuple[int, ...]]) -> _Scored:
        search = self.search
        task = search.task
        target, qf = task.target, task.qf
        stacked = np.stack(bases)
        statistics = self.statistics.model(stacked[:, self.split:], target)
        admissible = statistics.sizes > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            for f in search.pruning_filters:
                admissible &= f.is_satisfied_many(statistics, target)
            if self.quality_domain is not None:
                quality = qf.score(stacked[:, :self.split], target)
            elif isinstance(qf, AreaQF):
                quality = qf.evaluate_many(statistics, target, depth=len(patterns[0]))
            else:
                quality = qf.evaluate_many(statistics, target)
            if self.use_estimates:
                estimate = np.asarray(qf.optimistic_estimate_many(statistics, target), float)
            else:
                estimate = np.full(len(patterns), np.inf)
        batched = [f for f in search.result_filters if f.batchable]
        passing = np.ones(len(patterns), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            for f in batched:
                passing &= f.is_satisfied_many(statistics, target)
        others = [f for f in search.result_filters if not f.batchable]

        def passes(i: int) -> bool:
            if not passing[i]:
                return False
            return not others or search._passes(others, search.mask(patterns[i]), patterns[i])

        return _Scored(admissible, np.asarray(quality, dtype=float), estimate, passes)


class _GPNode:
    """GP-tree node: an item and the valuation basis of all rows whose paths pass through it."""

    __slots__ = ("item", "parent", "children", "basis")

    def __init__(self, item: int | None, parent: _GPNode | None, basis=None):
        self.item = item
        self.parent = parent
        self.children: dict[int, _GPNode] = {}
        self.basis = basis


class GPGrowth:
    """GP-growth: exhaustive search on generic pattern trees (Lemmerich et al., ECML PKDD 2012).

    ``valuation="auto"`` (default) chooses condensed valuation bases when the
    quality function is a ``ValuationQF`` or batchable (and all pruning
    filters are batchable), otherwise row sets; ``"rows"`` always uses row
    sets, the paper's trivial valuation basis. ``use_optimistic_estimates``
    prunes with the quality function's optimistic estimates (not in the
    paper; batchable bounded quality functions or row sets). Finds the same
    subgroups as the other exhaustive searches, with qualities equal up to
    the rounding of aggregated float statistics.
    """

    def __init__(self, valuation: str = "auto", use_optimistic_estimates: bool = False):
        if valuation not in VALUATIONS:
            raise ValueError(f"valuation must be one of {VALUATIONS}, got {valuation!r}")
        self.valuation = valuation
        self.use_optimistic_estimates = bool(use_optimistic_estimates)

    def _valuation(self, search: _Search):
        qf = search.task.qf
        condensed = (isinstance(qf, ValuationQF) or qf.batchable) and all(
            f.batchable for f in search.pruning_filters
        )
        if self.valuation == "auto" and condensed:
            return _Vectors(search, self.use_optimistic_estimates)
        return _RowSets(search, self.use_optimistic_estimates)

    def execute(self, task: SubgroupDiscoveryTask) -> list[tuple[float, Rule]]:
        search = _Search(task, batch_support=False)
        valuation = self._valuation(search)
        supports = search.masks.sum(axis=1)
        items = [i for i in range(len(search.conditions)) if supports[i] > 0]
        if not items:
            return search.results()
        bases = [valuation.of_rows(np.flatnonzero(search.masks[i])) for i in items]
        estimates = self._evaluate(search, valuation, (), items, bases)
        if task.depth > 1 and estimates:
            order = sorted(estimates, key=lambda i: (-supports[i], i))
            header = self._tree(search, valuation, order)
            self._mine(search, valuation, header, estimates, ())
        return search.results()

    def _evaluate(self, search, valuation, prefix, items, bases) -> dict[int, float]:
        """Score ``prefix + item`` for all items; returns the items to keep and their estimates."""
        patterns = [tuple(sorted((*prefix, item))) for item in items]
        scored = valuation.score(bases, patterns)
        for i in np.flatnonzero(scored.admissible):
            quality, pattern = float(scored.quality[i]), patterns[i]
            if search.result.accepts(quality, pattern) and scored.passes(i):
                search.result.add(quality, pattern)
        threshold = search.result.min_required_quality()
        return {
            item: float(scored.estimate[i])
            for i, item in enumerate(items)
            if scored.admissible[i]
            and (not self.use_optimistic_estimates or scored.estimate[i] > threshold)
        }

    @staticmethod
    def _tree(search, valuation, order: list[int]) -> dict[int, list[_GPNode]]:
        """The GP-tree of all rows over the items in ``order``; returns its header."""
        order = np.array(order)
        transactions = search.masks[order].T
        rows = np.flatnonzero(transactions.any(axis=1))
        _, inverse = np.unique(
            np.packbits(transactions[rows], axis=1), axis=0, return_inverse=True
        )
        inverse = inverse.reshape(-1)
        grouping = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[grouping], np.arange(inverse.max(initial=-1) + 2))
        root = _GPNode(None, None)
        header: dict[int, list[_GPNode]] = {int(item): [] for item in order}
        for group in range(len(bounds) - 1):
            positions = rows[grouping[bounds[group]:bounds[group + 1]]]
            path = order[transactions[positions[0]]]
            GPGrowth._insert(root, path, valuation.of_rows(positions), header)
        GPGrowth._aggregate_nodes(valuation, header)
        return header

    @staticmethod
    def _insert(root, path, basis, header) -> None:
        """Add a path; its nodes collect the basis, aggregated by ``_aggregate_nodes``."""
        node = root
        for item in path:
            item = int(item)
            child = node.children.get(item)
            if child is None:
                child = node.children[item] = _GPNode(item, node, [])
                header[item].append(child)
            child.basis.append(basis)
            node = child

    @staticmethod
    def _aggregate_nodes(valuation, header) -> None:
        for nodes in header.values():
            for node in nodes:
                bases = node.basis
                node.basis = bases[0] if len(bases) == 1 else valuation.combine(bases)

    def _mine(self, search, valuation, header, estimates, prefix) -> None:
        depth = search.task.depth
        items = list(header)
        if self.use_optimistic_estimates:
            items.sort(key=lambda item: -estimates[item])
        for item in items:
            pattern = (*prefix, item)
            if len(pattern) >= depth:
                return
            if self.use_optimistic_estimates and not (
                estimates[item] > search.result.min_required_quality()
            ):
                continue
            excluded = {search.groups[i] for i in pattern}
            paths = []
            collected: dict[int, list] = defaultdict(list)
            for node in header[item]:
                path = []
                ancestor = node.parent
                while ancestor.item is not None:
                    if search.groups[ancestor.item] not in excluded:
                        path.append(ancestor.item)
                        collected[ancestor.item].append(node.basis)
                    ancestor = ancestor.parent
                if path:
                    paths.append((path, node.basis))
            if not collected:
                continue
            candidates = list(collected)
            kept = self._evaluate(
                search, valuation, pattern, candidates,
                [valuation.combine(collected[c]) for c in candidates],
            )
            if len(pattern) + 1 >= depth or not kept:
                continue
            root = _GPNode(None, None)
            conditional: dict[int, list[_GPNode]] = {c: [] for c in candidates if c in kept}
            for path, basis in paths:
                self._insert(root, [c for c in reversed(path) if c in kept], basis, conditional)
            self._aggregate_nodes(valuation, conditional)
            self._mine(search, valuation, conditional, kept, pattern)
