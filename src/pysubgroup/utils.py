"""Convenience functions for inspecting search results.

``to_dataframe`` turns the ``(quality, Rule)`` pairs returned by a search
(plain ``Rule``s work too) into a table with one row per subgroup. Columns are
statistics chosen by name, see ``available_statistics(target)``, and the values
of any quality functions passed as ``measures``. Statistics depend on the
target type:

- every target: ``size``, ``relative_size``, and ``depth`` (number of conditions);
- labelled targets (binary, soft classifier), with ``p`` positives among ``n``
  rows in the subgroup and ``P`` among ``N`` in the dataset: ``positives``,
  ``target_share`` (``p/n``, the confidence or precision), ``target_share_complement``,
  ``lift`` (``(p/n) / (P/N)``), ``support`` (``p/N``), ``recall`` (``p/P``),
  ``false_positive_rate``, ``leverage`` (``p/N - n/N * P/N``, i.e. WRAcc),
  ``odds_ratio``, and ``conviction``;
- soft-classifier targets also: ``roc_auc``;
- numeric targets: ``mean``, ``median``, ``std``, ``min``, ``max``,
  ``mean_complement``, ``mean_difference`` and ``median_difference`` (subgroup
  minus dataset), and ``mean_lift`` (subgroup over dataset mean);
- multiclass targets: ``majority_class``, ``majority_share``, and
  ``class_shares`` (one column ``share <class>`` per class);
- vector targets: ``means`` (one column ``mean <attribute>`` per attribute);
- EMM correlation targets: ``correlation`` and ``correlation_complement``;
  regression targets also ``slope``, ``intercept``, ``slope_complement``, and
  ``intercept_complement``; general regression targets: ``coefficients`` (one
  column ``coefficient <name>`` per coefficient, intercept first); logistic
  regression targets: ``logit_slopes`` (columns ``logit_slope`` and
  ``logit_slope_complement``, the change of the log-odds per unit of ``x``).

Ratios with a zero denominator are ``nan`` or ``inf``; statistics of empty
subgroups are ``nan``.

Result filters drop subgroups from a result list and keep the order and the
item types (pairs or ``Rule``s) of the rest: ``overlap_filter`` (a subgroup
whose covered rows are too similar to those of an earlier kept one),
``unique_attributes`` (all but the first subgroup per set of features), and
``statistic_filter`` (minimum and maximum values of statistics or of the
quality). pysubgroup 0.x has functions of these names; these are written anew.

``count_calls(qf)`` returns a copy of a quality function that counts its calls
and the subgroups it scores, for debugging and comparing searches.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import wraps
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .emm import (
    CorrelationTarget,
    GeneralRegressionTarget,
    LikelihoodTarget,
    LogisticInteractionQF,
    LogisticRegressionTarget,
    RegressionTarget,
)
from .emm.correlation import correlations
from .emm.regression import linear_regression, simple_regression
from .encoding import Encoder
from .rules import Rule
from .targets import (
    AreaQF,
    BaseTarget,
    GeneralizationAwareMeasure,
    LabelledTarget,
    MulticlassTarget,
    NumericTarget,
    QualityFunction,
    SoftClassifierTarget,
    VectorTarget,
)


def rule_label(rule: Rule) -> str:
    """Readable subgroup description, e.g. ``age ∈ (30, 40] ∧ sex=f``."""
    return " ∧ ".join(map(str, rule.conditions)) if rule.conditions else "dataset"


def result_pairs(results: Iterable) -> list[tuple[float | None, Rule]]:
    """Search results as ``(quality, Rule)`` pairs; plain ``Rule``s get quality ``None``."""
    pairs = []
    for item in results:
        if isinstance(item, Rule):
            pairs.append((None, item))
        else:
            quality, rule = item
            pairs.append((quality, rule))
    return pairs


def subgroup_masks(results: Iterable, data: pd.DataFrame) -> np.ndarray:
    """Boolean row masks (one row per subgroup) of the given rules on ``data``."""
    rules = [rule for _, rule in result_pairs(results)]
    conditions = list(dict.fromkeys(c for rule in rules for c in rule.conditions))
    masks = np.ones((len(rules), len(data)), dtype=bool)
    if conditions:
        coverage = Encoder(conditions).fit_transform(data)
        position = {condition: index for index, condition in enumerate(conditions)}
        for row, rule in enumerate(rules):
            masks[row] = coverage.intersect(position[c] for c in rule.conditions)
    return masks


def subgroup_data(subgroup: Rule | tuple[float, Rule], data: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``data`` that one subgroup covers, as a DataFrame.

    ``subgroup`` is a ``Rule`` or a ``(quality, Rule)`` pair from a search;
    ``data`` must contain the features of its conditions. The rows keep their
    order and index.
    """
    rule = subgroup if isinstance(subgroup, Rule) else subgroup[1]
    return data.iloc[np.flatnonzero(subgroup_masks([rule], data)[0])]


def generalization_reference(
    qf: GeneralizationAwareMeasure, rule: Rule, data: pd.DataFrame, target: BaseTarget
) -> float:
    """Largest ``qf.value`` of a generalization of ``rule``, the reference of ``qf.evaluate``.

    The generalizations are the subgroups described by the proper subsets of
    the rule's conditions, including the empty one (the whole dataset); all
    ``2 ** len(rule) - 1`` of them are evaluated.
    """
    subsets = [
        Rule(conditions)
        for size in range(len(rule))
        for conditions in combinations(rule.conditions, size)
    ]
    return max(qf.value(mask, target) for mask in subgroup_masks(subsets, data))


# Statistics: name -> function(mask, target, rule) returning a value, or a dict
# of columns for statistics that expand into one column per class or attribute.

Statistic = Callable[[np.ndarray, BaseTarget, Rule], Any]


COUNTED_METHODS = (
    "evaluate",
    "evaluate_many",
    "optimistic_estimate",
    "optimistic_estimate_many",
    "value",
    "value_many",
)


@dataclass
class CallCounts:
    """Calls per method (``calls``) and subgroups scored per method (``subgroups``).

    A scalar call scores one subgroup, a batched (``*_many``) call one per
    returned value.
    """

    calls: Counter = field(default_factory=Counter)
    subgroups: Counter = field(default_factory=Counter)

    def reset(self) -> None:
        self.calls.clear()
        self.subgroups.clear()


def count_calls(qf: QualityFunction) -> QualityFunction:
    """A copy of ``qf`` that counts the calls of its scoring methods in ``call_counts``.

    The copy keeps the class of ``qf``, so searches take the same code paths
    (batching, bounds, specialized kernels) as with ``qf``. Counted are the
    methods in ``COUNTED_METHODS`` that ``qf`` has, including calls a method
    makes to another one of them. The counts are a ``CallCounts``::

        counted = count_calls(WRAccQF())
        DFS().execute(replace(task, qf=counted))
        counted.call_counts.subgroups["evaluate"]
    """
    counted = copy.copy(qf)
    counts = CallCounts()
    for name in COUNTED_METHODS:
        method = getattr(counted, name, None)
        if callable(method):
            setattr(counted, name, _counting(method, name, counts))
    counted.call_counts = counts
    return counted


def _counting(method: Callable, name: str, counts: CallCounts) -> Callable:
    batched = name.endswith("_many")

    @wraps(method)
    def counting(*args, **kwargs):
        result = method(*args, **kwargs)
        counts.calls[name] += 1
        counts.subgroups[name] += len(result) if batched else 1
        return result

    return counting


def _ratio(numerator: float, denominator: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(numerator) / np.float64(denominator))


def _labelled_counts(mask: np.ndarray, target: LabelledTarget) -> tuple[int, int, int, int]:
    """Subgroup size and positives, dataset size and positives."""
    size = int(np.count_nonzero(mask))
    positives = int(np.count_nonzero(mask & target.positives_))
    return size, positives, target.n_samples_, target.n_positives_


def _labelled(measure: Callable[[int, int, int, int], float]) -> Statistic:
    return lambda mask, target, rule: measure(*_labelled_counts(mask, target))


def _odds_ratio(n: int, p: int, big_n: int, big_p: int) -> float:
    # Positives and negatives inside versus outside the subgroup.
    return _ratio(p * ((big_n - n) - (big_p - p)), (n - p) * (big_p - p))


def _conviction(n: int, p: int, big_n: int, big_p: int) -> float:
    return _ratio(1 - big_p / big_n, 1 - _ratio(p, n))


def _roc_auc(mask: np.ndarray, target: SoftClassifierTarget, rule: Rule) -> float:
    labels = target.positives_[mask]
    positives = int(np.count_nonzero(labels))
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(target.scores_[mask])
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _values(function: Callable[[np.ndarray], float]) -> Statistic:
    def statistic(mask, target, rule) -> float:
        values = target.values_[mask]
        return float(function(values)) if len(values) else float("nan")

    return statistic


def _class_counts(mask: np.ndarray, target: MulticlassTarget) -> np.ndarray:
    return np.bincount(target.codes_[mask], minlength=len(target.classes_))


def _majority_class(mask, target, rule) -> Any:
    counts = _class_counts(mask, target)
    return target.classes_[int(np.argmax(counts))] if counts.sum() else None


def _majority_share(mask, target, rule) -> float:
    counts = _class_counts(mask, target)
    return _ratio(counts.max(), counts.sum())


def _class_shares(mask, target, rule) -> dict[str, float]:
    counts = _class_counts(mask, target)
    return {
        f"share {label}": _ratio(count, counts.sum())
        for label, count in zip(target.classes_, counts, strict=True)
    }


def _means(mask, target, rule) -> dict[str, float]:
    values = target.values_[mask]
    means = values.mean(axis=0) if len(values) else np.full(values.shape[1], np.nan)
    return {
        f"mean {name}": float(mean) for name, mean in zip(target.attributes_, means, strict=True)
    }


def _moment_statistic(function, part: int, complement: bool = False) -> Statistic:
    """Entry ``part`` of ``function(target, *moments)`` for the subgroup or its complement."""

    def statistic(mask, target, rule) -> float:
        moments = target.moments(mask)
        if complement:
            moments = target.complement(*moments)
        value = function(target, *moments)
        return float((value if part is None else value[part])[0])

    return statistic


def _coefficients(mask, target, rule) -> dict[str, float]:
    coefficients, _, _ = linear_regression(target, *target.moments(mask))
    return {
        f"coefficient {name}": float(value)
        for name, value in zip(target.coefficient_names_, coefficients[0], strict=True)
    }


def _logistic_slopes(mask, target, rule) -> dict[str, float]:
    """Slopes of ``logit P(y)`` on ``x`` (original units) in the subgroup and its complement."""
    fitted = LogisticInteractionQF().coefficients(mask, target)
    if fitted is None:
        return {"logit_slope": float("nan"), "logit_slope_complement": float("nan")}
    beta, _ = fitted
    scale = target.values_.std()
    return {"logit_slope": float((beta[2] + beta[3]) / scale),
            "logit_slope_complement": float(beta[2] / scale)}


def _model_parameters(complement: bool) -> Statistic:
    """Named maximum-likelihood parameters of a ``LikelihoodTarget``'s model on a group."""

    def statistic(mask, target, rule) -> dict[str, float]:
        group = ~mask if complement else mask
        model = target.model_
        parameters = model.fit(group) if group.any() else None
        suffix = "_complement" if complement else ""
        return {f"{name}{suffix}": value for name, value in model.describe(parameters).items()}

    return statistic


_CORRELATION: dict[str, Statistic] = {
    "correlation": _moment_statistic(correlations, None),
    "correlation_complement": _moment_statistic(correlations, None, complement=True),
}

_REGRESSION: dict[str, Statistic] = {
    "slope": _moment_statistic(simple_regression, 0),
    "intercept": _moment_statistic(simple_regression, 1),
    "slope_complement": _moment_statistic(simple_regression, 0, complement=True),
    "intercept_complement": _moment_statistic(simple_regression, 1, complement=True),
}

_GENERAL: dict[str, Statistic] = {
    "size": lambda mask, target, rule: int(np.count_nonzero(mask)),
    "relative_size": lambda mask, target, rule: np.count_nonzero(mask) / target.n_samples_,
    "depth": lambda mask, target, rule: len(rule),
}

_LABELLED: dict[str, Statistic] = {
    "positives": _labelled(lambda n, p, big_n, big_p: p),
    "target_share": _labelled(lambda n, p, big_n, big_p: _ratio(p, n)),
    "target_share_complement": _labelled(
        lambda n, p, big_n, big_p: _ratio(big_p - p, big_n - n)
    ),
    "lift": _labelled(lambda n, p, big_n, big_p: _ratio(_ratio(p, n), big_p / big_n)),
    "support": _labelled(lambda n, p, big_n, big_p: p / big_n),
    "recall": _labelled(lambda n, p, big_n, big_p: _ratio(p, big_p)),
    "false_positive_rate": _labelled(lambda n, p, big_n, big_p: _ratio(n - p, big_n - big_p)),
    "leverage": _labelled(lambda n, p, big_n, big_p: p / big_n - (n / big_n) * (big_p / big_n)),
    "odds_ratio": _labelled(_odds_ratio),
    "conviction": _labelled(_conviction),
}

_NUMERIC: dict[str, Statistic] = {
    "mean": _values(np.mean),
    "median": _values(np.median),
    "std": _values(np.std),
    "min": _values(np.min),
    "max": _values(np.max),
    "mean_complement": lambda mask, target, rule: (
        float(target.values_[~mask].mean()) if not mask.all() else float("nan")
    ),
    "mean_difference": lambda mask, target, rule: _values(np.mean)(mask, target, rule)
    - target.mean_,
    "median_difference": lambda mask, target, rule: _values(np.median)(mask, target, rule)
    - target.median_,
    "mean_lift": lambda mask, target, rule: _ratio(
        _values(np.mean)(mask, target, rule), target.mean_
    ),
}

# Statistics per target type, and the ones shown when none are selected.
_BY_TARGET: list[tuple[type[BaseTarget], dict[str, Statistic], tuple[str, ...]]] = [
    (BaseTarget, _GENERAL, ("size", "relative_size")),
    (LabelledTarget, _LABELLED, ("target_share", "lift")),
    (SoftClassifierTarget, {"roc_auc": _roc_auc}, ("roc_auc",)),
    (NumericTarget, _NUMERIC, ("mean", "mean_difference")),
    (
        MulticlassTarget,
        {
            "majority_class": _majority_class,
            "majority_share": _majority_share,
            "class_shares": _class_shares,
        },
        ("majority_class", "majority_share"),
    ),
    (VectorTarget, {"means": _means}, ("means",)),
    (CorrelationTarget, _CORRELATION, ("correlation", "correlation_complement")),
    (RegressionTarget, _CORRELATION | _REGRESSION, ("slope", "slope_complement")),
    (GeneralRegressionTarget, {"coefficients": _coefficients}, ("coefficients",)),
    (LogisticRegressionTarget, {"logit_slopes": _logistic_slopes}, ("logit_slopes",)),
    (
        LikelihoodTarget,
        {"parameters": _model_parameters(False), "parameters_complement": _model_parameters(True)},
        ("parameters",),
    ),
]


def _statistics_for(target: BaseTarget) -> tuple[dict[str, Statistic], list[str]]:
    available: dict[str, Statistic] = {}
    defaults: list[str] = []
    for target_type, statistics, default in _BY_TARGET:
        if isinstance(target, target_type):
            available |= statistics
            defaults += default
    return available, defaults


def available_statistics(target: BaseTarget) -> list[str]:
    """Names of the statistics ``to_dataframe`` can compute for ``target``."""
    return list(_statistics_for(target)[0])


def _measures(
    measures: Mapping[str, QualityFunction] | Iterable[QualityFunction] | None,
) -> dict[str, QualityFunction]:
    if measures is None:
        return {}
    if isinstance(measures, Mapping):
        return dict(measures)
    named: dict[str, QualityFunction] = {}
    for qf in measures:
        name = type(qf).__name__
        if name in named:
            raise ValueError(
                f"two measures are named {name!r}; pass a dict to name them explicitly"
            )
        named[name] = qf
    return named


def to_dataframe(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    statistics: str | Sequence[str] | None = None,
    measures: Mapping[str, QualityFunction] | Iterable[QualityFunction] | None = None,
) -> pd.DataFrame:
    """One row per subgroup with its quality, description, statistics, and measures.

    ``results`` are ``(quality, Rule)`` pairs as returned by a search, or plain
    ``Rule``s (then there is no ``quality`` column). ``target`` must be fitted on
    ``data``. ``statistics`` selects columns by name from
    ``available_statistics(target)``; ``None`` shows the sizes and the target's
    main statistics, ``"all"`` every available one. ``measures`` are quality
    functions evaluated on each subgroup, as a dict from column name to quality
    function, or a list (columns named after the classes). Generalization-aware
    measures get their reference from ``generalization_reference``.
    """
    if getattr(target, "n_samples_", None) != len(data):
        raise ValueError("target must be fitted on data")
    available, defaults = _statistics_for(target)
    if statistics is None:
        names = defaults
    elif isinstance(statistics, str):
        names = list(available) if statistics == "all" else [statistics]
    else:
        names = list(statistics)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"unknown statistics {unknown} for {type(target).__name__}; "
            f"available: {list(available)}"
        )
    named_measures = _measures(measures)
    for qf in named_measures.values():
        qf._check_target(target)

    pairs = result_pairs(results)
    masks = subgroup_masks(pairs, data)
    rows = []
    for (quality, rule), mask in zip(pairs, masks, strict=True):
        row: dict[str, Any] = {}
        if quality is not None:
            row["quality"] = quality
        row["subgroup"] = rule_label(rule)
        for name in names:
            value = available[name](mask, target, rule)
            row |= value if isinstance(value, dict) else {name: value}
        for name, qf in named_measures.items():
            extra = {"depth": len(rule)} if isinstance(qf, AreaQF) else {}
            if isinstance(qf, GeneralizationAwareMeasure):
                extra = {"reference": generalization_reference(qf, rule, data, target)}
            row[name] = qf.evaluate(mask, target, **extra)
        rows.append(row)
    if not rows:
        # The columns of a non-empty table; statistics may expand into several.
        everything, columns = np.ones(len(data), dtype=bool), ["subgroup"]
        for name in names:
            value = available[name](everything, target, Rule())
            columns.extend(value if isinstance(value, dict) else [name])
        return pd.DataFrame(columns=[*columns, *named_measures])
    return pd.DataFrame(rows)


def overlap_filter(
    results: Iterable,
    data: pd.DataFrame,
    max_similarity: float = 0.9,
    similarity: str = "jaccard",
) -> list:
    """Greedily drop subgroups whose rows overlap those of an earlier kept one.

    Subgroups are visited in the given order (searches return the best first);
    one is kept if its similarity to every kept subgroup is at most
    ``max_similarity``. The similarity of row sets ``A`` and ``B`` is
    ``"jaccard"``, ``|A ∩ B| / |A ∪ B|``, or ``"overlap"``, the overlap
    coefficient ``|A ∩ B| / min(|A|, |B|)``, which is 1 whenever one subgroup
    covers a subset of the other's rows. With a zero denominator (empty
    subgroups) the similarity is 1.
    """
    if similarity not in ("jaccard", "overlap"):
        raise ValueError('similarity must be "jaccard" or "overlap"')
    if not 0 <= max_similarity <= 1:
        raise ValueError("max_similarity must be in [0, 1]")
    items = list(results)
    masks = subgroup_masks(items, data)
    sizes = np.count_nonzero(masks, axis=1)
    kept: list[int] = []
    for index, mask in enumerate(masks):
        if kept:
            intersection = np.count_nonzero(masks[kept] & mask, axis=1)
            if similarity == "jaccard":
                denominator = sizes[kept] + sizes[index] - intersection
            else:
                denominator = np.minimum(sizes[kept], sizes[index])
            values = np.divide(
                intersection, denominator,
                out=np.ones(len(kept)), where=denominator > 0,
            )
            if np.any(values > max_similarity):
                continue
        kept.append(index)
    return [items[index] for index in kept]


def unique_attributes(results: Iterable) -> list:
    """The first subgroup (in the given order) for each set of features in its description.

    Subgroups on the same features, e.g. ``age ≤ 30`` and ``age ≤ 40``, or
    ``sex=f ∧ age ≤ 30`` and ``age > 50 ∧ sex=m``, count as one. Unlike
    pysubgroup 0.x's ``unique_attributes``, nominal features are not exempt.
    """
    items = list(results)
    seen: set[frozenset] = set()
    kept = []
    for item, (_, rule) in zip(items, result_pairs(items), strict=True):
        features = frozenset(condition.feature for condition in rule.conditions)
        if features not in seen:
            seen.add(features)
            kept.append(item)
    return kept


def statistic_filter(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    minimum: Mapping[str, float] | None = None,
    maximum: Mapping[str, float] | None = None,
) -> list:
    """Subgroups whose statistics lie within the given bounds (both inclusive).

    ``minimum`` and ``maximum`` map a column of ``to_dataframe`` to a bound: a
    statistic from ``available_statistics(target)``, a column of an expanding
    statistic (``"share <class>"``), or ``"quality"`` (the search quality; not
    for plain ``Rule``s). ``target`` must be fitted on ``data``. A subgroup
    whose statistic is ``nan`` fails every bound on it::

        statistic_filter(results, data, target, minimum={"size": 50, "lift": 1.2})
    """
    minimum, maximum = dict(minimum or {}), dict(maximum or {})
    items = list(results)
    if getattr(target, "n_samples_", None) != len(data):
        raise ValueError("target must be fitted on data")
    pairs = result_pairs(items)
    available = _statistics_for(target)[0]
    names = list(dict.fromkeys([*minimum, *maximum]))
    if "quality" in names and any(quality is None for quality, _ in pairs):
        raise ValueError("plain Rules have no quality to filter on")
    statistics = [name for name in names if name in available]
    unknown = [name for name in names if name not in available and name != "quality"]
    if unknown:
        # Columns of expanding statistics, named as in to_dataframe.
        everything = np.ones(len(data), dtype=bool)
        for name, statistic in available.items():
            value = statistic(everything, target, Rule())
            if isinstance(value, dict) and set(value) & set(unknown):
                statistics.append(name)
                unknown = [column for column in unknown if column not in value]
    if unknown:
        raise ValueError(
            f"unknown statistics {unknown} for {type(target).__name__}; "
            f"available: {list(available)}"
        )
    frame = to_dataframe(pairs, data, target, statistics)
    expanding = [name for name in names if name != "quality" and name not in frame]
    if expanding:
        raise ValueError(f"statistics {expanding} expand into several columns; name a column")
    keep = np.ones(len(items), dtype=bool)
    for name, bound in minimum.items():
        keep &= (frame[name] >= bound).to_numpy()
    for name, bound in maximum.items():
        keep &= (frame[name] <= bound).to_numpy()
    return [item for item, kept in zip(items, keep, strict=True) if kept]
