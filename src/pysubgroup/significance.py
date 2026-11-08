"""Significance filtering of search results on a holdout set.

Subgroup discovery compares exponentially many patterns, so the best subgroups
of a search often owe their quality to chance. Following SubROC (Siegl et al.,
https://doi.org/10.48550/arXiv.2505.11283, Section 3.3), the data are split into
a search set and a validation set (a holdout procedure, Webb: Discovering
significant patterns, MLJ 68, 2007). The search finds the top-k' subgroups on the
search set; each is then tested on the validation set, and the top-k of those
whose corrected p-value is at most ``alpha`` are kept.

- ``holdout_split``: a random, optionally stratified, split of a DataFrame.
- ``randomization_p_values``: empirical p-values of a randomization test
  (``randomization_null_distribution`` returns the sampled statistics). The
  null hypothesis is that the subgroup's statistic is drawn from its
  distribution over random subsets of the validation set with the same size,
  and, if stratified, the same number of rows in every stratum: target classes
  (``stratify_target=True``, the paper's "same size and class balance") and/or
  the values of columns (``stratify``).
- ``adjust_p_values``: multiple-testing corrections (Benjamini-Yekutieli,
  Benjamini-Hochberg, Bonferroni, Holm).
- ``significance_filter``: the significant subgroups among given results.
- ``holdout_search``: split, search, test, and filter in one call; returns a
  ``HoldoutResult`` with a result table (``HoldoutResult.to_dataframe``).

The tested statistic is the quality function itself, or, for generalization-aware
measures, the value they compare with the generalizations (``qf.value``),
since the reference depends on the search. SubROC tests the unweighted
measure; with size and class balance fixed by the null model
(``stratify_target=True``), the size and class-balance weights are a positive
constant, so the weighted measure gives the same p-value.

The empirical p-value is ``b / R`` for the ``b`` of ``R`` random subsets
scoring at least as high, as in the paper; it is 0 if no subset does, so such
subgroups pass any correction. ``plus_one=True`` gives ``(b + 1) / (R + 1)``
instead (Phipson and Smyth, Permutation p-values should never be zero, SAGMB 9,
2010), whose minimum ``1 / (R + 1)`` must survive the correction: e.g. with
Benjamini-Yekutieli, k' = 100 and ``alpha = 0.05``, a single significant
subgroup needs R >= 10,374.
Deviations from the paper: by default the null model keeps only the size, since with
fixed class counts every measure of size and target share only (WRAcc, ``StandardQF``)
is constant and gets p-value 1; ``stratify_target=True`` keeps the class balance as in
the paper. The null model may also stratify by columns. Subgroups whose
statistic is undefined on the validation set (``-inf`` or ``nan``, e.g. a ROC AUC without
negatives) get p-value 1. Implemented from the paper without reusing code.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .algorithms.base_algorithms import SubgroupDiscoveryTask
from .rules import Rule
from .targets import (
    AreaQF,
    BaseTarget,
    GeneralizationAwareMeasure,
    LabelledTarget,
    MulticlassTarget,
    QualityFunction,
)
from .utils import result_pairs, subgroup_masks, to_dataframe

CORRECTIONS = ("by", "bh", "bonferroni", "holm", "none")


def _columns(stratify: Hashable | Sequence[Hashable] | None) -> list[Hashable]:
    if stratify is None:
        return []
    if isinstance(stratify, str) or not isinstance(stratify, Sequence):
        return [stratify]
    return list(stratify)


def _strata(keys: list[np.ndarray], n_rows: int) -> np.ndarray:
    """Stratum index of every row: rows with equal keys share a stratum."""
    if not keys:
        return np.zeros(n_rows, dtype=np.intp)
    _, codes = np.unique(np.column_stack(keys), axis=0, return_inverse=True)
    return codes.reshape(-1)


def _column_keys(data: pd.DataFrame, stratify) -> list[np.ndarray]:
    columns = _columns(stratify)
    absent = [column for column in columns if column not in data.columns]
    if absent:
        raise ValueError(f"stratification columns are missing: {absent}")
    return [pd.factorize(data[column], use_na_sentinel=False)[0] for column in columns]


def holdout_split(
    data: pd.DataFrame,
    validation_size: float = 0.5,
    *,
    stratify: Hashable | Sequence[Hashable] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``data`` at random into a search set and a validation set.

    ``validation_size`` is the share of rows in the validation set, rounded per
    stratum. With ``stratify`` (a column or a list of columns), every
    combination of their values (missing values included) is split in that
    proportion. Both parts keep the row order and the index of ``data``.
    """
    if not 0 < validation_size < 1:
        raise ValueError(f"validation_size must be in (0, 1), got {validation_size}")
    rng = np.random.default_rng(random_state)
    strata = _strata(_column_keys(data, stratify), len(data))
    validation = np.zeros(len(data), dtype=bool)
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        count = int(np.round(validation_size * len(rows)))
        validation[rng.choice(rows, count, replace=False)] = True
    if validation.all() or not validation.any():
        raise ValueError("the split leaves the search or the validation set empty")
    return data.iloc[np.flatnonzero(~validation)], data.iloc[np.flatnonzero(validation)]


def _statistic(qf: QualityFunction) -> Callable[[np.ndarray, BaseTarget], float]:
    if isinstance(qf, GeneralizationAwareMeasure):
        return qf.value
    if isinstance(qf, AreaQF):
        raise TypeError("AreaQF depends on the description length and cannot be tested")
    return qf.evaluate


def _target_keys(target: BaseTarget, stratify_target: bool) -> list[np.ndarray]:
    if not stratify_target:
        return []
    classes = None
    if isinstance(target, LabelledTarget):
        classes = target.positives_
    elif isinstance(target, MulticlassTarget):
        classes = target.codes_
    if classes is None:
        raise TypeError(
            f"{type(target).__name__} has no target classes to stratify by; "
            "use stratify_target=False"
        )
    return [classes]


def randomization_null_distribution(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    qf: QualityFunction,
    *,
    n_samples: int = 1000,
    stratify_target: bool = False,
    stratify: Hashable | Sequence[Hashable] | None = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed statistics and their randomization null distributions on ``data``.

    Returns ``observed`` (one entry per subgroup) and ``samples`` of shape
    ``(n_subgroups, n_samples)``: the statistic of random subsets drawn as in
    ``randomization_p_values``, which uses the same random numbers for the same
    ``random_state``. Subgroups whose observed statistic is undefined (not
    finite) are not sampled; their row is ``nan``.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if getattr(target, "n_samples_", None) != len(data):
        raise ValueError("target must be fitted on data")
    qf._check_target(target)
    statistic = _statistic(qf)
    rng = np.random.default_rng(random_state)
    keys = _target_keys(target, stratify_target) + _column_keys(data, stratify)
    strata = _strata(keys, len(data))
    stratum_rows = [np.flatnonzero(strata == stratum) for stratum in range(strata.max() + 1)]
    masks = subgroup_masks(results, data)
    observed = np.array([statistic(mask, target) for mask in masks], dtype=float)
    samples = np.full((len(masks), n_samples), np.nan)
    for index, mask in enumerate(masks):
        if not np.isfinite(observed[index]):
            continue
        counts = np.bincount(strata[mask], minlength=len(stratum_rows))
        for draw in range(n_samples):
            sample = np.zeros(len(data), dtype=bool)
            for rows, count in zip(stratum_rows, counts, strict=True):
                if count:
                    sample[rng.choice(rows, count, replace=False)] = True
            samples[index, draw] = statistic(sample, target)
    return observed, samples


def _p_values(observed: np.ndarray, samples: np.ndarray, plus_one: bool) -> np.ndarray:
    tolerance = 1e-12 * np.maximum(1.0, np.abs(observed))
    at_least = (samples >= (observed - tolerance)[:, None]).sum(axis=1)
    p_values = (at_least + plus_one) / (samples.shape[1] + plus_one)
    return np.where(np.isfinite(observed), p_values, 1.0)


def randomization_p_values(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    qf: QualityFunction,
    *,
    n_samples: int = 1000,
    stratify_target: bool = False,
    stratify: Hashable | Sequence[Hashable] | None = None,
    random_state: int | np.random.Generator | None = None,
    plus_one: bool = False,
) -> np.ndarray:
    """Empirical p-value of every subgroup on ``data`` (one-sided, high statistics).

    ``target`` must be fitted on ``data``, usually the validation set. For each
    subgroup, ``n_samples`` random subsets of ``data`` are drawn without
    replacement with the subgroup's size. With ``stratify_target=True``
    (labelled and multiclass targets) or ``stratify`` columns, they also have
    the subgroup's number of rows in every stratum: the target classes,
    combined with the values of the ``stratify`` columns.
    The p-value is ``b / n_samples`` for the ``b`` subsets whose
    statistic is at least the subgroup's, or ``(b + 1) / (n_samples + 1)`` if
    ``plus_one``. See the module docstring for the statistic and
    ``randomization_null_distribution`` for the sampled statistics.
    """
    observed, samples = randomization_null_distribution(
        results,
        data,
        target,
        qf,
        n_samples=n_samples,
        stratify_target=stratify_target,
        stratify=stratify,
        random_state=random_state,
    )
    return _p_values(observed, samples, plus_one)


def adjust_p_values(p_values: Sequence[float], method: str = "by") -> np.ndarray:
    """P-values corrected for testing all of them (in the given order).

    ``"by"``: Benjamini-Yekutieli (false discovery rate under any dependency,
    SubROC's default); ``"bh"``: Benjamini-Hochberg (false discovery rate under
    independence or positive dependency); ``"bonferroni"`` and ``"holm"``
    (family-wise error rate); ``"none"``.
    """
    if method not in CORRECTIONS:
        raise ValueError(f"method must be one of {CORRECTIONS}, got {method!r}")
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if method == "none" or not m:
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    if method == "holm":
        adjusted = np.maximum.accumulate((m - ranks + 1) * ranked)
    else:
        factor = np.sum(1 / ranks) if method == "by" else 1.0
        adjusted = np.minimum.accumulate((factor * m / ranks * ranked)[::-1])[::-1]
    result = np.empty(m)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def significance_filter(
    results: Iterable,
    data: pd.DataFrame,
    target: BaseTarget,
    qf: QualityFunction,
    *,
    k: int | None = None,
    alpha: float = 0.05,
    correction: str = "by",
    **test_options: Any,
) -> list[tuple[float | None, Rule]]:
    """The first ``k`` (default: all) results whose corrected p-value is at most ``alpha``.

    ``data`` and ``target`` are the validation set; ``test_options`` go to
    ``randomization_p_values``. The correction counts every given result.
    """
    pairs = result_pairs(results)
    p_values = randomization_p_values(pairs, data, target, qf, **test_options)
    significant = adjust_p_values(p_values, correction) <= alpha
    return [pair for pair, keep in zip(pairs, significant, strict=True) if keep][:k]


@dataclass
class HoldoutResult:
    """Candidates found on the search set and their tests on the validation set.

    ``candidates`` are the top-k' subgroups of the search with their qualities
    on the search set; ``p_values``, ``adjusted_p_values``, and ``significant``
    hold one entry per candidate. ``results`` are the first ``k`` significant
    candidates.
    """

    candidates: list[tuple[float, Rule]]
    p_values: np.ndarray
    adjusted_p_values: np.ndarray
    significant: np.ndarray
    k: int
    qf: QualityFunction
    search_data: pd.DataFrame
    validation_data: pd.DataFrame
    search_target: BaseTarget
    validation_target: BaseTarget

    @property
    def results(self) -> list[tuple[float, Rule]]:
        return [self.candidates[i] for i in self._selected()]

    def _selected(self) -> np.ndarray:
        return np.flatnonzero(self.significant)[: self.k]

    def to_dataframe(
        self,
        statistics: str | Sequence[str] | None = None,
        measures: Mapping[str, QualityFunction] | Iterable[QualityFunction] | None = None,
        *,
        candidates: bool = False,
    ) -> pd.DataFrame:
        """Result table: search-set quality and statistics, validation statistics, and tests.

        Columns of the search set are those of ``utils.to_dataframe``; the
        validation set adds ``validation quality`` and ``validation <statistic>``
        columns, then ``p_value``, ``adjusted_p_value``, and ``significant``.
        Shows ``results``, or every candidate with ``candidates=True``.
        """
        rows = np.arange(len(self.candidates)) if candidates else self._selected()
        pairs = [self.candidates[i] for i in rows]
        table = to_dataframe(pairs, self.search_data, self.search_target, statistics, measures)
        if not pairs:
            table.insert(0, "quality", [])
        validation_measures = {"quality": self.qf}
        if isinstance(measures, Mapping):
            validation_measures |= measures
        elif measures is not None:
            validation_measures |= {type(qf).__name__: qf for qf in measures}
        validation = to_dataframe(
            [rule for _, rule in pairs],
            self.validation_data,
            self.validation_target,
            statistics,
            validation_measures,
        ).drop(columns="subgroup")
        validation.columns = [f"validation {column}" for column in validation.columns]
        table = pd.concat([table, validation], axis=1)
        table["p_value"] = self.p_values[rows]
        table["adjusted_p_value"] = self.adjusted_p_values[rows]
        table["significant"] = self.significant[rows]
        return table


def holdout_search(
    algorithm,
    task: SubgroupDiscoveryTask,
    *,
    k: int = 5,
    validation_size: float = 0.5,
    split_stratify: Hashable | Sequence[Hashable] | None = None,
    alpha: float = 0.05,
    correction: str = "by",
    n_samples: int = 1000,
    stratify_target: bool = False,
    stratify: Hashable | Sequence[Hashable] | None = None,
    random_state: int | np.random.Generator | None = None,
    plus_one: bool = False,
) -> HoldoutResult:
    """Search the top-k' subgroups on a search set, keep the top-k significant ones.

    ``task`` describes the search on the whole dataset; its ``result_set_size``
    is k' (e.g. 100) and ``k`` the number of significant subgroups to keep.
    ``task.data`` is split with ``holdout_split(validation_size,
    stratify=split_stratify)``, ``task.target`` is refitted on both parts, and
    ``algorithm`` (e.g. ``DFS()``) searches the search set. The candidates are
    tested on the validation set with ``randomization_p_values`` (``n_samples``,
    ``stratify_target``, ``stratify``, ``plus_one``) and corrected with ``adjust_p_values``
    (``correction``). The search space is kept as given; conditions generated
    from the whole dataset see the validation set's values only through their
    thresholds.
    """
    rng = np.random.default_rng(random_state)
    search_data, validation_data = holdout_split(
        task.data, validation_size, stratify=split_stratify, random_state=rng
    )
    search_target = task.target.refit(search_data)
    validation_target = task.target.refit(validation_data)
    candidates = algorithm.execute(replace(task, data=search_data, target=search_target))
    p_values = randomization_p_values(
        candidates,
        validation_data,
        validation_target,
        task.qf,
        n_samples=n_samples,
        stratify_target=stratify_target,
        stratify=stratify,
        random_state=rng,
        plus_one=plus_one,
    )
    adjusted = adjust_p_values(p_values, correction)
    return HoldoutResult(
        candidates=candidates,
        p_values=p_values,
        adjusted_p_values=adjusted,
        significant=adjusted <= alpha,
        k=k,
        qf=task.qf,
        search_data=search_data,
        validation_data=validation_data,
        search_target=search_target,
        validation_target=validation_target,
    )
