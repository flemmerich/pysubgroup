import itertools

import numpy as np
import pandas as pd
import pytest

from pysubgroup import (
    DFS,
    OPUS,
    VLSD,
    Apriori,
    BeamSearch,
    BestFirstSearch,
    BinaryTarget,
    ChiSquaredQF,
    ClassDistributionL1QF,
    Condition,
    CountQF,
    CustomFilter,
    DpSubgroup,
    GiniQF,
    GradientBoostingQF,
    GradientTarget,
    KolmogorovSmirnovQF,
    LabelledTarget,
    MannWhitneyQF,
    MaxConfidenceQF,
    MaxSizeFilter,
    MeanVectorL1QF,
    MinMajorityCountFilter,
    MinSizeFilter,
    MinTargetShareFilter,
    MulticlassChiSquaredQF,
    MulticlassTarget,
    NumBSD,
    NumericTarget,
    RankAUCQF,
    Rule,
    SDMap,
    SDMapStar,
    SplitQF,
    StandardDeviationQF,
    StandardQF,
    StandardQFNumeric,
    SubgroupDiscoveryTask,
    VarianceReductionQF,
    VectorTarget,
    WRAccQF,
    generate_conditions,
)
from pysubgroup.algorithms.base_algorithms import _Search

EXHAUSTIVE = [
    DFS(), BestFirstSearch(), OPUS(), Apriori(), SDMap(), SDMapStar(), VLSD(),
    VLSD("quality-descending", "quality-descending"), VLSD("quality-ascending", "no-order"),
    DpSubgroup(), DpSubgroup("optimistic_estimate"), DpSubgroup("frequency"),
    DpSubgroup(iterative_deepening=True),
]


def brute_force(task):
    """Best qualities of all conjunctions with distinct features, filters applied."""
    search = _Search(task)
    features = [c.feature for c in search.conditions]
    qualities = []
    for depth in range(1, task.depth + 1):
        for indices in itertools.combinations(range(len(search.conditions)), depth):
            if len({features[i] for i in indices}) < depth:
                continue
            mask = search.mask(indices)
            conditions = tuple(search.conditions[i] for i in indices)
            if not mask.any() or not all(
                f.is_satisfied(mask, task.target, conditions) for f in task.filters
            ):
                continue
            quality = search.evaluate(mask, indices)
            if quality > task.min_quality:
                qualities.append(quality)
    return sorted(qualities, reverse=True)[: task.result_set_size]


def data(seed):
    rng = np.random.default_rng(seed)
    rows = 120
    frame = pd.DataFrame({f"a{j}": rng.choice(list("xyz"), rows) for j in range(3)})
    frame["n"] = rng.integers(0, 6, rows).astype(float)
    frame.loc[rng.random(rows) < 0.1, "n"] = np.nan
    frame["label"] = (rng.random(rows) < 0.3 + 0.4 * (frame.a0 == "x")).astype(int)
    frame["value"] = rng.normal(size=rows) + (frame.n > 2)
    frame["cls"] = np.where(rng.random(rows) < 0.5, frame.a1, rng.choice(list("pqr"), rows))
    shares = rng.dirichlet([2, 1, 1], rows) + 0.3 * (frame.a2 == "y").to_numpy()[:, None]
    frame[["s0", "s1", "s2"]] = shares / shares.sum(axis=1, keepdims=True)
    frame["g"] = rng.normal(size=rows) - (frame.a0 == "y")
    frame["h"] = rng.uniform(0.1, 1.0, rows)
    features = frame[["a0", "a1", "a2", "n"]]
    return frame, generate_conditions(features, n_bins=3, numeric_candidates="thresholds")


FILTER_SETS = [
    [],
    [MinSizeFilter(5), MaxSizeFilter(0.5)],
    [MinTargetShareFilter(0.4),
     CustomFilter(lambda m, t, c: all(x.feature != "a1" for x in c), anti_monotone=True)],
]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("filters", FILTER_SETS)
@pytest.mark.parametrize("make", [
    lambda frame: (BinaryTarget().fit(frame, "label", 1), WRAccQF()),
    lambda frame: (BinaryTarget().fit(frame, "label", 1), StandardQF(0.5)),
    lambda frame: (NumericTarget().fit(frame, "value"), StandardQFNumeric(1)),
    lambda frame: (BinaryTarget().fit(frame, "label", 1), StandardQF(0.5, direction="negative")),
    lambda frame: (BinaryTarget().fit(frame, "label", 1), WRAccQF(direction="both")),
    lambda frame: (NumericTarget().fit(frame, "value"),
                   StandardQFNumeric(0.5, direction="negative")),
    lambda frame: (NumericTarget().fit(frame, "value"),
                   StandardQFNumeric(1, "max", "median", direction="both")),
    lambda frame: (NumericTarget().fit(frame, "value"),
                   StandardQFNumeric(0.5, "order", "median", direction="negative")),
    lambda frame: (NumericTarget().fit(frame, "value"), StandardQFNumeric(0.5, "combined")),
    lambda frame: (NumericTarget().fit(frame, "value"), VarianceReductionQF("order")),
    lambda frame: (NumericTarget().fit(frame, "value"), MannWhitneyQF(direction="both")),
    lambda frame: (NumericTarget().fit(frame, "value"), RankAUCQF("order", "negative")),
    lambda frame: (NumericTarget().fit(frame, "value"), StandardDeviationQF(0.5)),
    lambda frame: (NumericTarget().fit(frame, "value"), KolmogorovSmirnovQF()),
    lambda frame: (MulticlassTarget().fit(frame, "cls"), SplitQF()),
    lambda frame: (MulticlassTarget().fit(frame, "cls"), MulticlassChiSquaredQF()),
    lambda frame: (MulticlassTarget().fit(frame, "cls"), GiniQF()),
    lambda frame: (MulticlassTarget().fit(frame, "cls"), ClassDistributionL1QF()),
    lambda frame: (VectorTarget().fit(frame, "s0", "s1", "s2"), MeanVectorL1QF()),
    lambda frame: (GradientTarget().fit(frame, "g", "h"), GradientBoostingQF(2.0)),
])
def test_exhaustive_algorithms_find_the_best_qualities(seed, filters, make):
    frame, conditions = data(seed)
    target, qf = make(frame)
    if not isinstance(target, LabelledTarget) and any(
        isinstance(f, MinTargetShareFilter) for f in filters
    ):
        with pytest.raises(TypeError, match="positive labels"):
            DFS().execute(SubgroupDiscoveryTask(frame, target, conditions, qf, filters=filters))
        return
    task = SubgroupDiscoveryTask(frame, target, conditions, qf, result_set_size=7, depth=3,
                                 filters=filters)
    expected = brute_force(task)
    numeric = [NumBSD(), NumBSD(fast_bounds=False)] if isinstance(target, NumericTarget) else []
    for algorithm in EXHAUSTIVE + numeric:
        found = algorithm.execute(task)
        assert [q for q, _ in found] == pytest.approx(expected, rel=1e-12, abs=1e-12), (
            algorithm
        )
        for _, rule in found:
            features = [c.feature for c in rule.conditions]
            assert len(set(features)) == len(features)
    beam = [q for q, _ in BeamSearch(beam_width=10).execute(task)]
    assert all(b <= e + 1e-12 for b, e in zip(beam, expected, strict=False))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("batched", [None, False])
def test_beam_search_refines_only_subgroups_passing_anti_monotone_filters(
    seed, batched, monkeypatch
):
    frame, conditions = data(seed)
    filters = [MinSizeFilter(10), CustomFilter(
        lambda m, t, c: all(x.feature != "a1" for x in c), anti_monotone=True
    )]
    if batched is None:
        filters.pop()  # a custom filter disables batching
    task = SubgroupDiscoveryTask(frame, BinaryTarget().fit(frame, "label", 1), conditions,
                                 WRAccQF(), result_set_size=7, depth=3, filters=filters,
                                 batched=batched)
    refined = []
    expand = _Search.expand

    def recording_expand(search, parent, parent_mask, candidates):
        refined.append((search, parent))
        return expand(search, parent, parent_mask, candidates)

    monkeypatch.setattr(_Search, "expand", recording_expand)
    found = BeamSearch(beam_width=10).execute(task)
    assert found
    assert len(refined) > 1
    for search, parent in refined:
        if parent:
            mask = search.mask(parent)
            conds = tuple(search.conditions[i] for i in sorted(parent))
            assert all(f.is_satisfied(mask, task.target, conds) for f in filters), conds


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("min_count", [5, 0.2])
def test_class_association_rule_mining_finds_the_best_confidences(seed, min_count):
    # CBA/CMAR mining: frequent condition sets ranked by majority confidence.
    frame, conditions = data(seed)
    task = SubgroupDiscoveryTask(
        frame, MulticlassTarget().fit(frame, "cls"), conditions, MaxConfidenceQF(),
        result_set_size=25, depth=3, min_quality=0.5,
        filters=[MinMajorityCountFilter(min_count)],
    )
    expected = brute_force(task)
    for algorithm in EXHAUSTIVE:
        found = algorithm.execute(task)
        assert [round(q, 12) for q, _ in found] == [round(q, 12) for q in expected], algorithm


def test_vlsd_rejects_unknown_sort_criteria():
    with pytest.raises(ValueError, match="sort criteria"):
        VLSD(sort_criterion_in_s1="oe-descending")


def test_dpsubgroup_rejects_unknown_exploration_strategies():
    with pytest.raises(ValueError, match="exploration"):
        DpSubgroup("depth-first")


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("exploration", DpSubgroup.STRATEGIES)
def test_dpsubgroup_evaluates_every_subgroup_once_without_pruning(
    exploration, batched, monkeypatch
):
    # ChiSquaredQF has no optimistic estimate, so nothing is pruned and the
    # frequency-ordered candidate sets must enumerate each conjunction once.
    frame, conditions = data(0)
    task = SubgroupDiscoveryTask(frame, BinaryTarget().fit(frame, "label", 1), conditions,
                                 ChiSquaredQF(), result_set_size=3, depth=3, batched=batched)
    seen = []
    add_child = _Search.add_child

    def recording(self, children, i):
        seen.append(tuple(sorted(children.indices(i))))
        return add_child(self, children, i)

    monkeypatch.setattr(_Search, "add_child", recording)
    DpSubgroup(exploration).execute(task)
    search = _Search(task)
    expected = {
        indices
        for depth in range(1, 4)
        for indices in itertools.combinations(range(len(search.conditions)), depth)
        if len({search.groups[i] for i in indices}) == depth and search.mask(indices).any()
    }
    assert len(seen) == len(set(seen))
    assert set(seen) == expected


@pytest.mark.parametrize("algorithm", [DFS(), Apriori(), BeamSearch(), VLSD(), SDMap(), OPUS()])
def test_name_and_position_of_a_column_are_one_feature(algorithm):
    d = pd.DataFrame({"x": [0, 1, 2, 3], "y": [1, 1, 0, 0]})
    task = SubgroupDiscoveryTask(
        d, BinaryTarget().fit(d, "y", 1), [Condition("x", "gt", 0), Condition(0, "le", 2)],
        CountQF(), depth=2,
    )
    rules = {rule for _, rule in algorithm.execute(task)}
    assert rules == {Rule((Condition("x", "gt", 0),)), Rule((Condition("x", "le", 2),))}


def test_vlsd_sorts_two_condition_classes_by_their_qualities(monkeypatch):
    qualities = []
    sort = VLSD._sort

    def recording(subgroups, criterion):
        qualities.extend(s.quality for s in subgroups)
        return sort(subgroups, criterion)

    monkeypatch.setattr(VLSD, "_sort", staticmethod(recording))
    d, conditions = data(0)
    task = SubgroupDiscoveryTask(d, BinaryTarget().fit(d, "label", 1), conditions, WRAccQF(),
                                 depth=2)
    VLSD(sort_criterion_in_other_sizes="quality-descending").execute(task)
    assert qualities and not np.isnan(qualities).any()
