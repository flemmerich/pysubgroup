"""Generalization-aware measures and searches (Lemmerich 2014, Chapters 3 and 6)."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pysubgroup import (
    ARLQF,
    DFS,
    PRAUCQF,
    ROCAUCQF,
    BinaryTarget,
    ChiSquaredQF,
    Condition,
    Encoder,
    GeneralizationAwareApriori,
    GeneralizationAwareDFS,
    GeneralizationAwareQF,
    GeneralizationAwareStandardQF,
    GeneralizationAwareStandardQFNumeric,
    MinSizeFilter,
    NumericTarget,
    Rule,
    SoftClassifierTarget,
    StandardQF,
    StandardQFNumeric,
    SubgroupDiscoveryTask,
    WRAccQF,
    generalization_reference,
    generate_conditions,
    to_dataframe,
)
from pysubgroup.algorithms.base_algorithms import _Search
from pysubgroup.algorithms.generalization_aware_algorithms import (
    has_overlapping_or_negated_conditions,
)


def frame(seed, rows=200):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({f"a{j}": rng.choice(list("xyz"), rows) for j in range(4)})
    data["n"] = rng.integers(0, 6, rows).astype(float)
    signal = (data.a0 == "x") + (data.n > 2) + 0.5 * (data.a1 == "y")
    data["value"] = np.round(rng.normal(size=rows) + signal, 1)
    data["label"] = rng.random(rows) < 0.2 + 0.2 * signal
    # Classifier scores, rounded to create ties, that are noise where a0 == "y".
    score = data.label + rng.normal(scale=0.7, size=rows)
    data["score"] = np.round(np.where(data.a0 == "y", rng.normal(size=rows), score), 1)
    conditions = generate_conditions(
        data[["a0", "a1", "a2", "a3", "n"]], n_bins=3, numeric_candidates="thresholds"
    )
    return data, conditions


def target_for(qf, data):
    if qf.target_type is SoftClassifierTarget:
        return SoftClassifierTarget().fit(data, "label", "score", True)
    if qf.target_type is BinaryTarget:
        return BinaryTarget().fit(data, "label", True)
    return NumericTarget().fit(data, "value")


MEASURES = [
    *(GeneralizationAwareStandardQF(a) for a in (0, 0.3, 0.5, 1)),
    *(GeneralizationAwareStandardQFNumeric(a) for a in (0, 0.3, 0.5, 1)),
    GeneralizationAwareStandardQFNumeric(0.5, "max"),
    GeneralizationAwareStandardQFNumeric(1, "combined"),
    GeneralizationAwareStandardQFNumeric(0.5, "order"),
    GeneralizationAwareQF(WRAccQF()),
    GeneralizationAwareQF(StandardQF(0.5, direction="both")),
    GeneralizationAwareQF(StandardQFNumeric(1)),
    # SubROC's generalization-aware measures (Siegl et al., Sect. 3.1.3).
    *(GeneralizationAwareQF(qf(*weights)) for qf in (ARLQF, ROCAUCQF, PRAUCQF)
      for weights in ((0, 0), (1, 1), (1, 0.5))),
]


def brute_force(task, check_estimates=False):
    """Qualities of every subgroup (at most one condition per feature) from all its subsets.

    With ``check_estimates``, every estimate (with and without the summaries of
    the differences) is compared with the qualities of all refinements of its
    subgroup.
    """
    qf, target = task.qf, task.target
    masks = Encoder(task.search_space).fit_transform(task.data)

    def mask_of(indices):
        return masks.intersect(indices) if indices else np.ones(target.n_samples_, dtype=bool)

    features = [condition.feature for condition in task.search_space]
    values = {(): qf.value(np.ones(target.n_samples_, dtype=bool), target)}
    qualities = {}
    for depth in range(1, task.depth + 1):
        for indices in itertools.combinations(range(len(features)), depth):
            if len({features[i] for i in indices}) < depth:
                continue
            mask = masks.intersect(indices)
            if not mask.any() or not all(f.is_satisfied(mask, target, None) for f in task.filters):
                continue
            # Generalizations of admissible subgroups are admissible (anti-monotone filters).
            reference = max(
                values[subset] for size in range(depth)
                for subset in itertools.combinations(indices, size)
            )
            values[indices] = qf.value(mask, target)
            qualities[indices] = qf.evaluate(mask, target, reference=reference)
    if check_estimates:
        differences = {(): None}
        for indices in sorted(values, key=len):
            if not indices:
                continue
            mask = mask_of(indices)
            direct = [indices[:j] + indices[j + 1:] for j in range(len(indices))]
            differences[indices] = qf.merge_differences(
                [qf.difference(mask_of(g), mask, target) for g in direct]
                + [differences[g] for g in direct]
            )
            best = max(values[s] for size in range(len(indices) + 1)
                       for s in itertools.combinations(indices, size))
            refinements = [q for other, q in qualities.items()
                           if len(other) > len(indices) and set(indices) <= set(other)]
            highest = max(refinements, default=-np.inf)
            for summary in (None, differences[indices]):
                estimate = qf.optimistic_estimate(mask, target, reference=best,
                                                  differences=summary)
                assert highest <= estimate + 1e-9, (indices, summary, highest, estimate)
    found = sorted(qualities.values(), reverse=True)
    return [q for q in found if q > task.min_quality][: task.result_set_size]


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("qf", MEASURES, ids=lambda qf: f"{type(qf).__name__}{vars(qf)}")
def test_apriori_finds_the_best_subgroups_of_brute_force(seed, depth, qf):
    data, conditions = frame(seed)
    task = SubgroupDiscoveryTask(data, target_for(qf, data), conditions, qf,
                                 result_set_size=8, depth=depth)
    expected = brute_force(task)
    evaluated = {}
    for difference_based in ("on", "off"):
        algorithm = GeneralizationAwareApriori(difference_based)
        found = [q for q, _ in algorithm.execute(task)]
        np.testing.assert_allclose(found, expected, rtol=1e-9, atol=1e-12)
        evaluated[difference_based] = algorithm.n_evaluated_
    assert evaluated["on"] <= evaluated["off"]


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("qf", MEASURES, ids=lambda qf: f"{type(qf).__name__}{vars(qf)}")
def test_dfs_finds_the_best_subgroups_of_brute_force(seed, depth, qf):
    data, conditions = frame(seed)
    task = SubgroupDiscoveryTask(data, target_for(qf, data), conditions, qf,
                                 result_set_size=8, depth=depth)
    expected = brute_force(task)
    for difference_based in ("on", "off"):
        for batched in ((None, False) if qf.batchable else (None,)):
            task.batched = batched
            algorithm = GeneralizationAwareDFS(difference_based)
            found = [q for q, _ in algorithm.execute(task)]
            np.testing.assert_allclose(found, expected, rtol=1e-9, atol=1e-12)


def test_dfs_matches_apriori_on_deeper_searches():
    data, conditions = frame(4, rows=300)
    for qf in (GeneralizationAwareStandardQF(0.5), GeneralizationAwareStandardQFNumeric(1),
               GeneralizationAwareQF(WRAccQF())):
        task = SubgroupDiscoveryTask(data, target_for(qf, data), conditions, qf,
                                     result_set_size=5, depth=5)
        expected = GeneralizationAwareApriori().execute(task)
        found = GeneralizationAwareDFS().execute(task)
        np.testing.assert_allclose([q for q, _ in found], [q for q, _ in expected], rtol=1e-9)


def test_batchable_measures_agree_with_their_scalar_versions():
    data, conditions = frame(2)
    for qf in (GeneralizationAwareStandardQF(0.3), GeneralizationAwareQF(StandardQF(0.5))):
        target = target_for(qf, data)
        search = _Search(SubgroupDiscoveryTask(data, target, conditions, qf))
        children = search.expand((), None, range(len(conditions)))
        reference = np.linspace(0, 0.5, len(conditions))
        masks = [search.masks[i] for i in range(len(conditions))]
        np.testing.assert_allclose(
            qf.value_many(children.statistics, target), [qf.value(m, target) for m in masks])
        np.testing.assert_allclose(
            qf.evaluate_many(children.statistics, target, reference=reference),
            [qf.evaluate(m, target, reference=r) for m, r in zip(masks, reference, strict=True)])
        np.testing.assert_allclose(
            qf.optimistic_estimate_many(children.statistics, target, reference=reference),
            [qf.optimistic_estimate(m, target, reference=r)
             for m, r in zip(masks, reference, strict=True)])
    qf = GeneralizationAwareStandardQF(0.3)
    target = target_for(qf, data)
    search = _Search(SubgroupDiscoveryTask(data, target, conditions, qf))
    children = search.expand((), None, range(len(conditions)))
    general = np.ones(len(data), dtype=bool)
    states = qf.difference_states_many(children.statistics, target)
    removed = qf.differences_many(qf.difference_state(general, target), states)
    merged = qf.merge_differences_many([removed, np.where(np.arange(len(removed)) % 3, 4, np.inf)])
    expected = []
    for i, mask in enumerate(search.masks):
        summary = qf.merge_differences([qf.difference(general, mask, target),
                                        4 if i % 3 else None])
        assert summary == merged[i]
        expected.append(qf.optimistic_estimate(mask, target, reference=0.2, differences=summary))
    np.testing.assert_allclose(qf.optimistic_estimate_many(
        children.statistics, target, reference=0.2, differences=merged), expected)


def test_overlapping_or_negated_conditions():
    def check(*conditions):
        return has_overlapping_or_negated_conditions(list(conditions))

    assert check(Condition("x", "le", 3), Condition("x", "le", 5))
    assert check(Condition("x", "le", 3), Condition("x", "interval", lower=2, upper=4))
    assert check(Condition("x", "gt", 3), Condition("x", "interval", upper=4))
    assert check(Condition("a", "eq", "u"), Condition("b", "ne", "v"))
    # Bins, complementary thresholds, and intervals on different features are disjoint.
    assert not check(Condition("x", "interval", upper=1), Condition("x", "interval", lower=1,
                     upper=2), Condition("x", "interval", lower=2))
    assert not check(Condition("x", "le", 3), Condition("x", "gt", 3))
    assert not check(Condition("x", "le", 3), Condition("y", "le", 5),
                     Condition("a", "eq", "u"), Condition("a", "in", ("u", "v")))


def test_auto_uses_difference_based_estimates_only_for_overlaps():
    data, _ = frame(0)
    qf = GeneralizationAwareStandardQF(1)
    target = target_for(qf, data)
    features = data[["a0", "a1", "a2", "a3", "n"]]
    spaces = {
        "bins": (generate_conditions(features, n_bins=3), False),
        "thresholds": (generate_conditions(features, n_bins=3,
                                           numeric_candidates="thresholds"), True),
        "negated": (generate_conditions(features, n_bins=3)
                    + [Condition("a0", "ne", "x")], True),
    }
    for conditions, expected in spaces.values():
        task = SubgroupDiscoveryTask(data, target, conditions, qf, depth=2)
        algorithm = GeneralizationAwareApriori()
        algorithm.execute(task)
        assert algorithm.difference_based_ is expected
        for setting, used in (("on", True), ("off", False)):
            forced = GeneralizationAwareApriori(setting)
            forced.execute(task)
            assert forced.difference_based_ is used


@pytest.mark.parametrize("qf", MEASURES, ids=lambda qf: f"{type(qf).__name__}{vars(qf)}")
def test_estimates_bound_all_refinements(qf):
    data, conditions = frame(5, rows=120)
    task = SubgroupDiscoveryTask(data, target_for(qf, data), conditions, qf, depth=3)
    brute_force(task, check_estimates=True)


def test_filters_and_min_quality():
    data, conditions = frame(3)
    qf = GeneralizationAwareStandardQF(0.5)
    task = SubgroupDiscoveryTask(data, target_for(qf, data), conditions, qf, result_set_size=10,
                                 depth=3, filters=[MinSizeFilter(25)], min_quality=0.02)
    masks = Encoder(conditions).fit_transform(data)
    for algorithm in (GeneralizationAwareApriori(), GeneralizationAwareDFS()):
        found = algorithm.execute(task)
        np.testing.assert_allclose([q for q, _ in found], brute_force(task), rtol=1e-9)
        for _, rule in found:
            assert masks.intersect(conditions.index(c) for c in rule.conditions).sum() >= 25


def test_results_match_to_dataframe():
    data, conditions = frame(1)
    qf = GeneralizationAwareStandardQFNumeric(0.5)
    target = target_for(qf, data)
    task = SubgroupDiscoveryTask(data, target, conditions, qf, depth=3)
    results = GeneralizationAwareApriori().execute(task)
    table = to_dataframe(results, data, target, measures={"again": qf})
    np.testing.assert_allclose(table["again"], table["quality"], rtol=1e-12)


def test_theorem_2_example_15():
    # 16 positives and 8 negatives, largest share of a generalization 20%, a = 0.5:
    # the dissertation's bound tp ** a * (1 - 0.2) = 3.2, scaled here by n ** -a.
    data = pd.DataFrame({"x": [True] * 24 + [False] * 76,
                         "label": [True] * 16 + [False] * 8 + [True] * 4 + [False] * 72})
    target = BinaryTarget().fit(data, "label", True)
    qf = GeneralizationAwareStandardQF(0.5)
    estimate = qf.optimistic_estimate(data.x.to_numpy(), target, reference=0.2)
    assert estimate * 100**0.5 == pytest.approx(3.2)


def example_16_17(negatives_ab):
    # A covers 20 positives and 10 negatives, A ∧ B 16 positives and the given negatives.
    rows = 100
    a = np.zeros(rows, dtype=bool)
    a[:30] = True
    ab = np.zeros(rows, dtype=bool)
    ab[:16] = True
    ab[20:20 + negatives_ab] = True
    labels = np.zeros(rows, dtype=bool)
    labels[:20] = True
    labels[30:40] = True
    target = BinaryTarget().fit(pd.DataFrame({"label": labels}), "label", True)
    return target, a, ab


def test_theorem_3_examples_16_and_17():
    qf = GeneralizationAwareStandardQF(0.5)
    # Example 17: two negatives removed; the bound is sqrt(2) / 2, scaled by n ** -a.
    target, a, ab = example_16_17(8)
    differences = qf.merge_differences([qf.difference(a, ab, target)])
    assert differences == 2
    estimate = qf.optimistic_estimate(ab, target, reference=0.2, differences=differences)
    assert estimate * 100**0.5 == pytest.approx(2**0.5 / 2)
    # Theorem 2 alone gives 16 ** 0.5 * 0.8 = 3.2.
    assert qf.optimistic_estimate(ab, target, reference=0.2) * 100**0.5 == pytest.approx(3.2)
    # Example 16: no negatives removed, so no refinement can score above 0.
    target, a, ab = example_16_17(10)
    differences = qf.difference(a, ab, target)
    assert qf.optimistic_estimate(ab, target, reference=0.2, differences=differences) == 0


def test_theorem_4_and_the_pareto_set():
    qf = GeneralizationAwareStandardQFNumeric(1)
    data = pd.DataFrame({"value": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]})
    target = NumericTarget().fit(data, "value")
    generalization = np.array([True, True, True, True, True, False])
    subgroup = np.array([True, True, True, False, False, False])
    assert qf.difference(generalization, subgroup, target) == ((2, 1.0),)
    # 2 removed rows, 3 rows left: 2 * 3 / (3 + 2) * (5 - 1), below the sum bound 12.
    estimate = qf.optimistic_estimate(subgroup, target, reference=0.0, differences=((2, 1.0),))
    assert estimate == pytest.approx(6 / 5 * 4)
    # Removed values above the subgroup's largest: no refinement improves.
    assert qf.optimistic_estimate(subgroup, target, reference=0.0, differences=((1, 6.0),)) == 0
    merged = qf.merge_differences([((3, 1.0),), None, ((2, 0.5), (4, 2.0)), ((2, 1.0),)])
    assert merged == ((2, 1.0), (4, 2.0))


def test_a_condition_that_changes_nothing_scores_zero():
    data = pd.DataFrame({"a": list("xxxxyyyy"), "b": list("uuuuuuuv"),
                         "label": [1, 1, 1, 0, 0, 0, 1, 0]})
    target = BinaryTarget().fit(data, "label", 1)
    a, b = Condition("a", "eq", "x"), Condition("b", "eq", "u")
    for qf in (GeneralizationAwareStandardQF(1), GeneralizationAwareQF(WRAccQF())):
        mask = (data.a == "x").to_numpy()
        reference = generalization_reference(qf, Rule((a, b)), data, target)
        assert qf.evaluate(mask, target, reference=reference) == pytest.approx(0)
        assert qf.evaluate(mask, target, reference=generalization_reference(
            qf, Rule((a,)), data, target)) > 0


def test_errors():
    data, conditions = frame(0)
    qf = GeneralizationAwareStandardQF(1)
    target = target_for(qf, data)
    with pytest.raises(ValueError, match="reference"):
        qf.evaluate(np.ones(len(data), dtype=bool), target)
    with pytest.raises(ValueError, match="reference"):
        DFS().execute(SubgroupDiscoveryTask(data, target, conditions, qf))
    with pytest.raises(TypeError, match="generalization-aware"):
        GeneralizationAwareApriori().execute(
            SubgroupDiscoveryTask(data, target, conditions, StandardQF(1)))
    with pytest.raises(ValueError, match="finite"):
        GeneralizationAwareApriori().execute(
            SubgroupDiscoveryTask(data, target, conditions, GeneralizationAwareQF(ChiSquaredQF())))
    with pytest.raises(ValueError, match="a must be"):
        GeneralizationAwareStandardQFNumeric(2)
    with pytest.raises(ValueError, match="estimator"):
        GeneralizationAwareStandardQFNumeric(1, "median")
    with pytest.raises(ValueError, match="difference_based"):
        GeneralizationAwareApriori(True)
    with pytest.raises(ValueError, match="difference_based"):
        GeneralizationAwareDFS("yes")
    with pytest.raises(TypeError, match="generalization-aware"):
        GeneralizationAwareDFS().execute(
            SubgroupDiscoveryTask(data, target, conditions, StandardQF(1)))
    with pytest.raises(ValueError, match="reference"):
        qf.evaluate_many(None, target)
    with pytest.raises(TypeError):
        GeneralizationAwareQF(qf)
