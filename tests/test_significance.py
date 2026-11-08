"""Holdout split, randomization test, corrections, and holdout_search (SubROC, Sect. 3.3)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from pysubgroup import (
    DFS,
    ROCAUCQF,
    AreaQF,
    BinaryTarget,
    Condition,
    CorrelationTarget,
    FITarget,
    GeneralizationAwareApriori,
    GeneralizationAwareQF,
    GradientTarget,
    LogisticRegressionTarget,
    MinSizeFilter,
    MulticlassTarget,
    NumericTarget,
    Rule,
    SoftClassifierTarget,
    StandardQFNumeric,
    SubgroupDiscoveryTask,
    WRAccQF,
    adjust_p_values,
    generate_conditions,
    holdout_search,
    holdout_split,
    randomization_null_distribution,
    randomization_p_values,
    significance_filter,
)


def classifier_frame(seed=0, rows=600):
    """Scores that rank well except in the subgroup ``group == "b"``, where they are noise."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "group": rng.choice(list("abc"), rows),
        "sex": rng.choice(["f", "m"], rows),
        "noise": rng.choice(list("uvw"), rows),
    })
    data["label"] = rng.random(rows) < 0.4
    good = data.label + rng.normal(scale=0.5, size=rows)
    data["score"] = np.where(data.group == "b", rng.random(rows), good)
    return data


def binary_frame(seed=0, rows=400):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"a": rng.choice(list("xyz"), rows), "b": rng.choice(list("uv"), rows)})
    data["label"] = rng.random(rows) < np.where(data.a == "x", 0.8, 0.2)
    data["value"] = rng.normal(size=rows) + (data.a == "x")
    return data


def test_holdout_split_partitions_and_keeps_order():
    data = binary_frame().set_index(np.arange(400)[::-1] * 10)
    search, validation = holdout_split(data, 0.25, random_state=1)
    assert len(validation) == 100 and len(search) == 300
    assert set(search.index).isdisjoint(validation.index)
    assert set(search.index) | set(validation.index) == set(data.index)
    for part in (search, validation):
        assert (np.diff(data.index.get_indexer(part.index)) > 0).all()
    again, _ = holdout_split(data, 0.25, random_state=1)
    pd.testing.assert_frame_equal(again, search)


def test_holdout_split_is_stratified():
    data = binary_frame()
    data.loc[:20, "b"] = np.nan
    _, validation = holdout_split(data, 0.5, stratify=["label", "b"], random_state=0)
    counts = data.groupby(["label", "b"], dropna=False).size()
    validation_counts = validation.groupby(["label", "b"], dropna=False).size()
    assert (np.abs(validation_counts - counts / 2) <= 0.5).all()
    with pytest.raises(ValueError, match="missing"):
        holdout_split(data, 0.5, stratify="absent")
    with pytest.raises(ValueError, match="validation_size"):
        holdout_split(data, 1.0)


def test_corrections():
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5, 0.2])
    np.testing.assert_allclose(adjust_p_values(p, "bh"), false_discovery_control(p, method="bh"))
    np.testing.assert_allclose(adjust_p_values(p, "by"), false_discovery_control(p, method="by"))
    np.testing.assert_allclose(adjust_p_values(p, "bonferroni"), np.minimum(6 * p, 1))
    np.testing.assert_allclose(
        adjust_p_values(p[:4], "holm"), [0.03, 0.06, 0.06, 0.02]
    )
    np.testing.assert_array_equal(adjust_p_values(p, "none"), p)
    assert adjust_p_values([], "by").shape == (0,)
    with pytest.raises(ValueError, match="method"):
        adjust_p_values(p, "fdr")


def test_class_stratification_fixes_the_target_share():
    data = binary_frame()
    target = BinaryTarget().fit(data, "label", True)
    rules = [Rule((Condition("a", "eq", "x"),)), Rule((Condition("b", "eq", "u"),))]
    stratified = randomization_p_values(rules, data, target, WRAccQF(), n_samples=50,
                                        stratify_target=True, random_state=0)
    np.testing.assert_array_equal(stratified, 1.0)  # every random subset has the same share
    # By default only the size is kept.
    plain = randomization_p_values(rules, data, target, WRAccQF(), n_samples=200,
                                   random_state=0)
    assert plain[0] == 0
    assert plain[1] > 0.01


def test_column_stratification():
    data = classifier_frame()
    target = SoftClassifierTarget().fit(data, "label", "score")
    rule = Rule((Condition("sex", "eq", "f"),))
    # Random subsets with as many rows per sex and class as the stratum are the stratum.
    p = randomization_p_values([rule], data, target, ROCAUCQF(), n_samples=30, stratify="sex",
                               stratify_target=True, random_state=0)
    assert p[0] == 1.0


def test_subroc_detects_the_planted_subgroup():
    data = classifier_frame()
    target = SoftClassifierTarget().fit(data, "label", "score")
    rules = [Rule((Condition("group", "eq", "b"),)), Rule((Condition("noise", "eq", "u"),))]
    options = dict(n_samples=200, stratify_target=True, random_state=0)
    p = randomization_p_values(rules, data, target, ROCAUCQF(), **options)
    assert p[0] == 0
    assert p[1] > 0.05
    # The size and class-balance weights are constant under the paper's null model.
    weighted = randomization_p_values(rules, data, target, ROCAUCQF(1, 1), **options)
    np.testing.assert_array_equal(weighted, p)



def test_plus_one_p_values_are_never_zero():
    data = classifier_frame()
    target = SoftClassifierTarget().fit(data, "label", "score")
    rules = [Rule((Condition("group", "eq", "b"),)), Rule((Condition("noise", "eq", "u"),))]
    options = dict(n_samples=200, random_state=0)
    paper = randomization_p_values(rules, data, target, ROCAUCQF(), **options)
    corrected = randomization_p_values(rules, data, target, ROCAUCQF(), plus_one=True, **options)
    np.testing.assert_allclose(corrected, (paper * 200 + 1) / 201)
    assert corrected[0] == 1 / 201

def test_null_distribution_gives_the_p_values():
    data = classifier_frame()
    target = SoftClassifierTarget().fit(data, "label", "score")
    rules = [Rule((Condition("group", "eq", "b"),)), Rule((Condition("noise", "eq", "u"),))]
    options = dict(n_samples=50, random_state=1)
    observed, samples = randomization_null_distribution(rules, data, target, ROCAUCQF(),
                                                        **options)
    assert samples.shape == (2, 50)
    assert observed[0] == ROCAUCQF().evaluate((data.group == "b").to_numpy(), target)
    np.testing.assert_array_equal(
        randomization_p_values(rules, data, target, ROCAUCQF(), **options),
        (samples >= observed[:, None]).mean(axis=1),
    )


def test_statistic_of_generalization_aware_measures_is_the_value():
    data = binary_frame()
    target = BinaryTarget().fit(data, "label", True)
    rules = [Rule((Condition("a", "eq", "y"), Condition("b", "eq", "u")))]
    options = dict(n_samples=100, random_state=3)
    np.testing.assert_array_equal(
        randomization_p_values(rules, data, target, GeneralizationAwareQF(WRAccQF()), **options),
        randomization_p_values(rules, data, target, WRAccQF(), **options),
    )


def test_undefined_statistics_and_errors():
    data = classifier_frame()
    data.loc[data.sex == "f", "label"] = True
    target = SoftClassifierTarget().fit(data, "label", "score")
    rules = [Rule((Condition("sex", "eq", "f"),)), Rule((Condition("group", "eq", "zzz"),))]
    np.testing.assert_array_equal(
        randomization_p_values(rules, data, target, ROCAUCQF(), n_samples=10), 1.0
    )
    numeric = NumericTarget().fit(data, "score")
    with pytest.raises(TypeError, match="no target classes"):
        randomization_p_values(rules, data, numeric, StandardQFNumeric(1), stratify_target=True)
    with pytest.raises(TypeError, match="AreaQF"):
        randomization_p_values(rules, data, FITarget().fit(data), AreaQF())
    with pytest.raises(ValueError, match="fitted on data"):
        randomization_p_values(rules, data.iloc[:10], target, ROCAUCQF())


def test_significance_filter_keeps_the_order():
    data = binary_frame()
    target = BinaryTarget().fit(data, "label", True)
    results = [(0.3, Rule((Condition("b", "eq", "u"),))), (0.2, Rule((Condition("a", "eq", "x"),))),
               (0.1, Rule((Condition("a", "eq", "z"),)))]
    kept = significance_filter(results, data, target, WRAccQF(), n_samples=200, random_state=0)
    assert [rule for _, rule in kept] == [results[1][1]]
    assert significance_filter(results, data, target, WRAccQF(), k=0, n_samples=200) == []


@pytest.mark.parametrize(
    ("make", "columns"),
    [
        (lambda d: BinaryTarget().fit(d, "label", True), ["positives_"]),
        (lambda d: NumericTarget().fit(d, "value"), ["values_"]),
        (lambda d: MulticlassTarget().fit(d, "a"), ["codes_"]),
        (lambda d: SoftClassifierTarget().fit(d, "label", "value", False), ["positives_"]),
        (lambda d: FITarget().fit(d), []),
        (lambda d: GradientTarget().fit(d, "value", "weight"), ["gradients_"]),
        (lambda d: CorrelationTarget().fit(d, "value", "weight"), ["values_"]),
        (lambda d: LogisticRegressionTarget().fit(d, "value", "label", False), ["labels_"]),
    ],
)
def test_refit_uses_the_same_arguments(make, columns):
    data = binary_frame()
    data["weight"] = np.linspace(1, 2, len(data))
    target = make(data)
    part = data.iloc[::3]
    refitted = target.refit(part)
    expected = make(part)
    assert type(refitted) is type(target) and refitted.n_samples_ == len(part)
    for column in columns:
        np.testing.assert_array_equal(getattr(refitted, column), getattr(expected, column))


def test_holdout_search_finds_the_planted_subgroup():
    data = classifier_frame(rows=1200)
    target = SoftClassifierTarget().fit(data, "label", "score")
    conditions = generate_conditions(data[["group", "sex", "noise"]])
    task = SubgroupDiscoveryTask(data, target, conditions, ROCAUCQF(1, 1), result_set_size=10,
                                 depth=2, filters=[MinSizeFilter(20)])
    found = holdout_search(DFS(), task, k=3, split_stratify="label", n_samples=600,
                           random_state=0)
    assert len(found.candidates) == 10 and len(found.p_values) == 10
    assert len(found.search_data) + len(found.validation_data) == len(data)
    assert 1 <= len(found.results) <= 3
    assert found.results[0][1] == Rule((Condition("group", "eq", "b"),))
    table = found.to_dataframe(statistics=["size", "roc_auc"])
    assert list(table.columns) == [
        "quality", "subgroup", "size", "roc_auc", "validation size", "validation roc_auc",
        "validation quality", "p_value", "adjusted_p_value", "significant",
    ]
    assert table.significant.all() and len(table) == len(found.results)
    validation_target = found.validation_target
    mask = (found.validation_data.group == "b").to_numpy()
    assert table["validation quality"][0] == pytest.approx(
        ROCAUCQF(1, 1).evaluate(mask, validation_target)
    )
    everything = found.to_dataframe(candidates=True)
    assert len(everything) == 10
    np.testing.assert_array_equal(everything.adjusted_p_value, found.adjusted_p_values)


def test_holdout_search_with_generalization_awareness():
    data = classifier_frame(rows=1200)
    target = SoftClassifierTarget().fit(data, "label", "score")
    conditions = generate_conditions(data[["group", "sex", "noise"]])
    qf = GeneralizationAwareQF(ROCAUCQF(1, 1))
    task = SubgroupDiscoveryTask(data, target, conditions, qf, result_set_size=10, depth=2,
                                 filters=[MinSizeFilter(20)])
    found = holdout_search(GeneralizationAwareApriori(), task, k=2, n_samples=600,
                           random_state=1)
    assert found.results[0][1] == Rule((Condition("group", "eq", "b"),))
    # Refinements of the planted subgroup gain little over it.
    assert all(len(rule) == 1 or Condition("group", "eq", "b") not in rule.conditions
               for _, rule in found.results)
    assert "validation quality" in found.to_dataframe().columns


@pytest.mark.parametrize("conditions", [[Condition("x", "gt", 0)], []])
def test_holdout_table_without_results(conditions):
    d = pd.DataFrame({"x": [0, 1, 2, 3], "y": [1, 1, 0, 0]})
    task = SubgroupDiscoveryTask(d, BinaryTarget().fit(d, "y", 1), conditions, WRAccQF())
    found = holdout_search(DFS(), task, n_samples=4, random_state=0)
    assert found.results == []
    table = found.to_dataframe()
    assert table.empty
    d2 = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7], "y": [1, 1, 1, 1, 0, 0, 0, 0]})
    task2 = SubgroupDiscoveryTask(d2, BinaryTarget().fit(d2, "y", 1), [Condition("x", "le", 3)],
                                  WRAccQF())
    nonempty = holdout_search(DFS(), task2, n_samples=4, random_state=0).to_dataframe(
        candidates=True
    )
    assert list(table.columns) == list(nonempty.columns)
