import math

import numpy as np
import pandas as pd
import pytest

import pysubgroup as sd
from pysubgroup import Condition, Rule
from pysubgroup.utils import result_pairs, rule_label, subgroup_masks

# Ten rows; the subgroup x=a covers rows 0-3, three of them positive.
DATA = pd.DataFrame({
    "x": list("aaaabbbbbb"),
    "y": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    "v": [4.0, 6.0, 8.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "w": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "score": [0.9, 0.8, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.6],
})
X_A = Rule((Condition("x", "eq", "a"),))


def test_binary_statistics_match_hand_computed_values() -> None:
    target = sd.BinaryTarget().fit(DATA, "y", 1)
    row = sd.to_dataframe([(0.5, X_A)], DATA, target, "all").iloc[0]

    # n=4, p=3 in the subgroup; N=10, P=4 in the dataset.
    assert row["quality"] == 0.5
    assert row["subgroup"] == "x=a"
    assert (row["size"], row["relative_size"], row["depth"], row["positives"]) == (4, 0.4, 1, 3)
    assert row["target_share"] == pytest.approx(0.75)
    assert row["target_share_complement"] == pytest.approx(1 / 6)
    assert row["lift"] == pytest.approx(0.75 / 0.4)
    assert row["support"] == pytest.approx(0.3)
    assert row["recall"] == pytest.approx(0.75)
    assert row["false_positive_rate"] == pytest.approx(1 / 6)
    assert row["leverage"] == pytest.approx(0.3 - 0.4 * 0.4)
    assert row["odds_ratio"] == pytest.approx(3 * 5 / (1 * 1))
    assert row["conviction"] == pytest.approx(0.6 / 0.25)


def test_defaults_columns_and_measures() -> None:
    target = sd.BinaryTarget().fit(DATA, "y", 1)
    frame = sd.to_dataframe(
        [(0.5, X_A), (0.1, Rule())], DATA, target,
        measures={"chi2": sd.ChiSquaredQF(), "wracc": sd.WRAccQF()},
    )
    assert list(frame.columns) == [
        "quality", "subgroup", "size", "relative_size", "target_share", "lift", "chi2", "wracc",
    ]
    assert frame.subgroup.tolist() == ["x=a", "dataset"]
    assert frame.wracc[0] == pytest.approx(0.3 - 0.4 * 0.4)
    assert frame.wracc[1] == 0

    named = sd.to_dataframe([X_A], DATA, target, ["size"], measures=[sd.LiftQF()])
    assert list(named.columns) == ["subgroup", "size", "LiftQF"]
    with pytest.raises(ValueError, match="pass a dict"):
        sd.to_dataframe([X_A], DATA, target, measures=[sd.StandardQF(0.5), sd.StandardQF(1)])


def test_numeric_multiclass_vector_and_soft_classifier_statistics() -> None:
    numeric = sd.NumericTarget().fit(DATA, "v")
    row = sd.to_dataframe([X_A], DATA, numeric, "all").iloc[0]
    assert (row["mean"], row["median"], row["min"], row["max"]) == (5.0, 5.0, 2.0, 8.0)
    assert row["std"] == pytest.approx(np.std([4, 6, 8, 2]))
    assert row["mean_complement"] == 1.0
    assert row["mean_difference"] == pytest.approx(5.0 - 2.6)
    assert row["median_difference"] == pytest.approx(5.0 - 1.0)
    assert row["mean_lift"] == pytest.approx(5.0 / 2.6)

    multiclass = sd.MulticlassTarget().fit(DATA, "y")
    row = sd.to_dataframe([X_A], DATA, multiclass, "all").iloc[0]
    assert (row["majority_class"], row["majority_share"]) == (1, 0.75)
    assert (row["share 0"], row["share 1"]) == (0.25, 0.75)

    vector = sd.VectorTarget().fit(DATA, "v", "w")
    frame = sd.to_dataframe([X_A], DATA, vector)
    assert (frame["mean v"][0], frame["mean w"][0]) == (5.0, 1.0)

    soft = sd.SoftClassifierTarget().fit(DATA, "y", "score")
    # Positives score 0.9, 0.8, 0.3 and the negative 0.4: two of three pairs are ordered.
    assert sd.to_dataframe([X_A], DATA, soft).roc_auc[0] == pytest.approx(2 / 3)
    assert "roc_auc" not in sd.available_statistics(sd.BinaryTarget().fit(DATA, "y", 1))


def test_empty_subgroups_and_area_measure() -> None:
    target = sd.BinaryTarget().fit(DATA, "y", 1)
    empty = Rule((Condition("x", "eq", "a"), Condition("x", "eq", "b")))
    row = sd.to_dataframe([empty], DATA, target, "all", measures=[sd.AreaQF()]).iloc[0]
    assert row["size"] == 0
    assert math.isnan(row["target_share"]) and math.isnan(row["lift"])
    assert row["AreaQF"] == 0
    numeric = sd.NumericTarget().fit(DATA, "v")
    assert math.isnan(sd.to_dataframe([empty], DATA, numeric, "mean").iloc[0]["mean"])
    assert sd.to_dataframe([], DATA, target).empty


def test_invalid_inputs_are_rejected() -> None:
    target = sd.BinaryTarget().fit(DATA, "y", 1)
    with pytest.raises(ValueError, match="available"):
        sd.to_dataframe([X_A], DATA, target, ["mean"])
    with pytest.raises(ValueError, match="fitted on data"):
        sd.to_dataframe([X_A], DATA.iloc[:5], target)
    with pytest.raises(TypeError, match="NumericTarget"):
        sd.to_dataframe([X_A], DATA, target, measures=[sd.StandardQFNumeric(1)])


def test_helpers_are_shared_with_visualization() -> None:
    assert sd.visualization.subgroup_masks is subgroup_masks
    assert sd.visualization.rule_label is rule_label
    assert result_pairs([X_A, (0.2, X_A)]) == [(None, X_A), (0.2, X_A)]
    assert subgroup_masks([], DATA).shape == (0, 10)


def test_subgroup_data_selects_the_covered_rows() -> None:
    data = pd.DataFrame(
        {"age": [25, 35, 45, np.nan, 38], "sex": ["f", "f", "m", "f", "f"]},
        index=[10, 11, 12, 13, 14],
    )
    rule = Rule((Condition("age", "interval", lower=30, upper=40), Condition("sex", "eq", "f")))
    expected = data.loc[[11, 14]]
    pd.testing.assert_frame_equal(sd.subgroup_data(rule, data), expected)
    pd.testing.assert_frame_equal(sd.subgroup_data((0.5, rule), data), expected)
    pd.testing.assert_frame_equal(sd.subgroup_data(Rule(), data), data)


def test_count_calls_counts_scalar_and_batched_scoring() -> None:
    target = sd.BinaryTarget().fit(DATA, "y", 1)
    conditions = sd.generate_conditions(DATA[["x", "v"]], n_bins=3)
    task = sd.SubgroupDiscoveryTask(DATA, target, conditions, sd.WRAccQF(), 3, 2)
    counted = sd.count_calls(task.qf)
    assert isinstance(counted, sd.WRAccQF) and not hasattr(task.qf, "call_counts")
    assert counted.evaluate(np.ones(10, dtype=bool), target) == 0.0
    assert counted.call_counts.calls == {"evaluate": 1}
    counted.call_counts.reset()

    from dataclasses import replace

    results = sd.DFS().execute(replace(task, qf=counted))
    assert results == sd.DFS().execute(task)
    counts = counted.call_counts
    assert counts.calls["evaluate_many"] > 0
    assert counts.subgroups["evaluate_many"] >= counts.calls["evaluate_many"]


def test_overlap_filter_keeps_order_and_drops_similar_subgroups() -> None:
    x_b = Rule((Condition("x", "eq", "b"),))
    v_low = Rule((Condition("v", "le", 1.0),))  # rows 4-9, the same as x=b
    v_high = Rule((Condition("v", "gt", 5.0),))  # rows 1, 2: a subset of x=a
    results = [(0.5, X_A), (0.4, x_b), (0.3, v_low), (0.2, v_high), (0.1, Rule())]

    assert sd.overlap_filter(results, DATA) == [
        (0.5, X_A), (0.4, x_b), (0.2, v_high), (0.1, Rule()),
    ]
    # Jaccard(x=a, v>5) = 0.5, Jaccard(dataset, x=b) = 0.6.
    assert sd.overlap_filter(results, DATA, max_similarity=0.5) == [
        (0.5, X_A), (0.4, x_b), (0.2, v_high),
    ]
    assert sd.overlap_filter(results, DATA, similarity="overlap") == [(0.5, X_A), (0.4, x_b)]
    assert sd.overlap_filter([X_A, X_A, x_b], DATA, max_similarity=0) == [X_A, x_b]
    empty = Rule((Condition("x", "eq", "c"),))
    assert sd.overlap_filter([empty, empty, X_A], DATA) == [empty, X_A]
    assert sd.overlap_filter([], DATA) == []
    with pytest.raises(ValueError, match="similarity"):
        sd.overlap_filter(results, DATA, similarity="cosine")
    with pytest.raises(ValueError, match="max_similarity"):
        sd.overlap_filter(results, DATA, max_similarity=1.5)


def test_unique_attributes_keeps_first_subgroup_per_feature_set() -> None:
    x_b = Rule((Condition("x", "eq", "b"),))
    v_x = Rule((Condition("v", "le", 4.0), Condition("x", "eq", "a")))
    x_v = Rule((Condition("x", "eq", "b"), Condition("v", "gt", 1.0)))
    results = [(0.5, X_A), (0.4, v_x), (0.3, x_b), (0.2, x_v), (0.1, Rule()), (0.0, Rule())]
    assert sd.unique_attributes(results) == [(0.5, X_A), (0.4, v_x), (0.1, Rule())]
    assert sd.unique_attributes([x_b, X_A]) == [x_b]


def test_statistic_filter_bounds_statistics_columns_and_quality() -> None:
    target = sd.BinaryTarget().fit(DATA, "y", 1)
    x_b = Rule((Condition("x", "eq", "b"),))
    empty = Rule((Condition("x", "eq", "c"),))
    results = [(0.5, X_A), (0.4, x_b), (0.3, Rule()), (0.2, empty)]

    assert sd.statistic_filter(results, DATA, target, minimum={"lift": 1.0}) == [
        (0.5, X_A), (0.3, Rule()),
    ]
    # Bounds are inclusive; the empty subgroup's nan target share fails.
    assert sd.statistic_filter(
        results, DATA, target, minimum={"size": 4}, maximum={"target_share": 0.4}
    ) == [(0.4, x_b), (0.3, Rule())]
    assert sd.statistic_filter(results, DATA, target, maximum={"target_share": 1.0}) == [
        (0.5, X_A), (0.4, x_b), (0.3, Rule()),
    ]
    assert sd.statistic_filter(results, DATA, target, minimum={"quality": 0.4}) == results[:2]
    assert sd.statistic_filter(results, DATA, target) == results
    assert sd.statistic_filter([X_A, x_b], DATA, target, maximum={"size": 4}) == [X_A]

    multiclass = sd.MulticlassTarget().fit(DATA, "x")
    y_one = Rule((Condition("y", "eq", 1),))
    kept = sd.statistic_filter([y_one, Rule()], DATA, multiclass, minimum={"share a": 0.6})
    assert kept == [y_one]

    with pytest.raises(ValueError, match="unknown statistics"):
        sd.statistic_filter(results, DATA, target, minimum={"mean": 0})
    with pytest.raises(ValueError, match="no quality"):
        sd.statistic_filter([X_A], DATA, target, minimum={"quality": 0})
    with pytest.raises(ValueError, match="name a column"):
        sd.statistic_filter([y_one], DATA, multiclass, minimum={"class_shares": 0})
    with pytest.raises(ValueError, match="fitted"):
        sd.statistic_filter(results, DATA.iloc[:5], target, minimum={"size": 1})
