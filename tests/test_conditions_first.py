import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import KBinsDiscretizer

from pysubgroup import Condition, Encoder, generate_conditions


def test_generated_conditions_preserve_original_numeric_behavior():
    data = pd.DataFrame(
        {
            "measure": [0.0, 1.0, 2.0, 3.0, 4.0, np.nan],
            "category": ["red", "blue", "red", "blue", "red", "blue"],
        }
    )
    conditions = generate_conditions(data, n_bins=2)

    numeric = [condition for condition in conditions if condition.feature == "measure"]
    assert [(condition.operator, condition.lower, condition.upper) for condition in numeric] == [
        ("interval", None, 2.0),
        ("interval", 2.0, None),
        ("is_missing", None, None),
    ]
    assert [condition.value for condition in conditions if condition.operator == "eq"] == [
        "red", "blue"
    ]
    json.dumps([condition.to_dict() for condition in conditions])


def test_kbins_discretizer_is_cloned_and_overrides_apply_per_feature():
    data = pd.DataFrame({"a": np.arange(12), "b": np.arange(12) * 2})
    global_bins = KBinsDiscretizer(n_bins=2, strategy="uniform", encode="ordinal")
    override = KBinsDiscretizer(n_bins=3, strategy="uniform", encode="ordinal")
    conditions = generate_conditions(
        data, discretizer=global_bins, per_feature_discretizers={"b": override}
    )

    assert not hasattr(global_bins, "bin_edges_")
    assert not hasattr(override, "bin_edges_")
    assert sum(condition.feature == "a" for condition in conditions) == 2
    assert sum(condition.feature == "b" for condition in conditions) == 3
    assert [condition.upper for condition in conditions if condition.feature == "b"] == [
        11 / 1.5, 22 / 1.5, None
    ]


def test_high_cardinality_skips_auto_conditions_but_retains_missing():
    data = pd.DataFrame({"id": ["a", "b", "c", None], "other": [1, 2, 3, 4]})
    conditions = generate_conditions(data, max_categories=2, n_bins=None)

    assert conditions == [Condition("id", "is_missing")]


def test_optional_threshold_candidates_and_disabled_binning():
    data = pd.DataFrame({"a": [0, 1, 2, 3, 4]})
    thresholds = generate_conditions(data, n_bins=2, numeric_candidates="thresholds")
    equalities = generate_conditions(data, n_bins=None)

    assert [(item.operator, item.value) for item in thresholds] == [("le", 2), ("gt", 2)]
    assert [item.operator for item in equalities] == ["eq"] * 5


def test_both_representations_agree_on_manual_overlapping_and_duplicate_conditions():
    data = pd.DataFrame(
        {"age": [20, 30, 40, np.nan, 50, 20, 30, 60, 40, 10], "color": ["r", "b"] * 5}
    )
    conditions = [
        Condition("age", "le", 30),
        Condition("age", "le", 40),
        Condition("age", "interval", lower=20, upper=40),
        Condition("age", "gt", 100),
        Condition("color", "eq", "r"),
        Condition("age", "is_missing"),
        Condition("age", "le", 30),
    ]
    outputs = {}
    for representation in ("dense", "packed"):
        encoder = Encoder(conditions, representation=representation)
        coverage = encoder.fit_transform(data)
        assert coverage.shape == (10, len(conditions))
        assert encoder.conditions_[0] == encoder.conditions_[-1]
        assert coverage.count([3]) == 0
        assert coverage.count([]) == 10
        assert coverage.count([0], rows=np.asarray([True] * 5 + [False] * 5)) == 2
        unknown = encoder.transform(pd.DataFrame({"age": [100], "color": ["unseen"]}))
        assert not unknown.storage.any()
        outputs[representation] = coverage
        conditions.append(Condition("color", "eq", "different"))
        assert len(encoder.conditions_) == 7
        conditions.pop()
    dense, packed = outputs["dense"], outputs["packed"]
    for selection in ([], [0], [0, 1], [1, 2, 4], [5], [0, 6]):
        np.testing.assert_array_equal(dense.intersect(selection), packed.intersect(selection))
        assert dense.count(selection) == packed.count(selection)
    assert packed.storage.nbytes < dense.storage.nbytes


def test_sets_sort_deterministically_and_feature_positions_work():
    data = np.asarray([[1], [2], [3]])
    conditions = {Condition(0, "gt", 1), Condition(0, "le", 2)}
    left = Encoder(conditions).fit_transform(data)
    right = Encoder(set(reversed(tuple(conditions)))).fit_transform(data)

    np.testing.assert_array_equal(left.storage, right.storage)


def test_condition_generator_input_is_consumed_once():
    conditions = (Condition(0, "gt", value) for value in (1, 2))
    encoder = Encoder(conditions)

    assert encoder.fit_transform([[0], [2], [3]]).shape == (3, 2)


def test_datetime_categories_serialize_and_match():
    timestamp = pd.Timestamp("2026-01-01")
    conditions = [Condition("date", "eq", timestamp)]
    data = pd.DataFrame({"date": pd.Series([timestamp, pd.NaT], dtype=object)})

    coverage = Encoder(conditions).fit_transform(data)

    assert coverage.condition_mask(0).tolist() == [True, False]
    assert json.loads(json.dumps(conditions[0].to_dict()))["value"] == "2026-01-01T00:00:00"


def test_invalid_manual_schema_and_operator_are_rejected():
    with pytest.raises(ValueError, match="unknown feature"):
        Encoder([Condition("typo", "eq", "x")]).fit_transform(pd.DataFrame({"real": ["x"]}))
    with pytest.raises(ValueError, match="numeric predicate"):
        Encoder([Condition("name", "le", 5)]).fit_transform(pd.DataFrame({"name": ["x"]}))
    with pytest.raises(ValueError, match="unsupported"):
        Condition("name", "unknown", "x")


def test_equality_conditions_keep_first_seen_order_and_scalar_types():
    data = pd.DataFrame({
        "mixed": pd.Series([2, 1, 1.0, True, "x", None, 2], dtype=object),
        "nullable": pd.Series([2, None, 1, 2], dtype="Int64").reindex(range(7)),
        "category": pd.Series(list("zyxzyxz")).astype("category"),
    })
    conditions = generate_conditions(data, n_bins=None)
    values = {}
    for condition in conditions:
        if condition.operator == "eq":
            values.setdefault(condition.feature, []).append(condition.value)
    # Values equal as dict keys (1, 1.0, True) collapse to the first one.
    assert values["mixed"] == [2, 1, "x"] and type(values["mixed"][1]) is int
    # Scalars are boxed as iterating the column boxes them.
    assert [type(v) for v in values["nullable"]] == [type(v) for v in data.nullable.dropna()][:2]
    assert values["category"] == ["z", "y", "x"]


@pytest.mark.parametrize("values", [[[1], [2], [1]], ["a", {"b": 1}]])
def test_unhashable_values_are_rejected_with_the_scikit_learn_message(values):
    frame = pd.DataFrame({"x": pd.Series(values, dtype=object)})
    with pytest.raises(TypeError, match="argument must be a string or number"):
        generate_conditions(frame)


def test_infinity_is_rejected_in_object_and_categorical_columns():
    for column in (
        pd.Series(["a", float("inf"), "a"], dtype=object),
        pd.Series([1.0, -np.inf, 1.0]).astype("category"),
    ):
        with pytest.raises(ValueError, match="infinity"):
            generate_conditions(pd.DataFrame({"x": column}))
    # Only values that occur count, not unused categories.
    unused = pd.Series([1.0, 2.0]).astype(pd.CategoricalDtype([1.0, 2.0, np.inf]))
    assert len(generate_conditions(pd.DataFrame({"x": unused}))) == 2


def test_tuple_values_are_compared_as_one_value():
    frame = pd.DataFrame({"x": pd.Series([(1, 2), (3, 4), (5, 6), None], dtype=object)})
    conditions = generate_conditions(frame)
    assert conditions == [
        *(Condition("x", "eq", value) for value in [(1, 2), (3, 4), (5, 6)]),
        Condition("x", "is_missing"),
    ]
    extra = [Condition("x", "ne", (1, 2)), Condition("x", "in", ((1, 2),))]
    coverage = Encoder([*conditions, *extra]).fit_transform(frame)
    assert coverage.storage.tolist() == [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
        [False, True, True, False],
        [True, False, False, False],
    ]
