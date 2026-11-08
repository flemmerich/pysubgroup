import math

import numpy as np
import pandas as pd
import pytest
from helpers import assert_optimistic
from scipy.stats import chi2_contingency

from pysubgroup import (
    BinaryTarget,
    ChiSquaredQF,
    LabelledTarget,
    LiftQF,
    NumericTarget,
    SimpleBinomialQF,
    StandardQF,
    WRAccQF,
)


@pytest.fixture
def data():
    # Duplicate index labels: statistics must be positional.
    return pd.DataFrame(
        {"label": ["a", "b", "a", None, "a", "b", "b", "a", "b", "b"],
         "value": np.arange(10.0)},
        index=[5, 5, 2, 2, 9, 9, 1, 1, 0, 0],
    )


def test_binary_target_stores_positional_positive_labels(data):
    target = BinaryTarget().fit(data, "label", "a")
    assert isinstance(target, LabelledTarget)
    assert target.positives_.tolist() == [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    assert target.n_positives_ == 4 and target.n_samples_ == 10
    missing = BinaryTarget().fit(data, "label", None)
    assert missing.positives_.tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="missing"):
        BinaryTarget().fit(data, "absent", 1)


def test_standard_qf_matches_its_definition(data):
    target = BinaryTarget().fit(data, "label", "a")
    mask = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)  # 4 rows, 3 positives
    for a in (0, 0.5, 1):
        expected = (4 / 10) ** a * (3 / 4 - 4 / 10)
        assert StandardQF(a).evaluate(mask, target) == pytest.approx(expected)
    assert LiftQF().evaluate(mask, target) == StandardQF(0).evaluate(mask, target)
    assert SimpleBinomialQF().evaluate(mask, target) == StandardQF(0.5).evaluate(mask, target)
    assert WRAccQF().evaluate(mask, target) == StandardQF(1).evaluate(mask, target)
    assert math.isnan(StandardQF(1).evaluate(np.zeros(10, dtype=bool), target))


@pytest.mark.parametrize("direction", ["positive", "negative", "both"])
@pytest.mark.parametrize("a", [0, 0.5, 1])
def test_standard_qf_estimates_bound_every_subset(data, a, direction):
    target = BinaryTarget().fit(data, "label", "a")
    qf = StandardQF(a, direction=direction)
    rng = np.random.default_rng(0)
    for _ in range(5):
        mask = rng.random(10) < 0.7
        if mask.any():
            assert_optimistic(qf, target, mask)
    assert_optimistic(qf, target, ~target.positives_)
    assert_optimistic(qf, target, target.positives_)


@pytest.mark.parametrize("a", [0, 0.5, 1])
def test_decreases_are_increases_of_the_complementary_label(data, a):
    target = BinaryTarget().fit(data, "label", "a")
    complement = BinaryTarget().fit(data.assign(other=data.label != "a"), "other", True)
    rng = np.random.default_rng(3)
    for _ in range(5):
        mask = rng.random(10) < 0.6
        increase = StandardQF(a).evaluate(mask, target)
        decrease = StandardQF(a, direction="negative").evaluate(mask, target)
        assert decrease == pytest.approx(-increase, nan_ok=True)
        assert decrease == pytest.approx(StandardQF(a).evaluate(mask, complement), nan_ok=True)
        assert StandardQF(a, direction="both").evaluate(mask, target) == pytest.approx(
            abs(increase), nan_ok=True
        )
        assert StandardQF(a, direction="negative").optimistic_estimate(
            mask, target
        ) == pytest.approx(StandardQF(a).optimistic_estimate(mask, complement))
    assert (WRAccQF("negative").a, WRAccQF("negative").direction) == (1.0, "negative")
    assert (LiftQF("both").a, SimpleBinomialQF("both").a) == (0.0, 0.5)
    with pytest.raises(ValueError, match="direction"):
        StandardQF(a, direction="down")


@pytest.mark.parametrize("a", ["1", True, -0.5, 2])
def test_standard_qf_rejects_exponents_without_valid_estimates(a):
    with pytest.raises(ValueError, match="a must be"):
        StandardQF(a)


def test_chi_squared_matches_scipy(data):
    target = BinaryTarget().fit(data, "label", "a")
    mask = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
    table = [[np.sum(mask & target.positives_), np.sum(mask & ~target.positives_)],
             [np.sum(~mask & target.positives_), np.sum(~mask & ~target.positives_)]]
    statistic, p, _, _ = chi2_contingency(table, correction=False)
    assert ChiSquaredQF().evaluate(mask, target) == pytest.approx(statistic)
    assert ChiSquaredQF(stat="p").evaluate(mask, target) == pytest.approx(p)


def test_binary_measures_reject_other_targets(data):
    numeric = NumericTarget().fit(data, "value")
    with pytest.raises(TypeError, match="requires a BinaryTarget"):
        StandardQF(1).evaluate(np.ones(10, dtype=bool), numeric)


def test_tuple_labels_are_one_value():
    data = pd.DataFrame({"x": pd.Series([(1, 2), (3, 4), None], dtype=object)})
    assert BinaryTarget().fit(data, "x", (1, 2)).positives_.tolist() == [True, False, False]
