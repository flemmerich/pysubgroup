import math

import numpy as np
import pandas as pd
import pytest
from helpers import assert_optimistic, subsets

from pysubgroup import (
    ClassDistributionL1QF,
    MeanVectorL1QF,
    MulticlassTarget,
    VectorTarget,
)

SHARES = np.array([
    [0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.7, 0.2, 0.1], [0.3, 0.3, 0.4],
    [0.4, 0.4, 0.2], [0.1, 0.1, 0.8], [0.6, 0.1, 0.3], [0.2, 0.5, 0.3], [0.3, 0.4, 0.3],
])


@pytest.fixture
def frame():
    return pd.DataFrame(SHARES, columns=["spd", "cdu", "fdp"], index=[7] * 10)


@pytest.fixture
def target(frame):
    return VectorTarget().fit(frame, "spd", "cdu", "fdp")


def test_vector_target_stores_values_and_mean(target):
    assert target.values_ == pytest.approx(SHARES)
    assert target.mean_ == pytest.approx(SHARES.mean(axis=0))
    assert target.attributes_ == ("spd", "cdu", "fdp")


def test_vector_target_validates_columns(frame):
    with pytest.raises(ValueError, match="at least one"):
        VectorTarget().fit(frame)
    with pytest.raises(TypeError, match="numeric columns"):
        VectorTarget().fit(frame.assign(flag=True), "spd", "flag")
    with pytest.raises(ValueError, match="finite"):
        VectorTarget().fit(frame.assign(spd=np.nan), "spd")


def test_mean_vector_l1_matches_its_definition(target):
    mask = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    expected = 0.3 * np.abs(SHARES[mask].mean(axis=0) - SHARES.mean(axis=0)).sum()
    assert MeanVectorL1QF().evaluate(mask, target) == pytest.approx(expected)
    assert MeanVectorL1QF().evaluate(np.ones(10, dtype=bool), target) == 0.0
    assert math.isnan(MeanVectorL1QF().evaluate(np.zeros(10, dtype=bool), target))
    assert MeanVectorL1QF().optimistic_estimate(np.zeros(10, dtype=bool), target) == -math.inf


@pytest.mark.parametrize("max_exact_dimensions", [8, 0])
def test_mean_vector_l1_estimate_bounds_every_subset(target, max_exact_dimensions):
    qf = MeanVectorL1QF(max_exact_dimensions)
    rng = np.random.default_rng(0)
    for mask in [np.ones(10, dtype=bool), *(rng.random(10) < 0.7 for _ in range(6))]:
        if mask.any():
            assert_optimistic(qf, target, mask)


def test_mean_vector_l1_estimate_is_tight(target):
    qf = MeanVectorL1QF()
    for mask in (np.ones(10, dtype=bool), np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=bool)):
        best = max(qf.evaluate(subset, target) for subset in subsets(mask))
        assert qf.optimistic_estimate(mask, target) == pytest.approx(best)


def test_one_hot_vectors_match_the_class_distribution_measure():
    labels = pd.Series(list("abcaabccab"))
    one_hot = pd.get_dummies(labels, dtype=float)
    vector = VectorTarget().fit(one_hot, *one_hot.columns)
    multiclass = MulticlassTarget().fit(labels.to_frame("y"), "y")
    rng = np.random.default_rng(2)
    for _ in range(10):
        mask = rng.random(10) < 0.6
        if not mask.any():
            continue
        assert MeanVectorL1QF().evaluate(mask, vector) == pytest.approx(
            ClassDistributionL1QF().evaluate(mask, multiclass)
        )
        assert MeanVectorL1QF().optimistic_estimate(mask, vector) == pytest.approx(
            ClassDistributionL1QF().optimistic_estimate(mask, multiclass)
        )
