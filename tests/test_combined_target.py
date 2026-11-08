import math

import numpy as np
import pandas as pd
import pytest
from helpers import assert_optimistic

from pysubgroup import (
    DFS,
    AreaQF,
    BinaryTarget,
    ChiSquaredQF,
    CombinedQF,
    CountQF,
    GeneralizationAwareQF,
    LiftQF,
    MannWhitneyQF,
    NumericTarget,
    QualityFunction,
    StandardDeviationQF,
    StandardQF,
    StandardQFNumeric,
    SubgroupDiscoveryTask,
    WRAccQF,
    generate_conditions,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(3)
    rows = 11
    return pd.DataFrame(
        {"label": rng.integers(0, 2, rows), "value": rng.normal(size=rows).round(2)},
        index=rng.permutation(rows) % 4,
    )


def random_masks(rows, count=6, seed=0):
    rng = np.random.default_rng(seed)
    masks = [rng.random(rows) < 0.6 for _ in range(count)]
    return [*masks, np.ones(rows, dtype=bool), np.zeros(rows, dtype=bool)]


def test_combined_qf_is_the_weighted_sum(data):
    target = BinaryTarget().fit(data, "label", 1)
    qf = CombinedQF([WRAccQF(), LiftQF(), CountQF()], [2.0, 0.5, 0])
    for mask in random_masks(len(data))[:-1]:
        expected = 2 * WRAccQF().evaluate(mask, target) + 0.5 * LiftQF().evaluate(mask, target)
        assert qf.evaluate(mask, target) == pytest.approx(expected)
    assert math.isnan(qf.evaluate(np.zeros(len(data), dtype=bool), target))
    assert CombinedQF([WRAccQF()]).evaluate(mask, target) == WRAccQF().evaluate(mask, target)


BINARY = [
    ([WRAccQF(), LiftQF()], [1, 1]),
    ([WRAccQF(), LiftQF("negative")], [0.3, 1]),
    ([StandardQF(0.5, "both"), WRAccQF("negative")], [1, 2]),
    ([WRAccQF(), CountQF()], [1, 0.01]),
    ([CombinedQF([WRAccQF(), CountQF()], [1, 0.1]), LiftQF()], [1, 0]),
]
NUMERIC = [
    ([StandardQFNumeric(1), StandardDeviationQF(1)], [1, 0.5]),
    ([StandardQFNumeric(0.5, "order", direction="negative"), MannWhitneyQF()], [1, 1]),
    ([StandardQFNumeric(0, direction="both"), StandardDeviationQF(0.5, "both")], [2, 1]),
    ([StandardQFNumeric(0.5, centroid="median"), StandardQFNumeric(1, direction="negative")],
     [1, 1]),
]


@pytest.mark.parametrize(("measures", "weights"), BINARY + NUMERIC)
def test_combined_estimates_bound_every_subset(data, measures, weights):
    numeric = isinstance(measures[0], StandardQFNumeric)
    target = (NumericTarget().fit(data, "value") if numeric
              else BinaryTarget().fit(data, "label", 1))
    qf = CombinedQF(measures, weights)
    for mask in random_masks(len(data)):
        assert_optimistic(qf, target, mask)
        assert qf.optimistic_estimate(mask, target) < np.inf
    empty = np.zeros(len(data), dtype=bool)
    assert qf.optimistic_estimate(empty, target) == -np.inf


def test_estimate_is_tight_when_one_subset_attains_every_part(data):
    # Both StandardQF estimates are attained by the subgroup's positives.
    target = BinaryTarget().fit(data, "label", 1)
    qf = CombinedQF([WRAccQF(), StandardQF(0.5)], [1, 2])
    mask = np.ones(len(data), dtype=bool)
    mask[:3] = False
    positives = mask & target.positives_
    assert qf.optimistic_estimate(mask, target) == pytest.approx(qf.evaluate(positives, target))


def test_missing_bounds_make_the_estimate_infinite(data):
    target = BinaryTarget().fit(data, "label", 1)
    mask = random_masks(len(data))[0]
    assert CombinedQF([WRAccQF(), ChiSquaredQF()]).optimistic_estimate(mask, target) == np.inf
    # Weight 0 ignores the unbounded measure.
    qf = CombinedQF([WRAccQF(), ChiSquaredQF()], [1, 0])
    assert qf.optimistic_estimate(mask, target) == WRAccQF().optimistic_estimate(mask, target)
    # A part bounded by -inf (no subset scores above it) keeps the sum at -inf.
    empty = np.zeros(len(data), dtype=bool)
    assert CombinedQF([WRAccQF(), ChiSquaredQF()]).optimistic_estimate(empty, target) == -np.inf


def test_invalid_combinations_are_rejected():
    with pytest.raises(TypeError, match="incompatible"):
        CombinedQF([WRAccQF(), StandardQFNumeric(1)])
    with pytest.raises(TypeError, match="cannot be combined"):
        CombinedQF([WRAccQF(), AreaQF()])
    with pytest.raises(TypeError, match="cannot be combined"):
        CombinedQF([GeneralizationAwareQF(WRAccQF())])
    with pytest.raises(ValueError, match="weights for"):
        CombinedQF([WRAccQF(), LiftQF()], [1])
    with pytest.raises(ValueError, match="positive"):
        CombinedQF([WRAccQF()], [0])
    for weights in ([1, -0.5], [1, np.nan], [1, np.inf], [1, True]):
        with pytest.raises(ValueError, match="non-negative finite"):
            CombinedQF([WRAccQF(), LiftQF()], weights)
    with pytest.raises(ValueError, match="at least one"):
        CombinedQF([])
    assert CombinedQF([CountQF(), WRAccQF()]).target_type is BinaryTarget


def test_wrong_target_is_rejected(data):
    with pytest.raises(TypeError, match="BinaryTarget"):
        CombinedQF([WRAccQF(), CountQF()]).evaluate(
            np.ones(len(data), dtype=bool), NumericTarget().fit(data, "value")
        )


class Unbounded(QualityFunction):
    def __init__(self, qf):
        self.qf = qf

    def evaluate(self, coverage, target):
        return self.qf.evaluate(coverage, target)


@pytest.mark.parametrize("batched", [False, True])
def test_pruning_search_finds_the_unpruned_result(batched):
    rng = np.random.default_rng(5)
    rows = 300
    frame = pd.DataFrame({f"a{j}": rng.choice(list("xyz"), rows) for j in range(5)})
    frame["label"] = (rng.random(rows) < 0.2 + 0.5 * (frame.a0 == "x")).astype(int)
    conditions = generate_conditions(frame.drop(columns="label"))
    target = BinaryTarget().fit(frame, "label", 1)
    qf = CombinedQF([WRAccQF(), StandardQF(0.5, "negative"), CountQF()], [1, 0.2, 1e-4])
    task = SubgroupDiscoveryTask(frame, target, conditions, qf, result_set_size=10, depth=3,
                                 batched=batched)
    pruned = DFS().execute(task)
    exhaustive = DFS().execute(
        SubgroupDiscoveryTask(frame, target, conditions, Unbounded(qf), result_set_size=10,
                              depth=3, batched=False)
    )
    assert [rule for _, rule in pruned] == [rule for _, rule in exhaustive]
    np.testing.assert_allclose([q for q, _ in pruned], [q for q, _ in exhaustive])
