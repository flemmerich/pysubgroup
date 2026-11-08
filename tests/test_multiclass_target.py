import math

import numpy as np
import pandas as pd
import pytest
from helpers import assert_optimistic, subsets

from pysubgroup import (
    BinaryTarget,
    ClassDistributionL1QF,
    GiniQF,
    MaxConfidenceQF,
    MinMajorityCountFilter,
    MulticlassChiSquaredQF,
    MulticlassTarget,
    SplitQF,
)

LABELS = ["a", "b", "c", "a", "a", "b", "c", "c", "a", "b"]
QFS = [SplitQF(), MulticlassChiSquaredQF(), GiniQF(), ClassDistributionL1QF()]


@pytest.fixture
def target():
    frame = pd.DataFrame({"y": LABELS, "x": range(10)}, index=[3] * 10)
    return MulticlassTarget().fit(frame, "y")


def test_multiclass_target_stores_the_class_distribution(target):
    assert target.classes_.tolist() == ["a", "b", "c"]
    assert target.codes_.tolist() == [0, 1, 2, 0, 0, 1, 2, 2, 0, 1]
    assert target.class_counts_.tolist() == [4, 3, 3]
    assert target.priors_ == pytest.approx([0.4, 0.3, 0.3])
    assert target.attributes_ == ("y",)


def test_multiclass_target_validates_labels():
    with pytest.raises(ValueError, match="non-missing"):
        MulticlassTarget().fit(pd.DataFrame({"y": ["a", None]}), "y")
    mixed = MulticlassTarget().fit(pd.DataFrame({"y": [1, "a", 1]}), "y")
    assert mixed.class_counts_.tolist() == [2, 1]


def test_qualities_match_their_definitions(target):
    # Rows a, b, a, a: n = 4, p = (0.75, 0.25, 0), p0 = (0.4, 0.3, 0.3).
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    squared = 0.35**2 + 0.05**2 + 0.3**2
    assert SplitQF().evaluate(mask, target) == pytest.approx(4 * squared)
    assert MulticlassChiSquaredQF().evaluate(mask, target) == pytest.approx(
        4 * (0.35**2 / 0.4 + 0.05**2 / 0.3 + 0.3**2 / 0.3)
    )
    assert GiniQF().evaluate(mask, target) == pytest.approx(4 / 6 * squared)
    assert ClassDistributionL1QF().evaluate(mask, target) == pytest.approx(0.4 * 0.7)


@pytest.mark.parametrize("qf", QFS, ids=lambda qf: type(qf).__name__)
def test_whole_dataset_and_empty_subgroups(target, qf):
    assert qf.evaluate(np.ones(10, dtype=bool), target) == 0.0
    assert math.isnan(qf.evaluate(np.zeros(10, dtype=bool), target))
    assert qf.optimistic_estimate(np.zeros(10, dtype=bool), target) == -math.inf


@pytest.mark.parametrize("qf", [*QFS, ClassDistributionL1QF(max_exact_classes=0)],
                         ids=["split", "chi2", "gini", "l1", "l1-relaxed"])
def test_optimistic_estimates_bound_every_subset(target, qf):
    rng = np.random.default_rng(0)
    masks = [np.ones(10, dtype=bool), *(rng.random(10) < 0.7 for _ in range(6))]
    for mask in masks:
        if mask.any():
            assert_optimistic(qf, target, mask)


def test_class_distribution_l1_estimate_is_tight(target):
    qf = ClassDistributionL1QF()
    for mask in (np.ones(10, dtype=bool), np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=bool)):
        best = max(qf.evaluate(subset, target) for subset in subsets(mask))
        assert qf.optimistic_estimate(mask, target) == pytest.approx(best)
        assert ClassDistributionL1QF(0).optimistic_estimate(mask, target) >= best - 1e-12


def test_estimates_on_random_class_counts():
    """The bounds hold for every sub-multiset of random class counts (exhaustive)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        dataset = rng.integers(1, 6, size=rng.integers(2, 5))
        counts = np.array([rng.integers(0, c + 1) for c in dataset])
        if not counts.sum():
            continue
        grids = np.meshgrid(*[np.arange(c + 1) for c in counts], indexing="ij")
        refinements = [r for r in np.stack([g.ravel() for g in grids], axis=1) if r.sum()]
        for qf in QFS:
            best = max(qf.quality(r, dataset) for r in refinements)
            assert best <= qf.estimate(counts, dataset) + 1e-12, type(qf).__name__


def test_multiclass_qfs_require_a_multiclass_target():
    binary = BinaryTarget().fit(pd.DataFrame({"y": LABELS}), "y", "a")
    with pytest.raises(TypeError, match="MulticlassTarget"):
        SplitQF().evaluate(np.ones(10, dtype=bool), binary)
    with pytest.raises(ValueError, match="integer"):
        ClassDistributionL1QF(max_exact_classes=2.5)


def test_max_confidence_is_the_majority_share_and_bounded_by_one(target):
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)  # a, b, a, a
    qf = MaxConfidenceQF()
    assert qf.evaluate(mask, target) == pytest.approx(0.75)
    assert qf.evaluate(np.ones(10, dtype=bool), target) == pytest.approx(0.4)
    assert math.isnan(qf.evaluate(np.zeros(10, dtype=bool), target))
    assert qf.optimistic_estimate(mask, target) == 1.0
    assert qf.optimistic_estimate(np.zeros(10, dtype=bool), target) == -math.inf
    assert_optimistic(qf, target, mask)


def test_min_majority_count_filter_counts_the_most_frequent_class(target):
    mask = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=bool)  # a, b, a, a, c
    assert MinMajorityCountFilter(3).is_satisfied(mask, target, ())
    assert not MinMajorityCountFilter(4).is_satisfied(mask, target, ())
    assert MinMajorityCountFilter(0.3).is_satisfied(mask, target, ())
    with pytest.raises(TypeError, match="MulticlassTarget"):
        binary = BinaryTarget().fit(pd.DataFrame({"y": LABELS}), "y", "a")
        MinMajorityCountFilter(2).is_satisfied(mask, binary, ())
    with pytest.raises(ValueError):
        MinMajorityCountFilter(-1)
