import math

import numpy as np
import pandas as pd
import pytest
from helpers import assert_optimistic, subsets

from pysubgroup import (
    InterclassVarianceQF,
    KolmogorovSmirnovQF,
    MannWhitneyQF,
    NumericTarget,
    RankAUCQF,
    StandardDeviationQF,
    StandardQFNumeric,
    StandardQFNumericTscore,
    VarianceReductionQF,
    _numba,
)
from pysubgroup._evaluation import Backend, ChildStatistics

VALUES = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0, -5.0, 3.0]


@pytest.fixture
def target():
    frame = pd.DataFrame({"y": VALUES, "flag": [True, False] * 5}, index=[1, 1] * 5)
    return NumericTarget().fit(frame, "y")


def test_numeric_target_caches_dataset_centroids(target):
    assert target.values_.tolist() == VALUES
    assert target.mean_ == pytest.approx(np.mean(VALUES))
    assert target.median_ == pytest.approx(np.median(VALUES))


def test_numeric_target_rejects_bad_columns():
    with pytest.raises(TypeError, match="numeric column"):
        NumericTarget().fit(pd.DataFrame({"y": [True, False]}), "y")
    with pytest.raises(ValueError, match="finite"):
        NumericTarget().fit(pd.DataFrame({"y": [1.0, np.nan]}), "y")


def test_standard_qf_numeric_matches_its_definition(target):
    mask = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0], dtype=bool)
    chosen = np.array(VALUES)[mask]
    for a in (0, 0.5, 1):
        mean = StandardQFNumeric(a).evaluate(mask, target)
        assert mean == pytest.approx(4**a * (chosen.mean() - np.mean(VALUES)))
        median = StandardQFNumeric(a, centroid="median").evaluate(mask, target)
        assert median == pytest.approx(4**a * (np.median(chosen) - np.median(VALUES)))
    assert math.isnan(StandardQFNumeric(1).evaluate(np.zeros(10, dtype=bool), target))


@pytest.mark.parametrize("direction", ["positive", "negative", "both"])
@pytest.mark.parametrize("a", [0, 0.5, 1])
@pytest.mark.parametrize("estimator,centroid", [
    ("sum", "mean"), ("max", "mean"), ("average", "mean"), ("order", "mean"), ("max", "median"),
])
def test_numeric_estimates_bound_every_subset(target, a, estimator, centroid, direction):
    qf = StandardQFNumeric(a, estimator=estimator, centroid=centroid, direction=direction)
    rng = np.random.default_rng(1)
    for _ in range(4):
        mask = rng.random(10) < 0.8
        if mask.any():
            assert_optimistic(qf, target, mask)


def test_standard_qf_numeric_validates_parameters():
    for a in ("1", True, -0.5, 2):
        with pytest.raises(ValueError, match="a must be"):
            StandardQFNumeric(a)
        with pytest.raises(ValueError, match="a must be"):
            StandardQFNumeric(a, "order")
    with pytest.raises(ValueError, match="centroid"):
        StandardQFNumeric(1, centroid="mode")
    with pytest.raises(ValueError, match="estimator"):
        StandardQFNumeric(1, estimator="sum", centroid="median")


def test_t_score(target):
    mask = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    chosen = np.array(VALUES)[mask]
    expected = 4**0.5 * (chosen.mean() - np.mean(VALUES)) / chosen.std()
    assert StandardQFNumericTscore().evaluate(mask, target) == pytest.approx(expected)
    single = np.zeros(10, dtype=bool)
    single[0] = True
    assert StandardQFNumericTscore().evaluate(single, target) == 0.0
    assert StandardQFNumericTscore().optimistic_estimate(mask, target) == math.inf
    assert StandardQFNumericTscore().optimistic_estimate(~(mask | ~mask), target) == -math.inf


@pytest.mark.parametrize("kernels", [False, True])
@pytest.mark.parametrize("constant", [0.1, 3.0])
def test_t_score_of_constant_subgroups_is_zero(kernels, constant, monkeypatch):
    # np.std of three values 0.1 is about 1e-17, not 0.
    values = [constant] * 3 + [0.7, 0.3, 9.0]
    target = NumericTarget().fit(pd.DataFrame({"y": values}), "y")
    masks = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=bool)
    monkeypatch.setattr(_numba, "ENABLED", kernels and _numba.AVAILABLE)
    qf = StandardQFNumericTscore()
    qualities = [qf.evaluate(mask, target) for mask in masks]
    assert qualities[:2] == [0.0, 0.0] and qualities[2] != 0
    statistics = ChildStatistics(Backend(masks), None, range(len(masks)))
    batched = qf.evaluate_many(statistics, target)
    assert batched[:2].tolist() == [0.0, 0.0]
    assert batched[2] == pytest.approx(qualities[2], rel=1e-12)


@pytest.mark.parametrize("centroid", ["mean", "median"])
def test_decreases_are_increases_of_the_negated_values(target, centroid):
    negated = NumericTarget().fit(pd.DataFrame({"y": [-v for v in VALUES]}), "y")
    rng = np.random.default_rng(4)
    estimator = "max"
    for _ in range(5):
        mask = rng.random(10) < 0.6
        increase = StandardQFNumeric(0.5, estimator, centroid).evaluate(mask, target)
        decrease = StandardQFNumeric(0.5, estimator, centroid, "negative")
        assert decrease.evaluate(mask, target) == pytest.approx(-increase, nan_ok=True)
        assert decrease.evaluate(mask, target) == pytest.approx(
            StandardQFNumeric(0.5, estimator, centroid).evaluate(mask, negated), nan_ok=True
        )
        assert decrease.optimistic_estimate(mask, target) == pytest.approx(
            StandardQFNumeric(0.5, estimator, centroid).optimistic_estimate(mask, negated)
        )
        both = StandardQFNumeric(0.5, estimator, centroid, "both").evaluate(mask, target)
        assert both == pytest.approx(abs(increase), nan_ok=True)
    with pytest.raises(ValueError, match="direction"):
        StandardQFNumeric(1, direction="up")


def test_t_score_directions(target):
    mask = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    increase = StandardQFNumericTscore().evaluate(mask, target)
    assert StandardQFNumericTscore("negative").evaluate(mask, target) == -increase
    assert StandardQFNumericTscore("both").evaluate(mask, target) == abs(increase)


# Lemmerich, Atzmueller and Puppe (DMKD 2016): Table 1 and the subgroup P_A of sel_A.
TABLE_1 = [100.0, 75.0, 60.0, 53.0, 40.0, 35.0, 25.0, 12.0]
SEL_A = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=bool)


@pytest.fixture
def table_1():
    return NumericTarget().fit(pd.DataFrame({"y": TABLE_1}), "y")


def test_target_orders_and_ranks_rows(table_1):
    target = NumericTarget().fit(pd.DataFrame({"y": [2.0, 5.0, 2.0, 1.0]}), "y")
    assert target.order_.tolist() == [1, 0, 2, 3]  # ties keep the row order
    assert target.sorted_values_.tolist() == [5.0, 2.0, 2.0, 1.0]
    assert target.ranks_.tolist() == [2.5, 4.0, 2.5, 1.0]
    assert table_1.ranks_[SEL_A].tolist() == [8, 7, 5, 1]  # Example 9
    assert target.std_ == pytest.approx(np.std([2.0, 5.0, 2.0, 1.0]))


def test_paper_examples_for_mean_based_measures(table_1):
    # Example 3: the ordering-based bound of q_mean^0.5 is sqrt(2) * 37.5.
    order = StandardQFNumeric(0.5, "order")
    assert order.optimistic_estimate(SEL_A, table_1) == pytest.approx(2**0.5 * 37.5)
    # Example 4: fast bounds of the impact (a = 1) after one and two rows.
    impact = StandardQFNumeric(1)
    assert impact.fast_bound(1, 100.0, 100.0, 4, 50.0) == pytest.approx(200)
    assert impact.fast_bound(2, 175.0, 75.0, 4, 50.0) == pytest.approx(125)
    # Example 5: oe^1_mean = 78 and the bound of Theorem 12, 3^a * 50.
    for a in (0, 0.5, 1):
        assert StandardQFNumeric(a, "sum").optimistic_estimate(SEL_A, table_1) == 78
        assert StandardQFNumeric(a, "max").optimistic_estimate(SEL_A, table_1) == (
            pytest.approx(3**a * 50)
        )
        assert StandardQFNumeric(a, "combined").optimistic_estimate(SEL_A, table_1) == (
            pytest.approx(min(78, 3**a * 50))
        )
    # Theorem 13: the symmetric bound of the impact keeps the larger sum.
    both = StandardQFNumeric(1, "sum", direction="both")
    assert both.optimistic_estimate(SEL_A, table_1) == 78


def test_paper_example_6_variance_reduction(table_1):
    # Theorem 14 on Table 1 (i0 = 8): 4 / 4 * 50^2, and (50 + 25 + 3)^2 / 7. Example 6
    # computes the second bound from squared deviations instead (447.7), which is not
    # Theorem 14 and not a valid bound (see test_variance_reduction_example_6_is_no_bound).
    qf = VarianceReductionQF()
    assert qf._closed(4, 50.0, -38.0, 78.0, -38.0, table_1) == pytest.approx(78**2 / 7)
    assert qf.optimistic_estimate(SEL_A, table_1) == pytest.approx(min(2500, 78**2 / 7))
    chosen = np.array(TABLE_1)[SEL_A]
    expected = 4 / 4 * (chosen.mean() - 50) ** 2
    assert qf.evaluate(SEL_A, table_1) == pytest.approx(expected)
    assert InterclassVarianceQF().evaluate(SEL_A, table_1) == pytest.approx(8 * expected)


def test_variance_reduction_example_6_is_no_bound():
    values = [60.0] * 4 + [40.0] * 4
    target = NumericTarget().fit(pd.DataFrame({"y": values}), "y")
    mask = np.array([True] * 4 + [False] * 4)
    quality = VarianceReductionQF().evaluate(mask, target)
    assert quality == pytest.approx(100)
    assert quality > 4 * 10**2 / 7  # Example 6's formula
    assert VarianceReductionQF().optimistic_estimate(mask, target) >= quality


def test_interclass_variance_needs_two_passes(table_1):
    # Theorem 4 calls q_iv one-pass estimable, but the best subset of {60, 25, 12} is the
    # suffix {25, 12} (deviations -25, -38), not a prefix in descending order.
    mask = np.array([0, 0, 1, 0, 0, 0, 1, 1], dtype=bool)
    qf = InterclassVarianceQF("order")
    best = max(qf.evaluate(s, table_1) for s in subsets(mask))
    assert best == pytest.approx(8 * 63**2 / (2 * 6))
    prefixes = [[2], [2, 6], [2, 6, 7]]
    one_pass = max(qf.evaluate(np.isin(np.arange(8), rows), table_1) for rows in prefixes)
    assert one_pass < best
    assert qf.optimistic_estimate(mask, table_1) == pytest.approx(best)
    # q_iv as defined: i_P (mu_P - mu0)^2 + i_notP (mu_notP - mu0)^2.
    values = np.array(TABLE_1)
    definition = 4 * (values[SEL_A].mean() - 50) ** 2 + 4 * (values[~SEL_A].mean() - 50) ** 2
    assert qf.evaluate(SEL_A, table_1) == pytest.approx(definition)


def test_paper_example_7_median():
    values = [3.0, 2.0, 2.0, 0.0, -1.0, -1.0, 1.0, 1.0]
    target = NumericTarget().fit(pd.DataFrame({"y": values}), "y")
    assert target.median_ == 1
    mask = np.array([True] * 6 + [False] * 2)
    qf = StandardQFNumeric(1, "order", "median")
    assert qf.optimistic_estimate(mask, target) == 5  # the first five rows


def test_paper_examples_for_ks_and_rank_measures(table_1):
    # Example 8: Theorem 16 on four of eight rows.
    assert KolmogorovSmirnovQF().optimistic_estimate(SEL_A, table_1) == pytest.approx(2**0.5)
    # Example 9: oe^1 = 6.5 and sqrt(3) * 3.5 for q_mw', scaled to the z-statistic.
    scale = (12 / 9) ** 0.5
    assert MannWhitneyQF().optimistic_estimate(SEL_A, table_1) == pytest.approx(
        scale * min(6.5, 3**0.5 * 3.5)
    )
    ranks = np.array([8, 7, 5, 1])
    z = 4 * (ranks.mean() - 4.5) / (4 * 4 * 9 / 12) ** 0.5
    assert MannWhitneyQF().evaluate(SEL_A, table_1) == pytest.approx(z)
    # Example 10: the paper's q_auc (decrease) is bounded by (8 - 1) / 7 = 1.
    paper_auc = RankAUCQF(direction="negative")
    assert paper_auc.optimistic_estimate(SEL_A, table_1) == 1
    complement_ranks = np.array([6, 4, 3, 2])
    expected = (complement_ranks.sum() - 4 * 5 / 2) / 16
    assert paper_auc.evaluate(SEL_A, table_1) == pytest.approx(expected)
    assert RankAUCQF().evaluate(SEL_A, table_1) == pytest.approx(1 - expected)


def pairwise_auc(values, mask):
    inside, outside = values[mask], values[~mask]
    wins = (inside[:, None] > outside[None, :]).sum() + 0.5 * (
        inside[:, None] == outside[None, :]
    ).sum()
    return wins / (len(inside) * len(outside))


def test_rank_measures_match_their_definitions():
    rng = np.random.default_rng(3)
    values = rng.integers(0, 6, 30).astype(float)
    target = NumericTarget().fit(pd.DataFrame({"y": values}), "y")
    for _ in range(5):
        mask = rng.random(30) < 0.4
        auc = pairwise_auc(values, mask)
        assert RankAUCQF().evaluate(mask, target) == pytest.approx(auc)
        assert RankAUCQF(direction="negative").evaluate(mask, target) == pytest.approx(1 - auc)
        assert RankAUCQF(direction="both").evaluate(mask, target) == pytest.approx(
            max(auc, 1 - auc)
        )
        # Kolmogorov-Smirnov: the largest gap of the distribution functions.
        grid = np.unique(values)
        inside = (values[mask][:, None] <= grid).mean(axis=0)
        outside = (values[~mask][:, None] <= grid).mean(axis=0)
        n_in = mask.sum()
        expected = (n_in * (30 - n_in) / 30) ** 0.5 * np.abs(inside - outside).max()
        assert KolmogorovSmirnovQF().evaluate(mask, target) == pytest.approx(expected)
        sd = values[mask].std()
        assert StandardDeviationQF(0.5).evaluate(mask, target) == pytest.approx(
            n_in**0.5 * (sd - values.std())
        )


def test_new_measures_on_empty_and_whole_subgroups(target):
    empty, whole = np.zeros(10, dtype=bool), np.ones(10, dtype=bool)
    for qf in ALL_NEW_MEASURES:
        assert math.isnan(qf.evaluate(empty, target))
        assert qf.optimistic_estimate(empty, target) == -math.inf
        expected = 0.5 if isinstance(qf, RankAUCQF) else 0.0
        assert qf.evaluate(whole, target) == expected
        assert qf._evaluate_ordered(whole, target)[0] == pytest.approx(expected)


ALL_NEW_MEASURES = [
    *(StandardQFNumeric(a, estimator, direction=direction)
      for a in (0, 0.5, 1) for estimator in ("combined", "order")
      for direction in ("positive", "negative", "both")),
    *(StandardQFNumeric(a, "order", "median", direction)
      for a in (0, 0.5, 1) for direction in ("positive", "negative", "both")),
    *(qf(estimator) for qf in (VarianceReductionQF, InterclassVarianceQF)
      for estimator in ("closed", "order")),
    *(qf(estimator, direction) for qf in (MannWhitneyQF, RankAUCQF)
      for estimator in ("closed", "order") for direction in ("positive", "negative", "both")),
    *(StandardDeviationQF(a, direction) for a in (0, 0.5, 1)
      for direction in ("positive", "negative", "both")),
    KolmogorovSmirnovQF(),
]


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("qf", ALL_NEW_MEASURES, ids=lambda qf: f"{type(qf).__name__}{vars(qf)}")
def test_new_estimates_bound_every_subset(qf, ties):
    rng = np.random.default_rng(7)
    for _ in range(4):
        n = int(rng.integers(2, 10))
        values = rng.integers(0, 4, n).astype(float) if ties else rng.normal(size=n)
        target = NumericTarget().fit(pd.DataFrame({"y": values}), "y")
        mask = rng.random(n) < 0.8
        mask[rng.integers(n)] = True
        assert_optimistic(qf, target, mask)
        quality, estimate = qf._evaluate_ordered(mask, target)
        assert quality == pytest.approx(qf.evaluate(mask, target))
        best = max(qf.evaluate(s, target) for s in subsets(mask))
        assert estimate >= best - 1e-9
        if getattr(qf, "estimator", None) == "order":
            # Ordering-based bounds are tight (Theorems 1-6).
            assert qf.optimistic_estimate(mask, target) == pytest.approx(best)


def test_new_measures_validate_parameters():
    with pytest.raises(ValueError, match="estimator"):
        VarianceReductionQF("sum")
    with pytest.raises(ValueError, match="estimator"):
        MannWhitneyQF("max")
    with pytest.raises(ValueError, match="direction"):
        RankAUCQF(direction="up")
    with pytest.raises(ValueError, match="non-negative"):
        StandardDeviationQF(-1)
    with pytest.raises(ValueError, match="estimator"):
        StandardQFNumeric(1, "combined", "median")
    with pytest.raises(TypeError, match="NumericTarget"):
        from pysubgroup import BinaryTarget
        frame = pd.DataFrame({"y": [0, 1]})
        KolmogorovSmirnovQF().evaluate(np.ones(2, dtype=bool), BinaryTarget().fit(frame, "y", 1))
