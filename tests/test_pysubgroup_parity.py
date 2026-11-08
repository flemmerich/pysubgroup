"""Black-box parity with pysubgroup 0.x (https://github.com/flemmerich/pysubgroup).

pysubgroup 0.x is used only as a reference to compare results. As v2 took over its import
name, these tests expect a copy of the 0.x package renamed to ``pysubgroup_legacy``
on the Python path; without it they are skipped.

Known differences are excluded where they apply: pysubgroup 0.x's estimates of
``StandardQF`` without positives, of the numeric ``max`` estimator without values
above the mean and of the median (all too small), constant subgroups for the
t-score, and the empty "Dataset" subgroup that some pysubgroup 0.x searches report.
pysubgroup 0.x ignores ``invert``, so only increases are compared. Its ``"order"``
estimator sorts the data frame in place, so every pysubgroup 0.x quality function
gets its own copy.
"""

import math
import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest

from pysubgroup import (
    DFS,
    Apriori,
    BeamSearch,
    BestFirstSearch,
    BinaryTarget,
    ChiSquaredQF,
    CountQF,
    FITarget,
    LikelihoodQF,
    LikelihoodTarget,
    NumericTarget,
    PolynomialRegressionModel,
    Rule,
    StandardQF,
    StandardQFNumeric,
    StandardQFNumericTscore,
    SubgroupDiscoveryTask,
    WRAccQF,
    generate_conditions,
)
from pysubgroup.utils import subgroup_masks

ps = pytest.importorskip("pysubgroup_legacy")


def frame(seed, rows=300):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({f"a{j}": rng.choice(list("xyzw"), rows) for j in range(4)})
    # Integers, so x > b is exactly x >= floor(b) + 1 in pysubgroup 0.x's [lower, upper).
    data["n"] = rng.integers(0, 30, rows)
    data["label"] = (rng.random(rows) < 0.2 + 0.4 * (data.a0 == "x") * (data.n > 10)).astype(int)
    data["value"] = rng.normal(size=rows) + (data.a1 == "y") + 0.05 * data.n
    return data


def to_selector(condition):
    """The pysubgroup 0.x selector covering the same rows of an integer or categorical column."""
    def first_above(bound):
        return -math.inf if bound is None else math.floor(bound) + 1

    feature = condition.feature
    if condition.operator == "eq":
        return ps.EqualitySelector(feature, condition.value)
    if condition.operator == "le":
        return ps.IntervalSelector(feature, -math.inf, first_above(condition.value))
    if condition.operator == "gt":
        return ps.IntervalSelector(feature, first_above(condition.value), math.inf)
    assert condition.operator == "interval", condition
    upper = math.inf if condition.upper is None else first_above(condition.upper)
    return ps.IntervalSelector(feature, first_above(condition.lower), upper)


def search_space(data, numeric_candidates):
    conditions = generate_conditions(
        data[["a0", "a1", "a2", "a3", "n"]], n_bins=4, numeric_candidates=numeric_candidates
    )
    selectors = [to_selector(condition) for condition in conditions]
    for condition, selector, mask in zip(
        conditions, selectors, subgroup_masks([(0.0, Rule((c,))) for c in conditions], data),
        strict=True,
    ):
        assert np.array_equal(np.asarray(selector.covers(data), dtype=bool), mask), condition
    return conditions, selectors


def random_masks(seed, rows, count=25):
    rng = np.random.default_rng(seed)
    masks = rng.random((count, rows)) < rng.uniform(0.005, 0.95, (count, 1))
    masks[0] = False
    masks[0, rng.integers(rows)] = True  # a single row
    masks[1] = True  # the whole dataset
    return masks[masks.any(axis=1)]


def theirs(qf, target, data, masks):
    """pysubgroup 0.x's qualities and estimates on its own copy of the data."""
    copy = data.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qf.calculate_constant_statistics(copy, target)
        # Positions of the original rows in the (possibly re-sorted) copy.
        rows = copy.index.to_numpy()
        qualities = [float(qf.evaluate(mask[rows], target, copy)) for mask in masks]
        estimates = (
            [float(qf.optimistic_estimate(mask[rows], target, copy)) for mask in masks]
            if hasattr(qf, "optimistic_estimate") else None
        )
    return qualities, estimates


def ours(qf, target, masks):
    estimates = (
        [qf.optimistic_estimate(mask, target) for mask in masks]
        if hasattr(qf, "optimistic_estimate") else None
    )
    return [qf.evaluate(mask, target) for mask in masks], estimates


def binary(data):
    return BinaryTarget().fit(data, "label", 1), ps.BinaryTarget("label", 1)


def numeric(data):
    return NumericTarget().fit(data, "value"), ps.NumericTarget("value")


def frequency(data):
    return FITarget().fit(data), ps.FITarget()


def has_positives(data, mask):
    return data.label.to_numpy()[mask].any()


def above_mean(data, mask):
    return (data.value.to_numpy()[mask] > data.value.mean()).any()


def not_constant(data, mask):
    return np.ptp(data.value.to_numpy()[mask]) > 0


# (targets, our measure, pysubgroup 0.x's measure, rows whose estimates compare or None).
MEASURES = [
    *((binary, StandardQF(a), ps.StandardQF(a), has_positives) for a in (0, 0.5, 1)),
    *((binary, ChiSquaredQF(direction, min_instances, stat),
       ps.ChiSquaredQF(direction, min_instances, stat), None)
      for direction in ("both", "positive", "negative") for stat in ("chi2", "p")
      for min_instances in (1, 5)),
    (numeric, StandardQFNumeric(0.5), ps.StandardQFNumeric(0.5), lambda d, m: True),
    (numeric, StandardQFNumeric(1, "sum"), ps.StandardQFNumeric(1, estimator="sum"),
     lambda d, m: True),
    (numeric, StandardQFNumeric(0.5, "max"), ps.StandardQFNumeric(0.5, estimator="max"),
     above_mean),
    (numeric, StandardQFNumeric(0.5, "average"), ps.StandardQFNumeric(0.5, estimator="average"),
     above_mean),
    (numeric, StandardQFNumeric(0, "order"), ps.StandardQFNumeric(0, estimator="order"),
     lambda d, m: True),
    (numeric, StandardQFNumeric(0.5, "order"), ps.StandardQFNumeric(0.5, estimator="order"),
     lambda d, m: True),
    (numeric, StandardQFNumeric(0.5, "max", "median"),
     ps.StandardQFNumeric(0.5, estimator="max", centroid="median"), None),
    (numeric, StandardQFNumericTscore(), ps.StandardQFNumericTscore(), None),
    (frequency, CountQF(), ps.CountQF(), lambda d, m: True),
]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(("make", "qf", "reference", "estimated"), MEASURES,
                         ids=lambda x: type(x).__name__ if not callable(x) else None)
def test_qualities_and_estimates_equal_pysubgroup(seed, make, qf, reference, estimated):
    data = frame(seed)
    target, reference_target = make(data)
    masks = random_masks(seed, len(data))
    if isinstance(qf, StandardQFNumericTscore):
        masks = [mask for mask in masks if mask.sum() == 1 or not_constant(data, mask)]
    expected_qualities, expected_estimates = theirs(reference, reference_target, data, masks)
    qualities, estimates = ours(qf, target, masks)
    np.testing.assert_allclose(qualities, expected_qualities, rtol=1e-9, atol=1e-12)
    if estimated is not None:
        compared = [i for i, mask in enumerate(masks) if estimated(data, mask)]
        np.testing.assert_allclose(
            [estimates[i] for i in compared], [expected_estimates[i] for i in compared],
            rtol=1e-9, atol=1e-12,
        )


def run_pysubgroup(algorithm, data, target, selectors, qf, k, depth):
    copy = data.copy()
    task = ps.SubgroupDiscoveryTask(copy, target, selectors, qf, result_set_size=k + 1,
                                    depth=depth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = algorithm.execute(task)
    positions = copy.index.to_numpy()
    found = []
    for quality, description in result.to_descriptions():
        if not description.selectors:
            continue  # the empty "Dataset" subgroup
        mask = np.zeros(len(data), dtype=bool)
        mask[positions] = np.asarray(description.covers(copy), dtype=bool)
        found.append((float(quality), mask))
    return found[:k]


def assert_same_top_k(found, expected):
    np.testing.assert_allclose([q for q, _ in found], [q for q, _ in expected],
                               rtol=1e-9, atol=1e-12)
    # Above the last quality, ties cannot change which subgroups make the top k.
    cutoff = expected[-1][0] + 1e-9 * max(1.0, abs(expected[-1][0]))

    def above(results):
        return sorted(mask.tobytes() for quality, mask in results if quality > cutoff)

    assert above(found) == above(expected)


SEARCH_MEASURES = [
    (binary, WRAccQF(), ps.WRAccQF()),
    (binary, StandardQF(0.5), ps.StandardQF(0.5)),
    (numeric, StandardQFNumeric(0.5), ps.StandardQFNumeric(0.5)),
    (numeric, StandardQFNumeric(1, "order"), ps.StandardQFNumeric(1, estimator="order")),
]
ALGORITHMS = [
    (DFS, ps.DFS),
    (DFS, ps.SimpleDFS),
    (BestFirstSearch, ps.BestFirstSearch),
    (Apriori, ps.Apriori),
    # pysubgroup 0.x needs a beam at least as wide as the result set (k + 1 with "Dataset").
    (partial(BeamSearch, beam_width=11), partial(ps.BeamSearch, beam_width=11)),
]


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize(("make", "qf", "reference"), SEARCH_MEASURES,
                         ids=lambda x: type(x).__name__ if not callable(x) else None)
def test_searches_find_the_top_k_of_pysubgroup(seed, depth, make, qf, reference):
    data = frame(seed)
    conditions, selectors = search_space(data, "bins")
    target, reference_target = make(data)
    task = SubgroupDiscoveryTask(data, target, conditions, qf, result_set_size=10, depth=depth)
    for ours_type, theirs_type in ALGORITHMS:
        results = ours_type().execute(task)
        found = list(zip([q for q, _ in results], subgroup_masks(results, data), strict=True))
        expected = run_pysubgroup(theirs_type(), data, reference_target, selectors, reference,
                                  10, depth)
        assert_same_top_k(found, expected)


@pytest.mark.parametrize("seed", range(2))
def test_threshold_conditions_match_pysubgroup_intervals(seed):
    # Overlapping thresholds on one attribute; both DFS use each attribute once.
    data = frame(seed)
    conditions, selectors = search_space(data, "thresholds")
    target, reference_target = binary(data)
    task = SubgroupDiscoveryTask(data, target, conditions, WRAccQF(), result_set_size=10,
                                 depth=3)
    results = DFS().execute(task)
    found = list(zip([q for q, _ in results], subgroup_masks(results, data), strict=True))
    expected = run_pysubgroup(ps.DFS(), data, reference_target, selectors, ps.WRAccQF(), 10, 3)
    assert_same_top_k(found, expected)


@pytest.mark.parametrize("seed", range(3))
def test_density_likelihood_equals_emm_likelihood(seed):
    degree = 1  # pysubgroup 0.x's PolyRegression_ModelClass supports only degree 1
    data = frame(seed)
    data["x"] = data.n + np.random.default_rng(seed).normal(scale=0.1, size=len(data))
    target = LikelihoodTarget(PolynomialRegressionModel(degree, variance=1.0)).fit(
        data, "x", "value")
    qf = ps.EMM_Likelihood(ps.PolyRegression_ModelClass("x", "value", degree=degree))
    masks = random_masks(seed, len(data))
    expected, _ = theirs(qf, None, data, masks)
    actual, _ = ours(LikelihoodQF("density"), target, masks)
    compared = 0
    for mask, mine, reference in zip(masks, actual, expected, strict=True):
        # pysubgroup 0.x gives nan for at most degree + 1 rows, which pysubgroup fits.
        if mask.sum() > degree + 1:
            assert mine == pytest.approx(reference, rel=1e-9, nan_ok=True)
            compared += not math.isnan(reference)
    assert compared > 10
