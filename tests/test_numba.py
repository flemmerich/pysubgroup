import numpy as np
import pandas as pd
import pytest

from pysubgroup import (
    InterclassVarianceQF,
    MannWhitneyQF,
    NumericTarget,
    RankAUCQF,
    StandardQFNumeric,
    StandardQFNumericTscore,
    VarianceReductionQF,
    _numba,
)

pytestmark = pytest.mark.skipif(not _numba.AVAILABLE, reason="numba is not installed")


def test_pack64_sets_bit_i_of_word_i_div_64():
    mask = np.zeros(130, dtype=bool)
    mask[[0, 63, 64, 129]] = True
    words = _numba.pack64(mask)
    assert words.dtype == np.uint64 and words.shape == (3,)
    assert words.tolist() == [1 | 1 << 63, 1, 1 << 1]


@pytest.mark.parametrize("size", [0, 1, 5, 400, 1001])
def test_masked_statistics_match_numpy(size):
    rng = np.random.default_rng(size)
    values = rng.normal(size=1001)
    mask = np.zeros(1001, dtype=bool)
    mask[rng.choice(1001, size, replace=False)] = True
    selected = values[mask]
    n, total = _numba.masked_sum(values, mask)
    assert n == size and total == pytest.approx(selected.sum(), abs=1e-9)
    deviation = _numba.masked_squared_deviation(values, mask, 0.25)
    assert deviation == pytest.approx(np.sum((selected - 0.25) ** 2), abs=1e-9)
    n, n_above, excess = _numba.masked_excess(values, mask, 0.25)
    above = selected[selected > 0.25]
    assert (n, n_above) == (size, len(above))
    assert excess == pytest.approx(np.sum(above - 0.25), abs=1e-9)


@pytest.mark.parametrize(
    "qf", [StandardQFNumeric(0.5), StandardQFNumeric(1, "sum"), StandardQFNumericTscore(),
           StandardQFNumeric(1, "sum", direction="negative"),
           StandardQFNumeric(0.5, direction="both"), StandardQFNumericTscore("both")]
)
def test_numeric_qualities_match_numpy_path(qf, monkeypatch):
    rng = np.random.default_rng(1)
    target = NumericTarget().fit(pd.DataFrame({"y": rng.normal(size=500)}), "y")
    masks = [rng.random(500) < share for share in (0.0, 0.002, 0.3, 1.0)]
    with_numba = [(qf.evaluate(m, target), qf.optimistic_estimate(m, target)) for m in masks]
    monkeypatch.setattr(_numba, "ENABLED", False)
    without = [(qf.evaluate(m, target), qf.optimistic_estimate(m, target)) for m in masks]
    np.testing.assert_allclose(with_numba, without, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert with_numba[-1][0] == without[-1][0]  # the whole dataset scores exactly the same


@pytest.mark.parametrize("qf", [
    *(StandardQFNumeric(a, direction=direction)
      for a in (0, 0.5, 1) for direction in ("positive", "negative", "both")),
    VarianceReductionQF(), InterclassVarianceQF(),
    *(cls(direction=direction) for cls in (MannWhitneyQF, RankAUCQF)
      for direction in ("positive", "negative", "both")),
], ids=lambda qf: f"{type(qf).__name__}{vars(qf)}")
def test_prefix_scans_match_ordered_evaluation(qf):
    rng = np.random.default_rng(2)
    rows = 300  # not a multiple of 64
    # Rounded values give ties.
    target = NumericTarget().fit(pd.DataFrame({"y": np.round(rng.normal(size=rows), 1)}), "y")
    parent = rng.random(rows) < 0.7
    for condition in (np.zeros(rows, bool), rng.random(rows) < 0.01, rng.random(rows) < 0.5,
                      np.ones(rows, bool)):
        for outer in (parent, np.ones(rows, bool)):
            size, quality, estimate = qf._scan(
                _numba.pack64(outer[target.order_]), _numba.pack64(condition[target.order_]),
                target,
            )
            expected = qf._ordered((outer & condition)[target.order_], target)
            assert size == np.count_nonzero(outer & condition)
            np.testing.assert_allclose((quality, estimate), expected, rtol=1e-12, atol=1e-12)
    assert StandardQFNumeric(1, centroid="median")._scan(None, None, target) is None


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_ordered_mean_scan_matches_prefix_qualities(reverse, a):
    rng = np.random.default_rng(int(10 * a) + reverse)
    values = np.sort(rng.normal(size=300))[::-1].copy()
    parent = rng.random(300) < 0.8
    condition = rng.random(300) < 0.5
    target = NumericTarget().fit(pd.DataFrame({"y": values}), "y")
    centroid = -target.mean_ if reverse else target.mean_
    size, total, best, stopped = _numba.ordered_mean_scan(
        _numba.pack64(parent), _numba.pack64(condition), values,
        np.arange(301, dtype=float) ** a, centroid,
        int(min(parent.sum(), condition.sum())), -np.inf, reverse, True,
    )
    qf = StandardQFNumeric(a, "order", direction="negative" if reverse else "positive")
    quality, estimate = qf._ordered(parent & condition, target)
    assert not stopped
    assert size == np.count_nonzero(parent & condition)
    assert best == pytest.approx(estimate)
    assert size**a * (total / size - centroid) == pytest.approx(quality)


def test_ordered_mean_scan_stops_at_the_fast_bound():
    # Lemmerich et al. (2016), Example 4 on 70 rows: after the first word (64 rows,
    # all below the mean), no prefix of the remaining rows reaches the threshold.
    values = np.concatenate([np.full(64, -1.0), np.full(6, -2.0)])
    words = _numba.pack64(np.ones(70, dtype=bool))
    weights = np.arange(71, dtype=float)
    size, total, best, stopped = _numba.ordered_mean_scan(
        words, words, values, weights, 0.0, 70, 0.0, False, True
    )
    assert stopped and size == 64 and total == -64.0 and best == -1.0
    size, *_, stopped = _numba.ordered_mean_scan(
        words, words, values, weights, 0.0, 70, 0.0, False, False
    )
    assert not stopped and size == 70
