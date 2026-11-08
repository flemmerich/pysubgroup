"""NumBSD (Lemmerich, Atzmueller and Puppe, DMKD 2016, Algorithm 1)."""

import numpy as np
import pandas as pd
import pytest

from pysubgroup import (
    DFS,
    BinaryTarget,
    Condition,
    CustomFilter,
    InterclassVarianceQF,
    KolmogorovSmirnovQF,
    MannWhitneyQF,
    MinSizeFilter,
    NumBSD,
    NumericTarget,
    RankAUCQF,
    StandardDeviationQF,
    StandardQFNumeric,
    StandardQFNumericTscore,
    SubgroupDiscoveryTask,
    VarianceReductionQF,
    WRAccQF,
    _numba,
    generate_conditions,
)


def frame(seed, rows=300):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({f"a{j}": rng.choice(list("xyzw"), rows) for j in range(4)})
    data["n"] = rng.integers(0, 8, rows).astype(float)
    # Rounded values give ties.
    data["value"] = np.round(rng.normal(size=rows) + (data.n > 3) + (data.a0 == "x"), 1)
    conditions = generate_conditions(
        data[["a0", "a1", "a2", "a3", "n"]], n_bins=3, numeric_candidates="thresholds"
    )
    return data, conditions


MEASURES = [
    *(StandardQFNumeric(a, direction=direction)
      for a in (0, 0.1, 0.5, 1) for direction in ("positive", "negative", "both")),
    StandardQFNumeric(0.5, "order", "median"),
    StandardQFNumeric(1, "max", "median", "both"),
    VarianceReductionQF(),
    InterclassVarianceQF(),
    MannWhitneyQF(direction="negative"),
    RankAUCQF(direction="both"),
    StandardDeviationQF(0.5),
    KolmogorovSmirnovQF(),
    StandardQFNumericTscore(),  # no _ordered: evaluated on the unsorted mask
]


def qualities(algorithm, task):
    return [q for q, _ in algorithm.execute(task)]


@pytest.mark.parametrize("kernels", [True, False] if _numba.enabled() else [False])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("qf", MEASURES, ids=lambda qf: f"{type(qf).__name__}{vars(qf)}")
def test_numbsd_finds_the_qualities_of_dfs(seed, depth, qf, kernels, monkeypatch):
    if not kernels:
        monkeypatch.setattr(_numba, "enabled", lambda: False)
    data, conditions = frame(seed)
    target = NumericTarget().fit(data, "value")
    task = SubgroupDiscoveryTask(data, target, conditions, qf, result_set_size=7, depth=depth)
    expected = qualities(DFS(), task)
    for algorithm in (NumBSD(), NumBSD(fast_bounds=False)):
        np.testing.assert_allclose(qualities(algorithm, task), expected, rtol=1e-9, atol=1e-12)


def test_numbsd_applies_filters_and_min_quality():
    data, conditions = frame(3)
    target = NumericTarget().fit(data, "value")
    filters = [
        MinSizeFilter(20),
        CustomFilter(lambda m, t, c: all(x.feature != "a1" for x in c), anti_monotone=True),
        CustomFilter(lambda m, t, c: len(c) != 2),
    ]
    task = SubgroupDiscoveryTask(data, target, conditions, StandardQFNumeric(0.5),
                                 result_set_size=10, depth=3, filters=filters, min_quality=0.5)
    expected = DFS().execute(task)
    found = NumBSD().execute(task)
    np.testing.assert_allclose([q for q, _ in found], [q for q, _ in expected], rtol=1e-9)
    for _, rule in found:
        assert len(rule.conditions) != 2
        assert all(c.feature != "a1" for c in rule.conditions)


@pytest.mark.skipif(not _numba.enabled(), reason="fast bounds need numba")
@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_fast_bounds_stop_scans_without_changing_results(direction, monkeypatch):
    rng = np.random.default_rng(0)
    rows = 5000
    data = pd.DataFrame({f"a{j}": rng.choice(list("xyzw"), rows) for j in range(5)})
    data["value"] = rng.normal(size=rows) + 2 * (data.a0 == "x") * (data.a1 == "y")
    if direction == "negative":
        data["value"] *= -1
    conditions = generate_conditions(data.drop(columns="value"))
    target = NumericTarget().fit(data, "value")
    task = SubgroupDiscoveryTask(data, target, conditions,
                                 StandardQFNumeric(1, direction=direction), result_set_size=3,
                                 depth=3)
    fast = NumBSD()
    found = fast.execute(task)
    assert fast.n_stopped_early_ > 0
    slow = NumBSD(fast_bounds=False)
    assert slow.execute(task) == found
    assert slow.n_stopped_early_ == 0
    monkeypatch.setattr(_numba, "enabled", lambda: False)
    numpy_path = NumBSD()
    assert [q for q, _ in numpy_path.execute(task)] == pytest.approx([q for q, _ in found])
    assert numpy_path.n_stopped_early_ == 0


def test_numbsd_on_the_paper_toy_dataset():
    # Table 1 of the paper; sel_A, sel_B, sel_C as Boolean columns.
    data = pd.DataFrame({
        "y": [100.0, 75.0, 60.0, 53.0, 40.0, 35.0, 25.0, 12.0],
        "A": [1, 1, 0, 1, 0, 0, 0, 1],
        "B": [0, 0, 1, 1, 0, 0, 1, 0],
        "C": [0, 0, 0, 1, 0, 0, 0, 0],
    })
    conditions = [Condition(name, "eq", 1) for name in "ABC"]
    target = NumericTarget().fit(data, "y")
    task = SubgroupDiscoveryTask(data, target, conditions, StandardQFNumeric(0.5),
                                 result_set_size=1, depth=3)
    (quality, rule), = NumBSD().execute(task)
    # sel_A alone: 2 * (60 - 50) = 20; A and B: 1 * (53 - 50) = 3; B: sqrt(3) * (46 - 50)
    assert quality == pytest.approx(20)
    assert [c.feature for c in rule.conditions] == ["A"]


def test_numbsd_requires_a_numeric_target():
    data, conditions = frame(0)
    data["label"] = (data.value > 0).astype(int)
    task = SubgroupDiscoveryTask(data, BinaryTarget().fit(data, "label", 1), conditions,
                                 WRAccQF())
    with pytest.raises(TypeError, match="NumericTarget"):
        NumBSD().execute(task)
