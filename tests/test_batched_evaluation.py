"""Batched evaluation of search nodes: statistics, quality functions, filters, searches."""

import numpy as np
import pandas as pd
import pytest

from pysubgroup import (
    DFS,
    OPUS,
    VLSD,
    Apriori,
    AreaQF,
    BeamSearch,
    BestFirstSearch,
    BinaryTarget,
    ChiSquaredQF,
    ClassDistributionL1QF,
    CombinedQF,
    CountQF,
    CustomFilter,
    DpSubgroup,
    FITarget,
    GiniQF,
    InterclassVarianceQF,
    KolmogorovSmirnovQF,
    MannWhitneyQF,
    MaxConfidenceQF,
    MaxSizeFilter,
    MeanVectorL1QF,
    MinMajorityCountFilter,
    MinPositivesFilter,
    MinSizeFilter,
    MinTargetShareFilter,
    MulticlassChiSquaredQF,
    MulticlassTarget,
    NumericTarget,
    RankAUCQF,
    SDMap,
    SplitQF,
    StandardDeviationQF,
    StandardQF,
    StandardQFNumeric,
    StandardQFNumericTscore,
    SubgroupDiscoveryTask,
    VarianceReductionQF,
    VectorTarget,
    WRAccQF,
    _evaluation,
    _numba,
    generate_conditions,
)
from pysubgroup._evaluation import Backend, ChildStatistics

BATCHED_ALGORITHMS = [
    DFS(), BestFirstSearch(), OPUS(), Apriori(), DpSubgroup(), DpSubgroup("frequency"),
    BeamSearch(beam_width=12),
]


def frame(seed, rows=150):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({f"a{j}": rng.choice(list("xyzw"), rows) for j in range(4)})
    data["n"] = rng.integers(0, 8, rows).astype(float)
    data["label"] = (rng.random(rows) < 0.3 + 0.4 * (data.a0 == "x")).astype(int)
    data["value"] = rng.normal(size=rows) + (data.n > 3)
    data["cls"] = np.where(rng.random(rows) < 0.5, data.a1, rng.choice(list("pqr"), rows))
    shares = rng.dirichlet([2, 1, 1], rows) + 0.3 * (data.a2 == "y").to_numpy()[:, None]
    data[["s0", "s1", "s2"]] = shares / shares.sum(axis=1, keepdims=True)
    conditions = generate_conditions(
        data[["a0", "a1", "a2", "a3", "n"]], n_bins=3, numeric_candidates="thresholds"
    )
    return data, conditions


# (target factory, quality function, exact): exact measures use only integer counts; a
# float is the relative tolerance of an inexact measure (default 1e-12).
MEASURES = [
    (lambda d: BinaryTarget().fit(d, "label", 1), WRAccQF(), True),
    (lambda d: BinaryTarget().fit(d, "label", 1), StandardQF(0.5), True),
    (lambda d: BinaryTarget().fit(d, "label", 1), StandardQF(0.0, direction="negative"), True),
    (lambda d: BinaryTarget().fit(d, "label", 1), StandardQF(0.5, direction="both"), True),
    (lambda d: BinaryTarget().fit(d, "label", 1), ChiSquaredQF(), True),
    (lambda d: BinaryTarget().fit(d, "label", 1), ChiSquaredQF("positive", stat="p"), True),
    (lambda d: BinaryTarget().fit(d, "label", 1), CountQF(), True),
    (lambda d: FITarget().fit(d), CountQF(), True),
    (lambda d: NumericTarget().fit(d, "value"), StandardQFNumeric(1), False),
    (lambda d: NumericTarget().fit(d, "value"), StandardQFNumeric(0.5, "max"), False),
    (lambda d: NumericTarget().fit(d, "value"),
     StandardQFNumeric(0.5, "average", direction="negative"), False),
    (lambda d: NumericTarget().fit(d, "value"), StandardQFNumeric(0, direction="both"), False),
    (lambda d: NumericTarget().fit(d, "value"),
     StandardQFNumeric(0.5, "combined", direction="both"), False),
    (lambda d: NumericTarget().fit(d, "value"), VarianceReductionQF(), False),
    (lambda d: NumericTarget().fit(d, "value"), InterclassVarianceQF(), False),
    (lambda d: NumericTarget().fit(d, "value"), StandardDeviationQF(0.5), False),
    (lambda d: NumericTarget().fit(d, "value"), StandardDeviationQF(1, "both"), False),
    # The one-pass variance loses about eps * t**2 / n relative precision.
    (lambda d: NumericTarget().fit(d, "value"), StandardQFNumericTscore(), 1e-9),
    (lambda d: NumericTarget().fit(d, "value"), StandardQFNumericTscore("both"), 1e-9),
    # Rank deviations are half-integers, so their sums are exact.
    (lambda d: NumericTarget().fit(d, "value"), MannWhitneyQF(), True),
    (lambda d: NumericTarget().fit(d, "value"), MannWhitneyQF(direction="both"), True),
    (lambda d: NumericTarget().fit(d, "value"), RankAUCQF(direction="negative"), True),
    (lambda d: MulticlassTarget().fit(d, "cls"), SplitQF(), True),
    (lambda d: MulticlassTarget().fit(d, "cls"), MulticlassChiSquaredQF(), True),
    (lambda d: MulticlassTarget().fit(d, "cls"), GiniQF(), True),
    (lambda d: MulticlassTarget().fit(d, "cls"), ClassDistributionL1QF(), True),
    (lambda d: MulticlassTarget().fit(d, "cls"), MaxConfidenceQF(), True),
    (lambda d: MulticlassTarget().fit(d, "cls"), ClassDistributionL1QF(max_exact_classes=2),
     True),
    (lambda d: VectorTarget().fit(d, "s0", "s1", "s2"), MeanVectorL1QF(), False),
    (lambda d: VectorTarget().fit(d, "s0", "s1", "s2"), MeanVectorL1QF(max_exact_dimensions=1),
     False),
    (lambda d: BinaryTarget().fit(d, "label", 1),
     CombinedQF([WRAccQF(), StandardQF(0.5, direction="both"), CountQF()], [1, 0.3, 1e-3]),
     False),
    (lambda d: NumericTarget().fit(d, "value"),
     CombinedQF([StandardQFNumeric(1), StandardDeviationQF(1),
                 CombinedQF([MannWhitneyQF(direction="negative"), CountQF()], [1, 0.01])],
                [1, 0.5, 0.2]), False),
]


def assert_same(batched, scalar, exact):
    batched, scalar = np.asarray(batched, dtype=float), np.asarray(scalar, dtype=float)
    if exact is True:
        np.testing.assert_array_equal(batched, scalar)
    else:
        rtol = exact or 1e-12
        np.testing.assert_allclose(batched, scalar, rtol=rtol, atol=1e-12)


def random_children(seed, rows=150, n_conditions=40):
    rng = np.random.default_rng(seed)
    masks = rng.random((n_conditions, rows)) < rng.uniform(0.02, 0.9, (n_conditions, 1))
    masks[0] = False  # an empty child
    masks[1] = True  # the whole dataset
    parent = None if seed % 2 else rng.random(rows) < 0.7
    return masks, parent


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(("make", "qf", "exact"), MEASURES)
def test_batched_qualities_and_estimates_equal_scalar_ones(seed, make, qf, exact):
    data, _ = frame(seed)
    target = make(data)
    masks, parent = random_children(seed, rows=len(data))
    candidates = np.random.default_rng(seed).permutation(len(masks))
    statistics = ChildStatistics(Backend(masks), parent, candidates)
    children = [masks[c] if parent is None else masks[c] & parent for c in candidates]
    assert qf.batchable
    kwargs = {"depth": 2} if isinstance(qf, AreaQF) else {}
    assert_same(
        qf.evaluate_many(statistics, target, **kwargs),
        [qf.evaluate(mask, target, **kwargs) for mask in children],
        exact,
    )
    if hasattr(qf, "optimistic_estimate_many"):
        assert_same(
            qf.optimistic_estimate_many(statistics, target),
            [qf.optimistic_estimate(mask, target) for mask in children],
            exact,
        )


def test_area_qf_scales_sizes_by_depth():
    data, _ = frame(0)
    target = FITarget().fit(data)
    masks, parent = random_children(0, rows=len(data))
    statistics = ChildStatistics(Backend(masks), parent, range(len(masks)))
    expected = [AreaQF().evaluate(m & parent, target, depth=3) for m in masks]
    np.testing.assert_array_equal(AreaQF().evaluate_many(statistics, target, depth=3), expected)


@pytest.mark.parametrize("seed", range(3))
def test_batched_filters_equal_scalar_ones(seed):
    data, _ = frame(seed)
    target = BinaryTarget().fit(data, "label", 1)
    masks, parent = random_children(seed, rows=len(data))
    statistics = ChildStatistics(Backend(masks), parent, range(len(masks)))
    children = [m if parent is None else m & parent for m in masks]
    for f in (MinSizeFilter(10), MinSizeFilter(0.2), MaxSizeFilter(0.5), MaxSizeFilter(60),
              MinPositivesFilter(8), MinTargetShareFilter(0.45)):
        assert f.batchable
        np.testing.assert_array_equal(
            f.is_satisfied_many(statistics, target),
            [f.is_satisfied(mask, target, ()) for mask in children],
        )


@pytest.mark.parametrize("seed", range(3))
def test_batched_majority_count_filter_equals_scalar_one(seed):
    data, _ = frame(seed)
    target = MulticlassTarget().fit(data, "cls")
    masks, parent = random_children(seed, rows=len(data))
    statistics = ChildStatistics(Backend(masks), parent, range(len(masks)))
    children = [m if parent is None else m & parent for m in masks]
    for f in (MinMajorityCountFilter(12), MinMajorityCountFilter(0.15)):
        assert f.batchable and f.anti_monotone
        np.testing.assert_array_equal(
            f.is_satisfied_many(statistics, target),
            [f.is_satisfied(mask, target, ()) for mask in children],
        )


@pytest.mark.parametrize("kernels", [False, True] if _numba.enabled() else [False])
@pytest.mark.parametrize("parent_seed", [None, 4])
def test_statistics_split_into_blocks_and_gather_any_candidates(kernels, parent_seed, monkeypatch):
    rng = np.random.default_rng(3)
    rows_total = 130  # not a multiple of 64, so the last packed word is partial
    masks = rng.random((30, rows_total)) < 0.4
    parent = None if parent_seed is None else rng.random(rows_total) < 0.6
    values = rng.normal(size=rows_total)
    narrow = rng.normal(size=(rows_total, 3))
    wide = rng.normal(size=(rows_total, _evaluation._KERNEL_COLUMNS + 1))
    rows = rng.random(rows_total) < 0.5
    candidates = [7, 3, 3, 29, 0, 12, 13, 14]
    children = masks[candidates] if parent is None else masks[candidates] & parent
    expected = {
        "sizes": children.sum(axis=1),
        "count": (children & rows).sum(axis=1),
        "sum": np.array([values[c].sum() for c in children]),
        "narrow": np.array([narrow[c].sum(axis=0) for c in children]),
        "wide": np.array([wide[c].sum(axis=0) for c in children]),
        "maximum": np.array([values[c].max(initial=-np.inf) for c in children]),
    }
    for block in (1 << 24, 300, 1):  # one block, blocks of two children, one child each
        monkeypatch.setattr(_evaluation, "_BLOCK_ELEMENTS", block)
        backend = Backend(masks)
        backend.use_kernels = kernels
        statistics = ChildStatistics(backend, parent, candidates)
        np.testing.assert_array_equal(statistics.sizes, expected["sizes"])
        np.testing.assert_array_equal(statistics.count(rows), expected["count"])
        np.testing.assert_allclose(statistics.sum(values), expected["sum"], rtol=1e-12)
        np.testing.assert_allclose(statistics.sum(narrow), expected["narrow"], rtol=1e-12)
        np.testing.assert_allclose(statistics.sum(wide), expected["wide"], rtol=1e-12)
        np.testing.assert_array_equal(statistics.maximum(values), expected["maximum"])
    empty = ChildStatistics(Backend(masks), parent, [])
    assert empty.sizes.shape == (0,) and empty.count(rows).shape == (0,)
    assert empty.sum(values).shape == (0,) and empty.sum(wide).shape == (0, wide.shape[1])


def run(algorithm, task, **changes):
    task = SubgroupDiscoveryTask(**{**task.__dict__, **changes})
    return algorithm.execute(task)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("filters", [
    [],
    [MinSizeFilter(8), MaxSizeFilter(0.4)],
    [MinPositivesFilter(5), MinTargetShareFilter(0.4)],
])
@pytest.mark.parametrize(("make", "qf", "exact"), MEASURES)
def test_batched_searches_equal_scalar_searches(seed, filters, make, qf, exact):
    data, conditions = frame(seed)
    target = make(data)
    if any(isinstance(f, MinPositivesFilter | MinTargetShareFilter) for f in filters) and not (
        isinstance(target, BinaryTarget)
    ):
        return
    task = SubgroupDiscoveryTask(data, target, conditions, qf, result_set_size=8, depth=3,
                                 filters=filters)
    for algorithm in BATCHED_ALGORITHMS:
        scalar = run(algorithm, task, batched=False)
        batched = run(algorithm, task, batched=True)
        # The variance reduction scores a subgroup and its complement alike; float sums
        # break such ties differently, so only the qualities must agree.
        if not isinstance(qf, VarianceReductionQF):
            assert [rule for _, rule in batched] == [rule for _, rule in scalar], algorithm
        assert_same([q for q, _ in batched], [q for q, _ in scalar], exact)


def test_area_qf_searches_equal_scalar_searches():
    data, conditions = frame(1)
    task = SubgroupDiscoveryTask(data, FITarget().fit(data), conditions, AreaQF(), depth=3)
    for algorithm in BATCHED_ALGORITHMS:
        assert run(algorithm, task, batched=True) == run(algorithm, task, batched=False)


def test_unbatchable_parts_fall_back_or_raise():
    data, conditions = frame(0)
    binary = BinaryTarget().fit(data, "label", 1)
    numeric = NumericTarget().fit(data, "value")
    custom = CustomFilter(lambda m, t, c: len(c) < 3, anti_monotone=True)
    unbatchable = [
        SubgroupDiscoveryTask(data, numeric, conditions, StandardQFNumeric(1, centroid="median")),
        SubgroupDiscoveryTask(data, numeric, conditions, StandardQFNumeric(1, "order")),
        SubgroupDiscoveryTask(data, numeric, conditions, KolmogorovSmirnovQF()),
        SubgroupDiscoveryTask(data, numeric, conditions, VarianceReductionQF("order")),
        SubgroupDiscoveryTask(data, numeric, conditions, RankAUCQF("order")),
        SubgroupDiscoveryTask(data, binary, conditions, WRAccQF(), filters=[custom]),
    ]
    for task in unbatchable:
        # Automatic batching evaluates one subgroup at a time instead.
        assert run(DFS(), task) == run(DFS(), task, batched=False)
        with pytest.raises(ValueError, match="batched=True is not possible"):
            run(DFS(), task, batched=True)
    task = SubgroupDiscoveryTask(data, binary, conditions, WRAccQF())
    # Non-pruning custom filters are checked on the child's mask.
    filtered = [CustomFilter(lambda m, t, c: len(c) != 2)]
    assert run(DFS(), task, filters=filtered, batched=True) == run(
        DFS(), task, filters=filtered, batched=False
    )
    for algorithm in (SDMap(), VLSD()):
        assert run(algorithm, task) == run(DFS(), task)
        with pytest.raises(ValueError, match="one subgroup at a time"):
            run(algorithm, task, batched=True)
        with pytest.raises(ValueError, match="requires batched evaluation"):
            run(algorithm, task, device="cuda")
    with pytest.raises(ValueError, match="device must be one of"):
        run(DFS(), task, device="tpu")
    with pytest.raises(ValueError, match="requires batched evaluation"):
        run(DFS(), task, device="cuda", batched=False)
    with pytest.raises(ValueError, match="batched must be"):
        run(DFS(), task, batched="yes")


def cuda_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not cuda_available(), reason="requires CuPy and a CUDA device")
@pytest.mark.parametrize(("make", "qf", "exact"), MEASURES)
def test_gpu_searches_equal_cpu_searches(make, qf, exact):
    data, conditions = frame(2, rows=400)
    task = SubgroupDiscoveryTask(data, make(data), conditions, qf, result_set_size=8, depth=3,
                                 filters=[MinSizeFilter(5)])
    for algorithm in BATCHED_ALGORITHMS:
        cpu = run(algorithm, task, batched=False)
        gpu = run(algorithm, task, device="cuda")
        assert [rule for _, rule in gpu] == [rule for _, rule in cpu], algorithm
        assert_same([q for q, _ in gpu], [q for q, _ in cpu], exact)
