import itertools
import math

import numpy as np
import pandas as pd
import pytest

from pysubgroup import (
    DFS,
    DSSD,
    BinaryTarget,
    CustomFilter,
    MinSizeFilter,
    MulticlassTarget,
    NominalTarget,
    NumericTarget,
    StandardQFNumeric,
    SubgroupDiscoveryTask,
    WeightedKLQF,
    WRAccQF,
    cover_redundancy,
    generate_conditions,
    joint_entropy,
    subgroup_covers,
)
from pysubgroup.algorithms import dssd as dssd_module
from pysubgroup.algorithms.dssd import (
    _Candidate,
    _compression_selection,
    _cover_selection,
    _description_selection,
    _variable_description_selection,
)
from pysubgroup.rules import Condition, Rule
from pysubgroup.targets.nominal import discrete_codes


class _Masks:
    def __init__(self, masks):
        self.masks = masks
        self.space = type("Space", (), {"n_rows": masks.shape[1]})()

    def __getitem__(self, i):
        return self.masks[i]


def _random_candidates(rng, n_candidates, n_rows, negative=False):
    masks = rng.random((n_candidates, n_rows)) < rng.uniform(0.1, 0.6, (n_candidates, 1))
    masks[:, 0] = True
    low = -0.3 if negative else 0.0
    # Rounded qualities produce ties.
    qualities = np.round(rng.uniform(low, 1.0, n_candidates), 1)
    candidates = [_Candidate((i,), float(q), i) for i, q in enumerate(qualities)]
    order = sorted(range(n_candidates), key=lambda i: candidates[i].rank())
    return [candidates[i] for i in order], masks[order]


def _naive_cover(candidates, masks, alpha, size, threshold):
    counts = np.zeros(masks.shape[1], dtype=np.int64)
    weights = np.ones(masks.shape[1])
    remaining = [i for i, c in enumerate(candidates) if c.quality > 0]
    chosen = []
    while remaining and (size is None or len(chosen) < size):
        scores = [float(weights[masks[i]].mean()) * candidates[i].quality for i in remaining]
        best = int(np.argmax(scores))
        if threshold is not None and scores[best] < threshold:
            break
        i = remaining.pop(best)
        chosen.append(i)
        counts[masks[i]] += 1
        weights[masks[i]] = alpha ** counts[masks[i]]
    return chosen


@pytest.mark.parametrize("seed", range(20))
def test_lazy_cover_selection_equals_the_greedy_selection(seed):
    rng = np.random.default_rng(seed)
    candidates, masks = _random_candidates(rng, 40, 30, negative=seed % 2 == 1)
    for alpha, size, threshold in [(0.9, 10, None), (0.5, 35, None), (0.9, None, 0.3)]:
        selected = _cover_selection(candidates, _Masks(masks), alpha, size, threshold)
        expected = _naive_cover(candidates, masks, alpha, size, threshold)
        assert [candidates.index(c) for c in selected] == expected
        assert all(c.quality > 0 for c in selected)


def _encoded_size(selection, masks, attributes):
    """Encoded size of the dataset when each row uses the last selected subgroup covering it."""
    n_rows = masks.shape[1]
    owner = np.full(n_rows, -1)
    for i in selection:
        owner[masks[i]] = i
    total = 0.0
    for codes, _ in attributes:
        for t in range(n_rows):
            rows = np.ones(n_rows, bool) if owner[t] < 0 else masks[owner[t]]
            share = np.count_nonzero(codes[rows] == codes[t]) / np.count_nonzero(rows)
            total -= math.log2(share)
    return total


@pytest.mark.parametrize("seed", range(8))
def test_compression_selection_maximizes_the_compression_gain(seed):
    rng = np.random.default_rng(seed)
    candidates, masks = _random_candidates(rng, 15, 25)
    attributes = [(rng.integers(0, 3, 25), 3), (rng.integers(0, 2, 25), 2)]
    for size in (4, None):
        selected = _compression_selection(candidates, _Masks(masks), attributes, size)
        chosen = [candidates.index(c) for c in selected]
        # Replay the greedy choice with encoded sizes computed from scratch.
        replay = []
        while size is None or len(replay) < size:
            before = _encoded_size(replay, masks, attributes)
            gains = [
                -math.inf
                if i in replay
                else before - _encoded_size([*replay, i], masks, attributes)
                for i in range(len(candidates))
            ]
            best = int(np.argmax(gains))
            if size is None and gains[best] <= 1e-9:
                break
            assert gains[best] == pytest.approx(gains[chosen[len(replay)]])
            replay.append(chosen[len(replay)])
            if len(replay) == len(chosen):
                break
        assert replay == chosen
        if size is None:
            remaining = _encoded_size(chosen, masks, attributes)
            assert all(
                remaining - _encoded_size([*chosen, i], masks, attributes) <= 1e-9
                for i in range(len(candidates))
                if i not in chosen
            )


def test_description_selection_skips_equal_quality_variations():
    candidates = [
        _Candidate((0, 1), 0.5, 0),
        _Candidate((0, 2), 0.5, 1),  # differs from (0, 1) in one condition
        _Candidate((3, 4), 0.5, 2),
        _Candidate((0, 5), 0.4, 3),  # lower quality: kept
        _Candidate((0,), 0.4, 4),  # all but one of (0, 5)
        _Candidate((6, 7), 0.3, 5),
    ]
    assert [c.seq for c in _description_selection(candidates, 10)] == [0, 2, 3, 5]
    assert [c.seq for c in _description_selection(candidates, 2)] == [0, 2]


def test_description_selection_requires_exactly_equal_qualities():
    candidates = [_Candidate((0,), 0.3, 0), _Candidate((1,), 0.1 + 0.2, 1)]
    assert candidates[0].quality != candidates[1].quality
    assert [c.seq for c in _description_selection(candidates, 10)] == [0, 1]


def test_cover_selection_skips_non_positive_qualities():
    masks = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]], bool)
    candidates = [
        _Candidate((0,), 0.5, 0),
        _Candidate((1,), 0.0, 1),
        _Candidate((2,), -0.1, 2),
        _Candidate((3,), -0.2, 3),
    ]
    for size, threshold in [(4, None), (None, -1.0)]:
        selected = _cover_selection(candidates, _Masks(masks), 0.9, size, threshold)
        assert [c.seq for c in selected] == [0]


def test_variable_description_selection_limits_feature_uses():
    conditions = [Condition("a", "eq", 1), Condition("b", "eq", 1), Condition("c", "eq", 1)]
    space = type("Space", (), {"conditions": conditions, "features": ["a", "b", "c"]})()
    candidates = [
        _Candidate((0, 1), 0.9, 0),
        _Candidate((0, 2), 0.8, 1),
        _Candidate((0,), 0.7, 2),  # "a" already used twice
        _Candidate((1,), 0.6, 3),
        _Candidate((2,), 0.5, 4),
        _Candidate((2,), 0.4, 5),  # all features used up: stop
    ]
    selected = _variable_description_selection(candidates, 2, space)
    assert [c.seq for c in selected] == [0, 1, 3, 4]


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 300
    data = pd.DataFrame(
        {
            "a": rng.choice(["x", "y", "z"], n),
            "b": rng.choice(["u", "v"], n),
            "c": rng.choice(["p", "q", "r", "s"], n),
            "d": rng.normal(size=n),
        }
    )
    data["y"] = ((data.a == "x") & (data.d > 0)) | (rng.random(n) < 0.15)
    return data


def _task(frame, depth=2, k=10, qf=None, target=None, **kwargs):
    target = target or BinaryTarget().fit(frame, "y", True)
    return SubgroupDiscoveryTask(
        frame,
        target,
        generate_conditions(frame.drop(columns="y", errors="ignore"), n_bins=6),
        qf or WRAccQF(),
        result_set_size=k,
        depth=depth,
        **kwargs,
    )


def test_top_k_with_a_wide_beam_is_exhaustive(frame):
    nominal = frame[["a", "b", "c", "y"]]
    task = _task(nominal, depth=2, k=15)
    expected = DFS().execute(task)
    found = DSSD("top_k", beam_width=1000, pool_size=1000, dominance_pruning=False).execute(task)
    assert [q for q, _ in found] == pytest.approx([q for q, _ in expected])
    assert {frozenset(r.conditions) for _, r in found} == {
        frozenset(r.conditions) for _, r in expected
    }


@pytest.mark.parametrize("selection", dssd_module.SELECTIONS)
def test_batched_and_scalar_evaluation_agree(frame, selection):
    results = [
        DSSD(selection, beam_width=10, pool_size=100).execute(
            _task(frame, depth=3, filters=[MinSizeFilter(10)], batched=batched)
        )
        for batched in (True, False)
    ]
    assert [str(r) for _, r in results[0]] == [str(r) for _, r in results[1]]
    assert [q for q, _ in results[0]] == pytest.approx([q for q, _ in results[1]])
    for quality, rule in results[0]:
        mask = subgroup_covers([rule], frame)[0]
        assert np.count_nonzero(mask) >= 10
        assert WRAccQF().evaluate(mask, BinaryTarget().fit(frame, "y", True)) == pytest.approx(
            quality
        )


@pytest.mark.parametrize("selection", dssd_module.SELECTIONS)
@pytest.mark.parametrize("batched", [None, False])
def test_only_subgroups_passing_anti_monotone_filters_are_refined(
    frame, selection, batched, monkeypatch
):
    filters = [MinSizeFilter(80)]  # large enough to keep good subgroups out of the beam
    if batched is False:
        filters.append(CustomFilter(
            lambda m, t, c: all(x.feature != "a" for x in c), anti_monotone=True
        ))
    task = _task(frame, depth=3, filters=filters, batched=batched)
    refined = []
    refinements = dssd_module._Space.refinements

    def recording_refinements(space, ids, mask):
        refined.append((space.describe(ids), mask))
        return refinements(space, ids, mask)

    monkeypatch.setattr(dssd_module._Space, "refinements", recording_refinements)
    found = DSSD(selection, beam_width=10, pool_size=100).execute(task)
    assert found
    assert len(refined) > 1
    for conditions, mask in refined:
        if conditions:
            assert all(f.is_satisfied(mask, task.target, conditions) for f in filters)


def test_diverse_selection_finds_less_redundant_covers(frame):
    task = _task(frame, depth=3, k=10, filters=[MinSizeFilter(10)])
    top = DSSD("top_k", beam_width=20, pool_size=500, dominance_pruning=False).execute(task)
    cover = DSSD("cover", beam_width=20, pool_size=500).execute(task)
    assert top[0][0] == pytest.approx(cover[0][0])
    assert cover_redundancy(subgroup_covers(cover, frame)) < cover_redundancy(
        subgroup_covers(top, frame)
    )


def test_refinements_follow_the_paper(frame):
    data = frame.copy()
    data.loc[:4, "d"] = np.nan
    task = SubgroupDiscoveryTask(
        data,
        BinaryTarget().fit(data, "y", True),
        [
            Condition("a", "eq", "x"),
            Condition("a", "ne", "y"),
            Condition("a", "ne", "z"),
            Condition("b", "eq", "u"),
            Condition("d", "interval", upper=0.0),
        ],
        WRAccQF(),
    )
    search = dssd_module._Search(task)
    space = dssd_module._Space(search, data, 6)
    root = space.refinements((), np.ones(len(data), bool))
    names = [str(space.conditions[i]) for i in root]
    # The interval on numeric "d" is replaced by thresholds at five cut points.
    assert names[:4] == ["a=x", "a≠y", "a≠z", "b=u"]
    values = data.d.dropna().to_numpy()
    cuts = np.quantile(values, np.arange(1, 6) / 6, method="inverted_cdf")
    assert names[4:] == [f"d {op} {cut:.15g}" for cut in cuts for op in ("≤", ">")]
    ne_y = space.ids[Condition("a", "ne", "y")]
    after_ne = [str(space.conditions[i]) for i in space.refinements((ne_y,), space.mask((ne_y,)))]
    assert "a≠z" in after_ne and "a=x" not in after_ne and "a≠y" not in after_ne
    le = space.ids[Condition("d", "le", float(cuts[2]))]
    inside = space.mask((le,))
    after_le = [str(space.conditions[i]) for i in space.refinements((le,), inside)]
    deeper = np.quantile(values[values <= cuts[2]], np.arange(1, 6) / 6, method="inverted_cdf")
    assert [n for n in after_le if n.startswith("d")] == [
        f"d {op} {cut:.15g}" for cut in deeper for op in ("≤", ">")
    ]
    # Missing values match neither side of a threshold.
    assert not inside[:5].any()


def test_without_bins_the_search_space_is_used_as_given(frame):
    task = _task(frame, depth=2)
    found = DSSD("top_k", beam_width=50, n_bins=None, dominance_pruning=False).execute(task)
    space = set(task.search_space)
    assert all(set(rule.conditions) <= space for _, rule in found)


def test_dominance_pruning_drops_conditions_that_do_not_help():
    # "noise" splits the data evenly and is useless; y depends on "a" only.
    data = pd.DataFrame(
        {"a": ["x", "x", "x", "x", "y", "y", "y", "y"] * 5, "noise": ["n", "m"] * 20}
    )
    data["y"] = (data.a == "x") & (np.arange(40) % 4 != 1)
    task = SubgroupDiscoveryTask(
        data,
        BinaryTarget().fit(data, "y", True),
        generate_conditions(data.drop(columns="y")),
        WRAccQF(),
        result_set_size=3,
        depth=2,
    )
    search = dssd_module._Search(task)
    space = dssd_module._Space(search, data, 6)
    noise = space.ids[Condition("noise", "eq", "n")]
    a = space.ids[Condition("a", "eq", "x")]
    pruned = DSSD()._prune(search, space, _Candidate((noise, a), 0.0, 0))
    assert pruned.ids == (a,)
    assert pruned.quality == pytest.approx(WRAccQF().evaluate(space.mask((a,)), task.target))
    found = DSSD("cover", beam_width=5).execute(task)
    assert found[0][1].conditions == (Condition("a", "eq", "x"),)


def test_rules_keep_conditions_in_the_order_they_were_added(frame):
    found = DSSD("top_k", beam_width=5, pool_size=50, dominance_pruning=False).execute(
        _task(frame, depth=2)
    )
    pairs = [rule for _, rule in found if len(rule) == 2]
    assert pairs
    first = {rule.conditions[0] for rule in pairs}
    singles = DSSD("top_k", beam_width=5, pool_size=50).execute(_task(frame, depth=1))
    assert first <= {rule.conditions[0] for _, rule in singles}


def test_emm_with_weighted_kl_and_compression():
    rng = np.random.default_rng(1)
    n = 200
    data = pd.DataFrame({"g": rng.choice(["a", "b", "c"], n), "h": rng.normal(size=n)})
    data["m1"] = np.where(data.g == "a", rng.random(n) < 0.9, rng.random(n) < 0.3)
    data["m2"] = np.where(data.h > 0.5, "hi", rng.choice(["lo", "mid"], n))
    target = NominalTarget().fit(data, "m1", "m2")
    task = _task(data, depth=2, qf=WeightedKLQF(), target=target, filters=[MinSizeFilter(5)])
    for selection in ("compression", "var_compression"):
        found = DSSD(selection, beam_width=10, pool_size=100).execute(task)
        assert found and all(q > 0 for q, _ in found)
    # At depth 1 the pool holds every subgroup; the first compression choice is
    # the best WKL subgroup, since its gain is |S| times its WKL quality.
    task.depth = 1
    best = DSSD("top_k", pool_size=100).execute(task)[0][0]
    for selection in ("compression", "var_compression"):
        assert DSSD(selection, pool_size=100).execute(task)[0][0] == pytest.approx(best)
    with pytest.raises(TypeError, match="discrete model attributes"):
        numeric = NumericTarget().fit(data, "h")
        DSSD("compression").execute(
            _task(data[["g", "h"]], qf=StandardQFNumeric(1.0), target=numeric)
        )


def test_weighted_kl_matches_definition_4():
    data = pd.DataFrame({"m1": [1, 1, 0, 0, 0, 1], "m2": ["a", "b", "b", "b", None, "a"]})
    target = NominalTarget().fit(data, "m1", "m2")
    assert [list(v) for v in target.values_][0] == [0, 1]
    mask = np.array([1, 1, 0, 0, 0, 1], bool)
    # m1 is 1 on all three rows (dataset share 1/2); m2 is a, b, a (dataset 2/6, 3/6, 1/6).
    kl1 = math.log2(1 / 0.5)
    kl2 = 2 / 3 * math.log2((2 / 3) / (2 / 6)) + 1 / 3 * math.log2((1 / 3) / (3 / 6))
    assert WeightedKLQF().evaluate(mask, target) == pytest.approx(3 / 6 * (kl1 + kl2))
    assert WeightedKLQF().evaluate(np.ones(6, bool), target) == pytest.approx(0.0)
    assert math.isnan(WeightedKLQF().evaluate(np.zeros(6, bool), target))
    labels = MulticlassTarget().fit(data.assign(m2=data.m2.fillna("?")), "m2")
    binary = BinaryTarget().fit(data, "m1", 1)
    assert len(discrete_codes(labels)) == len(discrete_codes(binary)) == 1
    assert WeightedKLQF().evaluate(mask, binary) == pytest.approx(3 / 6 * kl1)


def test_weighted_kl_batches_agree_with_scalar_evaluation(frame):
    target = NominalTarget().fit(frame, "a", "y")
    task = _task(frame, depth=2, qf=WeightedKLQF(), target=target)
    batched = DSSD("top_k", beam_width=10).execute(task)
    task.batched = False
    scalar = DSSD("top_k", beam_width=10).execute(task)
    assert [q for q, _ in batched] == pytest.approx([q for q, _ in scalar])


def test_diversity_measures():
    covers = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], bool)
    assert cover_redundancy(covers) == 0.0
    assert joint_entropy(covers) == pytest.approx(1.0)
    independent = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], bool)
    assert joint_entropy(independent) == pytest.approx(2.0)
    same = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], bool)
    # Counts 2, 2, 0, 0 around the mean 1.
    assert cover_redundancy(same) == pytest.approx(1.0)
    assert joint_entropy(same) == pytest.approx(1.0)
    assert math.isnan(cover_redundancy(np.zeros((2, 3), bool)))
    for covers in itertools.product([0, 1], repeat=6):
        matrix = np.array(covers, bool).reshape(2, 3)
        assert joint_entropy(matrix) <= 2 + 1e-12


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"selection": "best"}, "selection"),
        ({"beam_width": 0}, "beam_width"),
        ({"pool_size": 5}, "pool_size"),
        ({"alpha": 0.0}, "alpha"),
        ({"min_fraction": 1.5}, "min_fraction"),
        ({"n_bins": 1}, "n_bins"),
    ],
)
def test_parameters_are_validated(frame, kwargs, message):
    with pytest.raises(ValueError, match=message):
        DSSD(**kwargs).execute(_task(frame, k=10))


@pytest.mark.parametrize("selection", dssd_module.SELECTIONS)
@pytest.mark.parametrize("min_quality", [None, 5.0])
def test_empty_pools_give_empty_results(selection, min_quality):
    d = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    conditions = [] if min_quality is None else [Condition("x", "eq", 1)]
    kwargs = {} if min_quality is None else {"min_quality": min_quality}
    task = SubgroupDiscoveryTask(d, BinaryTarget().fit(d, "y", 1), conditions, WRAccQF(), **kwargs)
    assert DSSD(selection).execute(task) == []


def test_subgroup_covers_names_array_columns_as_the_encoder_does():
    rule = Rule((Condition("x0", "eq", 1),))
    assert subgroup_covers([rule, Rule()], np.array([[1], [2]])).tolist() == [
        [True, False], [True, True]
    ]
