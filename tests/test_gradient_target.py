import math

import numpy as np
import pandas as pd
import pytest
from helpers import assert_optimistic, subsets

from pysubgroup import GradientBoostingQF, GradientTarget
from pysubgroup.targets.gradient import gradient_order


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"g": rng.normal(size=10), "h": rng.uniform(0.1, 1.0, 10), "x": range(10)},
        index=[3] * 10,
    )


def test_gradient_target_stores_columns_and_order(frame):
    target = GradientTarget().fit(frame, "g", "h")
    assert target.attributes_ == ("g", "h")
    assert target.gradients_ == pytest.approx(frame.g)
    ratios = (frame.g / frame.h).to_numpy()[target.order_]
    assert (np.diff(ratios) <= 0).all()


def test_gradient_target_validates_columns(frame):
    with pytest.raises(TypeError, match="numeric"):
        GradientTarget().fit(frame.assign(h=True), "g", "h")
    with pytest.raises(ValueError, match="finite"):
        GradientTarget().fit(frame.assign(g=np.nan), "g", "h")
    with pytest.raises(ValueError, match="non-negative"):
        GradientTarget().fit(frame.assign(h=-1.0), "g", "h")
    with pytest.raises(ValueError, match="reg_lambda"):
        GradientBoostingQF(-1.0)


def test_zero_hessians_sort_by_the_sign_of_the_gradient():
    order = gradient_order(np.array([1.0, -1.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    assert list(order) == [0, 3, 2, 1]


def test_objective_matches_eq_2_3(frame):
    target = GradientTarget().fit(frame, "g", "h")
    mask = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=bool)
    qf = GradientBoostingQF(0.5)
    expected = frame.g[mask].sum() ** 2 / (2 * 10 * (0.5 + frame.h[mask].sum()))
    assert qf.evaluate(mask, target) == pytest.approx(expected)
    assert qf.evaluate(np.zeros(10, dtype=bool), target) == 0.0
    assert qf.optimistic_estimate(np.zeros(10, dtype=bool), target) == -math.inf
    assert GradientBoostingQF(0.0).evaluate(np.zeros(10, dtype=bool), target) == 0.0


@pytest.mark.parametrize("reg_lambda", [0.0, 0.3, 5.0])
def test_estimate_is_the_best_subset_theorem_3_1(frame, reg_lambda):
    target = GradientTarget().fit(frame, "g", "h")
    qf = GradientBoostingQF(reg_lambda)
    rng = np.random.default_rng(1)
    for mask in [np.ones(10, dtype=bool), *(rng.random(10) < 0.7 for _ in range(6))]:
        if not mask.any():
            continue
        assert_optimistic(qf, target, mask)
        best = max(qf.evaluate(subset, target) for subset in subsets(mask))
        assert qf.optimistic_estimate(mask, target) == pytest.approx(best)
