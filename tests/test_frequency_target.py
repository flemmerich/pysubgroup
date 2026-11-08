import numpy as np
import pandas as pd
import pytest

from pysubgroup import AreaQF, BinaryTarget, CountQF, FITarget


@pytest.fixture
def data():
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 1]}, index=[7, 7, 3, 3, 1])


def test_fitarget_records_the_dataset_size(data):
    target = FITarget().fit(data)
    assert target.n_samples_ == 5 and target.attributes_ == ()


def test_count_and_area_use_only_the_subgroup_size(data):
    mask = np.array([True, False, True, True, False])
    for target in (FITarget().fit(data), BinaryTarget().fit(data, "y", 1)):
        assert CountQF().evaluate(mask, target) == 3
        assert CountQF().optimistic_estimate(mask, target) == 3
        assert AreaQF().evaluate(mask, target, depth=2) == 6


def test_masks_must_be_positional_booleans(data):
    target = FITarget().fit(data)
    with pytest.raises(ValueError, match="Boolean mask"):
        CountQF().evaluate(np.array([1, 0, 1, 1, 0]), target)
    with pytest.raises(ValueError, match="Boolean mask"):
        CountQF().evaluate(np.ones(4, dtype=bool), target)
    with pytest.raises(ValueError, match="fitted"):
        CountQF().evaluate(np.ones(5, dtype=bool), FITarget())
