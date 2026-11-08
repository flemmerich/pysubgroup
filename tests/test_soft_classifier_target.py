import numpy as np
import pandas as pd
import pytest
from helpers import assert_optimistic
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from pysubgroup import (
    ARLQF,
    PRAUCQF,
    ROCAUCQF,
    LabelledTarget,
    SoftClassifierTarget,
)
from pysubgroup.targets.soft_classifier import (
    average_ranking_loss,
    pr_auc_score,
)
from pysubgroup.targets.soft_classifier import roc_auc_score as rank_roc_auc_score

LABELS = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
SCORES = [0.9, 0.8, 0.4, 0.7, 0.4, 0.1, 0.3, 0.6, 0.2, 0.5]


@pytest.fixture
def target():
    frame = pd.DataFrame({"label": LABELS, "score": SCORES}, index=[3] * 10)
    return SoftClassifierTarget().fit(frame, "label", "score")


def test_soft_classifier_target_stores_labels_and_scores(target):
    assert isinstance(target, LabelledTarget)
    assert target.positives_.tolist() == [bool(v) for v in LABELS]
    assert target.n_positives_ == 5
    assert target.scores_.tolist() == SCORES
    with pytest.raises(ValueError, match="binary"):
        SoftClassifierTarget().fit(pd.DataFrame({"l": [0, 1, 2], "s": [0.1, 0.2, 0.3]}), "l", "s")
    with pytest.raises(ValueError, match="finite"):
        SoftClassifierTarget().fit(pd.DataFrame({"l": [0, 1], "s": [0.1, np.nan]}), "l", "s")


def test_average_ranking_loss_counts_ties_as_one_half():
    positives = np.array([True, False, True, False])
    scores = np.array([0.4, 0.4, 0.2, 0.9])
    # Positive 0.4: one negative above (0.9), one tied (0.4) -> 1.5; positive 0.2: 2.
    assert average_ranking_loss(positives, scores) == pytest.approx((1.5 + 2) / 2)


@pytest.mark.parametrize("seed", range(20))
def test_measures_match_scikit_learn_and_pairwise_counts(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 60))
    positives = rng.random(n) < rng.uniform(0.1, 0.9)
    positives[:2] = [True, False]
    scores = rng.integers(0, int(rng.integers(1, 8)), n) / 4  # many ties
    precision, recall, _ = metrics.precision_recall_curve(positives, scores)
    pairs = scores[positives][:, None] - scores[~positives][None, :]
    for order in (np.arange(n), np.argsort(-scores, kind="stable")):
        labels, values = positives[order], scores[order]
        assert rank_roc_auc_score(labels, values) == pytest.approx(
            roc_auc_score(positives, scores)
        )
        assert pr_auc_score(labels, values) == pytest.approx(metrics.auc(recall, precision))
        assert average_ranking_loss(labels, values) == pytest.approx(
            np.mean(np.sum((pairs < 0) + 0.5 * (pairs == 0), axis=1))
        )


def test_measures_reject_missing_labels():
    with pytest.raises(ValueError, match="positives and negatives"):
        rank_roc_auc_score(np.array([True, True]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="positive"):
        pr_auc_score(np.array([False]), np.array([0.1]))
    with pytest.raises(ValueError, match="positive"):
        average_ranking_loss(np.array([], dtype=bool), np.array([]))


def test_measures_report_worse_performance_as_positive_quality(target):
    mask = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
    positives, scores = target.positives_, target.scores_
    expected = roc_auc_score(positives, scores) - roc_auc_score(positives[mask], scores[mask])
    assert ROCAUCQF().evaluate(mask, target) == pytest.approx(expected)
    pr_expected = pr_auc_score(positives, scores) - pr_auc_score(positives[mask], scores[mask])
    assert PRAUCQF().evaluate(mask, target) == pytest.approx(pr_expected)
    only_negatives = mask & ~positives
    assert ROCAUCQF().evaluate(only_negatives, target) == float("-inf")


@pytest.mark.parametrize("qf", [
    ARLQF(), ROCAUCQF(), PRAUCQF(),
    ARLQF(subgroup_class_balance_weight=1, subgroup_size_weight=0.5),
    ROCAUCQF(subgroup_class_balance_weight=1, subgroup_size_weight=1),
])
def test_soft_classifier_estimates_bound_every_subset(target, qf):
    rng = np.random.default_rng(2)
    for _ in range(4):
        mask = rng.random(10) < 0.8
        if mask.any():
            assert_optimistic(qf, target, mask)
