import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pysubgroup import (  # noqa: E402  # noqa: E402
    DFS,
    BinaryTarget,
    Condition,
    CorrelationTarget,
    MinSizeFilter,
    NumericTarget,
    RegressionTarget,
    Rule,
    SoftClassifierTarget,
    StandardQF,
    StandardQFNumeric,
    SubgroupDiscoveryTask,
    WRAccQF,
    generate_conditions,
    randomization_p_values,
)
from pysubgroup import visualization as viz  # noqa: E402
from pysubgroup.targets.soft_classifier import (  # noqa: E402
    pr_auc_score,
    roc_auc_score,
)


@pytest.fixture(scope="module")
def setting():
    rng = np.random.default_rng(0)
    rows = 300
    data = pd.DataFrame(
        {
            "age": rng.integers(18, 80, rows),
            "smoker": rng.choice(["yes", "no"], rows),
            "region": rng.choice(["north", "south"], rows),
        }
    )
    data["disease"] = np.where(
        rng.random(rows) < 0.2 + 0.4 * (data.smoker == "yes"), "yes", "no"
    )
    data["cost"] = 50 * data.age + rng.normal(0, 100, rows)
    conditions = generate_conditions(data[["age", "smoker", "region"]], n_bins=3)
    binary = BinaryTarget().fit(data, "disease", "yes")
    results = DFS().execute(
        SubgroupDiscoveryTask(data, binary, conditions, WRAccQF(), 5, 2,
                              filters=[MinSizeFilter(5)])
    )
    numeric = NumericTarget().fit(data, "cost")
    numeric_results = DFS().execute(
        SubgroupDiscoveryTask(data, numeric, conditions, StandardQFNumeric(1), 4, 2)
    )
    return data, conditions, binary, results, numeric, numeric_results


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_all_visualizations_render(setting, theme) -> None:
    data, conditions, binary, results, numeric, numeric_results = setting
    calls = [
        lambda: viz.plot_subgroup_bars(results, data, binary, theme=theme),
        lambda: viz.plot_spineplot(results, data, binary, theme=theme),
        lambda: viz.plot_roc_space(results, data, binary, isometrics=WRAccQF(), theme=theme),
        lambda: viz.plot_roc_space(results, data, binary, counts=True, theme=theme),
        lambda: viz.plot_size_quality(results, data, binary, theme=theme),
        lambda: viz.plot_size_quality(results, data, binary, y="target_share", theme=theme),
        lambda: viz.plot_size_quality(numeric_results, data, numeric, y="mean", theme=theme),
        lambda: viz.plot_target_distributions(numeric_results, data, numeric, theme=theme),
        lambda: viz.plot_attribute_distribution(results[:3], data, binary, "age", theme=theme),
        lambda: viz.plot_overlap(results, data, binary, theme=theme),
        lambda: viz.plot_overlap(numeric_results, data, numeric, order="given", theme=theme),
        lambda: viz.plot_similarity(results, data, theme=theme),
        lambda: viz.plot_specialization_graph(results, data, binary, theme=theme),
        lambda: viz.plot_similarity_dendrogram(results, data, theme=theme),
        lambda: viz.plot_null_distribution(results, data, binary, WRAccQF(), n_samples=20,
                                           random_state=0, theme=theme),
    ]
    for call in calls:
        ax = call()
        assert isinstance(ax, matplotlib.axes.Axes)
        ax.figure.canvas.draw()
        plt.close(ax.figure)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_model_scatter_shows_subgroup_complement_and_lines(setting, theme) -> None:
    data, conditions, binary, results, _, _ = setting
    target = RegressionTarget().fit(data, "age", "cost")
    ax = viz.plot_model_scatter(results, data, target, subgroup=2, theme=theme)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    mask = viz.subgroup_masks(results, data)[1]
    assert labels[0].startswith(f"complement (n = {(~mask).sum()}, r = ")
    assert labels[2].startswith(f"#2  {viz.rule_label(results[1][1])} (n = {mask.sum()}")
    assert len(ax.lines) == 2 and labels[1].startswith("y = ")
    ax.figure.canvas.draw()
    plt.close(ax.figure)
    ax = viz.plot_model_scatter(results, data, CorrelationTarget().fit(data, "age", "cost"),
                                lines=False)
    assert not ax.lines
    plt.close(ax.figure)
    with pytest.raises(TypeError, match="EMM correlation or regression"):
        viz.plot_model_scatter(results, data, binary)
    with pytest.raises(ValueError, match="subgroup must be between"):
        viz.plot_model_scatter(results, data, target, subgroup=len(results) + 1)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("curve", ["roc", "pr"])
def test_performance_curves_end_at_the_areas(setting, curve, theme) -> None:
    data, _, _, results, _, _ = setting
    data = data.assign(score=np.round(data.age / 80 + 0.3 * (data.disease == "yes"), 1))
    target = SoftClassifierTarget().fit(data, "disease", "score", "yes")
    no_negatives = Rule((Condition("disease", "eq", "yes"),))
    entries = [*results[:3], no_negatives]
    ax = viz.plot_performance_curves(entries, data, target, curve=curve, theme=theme)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    score = roc_auc_score if curve == "roc" else pr_auc_score
    masks = viz.subgroup_masks(results[:3], data)
    for label, mask in zip(labels[1:4], masks, strict=True):
        assert f"{score(*target._subgroup(mask)):.3f}" in label
    # Trapezoids under the drawn curve (diagonal steps at ties) give the area.
    x, y = ax.lines[1].get_data()
    assert np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) == pytest.approx(
        score(target.positives_, target.scores_)
    )
    assert ("undefined" in labels[-1]) is (curve == "roc")
    ax.figure.canvas.draw()
    plt.close(ax.figure)
    with pytest.raises(TypeError, match="soft-classifier"):
        viz.plot_performance_curves(results, data, setting[2])
    with pytest.raises(ValueError, match="curve"):
        viz.plot_performance_curves(results, data, target, curve="det")


def test_model_scatter_jitter_keeps_points_near_their_values(setting) -> None:
    data, _, _, results, _, _ = setting
    data = data.assign(hours=np.round(data.cost / 100))
    target = RegressionTarget().fit(data, "age", "hours")
    mask = viz.subgroup_masks(results, data)[0]
    exact = np.column_stack([data.age, data.hours])
    for jitter in (0.0, 0.3):
        ax = viz.plot_model_scatter(results, data, target, jitter=jitter)
        drawn = np.vstack([ax.collections[0].get_offsets(), ax.collections[1].get_offsets()])
        expected = np.vstack([exact[~mask], exact[mask]])
        offset = np.abs(drawn - expected)
        assert offset.max() <= jitter + 1e-12  # both attributes are whole numbers
        assert bool(offset.max() > 0) is (jitter > 0)
        plt.close(ax.figure)
    with pytest.raises(ValueError, match="jitter"):
        viz.plot_model_scatter(results, data, target, jitter=-1)


def test_dendrogram_joins_identical_subgroups_at_zero(setting) -> None:
    data = setting[0]
    yes = Rule((Condition("smoker", "eq", "yes"),))
    not_no = Rule((Condition("smoker", "ne", "no"),))
    north = Rule((Condition("region", "eq", "north"),))
    ax = viz.plot_similarity_dendrogram([yes, north, not_no], data)
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert sorted(labels) == ["#1  smoker=yes", "#2  region=north", "#3  smoker≠no"]
    assert abs(labels.index("#1  smoker=yes") - labels.index("#3  smoker≠no")) == 1
    with pytest.raises(ValueError, match="two subgroups"):
        viz.plot_similarity_dendrogram([yes], data)
    plt.close("all")


def test_null_distribution_labels_the_p_values(setting) -> None:
    data, _, binary, results, _, _ = setting
    ax = viz.plot_null_distribution(results[:2], data, binary, WRAccQF(), n_samples=30,
                                    random_state=4)
    expected = randomization_p_values(results[:2], data, binary, WRAccQF(), n_samples=30,
                                      random_state=4)
    texts = [text.get_text() for text in ax.texts]
    assert texts == [f"p = {p:.3g}" for p in expected]
    plt.close("all")


def test_masks_and_labels_follow_condition_semantics(setting) -> None:
    data = setting[0]
    rule = Rule(
        (Condition("age", "interval", lower=30, upper=40), Condition("smoker", "eq", "yes"))
    )
    mask = viz.subgroup_masks([rule], data)[0]
    expected = (data.age > 30) & (data.age <= 40) & (data.smoker == "yes")
    assert np.array_equal(mask, expected.to_numpy())
    assert viz.rule_label(rule) == "age ∈ (30, 40] ∧ smoker=yes"
    assert viz.rule_label(Rule((Condition("age", "interval", upper=30),))) == "age ∈ (-∞, 30]"


def test_wrong_targets_and_inputs_are_rejected(setting) -> None:
    data, conditions, binary, results, numeric, _ = setting
    with pytest.raises(TypeError, match="binary"):
        viz.plot_subgroup_bars(results, data, numeric)
    with pytest.raises(TypeError, match="numeric target"):
        viz.plot_target_distributions(results, data, binary)
    with pytest.raises(ValueError, match="fitted on the visualized data"):
        viz.plot_spineplot(results, data.iloc[:10], binary)
    with pytest.raises(ValueError, match="no subgroups"):
        viz.plot_spineplot([], data, binary)
    with pytest.raises(TypeError, match="StandardQF"):
        viz.plot_roc_space(results, data, binary, isometrics=StandardQFNumeric(1))
    with pytest.raises(ValueError, match="theme"):
        viz.plot_spineplot(results, data, binary, theme="sepia")
    plt.close("all")


def test_roc_isometrics_follow_the_direction(monkeypatch):
    d = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "y": [1, 1, 0, 1, 0, 0]})
    target = BinaryTarget().fit(d, "y", 1)
    results = [Rule((Condition("x", "le", 2),))]
    grids = []
    contour = matplotlib.axes.Axes.contour

    def recording(self, x, y, z, **kwargs):
        grids.append(z)
        return contour(self, x, y, z, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "contour", recording)
    for direction in ("positive", "negative", "both"):
        viz.plot_roc_space(results, d, target, isometrics=StandardQF(1, direction=direction))
    positive, negative, both = grids
    np.testing.assert_allclose(negative, -positive)
    np.testing.assert_allclose(both, np.abs(positive))
    plt.close("all")
