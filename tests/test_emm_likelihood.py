import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pysubgroup import (
    DFS,
    BeamSearch,
    LikelihoodModel,
    LikelihoodQF,
    LikelihoodTarget,
    PolynomialRegressionModel,
    SubgroupDiscoveryTask,
    generate_conditions,
    holdout_search,
    to_dataframe,
)
from pysubgroup.utils import subgroup_masks


def frame(seed, rows=150):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({f"a{j}": rng.choice(list("xyz"), rows) for j in range(3)})
    x = rng.normal(size=rows)
    data["x"] = x
    data["y"] = np.where(data.a0 == "x", -x + 0.5 * x**2, x) + rng.normal(scale=0.6, size=rows)
    return data


def random_masks(seed, rows, count=10):
    rng = np.random.default_rng(seed + 100)
    return [rng.random(rows) < rng.uniform(0.15, 0.85) for _ in range(count)]


def residual_sum(x, y, degree):
    return np.sum(np.square(y - np.polyval(np.polyfit(x, y, degree), x)))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_likelihood_ratio_matches_its_closed_form(seed, degree):
    data = frame(seed)
    x, y = data.x.to_numpy(), data.y.to_numpy()
    target = LikelihoodTarget(PolynomialRegressionModel(degree)).fit(data, "x", "y")
    for mask in random_masks(seed, len(data)):
        n, n_c, big_n = mask.sum(), (~mask).sum(), len(data)
        # Normal errors with variance RSS / n: l = -n/2 (log(2 pi RSS / n) + 1).
        ratio = (big_n * np.log(residual_sum(x, y, degree) / big_n)
                 - n * np.log(residual_sum(x[mask], y[mask], degree) / n)
                 - n_c * np.log(residual_sum(x[~mask], y[~mask], degree) / n_c))
        assert LikelihoodQF(score="statistic").evaluate(mask, target) == pytest.approx(ratio)
        assert LikelihoodQF().evaluate(mask, target) == pytest.approx(
            stats.chi2.cdf(ratio, degree + 2))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("variance", [None, 1.0, 0.3])
def test_density_difference_matches_its_definition(seed, variance):
    data = frame(seed)
    x, y = data.x.to_numpy(), data.y.to_numpy()
    target = LikelihoodTarget(PolynomialRegressionModel(2, variance)).fit(data, "x", "y")
    for mask in random_masks(seed, len(data)):
        residuals = y - np.polyval(np.polyfit(x[mask], y[mask], 2), x)
        scale = np.sqrt(variance or np.mean(np.square(residuals[mask])))
        density = stats.norm.pdf(residuals, scale=scale)
        expected = density[mask].mean() - density[~mask].mean()
        assert LikelihoodQF("density").evaluate(mask, target) == pytest.approx(expected)


def test_likelihood_ratio_with_a_fixed_variance_is_a_difference_of_residuals():
    data = frame(4)
    x, y = data.x.to_numpy(), data.y.to_numpy()
    target = LikelihoodTarget(PolynomialRegressionModel(1, variance=2.0)).fit(data, "x", "y")
    mask = (data.a0 == "x").to_numpy()
    expected = (residual_sum(x, y, 1) - residual_sum(x[mask], y[mask], 1)
                - residual_sum(x[~mask], y[~mask], 1)) / 2.0
    assert LikelihoodQF(score="statistic").evaluate(mask, target) == pytest.approx(expected)


def test_degenerate_groups_score_nan():
    data = frame(0, rows=40)
    data.loc[:4, "x"] = 1.0
    data.loc[5:7, "y"] = 3 * data.loc[5:7, "x"]
    target = LikelihoodTarget(PolynomialRegressionModel(1)).fit(data, "x", "y")
    rows = np.arange(len(data))
    for qf in (LikelihoodQF(), LikelihoodQF("density")):
        assert np.isnan(qf.evaluate(np.zeros(len(data), bool), target))
        assert np.isnan(qf.evaluate(np.ones(len(data), bool), target))
        # Two rows: too few for slope, intercept, and variance.
        assert np.isnan(qf.evaluate((rows >= 10) & (rows < 12), target))
        # One distinct x value: the slope is not identified.
        assert np.isnan(qf.evaluate(rows < 5, target))
        # An exact fit has an unbounded likelihood.
        assert np.isnan(qf.evaluate((rows >= 5) & (rows < 8), target))
        assert np.isfinite(qf.evaluate(rows < 6, target))
    # A fixed variance needs only as many rows as coefficients.
    fixed = LikelihoodTarget(PolynomialRegressionModel(1, 1.0)).fit(data, "x", "y")
    for qf in (LikelihoodQF(), LikelihoodQF("density")):
        assert np.isfinite(qf.evaluate((rows >= 10) & (rows < 12), fixed))
        assert np.isfinite(qf.evaluate((rows >= 5) & (rows < 8), fixed))


def test_validation():
    data = frame(0)
    with pytest.raises(ValueError, match="statistic"):
        LikelihoodQF("mean")
    with pytest.raises(ValueError, match="score"):
        LikelihoodQF("density", score="statistic")
    with pytest.raises(ValueError, match="degree"):
        PolynomialRegressionModel(-1)
    with pytest.raises(ValueError, match="variance"):
        PolynomialRegressionModel(1, variance=0.0)
    with pytest.raises(TypeError, match="LikelihoodModel"):
        LikelihoodTarget(object())
    model = PolynomialRegressionModel()
    with pytest.raises(ValueError, match="x and y"):
        LikelihoodTarget(model).fit(data, "x")
    with pytest.raises(TypeError, match="numeric"):
        LikelihoodTarget(model).fit(data, "a0", "y")
    with pytest.raises(ValueError, match="constant"):
        LikelihoodTarget(model).fit(data.assign(x=1.0), "x", "y")
    target = LikelihoodTarget(model).fit(data, "x", "y")
    # The target prepares a copy; the model passed in keeps only its configuration.
    assert not hasattr(model, "design_") and hasattr(target.model_, "design_")
    refitted = target.refit(data.iloc[:80])
    assert refitted.n_samples_ == 80 and refitted.model is model


class NormalMeanModel(LikelihoodModel):
    """A user-defined model class: one numeric attribute with a normal distribution."""

    def prepare(self, data, attributes):
        (name,) = attributes
        self.values_ = data[name].to_numpy(float)

    @property
    def n_parameters(self):
        return 2

    def fit(self, mask):
        values = self.values_[mask]
        return (values.mean(), values.var()) if len(values) > 1 else None

    def log_likelihood(self, parameters, mask=None):
        values = self.values_ if mask is None else self.values_[mask]
        return stats.norm.logpdf(values, parameters[0], np.sqrt(parameters[1]))


def test_custom_model_class_in_a_search():
    data = frame(3)
    data["y"] += np.where(data.a1 == "z", 3.0, 0.0)
    target = LikelihoodTarget(NormalMeanModel()).fit(data, "y")
    y = data.y.to_numpy()
    mask = (data.a1 == "z").to_numpy()
    n, n_c, big_n = mask.sum(), (~mask).sum(), len(y)
    expected = big_n * np.log(y.var()) - n * np.log(y[mask].var()) - n_c * np.log(y[~mask].var())
    qf = LikelihoodQF(score="statistic")
    assert qf.evaluate(mask, target) == pytest.approx(expected)
    conditions = generate_conditions(data[["a0", "a1", "a2", "x"]], n_bins=3)
    task = SubgroupDiscoveryTask(data, target, conditions, qf, result_set_size=5, depth=2)
    dfs, beam = DFS().execute(task), BeamSearch(beam_width=10).execute(task)
    assert str(dfs[0][1].conditions[0]) in {"a1=z", "a1≠z"}
    assert dfs[0][0] == pytest.approx(expected)
    assert [q for q, _ in beam] == pytest.approx([q for q, _ in dfs])


def test_search_and_holdout_with_polynomial_regression():
    data = frame(5, rows=300)
    target = LikelihoodTarget(PolynomialRegressionModel(2)).fit(data, "x", "y")
    conditions = generate_conditions(data[["a0", "a1", "a2", "x", "y"]], n_bins=3)
    task = SubgroupDiscoveryTask(data, target, conditions, LikelihoodQF(score="statistic"),
                                 result_set_size=3, depth=1)
    results = DFS().execute(task)
    assert str(results[0][1].conditions[0]) in {"a0=x", "a0≠x"}
    for _, rule in results:
        assert not {c.feature for c in rule.conditions} & {"x", "y"}
    # Significance filtering refits the target, with its model class, on both parts.
    result = holdout_search(DFS(), task, k=3, n_samples=20, random_state=0)
    assert len(result.p_values) == len(result.candidates) > 0


def test_to_dataframe_reports_model_parameters():
    data = frame(6)
    x, y = data.x.to_numpy(), data.y.to_numpy()
    target = LikelihoodTarget(PolynomialRegressionModel(2)).fit(data, "x", "y")
    conditions = generate_conditions(data[["a0", "a1"]])
    results = DFS().execute(SubgroupDiscoveryTask(data, target, conditions, LikelihoodQF(),
                                                  result_set_size=3, depth=1))
    table = to_dataframe(results, data, target, statistics="all")
    for row, mask in zip(table.to_dict("records"), subgroup_masks(results, data), strict=True):
        for group, suffix in ((mask, ""), (~mask, "_complement")):
            coefficients = np.polyfit(x[group], y[group], 2)[::-1]
            residuals = y[group] - np.polyval(coefficients[::-1], x[group])
            assert [row[f"b{j}{suffix}"] for j in range(3)] == pytest.approx(coefficients)
            assert row[f"variance{suffix}"] == pytest.approx(np.mean(np.square(residuals)))
