import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from pysubgroup import (
    DFS,
    AssociationDevianceQF,
    AssociationTarget,
    BDeuQF,
    BeamSearch,
    CooksDistanceQF,
    CorrelationDifferenceQF,
    CorrelationSignificanceQF,
    CorrelationTarget,
    DecisionTableTarget,
    GeneralRegressionTarget,
    HellingerQF,
    LogisticInteractionQF,
    LogisticRegressionTarget,
    MinSizeFilter,
    RegressionTarget,
    SlopeSignificanceQF,
    SubgroupDiscoveryTask,
    generate_conditions,
    to_dataframe,
)
from pysubgroup._evaluation import Backend, ChildStatistics
from pysubgroup.emm import entropy_weight
from pysubgroup.utils import subgroup_masks

HOUSING_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/HousePrices.csv"
HOUSING_CACHE = Path(__file__).resolve().parents[1] / "data-cache" / "HousePrices.csv"


@pytest.fixture(scope="module")
def housing():
    """The Windsor housing data (Anglin and Gencay 1996) of the EMM papers, cached locally."""
    if not HOUSING_CACHE.exists():
        try:
            HOUSING_CACHE.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(HOUSING_URL, HOUSING_CACHE)
        except OSError as error:
            pytest.skip(f"housing data not available: {error}")
    return pd.read_csv(HOUSING_CACHE)


def frame(seed, rows=120):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({f"a{j}": rng.choice(list("xyz"), rows) for j in range(3)})
    data["n"] = rng.integers(0, 6, rows).astype(float)
    x = rng.normal(size=rows)
    slope = np.where(data.a0 == "x", -1.0, 1.0)
    data["x"] = x
    data["w"] = rng.normal(size=rows)
    data["y"] = slope * x + 0.5 * data.w + rng.normal(scale=0.7, size=rows)
    data["label"] = rng.random(rows) < 1 / (1 + np.exp(-(slope * x)))
    data["u"] = np.where(rng.random(rows) < 0.6, data.a1, rng.choice(list("pq"), rows))
    data["v"] = np.where(data.a0 == "y", rng.choice(list("ab"), rows), data.u.isin(["x", "p"]))
    data["v"] = data.v.astype(str)
    return data


def random_masks(seed, rows, count=12):
    rng = np.random.default_rng(seed + 100)
    return [rng.random(rows) < rng.uniform(0.15, 0.85) for _ in range(count)]


# Brute-force references from the papers' formulas.

def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def slope_and_variance(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)
    return slope, np.sum(residuals**2) / ((len(x) - 2) * np.sum((x - x.mean()) ** 2))


def cooks_distance(predictors, response, mask, selected):
    design = np.column_stack([np.ones(len(response)), predictors])
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    beta_g = np.linalg.lstsq(design[mask], response[mask], rcond=None)[0]
    s2 = np.sum((response - design @ beta) ** 2) / (len(response) - design.shape[1])
    inverse = np.linalg.inv(design.T @ design)[np.ix_(selected, selected)]
    difference = (beta_g - beta)[selected]
    return difference @ np.linalg.solve(inverse, difference) / (len(selected) * s2)


def hellinger(inputs, output, mask, reference):
    prior = output.value_counts(normalize=True)
    cells = inputs.drop_duplicates().itertuples(index=False)
    total = 0.0
    for cell in cells:
        in_cell = (inputs == pd.Series(cell, index=inputs.columns)).all(axis=1).to_numpy()
        estimates = []
        for group in (mask, reference):
            rows = output[in_cell & group]
            estimates.append(
                rows.value_counts(normalize=True).reindex(prior.index, fill_value=0.0)
                if len(rows) else prior
            )
        total += float(np.sum((np.sqrt(estimates[0]) - np.sqrt(estimates[1])) ** 2))
    return total


def test_entropy_weight():
    assert entropy_weight(5, 10) == pytest.approx(1.0)
    assert entropy_weight(np.array([0, 10]), 10).tolist() == [0.0, 0.0]
    assert entropy_weight(1, 4) == pytest.approx(-0.25 * math.log2(0.25) - 0.75 * math.log2(0.75))


@pytest.mark.parametrize("seed", range(3))
def test_correlation_measures_match_their_definitions(seed):
    data = frame(seed)
    target = CorrelationTarget().fit(data, "x", "y")
    x, y = data.x.to_numpy(), data.y.to_numpy()
    for mask in random_masks(seed, len(data)):
        r_g, r_c = correlation(x[mask], y[mask]), correlation(x[~mask], y[~mask])
        n, n_c = mask.sum(), (~mask).sum()
        z = (np.arctanh(r_g) - np.arctanh(r_c)) / np.sqrt(1 / (n - 3) + 1 / (n_c - 3))
        assert CorrelationSignificanceQF().evaluate(mask, target) == pytest.approx(
            1 - 2 * stats.norm.sf(abs(z))
        )
        assert CorrelationSignificanceQF("statistic").evaluate(mask, target) == pytest.approx(
            abs(z)
        )
        assert CorrelationDifferenceQF().evaluate(mask, target) == pytest.approx(abs(r_g - r_c))
        assert CorrelationDifferenceQF(entropy=True).evaluate(mask, target) == pytest.approx(
            entropy_weight(n, len(data)) * abs(r_g - r_c)
        )
        assert CorrelationDifferenceQF("dataset").evaluate(mask, target) == pytest.approx(
            abs(r_g - correlation(x, y))
        )


def test_correlation_measures_of_degenerate_groups():
    data = frame(0)
    target = CorrelationTarget().fit(data, "x", "y")
    everything, nothing = np.ones(len(data), dtype=bool), np.zeros(len(data), dtype=bool)
    small = nothing.copy()
    small[:3] = True
    for qf in (CorrelationSignificanceQF(), CorrelationDifferenceQF()):
        for mask in (everything, nothing):
            assert math.isnan(qf.evaluate(mask, target))
    assert math.isnan(CorrelationSignificanceQF().evaluate(small, target))
    assert CorrelationDifferenceQF("dataset").evaluate(everything, target) == 0.0
    constant = data.assign(x=np.where(np.arange(len(data)) < 10, 1.0, data.x))
    target = CorrelationTarget().fit(constant, "x", "y")
    first = np.arange(len(data)) < 10
    assert math.isnan(CorrelationDifferenceQF().evaluate(first, target))


def test_numeric_model_targets_validate_columns():
    data = frame(0)
    with pytest.raises(ValueError, match="requires 2 attributes"):
        CorrelationTarget().fit(data, "x")
    with pytest.raises(TypeError, match="numeric"):
        RegressionTarget().fit(data, "x", "a0")
    with pytest.raises(ValueError, match="finite"):
        RegressionTarget().fit(data.assign(y=np.nan), "x", "y")
    with pytest.raises(ValueError, match="constant"):
        RegressionTarget().fit(data.assign(y=1.0), "x", "y")
    with pytest.raises(ValueError, match="score"):
        SlopeSignificanceQF("p")
    with pytest.raises(ValueError, match="reference"):
        CorrelationDifferenceQF("parent")


@pytest.mark.parametrize("seed", range(3))
def test_slope_significance_matches_its_definition(seed):
    data = frame(seed)
    target = RegressionTarget().fit(data, "x", "y")
    x, y = data.x.to_numpy(), data.y.to_numpy()
    for mask in random_masks(seed, len(data)):
        (b_g, s_g), (b_c, s_c) = slope_and_variance(x[mask], y[mask]), slope_and_variance(
            x[~mask], y[~mask]
        )
        t = (b_g - b_c) / np.sqrt(s_g + s_c)
        df = (s_g + s_c) ** 2 / (s_g**2 / (mask.sum() - 2) + s_c**2 / ((~mask).sum() - 2))
        assert SlopeSignificanceQF().evaluate(mask, target) == pytest.approx(
            1 - 2 * stats.t.sf(abs(t), df)
        )
        assert SlopeSignificanceQF("statistic").evaluate(mask, target) == pytest.approx(abs(t))
    two = np.arange(len(data)) < 2
    assert math.isnan(SlopeSignificanceQF().evaluate(two, target))


@pytest.mark.parametrize("seed", range(3))
def test_cooks_distance_matches_its_definition(seed):
    data = frame(seed)
    target = GeneralRegressionTarget().fit(data, "x", "w", "y")
    predictors, response = data[["x", "w"]].to_numpy(), data.y.to_numpy()
    assert target.coefficient_names_ == ("intercept", "x", "w")
    for mask in random_masks(seed, len(data)):
        for names, selected in ((None, [0, 1, 2]), (["x"], [1]), (["w", "intercept"], [2, 0])):
            assert CooksDistanceQF(names).evaluate(mask, target) == pytest.approx(
                cooks_distance(predictors, response, mask, selected)
            )
    assert CooksDistanceQF().evaluate(np.ones(len(data), dtype=bool), target) == 0.0
    tiny = np.arange(len(data)) < 2
    assert math.isnan(CooksDistanceQF().evaluate(tiny, target))
    with pytest.raises(ValueError, match="unknown coefficients"):
        CooksDistanceQF(["z"]).evaluate(tiny, target)


def test_cooks_distance_of_collinear_subgroups_is_nan():
    data = frame(0)
    data["w"] = np.where(data.a0 == "x", 2 * data.x, data.w)
    target = GeneralRegressionTarget().fit(data, "x", "w", "y")
    assert math.isnan(CooksDistanceQF().evaluate((data.a0 == "x").to_numpy(), target))


def logistic_reference(x, labels, mask):
    """Maximum-likelihood fit by a generic optimizer and the Wald z of the interaction."""
    d = mask.astype(float)
    design = np.column_stack([np.ones_like(x), d, x, d * x])
    y = labels.astype(float)

    def negative_log_likelihood(beta):
        eta = design @ beta
        return np.sum(np.logaddexp(0.0, eta) - y * eta)

    beta = optimize.minimize(negative_log_likelihood, np.zeros(4), method="BFGS",
                             options={"gtol": 1e-10}).x
    p = 1 / (1 + np.exp(-design @ beta))
    covariance = np.linalg.inv(design.T @ (design * (p * (1 - p))[:, None]))
    return beta[3] / np.sqrt(covariance[3, 3])


@pytest.mark.parametrize("seed", range(3))
def test_logistic_interaction_matches_a_generic_fit(seed):
    data = frame(seed)
    target = LogisticRegressionTarget().fit(data, "x", "label")
    standardized = (data.x - data.x.mean()) / data.x.std(ddof=0)
    for mask in random_masks(seed, len(data), count=5):
        z = logistic_reference(standardized.to_numpy(), data.label.to_numpy(), mask)
        assert LogisticInteractionQF("statistic").evaluate(mask, target) == pytest.approx(
            abs(z), rel=1e-5, abs=1e-6
        )
        assert LogisticInteractionQF().evaluate(mask, target) == pytest.approx(
            1 - 2 * stats.norm.sf(abs(z)), rel=1e-5, abs=1e-6
        )
        # The Wald statistic does not depend on the scale and origin of x.
        shifted = LogisticRegressionTarget().fit(data.assign(x=3 * data.x + 7), "x", "label")
        assert LogisticInteractionQF("statistic").evaluate(mask, shifted) == pytest.approx(
            abs(z), rel=1e-5, abs=1e-6
        )


def test_logistic_interaction_of_separated_or_pure_groups_is_nan():
    data = frame(0)
    separated = data.assign(label=data.x > 0)
    target = LogisticRegressionTarget().fit(separated, "x", "label")
    assert math.isnan(LogisticInteractionQF().evaluate((data.a0 == "x").to_numpy(), target))
    target = LogisticRegressionTarget().fit(data, "x", "label")
    pure = data.label.to_numpy()
    assert math.isnan(LogisticInteractionQF().evaluate(pure, target))
    target = LogisticRegressionTarget().fit(data.assign(label=data.a0), "x", "label", "y")
    assert target.labels_.tolist() == (data.a0 == "y").tolist()


def admissions():
    """Table 2 of DMKD Sect. 6.2: gender, admission, and program type."""
    rows = []
    counts = {("F", "No"): (30, 80, 10), ("F", "Yes"): (10, 80, 40),
              ("M", "No"): (20, 120, 30), ("M", "Yes"): (40, 120, 20)}
    for (gender, admitted), per_type in counts.items():
        for program, count in zip("ABC", per_type, strict=True):
            rows += [(gender, admitted, program)] * count
    return pd.DataFrame(rows, columns=["gender", "admitted", "type"])


def test_association_deviance_reproduces_table_3():
    data = admissions()
    target = AssociationTarget().fit(data, "gender", "admitted")
    qf = AssociationDevianceQF()
    assert qf.evaluate((data.type == "A").to_numpy(), target) == pytest.approx(21.09, abs=0.005)
    assert qf.evaluate((data.type == "C").to_numpy(), target) == pytest.approx(20.60, abs=0.005)
    # The paper reports 0.28 for type = B; the maximum-likelihood fit (also by a
    # root of the common odds ratio, see below) gives 0.0135.
    assert qf.evaluate((data.type == "B").to_numpy(), target) == pytest.approx(0.013513, abs=1e-6)


def common_odds_ratio_deviance(tables):
    """Deviance of the homogeneous 2 x 2 x K model from its common odds ratio."""
    def fitted(table, ratio):
        row, column, total = table[0].sum(), table[:, 0].sum(), table.sum()
        low, high = max(0, row + column - total), min(row, column)
        a = optimize.brentq(lambda a: a * (total - row - column + a)
                            - ratio * (row - a) * (column - a), low + 1e-12, high - 1e-12)
        return np.array([[a, row - a], [column - a, total - row - column + a]])

    observed = sum(table[0, 0] for table in tables)
    ratio = optimize.brentq(lambda r: sum(fitted(t, r)[0, 0] for t in tables) - observed,
                            1e-4, 1e4)
    return 2 * sum(np.sum(t * np.log(t / fitted(t, ratio))) for t in tables)


@pytest.mark.parametrize("seed", range(3))
def test_association_deviance_matches_the_common_odds_ratio_fit(seed):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({"x": rng.choice(list("ab"), 300), "y": rng.choice(list("cd"), 300),
                         "g": rng.random(300) < 0.4})
    data["y"] = np.where(data.g & (rng.random(300) < 0.3), np.where(data.x == "a", "c", "d"),
                         data.y)
    target = AssociationTarget().fit(data, "x", "y")
    mask = data.g.to_numpy()
    tables = [pd.crosstab(data.x[group], data.y[group]).to_numpy() for group in (mask, ~mask)]
    assert AssociationDevianceQF().evaluate(mask, target) == pytest.approx(
        common_odds_ratio_deviance(tables), rel=1e-7
    )


def test_association_fit_keeps_margins_and_a_common_association():
    from pysubgroup.emm.association import homogeneous_association

    rng = np.random.default_rng(4)
    table = rng.integers(1, 30, (3, 4, 2)).astype(float)
    fitted = homogeneous_association(table)
    for axis in (0, 1, 2):
        np.testing.assert_allclose(fitted.sum(axis=axis), table.sum(axis=axis), rtol=1e-8)
    log = np.log(fitted)
    interaction = log[1:, 1:] - log[1:, :1] - log[:1, 1:] + log[:1, :1]
    np.testing.assert_allclose(interaction[..., 0], interaction[..., 1], atol=1e-8)


def test_association_target_and_degenerate_groups():
    data = admissions()
    with pytest.raises(ValueError, match="requires 2 attributes"):
        AssociationTarget().fit(data, "gender")
    target = AssociationTarget().fit(data, "gender", "admitted")
    for mask in (np.zeros(len(data), dtype=bool), np.ones(len(data), dtype=bool)):
        assert math.isnan(AssociationDevianceQF().evaluate(mask, target))


@pytest.mark.parametrize("seed", range(3))
def test_hellinger_matches_its_definition(seed):
    data = frame(seed)
    target = DecisionTableTarget().fit(data, "a1", "a2", "v")
    inputs, output = data[["a1", "a2"]], data.v
    for mask in random_masks(seed, len(data), count=6):
        mask = mask & ~((data.a1 == "x") & (data.a2 == "z")).to_numpy()  # an empty cell
        assert HellingerQF().evaluate(mask, target) == pytest.approx(
            hellinger(inputs, output, mask, ~mask)
        )
        assert HellingerQF("dataset").evaluate(mask, target) == pytest.approx(
            hellinger(inputs, output, mask, np.ones(len(data), dtype=bool))
        )
        assert HellingerQF(entropy=True).evaluate(mask, target) == pytest.approx(
            entropy_weight(mask.sum(), len(data)) * hellinger(inputs, output, mask, ~mask)
        )


def test_bdeu_reproduces_the_paper_and_its_definition():
    # DMKD Sect. 6.7.2: 16 rows in one of I = 4 cells, all with the same output.
    data = pd.DataFrame({"band": ["deletion", "no change"] * 32,
                         "mycn": ["amplified"] * 32 + ["not amplified"] * 32,
                         "status": ["event", "no event"] * 32})
    data.loc[data.index[-16:], ["band", "mycn", "status"]] = ["no change", "not amplified",
                                                              "no event"]
    target = DecisionTableTarget().fit(data, "band", "mycn", "status")
    mask = np.arange(len(data)) >= len(data) - 16
    assert BDeuQF().evaluate(mask, target) == pytest.approx(-1.075, abs=6e-4)
    other = np.arange(len(data)) % 3 == 0
    a, i, j = 2.5, 4, 2
    expected = 0.0
    for _, cell in data[other].groupby(["band", "mycn"]):
        expected += math.lgamma(a / i) - math.lgamma(a / i + len(cell))
        for count in cell.status.value_counts():
            expected += math.lgamma(a / (i * j) + count) - math.lgamma(a / (i * j))
    assert BDeuQF(a).evaluate(other, target) == pytest.approx(expected)
    assert math.isnan(BDeuQF().evaluate(np.zeros(len(data), dtype=bool), target))


# (target factory, quality function, relative tolerance of batched against scalar results)
MEASURES = [
    (lambda d: CorrelationTarget().fit(d, "x", "y"), CorrelationSignificanceQF(), 1e-9),
    (lambda d: CorrelationTarget().fit(d, "x", "y"), CorrelationSignificanceQF("statistic"),
     1e-9),
    (lambda d: CorrelationTarget().fit(d, "x", "y"), CorrelationDifferenceQF(entropy=True), 1e-9),
    (lambda d: CorrelationTarget().fit(d, "x", "y"), CorrelationDifferenceQF("dataset"), 1e-9),
    (lambda d: RegressionTarget().fit(d, "x", "y"), SlopeSignificanceQF("statistic"), 1e-9),
    (lambda d: RegressionTarget().fit(d, "x", "y"), SlopeSignificanceQF(), 1e-9),
    (lambda d: GeneralRegressionTarget().fit(d, "x", "w", "y"), CooksDistanceQF(), 1e-9),
    (lambda d: GeneralRegressionTarget().fit(d, "x", "w", "y"), CooksDistanceQF(["w"]), 1e-9),
    (lambda d: AssociationTarget().fit(d, "u", "v"), AssociationDevianceQF(), 1e-9),
    (lambda d: DecisionTableTarget().fit(d, "a1", "u", "v"), HellingerQF(), 1e-12),
    (lambda d: DecisionTableTarget().fit(d, "a1", "u", "v"), HellingerQF("dataset"), 1e-12),
    (lambda d: DecisionTableTarget().fit(d, "a1", "u", "v"), HellingerQF(entropy=True), 1e-12),
    (lambda d: DecisionTableTarget().fit(d, "a1", "u", "v"), BDeuQF(), 1e-12),
]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(("make", "qf", "rtol"), MEASURES)
def test_batched_qualities_equal_scalar_ones(seed, make, qf, rtol):
    data = frame(seed)
    target = make(data)
    rng = np.random.default_rng(seed)
    masks = rng.random((30, len(data))) < rng.uniform(0.02, 0.9, (30, 1))
    masks[0], masks[1] = False, True
    parent = None if seed % 2 else rng.random(len(data)) < 0.7
    statistics = ChildStatistics(Backend(masks), parent, range(len(masks)))
    children = [m if parent is None else m & parent for m in masks]
    assert qf.batchable
    np.testing.assert_allclose(
        qf.evaluate_many(statistics, target),
        [qf.evaluate(mask, target) for mask in children],
        rtol=rtol, atol=1e-9,
    )


@pytest.mark.parametrize(("make", "qf", "rtol"), MEASURES)
def test_searches_find_the_same_subgroups_batched_and_scalar(make, qf, rtol):
    data = frame(1)
    target = make(data)
    conditions = generate_conditions(data[["a0", "a1", "a2", "n", "x"]], n_bins=3)
    task = SubgroupDiscoveryTask(data, target, conditions, qf, result_set_size=5, depth=2)
    for algorithm in (DFS(), BeamSearch(beam_width=5)):
        scalar = algorithm.execute(SubgroupDiscoveryTask(**{**task.__dict__, "batched": False}))
        batched = algorithm.execute(SubgroupDiscoveryTask(**{**task.__dict__, "batched": True}))
        np.testing.assert_allclose([q for q, _ in batched], [q for q, _ in scalar], rtol=rtol)
        for _, rule in batched:
            assert not {c.feature for c in rule.conditions} & set(target.attributes_)


def test_logistic_interaction_in_a_search():
    data = frame(2)
    target = LogisticRegressionTarget().fit(data, "x", "label")
    conditions = generate_conditions(data[["a0", "a1"]])
    results = DFS().execute(SubgroupDiscoveryTask(
        data, target, conditions, LogisticInteractionQF("statistic"), result_set_size=3, depth=1
    ))
    assert str(results[0][1].conditions[0]) in {"a0=x", "a0≠x"}


def test_housing_correlations_reproduce_table_1(housing):
    target = CorrelationTarget().fit(housing, "lotsize", "price")
    x, y = housing.lotsize.to_numpy(float), housing.price.to_numpy(float)
    nbath, drive, rec = (housing.bathrooms >= 2, housing.driveway == "yes",
                         housing.recreation == "yes")
    expected = [(nbath, 0.564, 144), (drive, 0.502, 469), (rec, 0.375, 97),
                (nbath & drive, 0.509, 128), (nbath & rec, 0.129, 38), (drive & rec, 0.304, 90),
                (nbath & rec & ~drive, -0.894, 3), (nbath & rec & drive, -0.090, 35)]
    assert correlation(x, y) == pytest.approx(0.536, abs=5e-4)
    for mask, r, size in expected:
        mask = mask.to_numpy()
        assert mask.sum() == size
        assert correlation(x[mask], y[mask]) == pytest.approx(r, abs=6e-4)
    d1 = (nbath & rec & drive).to_numpy()
    assert CorrelationDifferenceQF().evaluate(d1, target) == pytest.approx(0.090 + 0.549, abs=1e-3)
    # The papers report phi_scd(D1) = 0.9993; the formula of DMKD Sect. 5.1 gives 0.99989.
    assert CorrelationSignificanceQF().evaluate(d1, target) == pytest.approx(0.999893, abs=1e-6)


def test_housing_slopes_reproduce_section_6_3(housing):
    from pysubgroup.emm.regression import simple_regression

    target = RegressionTarget().fit(housing, "lotsize", "price")
    d2 = ((housing.driveway == "yes") & (housing.fullbase == "no")
          & (housing.bathrooms <= 1)).to_numpy()
    assert d2.sum() == 226
    slope, intercept, _ = simple_regression(target, *target.moments(d2))
    slope_c, intercept_c, _ = simple_regression(target, *target.complement(*target.moments(d2)))
    assert (slope[0], intercept[0]) == pytest.approx((3.31, 41568), abs=0.5)
    assert slope[0] == pytest.approx(3.31, abs=5e-3)
    assert (slope_c[0], intercept_c[0]) == pytest.approx((8.45, 30723), abs=0.5)
    assert slope_c[0] == pytest.approx(8.45, abs=5e-3)
    assert SlopeSignificanceQF().evaluate(d2, target) > 0.9999


# Searches with the standard algorithms

def emm_task(data, qf, **changes):
    target = RegressionTarget().fit(data, "x", "y")
    conditions = generate_conditions(data[["a0", "a1", "a2", "n", "w"]], n_bins=4,
                                     numeric_candidates="thresholds")
    settings = {"result_set_size": 6, "depth": 2, **changes}
    return SubgroupDiscoveryTask(data, target, conditions, qf, **settings)


def test_dfs_finds_the_papers_slope_subgroup(housing):
    data = housing.drop(columns="rownames")
    target = RegressionTarget().fit(data, "lotsize", "price")
    conditions = generate_conditions(data.drop(columns=["price", "lotsize"]), n_bins=4,
                                     numeric_candidates="thresholds")
    task = SubgroupDiscoveryTask(data, target, conditions, SlopeSignificanceQF("statistic"),
                                 result_set_size=5, depth=3, filters=[MinSizeFilter(26)])
    quality, rule = DFS().execute(task)[0]
    # D2 of DMKD Sect. 6.3: driveway, no basement, at most one bathroom (226 houses).
    assert set(map(str, rule.conditions)) == {"bathrooms=1", "driveway=yes", "fullbase=no"}
    assert quality == pytest.approx(6.578, abs=1e-3)


# Result statistics (utils.to_dataframe)

def test_to_dataframe_reports_emm_model_statistics():
    data = frame(5)
    x, y, w = data.x.to_numpy(), data.y.to_numpy(), data.w.to_numpy()
    rules = [r for _, r in BeamSearch(beam_width=3).execute(
        emm_task(data, SlopeSignificanceQF(), result_set_size=3))]
    masks = subgroup_masks(rules, data)
    table = to_dataframe(rules, data, RegressionTarget().fit(data, "x", "y"), statistics="all")
    for row, mask in zip(table.itertuples(), masks, strict=True):
        assert row.correlation == pytest.approx(correlation(x[mask], y[mask]))
        assert row.correlation_complement == pytest.approx(correlation(x[~mask], y[~mask]))
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        assert (row.slope, row.intercept) == pytest.approx((slope, intercept))
        slope, intercept = np.polyfit(x[~mask], y[~mask], 1)
        assert (row.slope_complement, row.intercept_complement) == pytest.approx(
            (slope, intercept))
    table = to_dataframe(rules, data, GeneralRegressionTarget().fit(data, "x", "w", "y"))
    for row, mask in zip(table.to_dict("records"), masks, strict=True):
        design = np.column_stack([np.ones(mask.sum()), x[mask], w[mask]])
        beta = np.linalg.lstsq(design, y[mask], rcond=None)[0]
        assert [row["coefficient intercept"], row["coefficient x"], row["coefficient w"]] == (
            pytest.approx(beta))
    target = LogisticRegressionTarget().fit(data, "x", "label")
    table = to_dataframe(rules, data, target)
    for row, mask in zip(table.itertuples(), masks, strict=True):
        beta = LogisticInteractionQF().coefficients(mask, target)[0]
        scale = x.std()
        assert row.logit_slope == pytest.approx((beta[2] + beta[3]) / scale)
        assert row.logit_slope_complement == pytest.approx(beta[2] / scale)
    table = to_dataframe(rules, data, CorrelationTarget().fit(data, "x", "y"))
    assert list(table.columns) == ["subgroup", "size", "relative_size", "correlation",
                                   "correlation_complement"]


@pytest.mark.parametrize(("make", "qf"), [
    (AssociationTarget, AssociationDevianceQF()),
    (DecisionTableTarget, HellingerQF()),
])
def test_fitting_again_drops_cached_cell_masks(make, qf):
    old = pd.DataFrame({"x": [0, 0, 1, 1, 0, 0, 1, 1], "y": [0, 1, 0, 1, 0, 1, 0, 1]})
    new = old.assign(y=[0, 0, 1, 1, 0, 1, 0, 1])
    masks = np.array([[True] * 3 + [False] * 5, [False] * 3 + [True] * 5])
    target = make().fit(old, "x", "y")
    qf.evaluate_many(ChildStatistics(Backend(masks), None, range(2)), target)
    target.fit(new, "x", "y")
    np.testing.assert_allclose(
        qf.evaluate_many(ChildStatistics(Backend(masks), None, range(2)), target),
        [qf.evaluate(mask, make().fit(new, "x", "y")) for mask in masks],
    )
