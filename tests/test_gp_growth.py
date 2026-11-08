import numpy as np
import pandas as pd
import pytest
from test_algorithms import FILTER_SETS, brute_force, data
from test_emm import frame, random_masks

from pysubgroup import (
    DFS,
    AssociationDevianceQF,
    AssociationTarget,
    BDeuQF,
    BinaryTarget,
    ClassDistributionL1QF,
    CooksDistanceQF,
    CorrelationDifferenceQF,
    CorrelationDomain,
    CorrelationSignificanceQF,
    CorrelationTarget,
    CovarianceDomain,
    CustomFilter,
    DecisionTableDomain,
    DecisionTableTarget,
    GeneralRegressionTarget,
    GiniQF,
    GPGrowth,
    GradientBoostingQF,
    GradientTarget,
    HellingerQF,
    KolmogorovSmirnovQF,
    LabelledTarget,
    LikelihoodQF,
    LikelihoodTarget,
    LogisticInteractionQF,
    LogisticRegressionTarget,
    MeanVectorL1QF,
    MinSizeFilter,
    MinTargetShareFilter,
    MulticlassTarget,
    NominalTarget,
    NumericTarget,
    PolynomialRegressionModel,
    RegressionTarget,
    SlopeDomain,
    SlopeSignificanceQF,
    SplitQF,
    StandardDeviationQF,
    StandardQF,
    StandardQFNumeric,
    SubgroupDiscoveryTask,
    ValuationQF,
    VarianceDomain,
    VarianceReductionQF,
    VectorTarget,
    WeightedKLQF,
    WRAccQF,
    generate_conditions,
)
from pysubgroup._evaluation import Backend, ChildStatistics

ALGORITHMS = [
    GPGrowth(), GPGrowth("rows"), GPGrowth(use_optimistic_estimates=True),
    GPGrowth("rows", use_optimistic_estimates=True),
]


def assert_finds(task, expected, rel=1e-9):
    for algorithm in ALGORITHMS:
        found = algorithm.execute(task)
        assert [q for q, _ in found] == pytest.approx(expected, rel=rel, abs=1e-12), (
            algorithm.valuation, algorithm.use_optimistic_estimates
        )
        for _, rule in found:
            features = [c.feature for c in rule.conditions]
            assert len(set(features)) == len(features)


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("filters", FILTER_SETS)
@pytest.mark.parametrize("make", [
    lambda d: (BinaryTarget().fit(d, "label", 1), WRAccQF()),
    lambda d: (BinaryTarget().fit(d, "label", 1), StandardQF(0.5, direction="negative")),
    lambda d: (NumericTarget().fit(d, "value"), StandardQFNumeric(1)),
    lambda d: (NumericTarget().fit(d, "value"),
               StandardQFNumeric(0.5, "order", "median", direction="negative")),
    lambda d: (NumericTarget().fit(d, "value"), StandardQFNumeric(0.5, "combined")),
    lambda d: (NumericTarget().fit(d, "value"), VarianceReductionQF("order")),
    lambda d: (NumericTarget().fit(d, "value"), StandardDeviationQF(0.5)),
    lambda d: (NumericTarget().fit(d, "value"), KolmogorovSmirnovQF()),
    lambda d: (MulticlassTarget().fit(d, "cls"), SplitQF()),
    lambda d: (MulticlassTarget().fit(d, "cls"), GiniQF()),
    lambda d: (MulticlassTarget().fit(d, "cls"), ClassDistributionL1QF()),
    lambda d: (VectorTarget().fit(d, "s0", "s1", "s2"), MeanVectorL1QF()),
    lambda d: (GradientTarget().fit(d, "g", "h"), GradientBoostingQF(2.0)),
    lambda d: (BinaryTarget().fit(d, "label", 1), WeightedKLQF()),
    lambda d: (MulticlassTarget().fit(d, "cls"), WeightedKLQF()),
    lambda d: (NominalTarget().fit(d, "label", "cls"), WeightedKLQF()),
])
def test_gp_growth_finds_the_best_qualities(seed, filters, make):
    d, conditions = data(seed)
    target, qf = make(d)
    if not isinstance(target, LabelledTarget) and any(
        isinstance(f, MinTargetShareFilter) for f in filters
    ):
        return
    task = SubgroupDiscoveryTask(d, target, conditions, qf, result_set_size=7, depth=3,
                                 filters=filters)
    assert_finds(task, brute_force(task))


EMM = [
    (lambda d: CorrelationTarget().fit(d, "x", "y"), CorrelationSignificanceQF()),
    (lambda d: CorrelationTarget().fit(d, "x", "y"), CorrelationDifferenceQF(entropy=True)),
    (lambda d: RegressionTarget().fit(d, "x", "y"), SlopeSignificanceQF("statistic")),
    (lambda d: GeneralRegressionTarget().fit(d, "x", "w", "y"), CooksDistanceQF()),
    (lambda d: AssociationTarget().fit(d, "u", "v"), AssociationDevianceQF()),
    (lambda d: DecisionTableTarget().fit(d, "a1", "u", "v"), HellingerQF()),
    (lambda d: DecisionTableTarget().fit(d, "a1", "u", "v"), BDeuQF()),
    (lambda d: LogisticRegressionTarget().fit(d, "x", "label"), LogisticInteractionQF()),
    (lambda d: LikelihoodTarget(PolynomialRegressionModel(1)).fit(d, "x", "y"), LikelihoodQF()),
    (lambda d: CorrelationTarget().fit(d, "x", "y"), ValuationQF(CorrelationDomain())),
    (lambda d: CorrelationTarget().fit(d, "x", "y"), ValuationQF(CovarianceDomain(), "dataset")),
    (lambda d: RegressionTarget().fit(d, "x", "y"), ValuationQF(SlopeDomain())),
    (lambda d: NumericTarget().fit(d, "y"), ValuationQF(VarianceDomain())),
    (lambda d: DecisionTableTarget().fit(d, "a1", "u", "v"), ValuationQF(DecisionTableDomain())),
]


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize(("make", "qf"), EMM)
def test_gp_growth_on_emm_targets(seed, make, qf):
    d = frame(seed)
    conditions = generate_conditions(d[["a0", "a1", "a2", "n"]], n_bins=3)
    task = SubgroupDiscoveryTask(d, make(d), conditions, qf, result_set_size=6, depth=3,
                                 filters=[MinSizeFilter(10)])
    assert_finds(task, brute_force(task), rel=1e-7)


def test_likelihood_qualities_match_dfs():
    # pysubgroup 0.x's GpGrowth scores EMM_Likelihood wrongly.
    d = frame(0)
    conditions = generate_conditions(d[["a0", "a1", "a2"]])
    task = SubgroupDiscoveryTask(d, LikelihoodTarget(PolynomialRegressionModel(1)).fit(d, "x", "y"),
                                 conditions, LikelihoodQF(), result_set_size=6, depth=2)
    expected = DFS().execute(task)
    found = GPGrowth().execute(task)
    assert [r for _, r in found] == [r for _, r in expected]
    assert [q for q, _ in found] == pytest.approx([q for q, _ in expected], rel=1e-12)


def test_task_is_not_modified():
    # pysubgroup 0.x's GpGrowth raises task.min_quality.
    d, conditions = data(0)
    task = SubgroupDiscoveryTask(d, BinaryTarget().fit(d, "label", 1), conditions, WRAccQF(),
                                 result_set_size=1, depth=2)
    GPGrowth(use_optimistic_estimates=True).execute(task)
    assert task.min_quality == float("-inf")


@pytest.mark.parametrize("seed", range(3))
def test_paper_models_match_numpy(seed):
    d = frame(seed)
    target = CorrelationTarget().fit(d, "x", "y")
    x, y = d.x.to_numpy(), d.y.to_numpy()
    for mask in random_masks(seed, len(d)):
        xs, ys = x[mask], y[mask]
        expected = {
            VarianceDomain(): np.var(xs),
            VarianceDomain(1): np.var(ys),
            CovarianceDomain(): np.cov(xs, ys, bias=True)[0, 1],
            CorrelationDomain(): np.corrcoef(xs, ys)[0, 1],
            SlopeDomain(): np.polyfit(xs, ys, 1)[0],
        }
        for domain, value in expected.items():
            rows = domain.project(target)
            assert domain.model(domain.aggregate_all(rows[mask]), target) == pytest.approx(value)


@pytest.mark.parametrize("domain", [CovarianceDomain(), CorrelationDomain(), SlopeDomain()])
def test_comoment_aggregation_and_subtraction(domain):
    d = frame(1)
    target = RegressionTarget().fit(d, "x", "y")
    rows = domain.project(target)
    rng = np.random.default_rng(0)
    parts = np.array_split(rng.permutation(len(d)), 4)
    bases = [domain.aggregate_all(rows[part]) for part in parts]
    total = domain.aggregate_all(rows)
    pairwise = domain.aggregate(domain.aggregate(bases[0], bases[1]),
                                domain.aggregate(bases[2], bases[3]))
    assert pairwise == pytest.approx(total)
    assert domain.aggregate_all(np.stack(bases)) == pytest.approx(total)
    rest = domain.subtract(total, bases[0])
    assert rest == pytest.approx(domain.aggregate_all(rows[np.concatenate(parts[1:])]))
    assert domain.aggregate(bases[0], np.zeros_like(bases[0])) == pytest.approx(bases[0])


@pytest.mark.parametrize("seed", range(3))
def test_valuation_qf_reproduces_existing_measures(seed):
    d = frame(seed)
    correlation = CorrelationTarget().fit(d, "x", "y")
    table = DecisionTableTarget().fit(d, "a1", "u", "v")
    pairs = [
        (ValuationQF(CorrelationDomain()), CorrelationDifferenceQF(), correlation),
        (ValuationQF(CorrelationDomain(), "dataset"), CorrelationDifferenceQF("dataset"),
         correlation),
        (ValuationQF(DecisionTableDomain()), HellingerQF(), table),
        (ValuationQF(DecisionTableDomain(), "dataset"), HellingerQF("dataset"), table),
    ]
    for valuation, reference, target in pairs:
        for mask in random_masks(seed, len(d)):
            assert valuation.evaluate(mask, target) == pytest.approx(
                reference.evaluate(mask, target), rel=1e-9
            )
        assert np.isnan(valuation.evaluate(np.zeros(len(d), dtype=bool), target))
    assert np.isnan(ValuationQF(CorrelationDomain()).evaluate(np.ones(len(d), dtype=bool),
                                                              correlation))


def test_condensed_valuation_never_evaluates_row_masks(monkeypatch):
    d, conditions = data(0)
    task = SubgroupDiscoveryTask(d, NumericTarget().fit(d, "value"), conditions,
                                 StandardQFNumeric(1), result_set_size=5, depth=3)
    expected = brute_force(task)

    def fail(*args, **kwargs):
        raise AssertionError("scalar evaluation")

    monkeypatch.setattr(StandardQFNumeric, "evaluate", fail)
    for algorithm in (GPGrowth(), GPGrowth(use_optimistic_estimates=True)):
        assert [q for q, _ in algorithm.execute(task)] == pytest.approx(expected)


def test_unbatchable_filters_and_errors():
    d, conditions = data(1)
    target = BinaryTarget().fit(d, "label", 1)
    no_a1 = CustomFilter(lambda m, t, c: all(x.feature != "a1" for x in c), anti_monotone=True)
    result_only = CustomFilter(lambda m, t, c: m.sum() % 2 == 0)
    task = SubgroupDiscoveryTask(d, target, conditions, ValuationQF(CorrelationDomain()),
                                 filters=[no_a1])
    with pytest.raises(TypeError, match="NumericTarget"):
        GPGrowth().execute(task)
    task = SubgroupDiscoveryTask(d, target, conditions, WRAccQF(), result_set_size=6, depth=3,
                                 filters=[MinSizeFilter(5), result_only])
    assert_finds(task, brute_force(task))
    with pytest.raises(ValueError, match="valuation"):
        GPGrowth("frequencies")
    with pytest.raises(ValueError, match="batched"):
        GPGrowth().execute(SubgroupDiscoveryTask(d, target, conditions, WRAccQF(), batched=True))
    empty = SubgroupDiscoveryTask(d, target, [], WRAccQF())
    assert GPGrowth().execute(empty) == []


def test_depth_one_and_constant_conditions():
    d = pd.DataFrame({"a": ["x"] * 6 + ["y"] * 4, "b": list("pqpqpqpqpq"),
                      "label": [1, 1, 1, 0, 1, 0, 0, 0, 1, 0]})
    conditions = generate_conditions(d[["a", "b"]])
    target = BinaryTarget().fit(d, "label", 1)
    for depth in (1, 2):
        task = SubgroupDiscoveryTask(d, target, conditions, WRAccQF(), depth=depth)
        assert_finds(task, brute_force(task))


def test_weighted_kl_reuses_its_value_masks_across_expansions():
    d, _ = data(0)
    target = NominalTarget().fit(d, "label", "cls")
    backend = Backend(np.array(random_masks(0, len(d), 4)))
    for _ in range(3):
        WeightedKLQF().evaluate_many(ChildStatistics(backend, None, range(4)), target)
    assert len(backend._arrays) == 2 + d.cls.nunique()


def test_fitting_again_drops_cached_valuation_bases():
    target = NumericTarget().fit(pd.DataFrame({"y": [0.0, 1.0, 2.0, 4.0]}), "y")
    qf = ValuationQF(VarianceDomain())
    mask = np.array([True, True, False, False])
    assert qf.evaluate(mask, target) == pytest.approx(0.75)
    new = pd.DataFrame({"y": [0.0, 10.0, 20.0, 40.0]})
    assert qf.evaluate(mask, target.fit(new, "y")) == pytest.approx(
        qf.evaluate(mask, NumericTarget().fit(new, "y"))
    )
