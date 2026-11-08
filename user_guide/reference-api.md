# API quick reference

The public classes and helpers below are imported from `pysubgroup` unless a
module is explicitly named. This is a workflow-oriented reference; executable
examples and interpretive detail are in the linked chapters.

## Descriptions and encoding

| API | Purpose |
| --- | --- |
| `Condition(feature, operator="eq", value=None, lower=None, upper=None)` | Immutable predicate; intervals are `(lower, upper]` |
| `Rule(conditions=(), consequent=None)` | Immutable conjunction; empty rule covers all rows |
| `Ruleset(...)` | Predictive collection with consequent/default semantics; not a discovery result list |
| `generate_conditions(X, n_bins=10, numeric_candidates="bins", ...)` | Generate an editable candidate list |
| `Encoder(conditions, representation="dense")` | Compile condition evaluation; also supports `"packed"` |
| `encoder.fit_transform(X)` / `encoder.transform(X)` | Create a `Coverage` object |
| `coverage.intersect(indices)` | Positional Boolean conjunction mask |
| `coverage.count(indices, rows=None)` | Count a conjunction, optionally within a row mask |
| `coverage.condition_mask(index)` | Positional mask of one condition |
| `prepare_frame(X)` | Validate input and normalize supported datetime/timedelta columns |

See [Data and descriptions](03-data-and-descriptions.md) for missing values,
feature naming, numeric candidates, and index semantics.

## Fitting targets

All target fitting calls begin with a DataFrame:

| Target | Typical fitting call after construction |
| --- | --- |
| `BinaryTarget` | `.fit(data, attribute, positive_value)` |
| `NumericTarget`, `MulticlassTarget` | `.fit(data, attribute)` |
| `VectorTarget`, `NominalTarget` | `.fit(data, *attributes)` |
| `SoftClassifierTarget` | `.fit(data, label_attribute, score_attribute, positive_value=1)` |
| `GradientTarget` | `.fit(data, gradient_attribute, hessian_attribute)` |
| `FITarget` | `.fit(data)` |
| `CorrelationTarget`, `RegressionTarget` | `.fit(data, x_attribute, y_attribute)` |
| `GeneralRegressionTarget` | `.fit(data, *predictors, response)` with response last |
| `LogisticRegressionTarget` | `.fit(data, x_attribute, y_attribute, positive_value)` |
| `AssociationTarget` | `.fit(data, x_attribute, y_attribute)` |
| `DecisionTableTarget` | `.fit(data, *inputs, output)` with output last |
| `LikelihoodTarget(model)` | `.fit(data, *model_attributes)` |

`target.refit(new_data)` creates a new target with the same fitting specification.
`attributes_` identifies the columns excluded from descriptions; `n_samples_`
records the number of positional target rows.

## Quality function families

| Family | Public classes |
| --- | --- |
| Frequency | `CountQF`, `AreaQF` |
| Binary | `StandardQF(a, direction="positive")`, `LiftQF`, `SimpleBinomialQF`, `WRAccQF`, `ChiSquaredQF` |
| Numeric | `StandardQFNumeric(a, ...)`, `StandardQFNumericTscore`, `VarianceReductionQF`, `InterclassVarianceQF`, `StandardDeviationQF(a, ...)`, `MannWhitneyQF`, `RankAUCQF`, `KolmogorovSmirnovQF` |
| Multiclass | `SplitQF`, `MulticlassChiSquaredQF`, `GiniQF`, `ClassDistributionL1QF`, `MaxConfidenceQF` |
| Multiple targets | `MeanVectorL1QF`, `WeightedKLQF` |
| Classifier performance | `ARLQF`, `ROCAUCQF`, `PRAUCQF` |
| Gradients | `GradientBoostingQF(reg_lambda=1)` |
| Generalization-aware | `GeneralizationAwareQF(qf)`, `GeneralizationAwareStandardQF(a)`, `GeneralizationAwareStandardQFNumeric(a, ...)` |
| EMM | `CorrelationDifferenceQF`, `CorrelationSignificanceQF`, `SlopeSignificanceQF`, `CooksDistanceQF`, `LogisticInteractionQF`, `AssociationDevianceQF`, `HellingerQF`, `BDeuQF`, `LikelihoodQF`, `ValuationQF` |

`qf.evaluate(mask, target)` returns the scalar quality. Bounded measures add
`optimistic_estimate(mask, target)`. Generalization-aware measures require
`reference=...`; `AreaQF` requires `depth=...`. Batchable classes additionally
provide `evaluate_many` and, if bounded, `optimistic_estimate_many`.

## Task and search

```python
task = sd.SubgroupDiscoveryTask(
    data, target, search_space, qf,
    result_set_size=10, depth=3, min_quality=float("-inf"),
    filters=[], batched=None, device="cpu",
)
results = sd.DFS().execute(task)
```

`results` is a descending list of `(float_quality, Rule)` pairs. Searches return
up to the requested number; scores must be above `min_quality`.

Search classes: `DFS`, `BestFirstSearch`, `OPUS`, `DpSubgroup`, `Apriori`, `SDMap`,
`SDMapStar`, `VLSD`, `BeamSearch`, `NumBSD`, `DSSD`, `GPGrowth`,
`GeneralizationAwareApriori`, and `GeneralizationAwareDFS`.

Selected options worth looking up before comparison:

- `BeamSearch(beam_width=20, beam_width_adaptive=False)`.
- `DpSubgroup(exploration="term_rank", iterative_deepening=False)`.
- `VLSD(sort_criterion_in_s1="no-order", sort_criterion_in_other_sizes="no-order")`.
- `NumBSD(fast_bounds=True)`.
- `DSSD(selection="cover", beam_width=100, pool_size=10_000, n_bins=6, ...)`.
- `GPGrowth(valuation="auto", use_optimistic_estimates=False)`.
- Generalization-aware searches: `difference_based="auto"`.

See [Search algorithms](06-search-algorithms.md), [Diversity](12-diverse-subgroups.md),
and [Performance](17-performance.md) for compatibility and guarantees.

## Filters and result helpers

Task filters: `MinSizeFilter`, `MaxSizeFilter`, `MinPositivesFilter`,
`MinTargetShareFilter`, `MinMajorityCountFilter`, `CustomFilter`.

| Helper | Return/use |
| --- | --- |
| `to_dataframe(results, data, target, statistics=None, measures=None)` | One result row per subgroup |
| `available_statistics(target)` | Available target-specific statistic names |
| `subgroup_data(rule_or_result_pair, data)` | Covered rows with original index |
| `utils.subgroup_masks(results, data)` | Boolean array shaped `(subgroups, rows)` |
| `overlap_filter(results, data, max_similarity=0.9, similarity="jaccard")` | Greedy row-overlap selection |
| `unique_attributes(results)` | One result per descriptor feature set |
| `statistic_filter(results, data, target, minimum=None, maximum=None)` | Inclusive bounds on result statistics |
| `generalization_reference(qf, rule, data, target)` | Maximum value over proper generalizations |
| `count_calls(qf)` | Counting copy with `call_counts` |
| `subgroup_covers(results, data)` | Coverage array for set diversity helpers |
| `cover_redundancy(covers)`, `joint_entropy(covers)` | Cover-set statistics |

## Significance

| API | Purpose |
| --- | --- |
| `holdout_split(data, validation_size=0.5, stratify=None, random_state=None)` | Discovery/validation split |
| `randomization_p_values(results, data, target, qf, ...)` | Test fixed covers against randomized row subsets |
| `randomization_null_distribution(...)` | Observed statistics and random-sample statistics |
| `adjust_p_values(p_values, method="by")` | Multiple-testing adjustment |
| `significance_filter(results, data, target, qf, ...)` | Return candidates passing the specified test/correction |
| `holdout_search(algorithm, task, ...)` | Combined workflow returning `HoldoutResult` |
| `HoldoutResult.to_dataframe(..., candidates=False)` | Discovery, validation, and significance columns |

Common randomization options: `n_samples=1000`, `stratify_target=False`,
`stratify=None`, `random_state=None`, `plus_one=False`. The combined workflow
uses `k` for the final result limit and `task.result_set_size` for the candidate
pool. See [Validation](14-validation.md) for the null model and fixed-space
behavior.

## Visualization and extension protocols

Plots live in `pysubgroup.visualization`, conventionally imported as `viz`.
They take results and data, with a fitted target for target-dependent plots,
accept `ax=` and `theme=`, and return a matplotlib `Axes`. In particular,
`plot_similarity` and `plot_similarity_dendrogram` do not take a target.
See the [plot selection table](11-results-and-visualization.md).

Extension bases: `BaseTarget`, `LabelledTarget`, `QualityFunction`,
`BoundedQualityFunction`, `Filter`, `LikelihoodModel`, and `ValuationDomain`.
See [Custom quality functions](18-custom-quality-functions.md) and
[Custom targets and models](19-custom-extensions.md) for complete examples.
