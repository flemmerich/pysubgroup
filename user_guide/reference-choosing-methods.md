# Choosing a target, measure, and algorithm

Start with the question. Choose a target representation and a score, then select
a compatible search. The fastest algorithm for an unrelated measure does not
answer the intended question.

## Target and quality selection

| Question | Target | Suggested starting measure | Explanation |
| --- | --- | --- | --- |
| Higher/lower frequency of one label? | `BinaryTarget` | `WRAccQF` or `StandardQF(a)` | [Binary quality](04-binary-quality.md) |
| Higher/lower numeric values? | `NumericTarget` | `StandardQFNumeric(a)` | [Numeric targets](07-numeric-targets.md) |
| Different numeric distribution? | `NumericTarget` | `KolmogorovSmirnovQF`, rank or spread measure | [Numeric targets](07-numeric-targets.md) |
| Different class composition? | `MulticlassTarget` | `ClassDistributionL1QF` or `SplitQF` | [Multiple targets](08-multiple-targets-and-wkl.md) |
| Different numeric mean vector? | `VectorTarget` | `MeanVectorL1QF` | [Multiple targets](08-multiple-targets-and-wkl.md) |
| Different marginal distributions of discrete attributes? | `NominalTarget` | `WeightedKLQF` | [WKL](08-multiple-targets-and-wkl.md) |
| Different correlation? | `CorrelationTarget` | `CorrelationDifferenceQF` | [EMM](09-exceptional-model-mining.md) |
| Different regression slope? | `RegressionTarget` | `SlopeSignificanceQF` | [EMM](09-exceptional-model-mining.md) |
| Different multiple-regression coefficients? | `GeneralRegressionTarget` | `CooksDistanceQF` | [EMM](09-exceptional-model-mining.md) |
| Different logistic-regression relationship? | `LogisticRegressionTarget` | `LogisticInteractionQF` | [EMM](09-exceptional-model-mining.md) |
| Different nominal association? | `AssociationTarget` | `AssociationDevianceQF` | [EMM](09-exceptional-model-mining.md) |
| Different conditional output distributions? | `DecisionTableTarget` | `HellingerQF` | [EMM](09-exceptional-model-mining.md) |
| Different parameters in a likelihood family? | `LikelihoodTarget` | `LikelihoodQF(statistic="ratio")` | [EMM](09-exceptional-model-mining.md) |
| Poorer classifier ranking? | `SoftClassifierTarget` | `ROCAUCQF`, `PRAUCQF`, or `ARLQF` | [Classifier auditing](10-classifier-auditing.md) |
| Frequent descriptions without a target? | `FITarget` | `CountQF` | [Specialized tasks](20-specialized-tasks.md) |
| Best rule for current loss derivatives? | `GradientTarget` | `GradientBoostingQF` | [Specialized tasks](20-specialized-tasks.md) |

## Search selection

| Situation | Starting choice | Guarantee | Important qualification |
| --- | --- | --- | --- |
| Ordinary target, small/moderate search space | `DFS` | **Exhaustive** | Within the configured language |
| Need alternative exhaustive ordering/pruning | `OPUS`, `DpSubgroup`, `BestFirstSearch` | **Exhaustive** | Compare on the actual task; memory needs differ |
| Prefer level-wise candidate generation | `Apriori` | **Exhaustive** | Large frontiers can consume memory |
| Prefer pattern-tree or vertical organization | `SDMapStar`, `SDMap`, `VLSD` | **Exhaustive** | Standard child batching/CUDA is unsupported |
| Numeric measure with ordering bounds | `NumBSD` | **Exhaustive** | Requires a supported numeric quality function |
| Model recoverable from aggregated statistics, especially EMM | `GPGrowth` | **Exhaustive** | Condensed valuations where supported; row sets otherwise |
| Generalization-aware measure | `GeneralizationAwareDFS`, `GeneralizationAwareApriori` | **Exhaustive** | These searches supply the required references |
| Heuristic search is acceptable | `BeamSearch` | **Heuristic** | Beam width does not guarantee the optimum |
| Diversity should affect exploration and final selection | `DSSD` | **Heuristic** | Also has different refinement semantics |

For the same task, all compatible exhaustive algorithms find the same optimal
top-k quality values. Their choice affects runtime, not solution quality; memory
requirements and the descriptions selected among equal-quality ties can differ.
Heuristic choices can change the attained quality as well as runtime. See the
[search chapter](06-search-algorithms.md) for the complete algorithm overview.

Exhaustive algorithms with no quality bound can still enumerate, but may be
expensive. No general-purpose compatibility table can predict which algorithm
will be fastest on every dataset.

## Which redundancy or validation operation?

| Need | Operation |
| --- | --- |
| Remove results covering nearly identical rows | `overlap_filter` |
| Keep one result per set of descriptor features | `unique_attributes` |
| Apply bounds on table statistics after search | `statistic_filter` |
| Favor additions that improve over simpler descriptions | Generalization-aware quality plus its dedicated search |
| Favor diversity while exploring | `DSSD` |
| Test fixed candidates on separate data | `randomization_p_values` plus `adjust_p_values` |
| Split, search, test, and select with a fixed supplied space | `holdout_search` |

Generalization awareness and diversity do not substitute for validation. A
test-based discovery score is also not an automatic correction for the search.
