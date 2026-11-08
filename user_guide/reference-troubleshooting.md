# Troubleshooting and migration

## “No subgroups were returned”

Check the target label, the generated conditions, minimum size, maximum depth,
and quality floor. With `min_quality=0`, zero and negative qualities are excluded.
An undefined quality is also excluded. Inspect a few manually chosen rules with
`to_dataframe` and score their masks directly before enlarging the search.

For ROC AUC, a group needs positives and negatives. For EMM, a group may be too
small, constant, singular, or impossible to fit. A larger result limit cannot
create eligible candidates.

## “The target has the right length, but the scores look wrong”

Targets store positional arrays. They must be fitted on the exact row order
being evaluated. Sorting a DataFrame, taking a same-length sample, or replacing
its rows invalidates a previously fitted target even if the index or length
looks compatible. Use `target.refit(new_data)`.

Masks are positional Boolean arrays, not index-label selections. Prefer
`subgroup_data(rule, data)` or `data.iloc[np.flatnonzero(mask)]` when the index
is nontrivial.

## “A numeric interval includes a boundary I did not expect”

Intervals are `(lower, upper]`: lower excluded, upper included. Missing values
are outside every numeric interval. `ne` also excludes missing values. Use
`is_missing` to represent missingness explicitly.

## “A feature did not appear in the search space”

Inspect the output of `generate_conditions`. High-cardinality categories can
be skipped through `max_categories`; numeric behavior depends on `n_bins` and
`numeric_candidates`. Target attributes are excluded from descriptions. Arrays
generate feature names `x0`, `x1`, and so on; their first feature is not named
the integer `0`.

For DataFrames, integer-valued `Condition.feature` means column **position**,
not an integer column label. Prefer readable string column names.

## “I need a bounded interval, but the search will not add a second threshold”

Standard searches permit one condition per feature. Supply an `interval`
candidate as one condition. DSSD allows some repeated conditions and can create
numeric cuts during refinement, so it has different language semantics.

## “Batching or CUDA raises an error”

`batched=True` requires support from the algorithm, measure, and pruning filters.
A custom scalar anti-monotone filter can disable batching. CUDA requires that
batched path and a compatible CuPy installation. Median/order numeric bounds,
SubROC, logistic fits, and several specialized searches cannot use it.

Use `batched=None` for automatic CPU selection and compare against
`batched=False` while debugging. See [Performance](17-performance.md).

## “Two exhaustive algorithms returned different rules”

First ensure identical tasks, candidate ordering, preprocessing, and constraints.
Compare quality values. Equal-quality descriptions at the cutoff can differ
because a bound equal to the current threshold is pruned. Numeric aggregation
order can also introduce small rounding differences.

Beam search and DSSD are heuristic. DSSD's default numeric thresholds and
same-feature refinements can additionally make its description space different.

## “My p-values are all one”

With size and class counts fixed, WRAcc and other functions of only those counts
are constant. `stratify_target=True` therefore gives p-value one for them.
Undefined validation statistics also get p-value one. Inspect the statistic,
validation counts, and the randomization constraints before changing the test.

## “No result passes after correction”

This can be an appropriate outcome. Check the Monte Carlo resolution: with
`plus_one=True`, the minimum p-value is `1/(R+1)`, and correction can make that
insufficient. Choose the number of draws and the correction as part of the
validation design, rather than repeatedly relaxing them after inspecting results.

## Moving from pysubgroup 0.x

The libraries share many concepts, but their APIs and some semantics differ:

| Topic | pysubgroup behavior |
| --- | --- |
| Target construction | Instantiate and then `.fit(data, ...)` |
| Candidate language | Immutable `Condition` objects; not pysubgroup 0.x selectors |
| Descriptions | `Rule` conjunctions; no general disjunction/DNF discovery language |
| Task results | Descending list of `(quality, Rule)` pairs |
| Result table | `sd.to_dataframe(results, data, target, ...)` |
| Numeric intervals | `(lower, upper]`, whereas pysubgroup 0.x uses `[lower, upper)` |
| Missing `ne` values | Do not match; `ne` is not a complete logical complement of `eq` |
| Row masks | Positional, independent of DataFrame index labels |
| Binary `LiftQF` | Scores share difference; table `lift` is a ratio |
| Standard size exponent | Restricted to `[0,1]`, where implemented bounds are valid |
| Example datasets | Load through scikit-learn/OpenML rather than a bundled datasets module |

For interval conversion between different inclusivity conventions, use
floating-point `nextafter` where appropriate rather than merely copying
endpoints. Always compare the covered rows, not only rendered descriptions.

There is no guarantee of identical tied descriptions, numeric defaults, or
undefined-case handling across libraries. Reproduce the target, search space,
measure parameters, and constraints before interpreting numerical differences.
