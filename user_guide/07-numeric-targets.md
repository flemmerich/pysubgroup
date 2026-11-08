---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Numeric targets
---

# Numeric targets

When the target is a measurement rather than a label, “unusual” can mean a
higher mean, a lower median, greater variability, or a differently shaped
distribution. Decide which effect matters before selecting a measure.
The numeric measures and ordering-based bounds follow the framework of
{cite:t}`lemmerich2016`.

## A mean-shift search

The scikit-learn diabetes dataset contains a numeric disease-progression
response and standardized descriptors. We use only four descriptors to keep the
search small. This example explores associations, not treatment effects.

```{code-cell} python
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pysubgroup as sd
from pysubgroup import visualization as viz

X, y = load_diabetes(return_X_y=True, as_frame=True)
X = X[["age", "sex", "bmi", "bp"]]
data = X.assign(progression=y)
target = sd.NumericTarget().fit(data, "progression")
conditions = sd.generate_conditions(X, n_bins=4, numeric_candidates="thresholds")
qf = sd.StandardQFNumeric(a=0.5, estimator="combined")
task = sd.SubgroupDiscoveryTask(data, target, conditions, qf,
                                depth=2, result_set_size=5,
                                filters=[sd.MinSizeFilter(30)])
results = sd.DFS().execute(task)
sd.to_dataframe(results, data, target, ["size", "mean", "mean_complement", "mean_difference", "std"])
```

For a mean target, the standard measure is

$$q_a(G)=|G|^a\left(\bar y_G-\bar y_D\right).$$

Unlike binary `StandardQF`, its size term is the **absolute count** $|G|^a$.
This changes score units across datasets, though using relative size would
multiply all scores within one dataset by the same constant.

## Mean versus median

The mean is sensitive to large values. The median describes a typical central
value and can be more robust to a few extremes. Both can be searched:

```{code-cell} python
from dataclasses import replace

median_qf = sd.StandardQFNumeric(a=0.5, centroid="median", estimator="order")
median_results = sd.DFS().execute(replace(task, qf=median_qf))
sd.to_dataframe(median_results, data, target, ["size", "mean", "median", "median_difference"])
```

Use `direction="negative"` for decreases and `"both"` for absolute deviations.
The exponent must lie in $[0,1]$. A median shift and a mean shift are different
questions; do not interpret a high mean score as evidence that most members
have high target values.

## Beyond location shifts

| Measure | What a large score emphasizes |
| --- | --- |
| `StandardQFNumericTscore` | Mean shift standardized by within-subgroup variation and size |
| `VarianceReductionQF` | Separation of subgroup and complement means |
| `InterclassVarianceQF` | The same ordering as variance reduction, scaled by dataset size |
| `StandardDeviationQF(a)` | A change in population standard deviation, size-weighted |
| `MannWhitneyQF` | Standardized rank difference relative to the complement |
| `RankAUCQF` | Probability a subgroup value exceeds a complement value, ties counting half |
| `KolmogorovSmirnovQF` | A two-sided difference between cumulative distributions, size-scaled |

For example, a bimodal subgroup may have an ordinary mean but an unusual spread
or cumulative distribution. Rank measures reduce sensitivity to target units.
The package's t-score gives exactly zero for a constant subgroup; inspect the
raw distribution rather than relying on a degenerate standardized statistic.

```{code-cell} python
sd.to_dataframe(results, data, target, ["size", "mean", "median", "std"],
    measures={"rank_auc": sd.RankAUCQF(), "KS": sd.KolmogorovSmirnovQF()})
```

These extra columns **rescore the discovered list**. They do not turn it into
the best list for the additional measures. To answer a different question,
rerun the search with that quality function.

```{code-cell} python
viz.plot_target_distributions(results, data, target)
plt.show()
```

Compare medians, spread, and extreme values in the plot. The listed subgroups
can overlap, so their distributions are not independent samples.

## Ordering bounds and NumBSD

For many measures, the best arbitrary subset of a given size consists of the
largest target values, or of either the largest or smallest values. Checking
these prefixes/suffixes yields a tight bound on arbitrary row subsets. The
target is sorted once and subgroup rows are gathered in that order.

- `estimator="order"` requests ordering-based bounds where supported.
- `estimator="closed"` uses cheaper closed-form bounds for applicable measures.
- Mean-based `StandardQFNumeric` offers `"sum"`, `"max"`, and `"combined"`;
  `"combined"` takes the smaller of the sum and max bounds.

`NumBSD` uses this ordered structure directly, with look-ahead and optional fast
numba bounds for suitable mean measures:

```{code-cell} python
ordered_task = replace(task, qf=sd.StandardQFNumeric(0.5, estimator="order"))
ordered_results = sd.NumBSD().execute(ordered_task)
sd.to_dataframe(ordered_results, data, target, ["size", "mean", "mean_difference"])
```

NumBSD requires a supported numeric measure; it is not a general replacement
for DFS on other target families. Faster bounds do not change the quality
definition. Measure their cost on the actual task before assuming tighter
bounds will make the search faster.

**Experiment.** Search with `direction="both"` and with a rank-based measure.
Which groups move up? Explain the change in terms of target distributions and
reference groups, rather than the scales of incomparable quality scores.
