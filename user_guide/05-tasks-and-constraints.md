---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Tasks and constraints
---

# Tasks, constraints, and the search space

A discovery task is a precise statement of which descriptions are eligible and
how to rank them. The same task can be passed to different compatible algorithms.

```{code-cell} python
from dataclasses import replace
from sklearn.datasets import load_iris
import pysubgroup as sd

X, y = load_iris(return_X_y=True, as_frame=True)
data = X.assign(species=y)
target = sd.BinaryTarget().fit(data, "species", 2)
conditions = sd.generate_conditions(X, n_bins=4, numeric_candidates="thresholds")
task = sd.SubgroupDiscoveryTask(
    data, target, conditions, sd.WRAccQF(),
    result_set_size=8, depth=2, min_quality=0,
    filters=[sd.MinSizeFilter(0.1)],
)
results = sd.DFS().execute(task)
sd.to_dataframe(results, data, target)
```

## Three different limits

**`result_set_size`** limits the number of returned descriptions, not the
number examined. A request for ten results can still require exploring many
thousands of candidates.

**`depth`** limits the number of conditions. It controls both interpretability
and combinatorial complexity. Start shallow and increase it for a reason.

**`min_quality`** supplies a quality floor. Returned qualities must be strictly
above it. The effective threshold can rise during top-k search once enough
better candidates have been found. An eligible result list may contain fewer
than the requested number of subgroups.

## Absolute and relative size constraints

`MinSizeFilter(10)` requires ten rows. `MinSizeFilter(0.1)` requires at least
10% of the task dataset. Integer and floating-point arguments deliberately have
different meanings; `10.0` is not an alternative spelling of ten rows.

For relative minimum size, a fractional threshold is rounded up to a row count;
relative maximum size is rounded down. A relative threshold on validation or
discovery subsets refers to their own sizes when their tasks are constructed.

## Pruning constraints versus result filters

A refinement adds conditions, so it cannot gain rows. Therefore a group that
already has fewer than ten rows cannot produce a refinement with ten rows.
Minimum size is **anti-monotone** and can safely prune the entire branch.

Target share behaves differently. A subgroup with a low positive share can
contain a strongly positive refinement. Rejecting its whole branch would lose
valid results.

| Filter | Prunes a failing branch? | Reason |
| --- | --- | --- |
| `MinSizeFilter` | Yes | Refinements cannot gain rows |
| `MinPositivesFilter` | Yes | Refinements cannot gain positives |
| `MinMajorityCountFilter` | Yes | Refinements cannot gain rows of any class |
| `MaxSizeFilter` | No | A refinement can become small enough |
| `MinTargetShareFilter` | No | A refinement can become enriched |
| `CustomFilter` | Only if explicitly declared | The analyst must justify anti-monotonicity |

```{code-cell} python
restricted = replace(task, filters=[sd.MinSizeFilter(15), sd.MinTargetShareFilter(0.8)])
sd.to_dataframe(sd.DFS().execute(restricted), data, target)
```

Filters supplied to a task participate in search and result eligibility.
[Post-hoc filters](12-diverse-subgroups.md) instead process an already ranked
list; they cannot recover a candidate that never entered that list.

## The language is part of the question

For the standard searches:

- Descriptions are nonempty conjunctions of at most `depth` conditions.
- A description uses at most one condition per feature.
- Target attributes are excluded.
- Only supplied candidate conditions can be selected.

With fixed numeric bins, the search sees those bins. With thresholds, it sees
the supplied one-sided cuts. An exhaustive search over one language need not
match a search over the other. Build interval candidates explicitly when a
bounded interval should be available as one condition.

DSSD differs: its refinement operator permits certain repeated conditions on a
feature and, by default, recomputes numeric thresholds inside each subgroup.
Use `n_bins=None` to keep its numeric candidates fixed. This makes search-space
choices important when comparing DSSD with other methods.

## Change one factor at a time

```{code-cell} python
summary = []
for depth in [1, 2, 3]:
    found = sd.DFS().execute(replace(task, depth=depth))
    summary.append({"depth": depth, "returned": len(found),
                    "best_quality": found[0][0] if found else None})
import pandas as pd
pd.DataFrame(summary)
```

The best attainable quality cannot decrease when the allowed depth grows and
all other eligibility rules remain fixed. But the computation can grow quickly,
and more complex descriptions are not automatically more useful.

**Experiment.** Choose a constraint that is not anti-monotone, such as “at least
80% positives.” Construct a small parent subgroup that fails and a refinement
that passes. Explain why treating it as a pruning filter would be incorrect.
