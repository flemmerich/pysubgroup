---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Data and descriptions
---

# Data and subgroup descriptions

Before choosing an algorithm, decide **what a useful description may say**.
An exhaustive search cannot repair an unsuitable description language. A customer
identifier can isolate individual people; a measurement collected after the
outcome can turn a meaningful question into a trivial one.

## Descriptor and target roles

Targets and discovery tasks require a pandas DataFrame. Condition generation
and encoding also accept two-dimensional arrays, whose generated names are
`x0`, `x1`, and so on. Use a DataFrame when labels carry meaning.

Exclude target columns explicitly when generating descriptors. Searches also
exclude the fitted target's `attributes_`, but a target-derived proxy remains
the analyst's responsibility. A useful starting list contains attributes known
at the relevant time and understandable to the intended reader.

## Conditions and conjunctions

```{code-cell} python
import numpy as np
import pandas as pd
import pysubgroup as sd

data = pd.DataFrame({
    "age": [20, 30, 40, np.nan, 50],
    "plan": ["basic", "plus", "basic", None, "plus"],
    "outcome": [0, 1, 1, 0, 1],
}, index=[101, 305, 305, 900, 1200])

rule = sd.Rule((sd.Condition("age", "gt", 25), sd.Condition("plan", "eq", "plus")))
print(rule)
sd.subgroup_data(rule, data)
```

Every condition must match. Duplicate index labels are allowed: coverage is
defined by **row position**, not by index label. `Rule` is immutable and hashable;
subgroup discovery returns rules without a consequent. A list of discovered
subgroups is not a predictive `Ruleset` with a default class.

| Operator | Meaning | Construction |
| --- | --- | --- |
| `eq` | Present value equals a value | `Condition("plan", "eq", "plus")` |
| `ne` | Present value differs from a value | `Condition("plan", "ne", "plus")` |
| `in` | Present value belongs to a set | `Condition("plan", "in", ("basic", "plus"))` |
| `le`, `gt` | Numeric threshold | `Condition("age", "le", 30)` |
| `interval` | Left-open, right-closed interval | `Condition("age", "interval", lower=20, upper=40)` |
| `is_missing` | Value is missing | `Condition("age", "is_missing")` |

An unbounded interval endpoint is `None`. For `eq`, `ne`, and `in`, values must
be suitable hashable category values; missingness uses its own operator.

## Choosing candidate conditions manually

You do not have to use automatic condition generation. Write an ordinary Python
list, specifying each permitted condition individually:

```{code-cell} python
manual_conditions = [
    sd.Condition("plan", "eq", "basic"),
    sd.Condition("plan", "eq", "plus"),
    sd.Condition("age", "le", 30),
    sd.Condition("age", "gt", 30),
    sd.Condition("age", "interval", lower=20, upper=40),
    sd.Condition("age", "is_missing"),
]
print("All manually specified candidates:")
print(*map(str, manual_conditions), sep="\n")
```

Pass this list as `search_space=manual_conditions` when constructing a
`SubgroupDiscoveryTask`. Each list entry is a separate candidate predicate;
the algorithm decides which candidates to combine into conjunctions. This lets
you supply domain-specific thresholds or intervals instead of accepting the
cuts produced by `generate_conditions`.

## Missing values and interval boundaries

Missing descriptors match **only** `is_missing`. In particular, `ne` is not the
Boolean complement of `eq` when values are missing. This prevents “unknown plan”
from silently becoming “a plan other than plus.”

```{code-cell} python
checks = [
    sd.Condition("plan", "eq", "plus"),
    sd.Condition("plan", "ne", "plus"),
    sd.Condition("plan", "is_missing"),
    sd.Condition("age", "interval", lower=20, upper=40),
]
coverage = sd.Encoder(checks).fit_transform(data)
pd.DataFrame(
    {str(condition): coverage.condition_mask(i) for i, condition in enumerate(checks)},
    index=data.index,
)
```

The interval includes ages 30 and 40, excludes 20, and does not match the missing
age. These are `(20, 40]` semantics. Missing target values follow target-specific
rules: for example, numeric targets require finite values, while `BinaryTarget`
treats a missing label as negative unless the selected positive value is missing.

### Choosing which endpoints to include

When writing candidates manually, you can also express all four choices of
open or closed endpoints. The `interval` operator itself always uses
`(lower, upper]`; it has no flags that change inclusivity. Instead, choose its
cut points to represent the desired set of values.

For the finite `float64` ages in this example, `np.nextafter(value, -np.inf)`
returns the immediately preceding representable number. Moving the lower cut
from 20 to that predecessor includes 20; moving the upper cut from 40 to its
predecessor excludes 40:

```{code-cell} python
before_20 = np.nextafter(20.0, -np.inf)
before_40 = np.nextafter(40.0, -np.inf)
boundary_conditions = {
    "(20, 40]": sd.Condition("age", "interval", lower=20, upper=40),
    "[20, 40]": sd.Condition("age", "interval", lower=before_20, upper=40),
    "(20, 40)": sd.Condition("age", "interval", lower=20, upper=before_40),
    "[20, 40)": sd.Condition("age", "interval", lower=before_20, upper=before_40),
}
boundary_coverage = sd.Encoder(list(boundary_conditions.values())).fit_transform(data)
pd.DataFrame(
    {"age": data["age"].to_numpy(),
     **{label: boundary_coverage.condition_mask(i)
        for i, label in enumerate(boundary_conditions)}},
    index=data.index,
)
```

The labels in the table name the intended intervals; internally, each condition
still has the library's `(lower, upper]` form with its chosen numeric cuts.
No other `float64` value lies between a boundary and its predecessor, so these
choices express inclusion or exclusion exactly for such data. For whole-number
ages, the same idea is simpler: `[20, 40]` can be written with `lower=19,
upper=40`, and `[20, 40)` with `lower=19, upper=39`. Those integer cut points
would not represent the same intervals for fractional ages.

Each interval is a single candidate, so it can be used directly in standard
searches that allow at most one condition per feature. Missing values remain
excluded from all four variants.

## Generating conditions

```{code-cell} python
descriptors = data.drop(columns="outcome")
conditions = sd.generate_conditions(descriptors, n_bins=3)
print(*map(str, conditions), sep="\n")
```

Categorical and low-cardinality numeric columns generate equality conditions.
Higher-cardinality numeric columns are discretized:

- `numeric_candidates="bins"` gives disjoint interval conditions.
- `numeric_candidates="thresholds"` gives `le` and `gt` at each cut point.
- `n_bins=None` treats numeric values as categories.

By default, integer cuts use positions among sorted distinct values and float
cuts use quantiles. A supplied scikit-learn `KBinsDiscretizer`, or
`per_feature_discretizers`, gives more control. Categorical columns exceeding
`max_categories` are skipped; inspect the generated list rather than assuming
every column contributed candidates.

## Coverage for manual calculations

```{code-cell} python
encoded = sd.Encoder(checks, representation="packed").fit_transform(data)
mask = encoded.intersect([0, 3])
print("Positional mask:", mask)
print("Count:", encoded.count([0, 3]))
data.iloc[np.flatnonzero(mask)]
```

`dense` stores Boolean condition masks; `packed` stores them bit-packed.
Both expose the same coverage semantics. Packed encoding is useful when you
need many condition counts with reduced storage, but does not change the
description language.

The standard discovery searches permit at most one condition per feature.
To express a bounded age range there, supply an interval as one candidate.
[DSSD](12-diverse-subgroups.md) has a different refinement operator and can
combine certain conditions on the same feature.

**Experiment.** Replace the interval's upper endpoint with 30. Then set its
lower endpoint to `None`. Predict the matching rows before running the code.
