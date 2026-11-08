---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Custom quality functions
---

# Writing a quality function

A quality function receives a positional Boolean coverage mask and a fitted
target. It returns a number, with **larger meaning more interesting**. This
chapter derives a small linear cost measure directly, then adds a valid bound
and batching without changing the score.

## Define the objective first

Suppose each covered positive has benefit one and each covered negative has
cost $c\ge0$. Let $t$ and $f$ be their counts. Define

$$q(G)=\frac{t-cf}{N}.$$

This differs from WRAcc: its trade-off is an explicit negative-row cost, not
the population target rate. It is useful when that cost interpretation matches
the application.

```{code-cell} python
import numpy as np
import pandas as pd
import pysubgroup as sd

class LinearCostQF(sd.QualityFunction):
    """Covered positive benefit minus negative cost, divided by dataset size."""

    target_type = sd.BinaryTarget

    def __init__(self, cost=1.0):
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("cost must be finite and nonnegative")
        self.cost = float(cost)

    def evaluate(self, coverage, target):
        mask = self._mask(coverage, target)
        size = np.count_nonzero(mask)
        if size == 0:
            return float("nan")
        positives = np.count_nonzero(mask & target.positives_)
        return float((positives - self.cost * (size - positives)) / target.n_samples_)

data = pd.DataFrame({"x": ["A", "A", "A", "B"], "label": [1, 1, 0, 0]})
target = sd.BinaryTarget().fit(data, "label", 1)
mask = np.array([True, True, True, False])
qf = LinearCostQF(cost=0.5)
print("Score:", qf.evaluate(mask, target))
assert qf.evaluate(mask, target) == (2 - 0.5) / 4
```

`target_type` declares compatibility. `_mask` checks that the target is fitted
and coverage has the correct Boolean shape. Keep configuration in the quality
function and per-row arrays in the target.

This class works with ordinary exhaustive searches, which enumerate without a
quality bound, and with heuristics. It can also be used as an additional measure
in result tables and as a statistic in randomization tests.

## Derive an optimistic estimate

For any refinement $H$ of $G$, $t_H\le t_G$ and $f_H\ge0$. Since $c\ge0$,

$$q(H)=\frac{t_H-cf_H}{N}\le\frac{t_G}{N}.$$

Thus $t_G/N$ is a valid bound, attained by the arbitrary subset containing only
the parent's positives when there are any. Such a subset need not be expressible
by a legal description, which is why the estimate remains optimistic.

```{code-cell} python
class BoundedLinearCostQF(LinearCostQF, sd.BoundedQualityFunction):
    """Linear cost with an upper bound obtained by discarding all negatives."""

    def optimistic_estimate(self, coverage, target):
        mask = self._mask(coverage, target)
        if not mask.any():
            return float("-inf")
        return float(np.count_nonzero(mask & target.positives_) / target.n_samples_)

bounded = BoundedLinearCostQF(0.5)
print("Bound:", bounded.optimistic_estimate(mask, target))
```

## Check the claim independently on every subset

Testing only the score of the parent cannot establish a refinement bound. For
this tiny example, enumerate every nonempty subset of its rows:

```{code-cell} python
from itertools import combinations

covered_rows = np.flatnonzero(mask)
scores = []
for size in range(1, len(covered_rows) + 1):
    for subset in combinations(covered_rows, size):
        child_mask = np.zeros(len(data), dtype=bool)
        child_mask[list(subset)] = True
        scores.append(bounded.evaluate(child_mask, target))
assert max(scores) <= bounded.optimistic_estimate(mask, target)
print("Best arbitrary subset:", max(scores))
```

For a production measure, include boundary cases, ties, missing-target policies,
all-negative parents, and both scalar and batched paths. A proof defines why
pruning is correct; tests catch implementation mistakes in that proof's use.

## Add batching from the same sufficient statistics

All scores here depend only on child sizes and positive counts. The batched
protocol supplies these without assembling every child mask in Python:

```{code-cell} python
class BatchedLinearCostQF(BoundedLinearCostQF):
    """The same linear cost and bound, evaluated from batched counts."""

    batchable = True

    def evaluate_many(self, statistics, target):
        self._check_target(target)
        sizes = statistics.sizes
        positives = statistics.count(target.positives_)
        scores = (positives - self.cost * (sizes - positives)) / target.n_samples_
        return np.where(sizes > 0, scores, np.nan)

    def optimistic_estimate_many(self, statistics, target):
        self._check_target(target)
        bounds = statistics.count(target.positives_) / target.n_samples_
        return np.where(statistics.sizes > 0, bounds, -np.inf)

conditions = sd.generate_conditions(data[["x"]])
task = sd.SubgroupDiscoveryTask(data, target, conditions, BatchedLinearCostQF(0.5),
    depth=1, result_set_size=2, batched=True)
results = sd.DFS().execute(task)
sd.to_dataframe(results, data, target, measures={"scalar_check": LinearCostQF(0.5)})
```

The corresponding statistics interface also provides sums and maxima of target
arrays. Batch methods should return one NumPy value per child and reproduce the
scalar definition, including undefined cases. A misleading `batchable=True`
declaration can break searches that automatically choose this path.

## Integrate with the rest of the workflow

```{code-cell} python
p_values = sd.randomization_p_values(results, data, target, LinearCostQF(0.5),
    n_samples=99, random_state=4, plus_one=True)
print("Illustrative resampling p-values:", p_values)
```

This last computation illustrates API compatibility on a tiny dataset. It is
not holdout validation: the same rows were used to discover these results.
Use the [separate-data procedure](14-validation.md) for a validation analysis.

**Experiment.** Add a reward for negatives instead of a penalty. Why does the
bound derived above fail when `cost` is negative? Derive a replacement before
allowing that parameter range.
