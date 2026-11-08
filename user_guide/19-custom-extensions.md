---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Custom targets and models
---

# Custom targets, filters, and model classes

The fitted target stores the row information needed by a measure. Filters
control eligibility. A likelihood model encapsulates a fitting procedure. Keep
these roles separate so that a new component can reuse the rest of the workflow.

## A weighted numeric target

As an example, suppose rows have nonnegative importance weights and we want a
weighted mean shift. This is a custom statistic; the built-in numeric measures
do not silently apply a weight column.

```{code-cell} python
import numpy as np
import pandas as pd
import pysubgroup as sd

class WeightedMeanTarget(sd.BaseTarget):
    """Finite numeric values and nonnegative weights, addressed by row position."""

    def fit(self, data, value_attribute, weight_attribute):
        super().fit(data, value_attribute, weight_attribute)
        self.values_ = data[value_attribute].to_numpy(dtype=float)
        self.weights_ = data[weight_attribute].to_numpy(dtype=float)
        if not np.isfinite(self.values_).all() or not np.isfinite(self.weights_).all():
            raise ValueError("values and weights must be finite")
        if (self.weights_ < 0).any() or self.weights_.sum() <= 0:
            raise ValueError("weights must be nonnegative with positive total")
        self.mean_ = float(np.average(self.values_, weights=self.weights_))
        return self

class WeightedMeanDifferenceQF(sd.QualityFunction):
    """Weighted subgroup mean minus weighted dataset mean; no quality bound."""

    target_type = WeightedMeanTarget

    def evaluate(self, coverage, target):
        mask = self._mask(coverage, target)
        if target.weights_[mask].sum() <= 0:
            return float("nan")
        return float(np.average(target.values_[mask], weights=target.weights_[mask]) - target.mean_)

data = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"],
    "value": [2.0, 4.0, 5.0, 1.0, 2.0, 3.0], "weight": [1., 2., 1., 1., 1., 2.]})
target = WeightedMeanTarget().fit(data, "value", "weight")
conditions = sd.generate_conditions(data[["group"]])
qf = WeightedMeanDifferenceQF()
task = sd.SubgroupDiscoveryTask(data, target, conditions, qf, depth=1, result_set_size=2)
results = sd.DFS().execute(task)
sd.to_dataframe(results, data, target, measures={"weighted_difference": qf})
```

`BaseTarget.fit` records `attributes_` and the number of rows. Both value and
weight columns are thereby excluded from descriptions. Inherited `refit`
constructs the same class and calls `fit` with these attributes. A configurable
constructor or additional non-attribute fitting arguments require a corresponding
refit implementation or `_fit_arguments` specification.

```{code-cell} python
refitted = target.refit(data.iloc[:4])
print("Original reference mean:", target.mean_)
print("Refitted reference mean:", refitted.mean_)
```

Targets must be refitted whenever the data changes, including row order changes.
Equal lengths alone do not make targets interchangeable. A custom target gets
the common table statistics (`size`, `relative_size`, `depth`); the built-in
table helper does not infer custom model parameters automatically.

## Custom constraints

A result may be required to mention a feature or to satisfy an application
condition. Consider a filter requiring the descriptor `group`:

```{code-cell} python
requires_group = sd.CustomFilter(
    lambda mask, target, conditions: any(c.feature == "group" for c in conditions),
    anti_monotone=False,
)
print(requires_group.is_satisfied(np.ones(len(data), dtype=bool), target, ()))
print(requires_group.is_satisfied(np.ones(len(data), dtype=bool), target,
                                  (sd.Condition("group", "eq", "A"),)))
```

The empty description fails, but adding `group=A` can pass. Therefore this is a
result-only constraint. By contrast, “at least ten covered rows” is safe for
pruning. A custom class can subclass `Filter` and provide `is_satisfied_many`
if it can decide from batch statistics; the simple callable wrapper is scalar.

## A pluggable likelihood model

To illustrate the model protocol, fit a univariate Gaussian with **known unit
variance** and an estimated mean. Only one free parameter is fitted. This
implements the normal log-density directly; it is intentionally simpler than
the built-in polynomial regression model.

```{code-cell} python
class UnitVarianceNormal(sd.LikelihoodModel):
    """Normal location family with known variance one and one fitted mean."""

    def prepare(self, data, attributes):
        if len(attributes) != 1:
            raise ValueError("exactly one model attribute is required")
        self.values_ = data[attributes[0]].to_numpy(dtype=float)
        if not np.isfinite(self.values_).all():
            raise ValueError("model values must be finite")

    @property
    def n_parameters(self):
        return 1

    def fit(self, mask):
        return float(self.values_[mask].mean()) if np.any(mask) else None

    def log_likelihood(self, parameters, mask=None):
        values = self.values_ if mask is None else self.values_[mask]
        return -0.5 * (np.log(2 * np.pi) + (values - parameters) ** 2)

    def describe(self, parameters):
        return {"mean": float("nan") if parameters is None else parameters}

likelihood_target = sd.LikelihoodTarget(UnitVarianceNormal()).fit(data, "value")
likelihood_task = sd.SubgroupDiscoveryTask(data, likelihood_target, conditions,
    sd.LikelihoodQF(score="statistic"), depth=1, result_set_size=2)
likelihood_results = sd.DFS().execute(likelihood_task)
sd.to_dataframe(likelihood_results, data, likelihood_target)
```

The target prepares a copy of the model on its data. `fit(mask)` must return
`None` for an undefined fit; zero is a legitimate parameter value. Return one
log-likelihood per requested row, not the sum. The likelihood quality function
performs the required aggregation and compares fits. `describe` supplies names
for parameter columns in result tables.

A likelihood-ratio interpretation additionally needs the chosen model and its
regularity assumptions to make sense. The number of free parameters is a
property of the fitted family, not merely the length of an arbitrary return
object.

## Optional: a valuation domain for GP-growth

If a model can be recovered from condensed statistics, a valuation domain can
make those statistics available to GP-growth {cite:p}`lemmerich2012`. A numeric
mean needs only count and sum:

```{code-cell} python
class MeanDomain(sd.ValuationDomain):
    """Additive (count, sum) bases with a mean model projector."""

    def project(self, target):
        return np.column_stack([np.ones(target.n_samples_), target.values_])

    def model(self, bases, target):
        with np.errstate(divide="ignore", invalid="ignore"):
            return bases[..., 1] / bases[..., 0]

numeric_target = sd.NumericTarget().fit(data, "value")
valuation_qf = sd.ValuationQF(MeanDomain(), reference="complement")
valuation_task = sd.SubgroupDiscoveryTask(data, numeric_target, conditions, valuation_qf,
    depth=1, result_set_size=2)
sd.to_dataframe(sd.GPGrowth().execute(valuation_task), data, numeric_target)
```

The default domain aggregation and subtraction are vector addition and
subtraction. Its default distance is absolute model difference, so this example
scores subgroup-versus-complement mean distance without a size factor. An
associative aggregation operation and a correct model projector are essential;
arbitrary non-associative state cannot be condensed in the same way.

**Experiment.** Extend the mean domain with a sum-of-squares component and
derive a variance projector. Consider numerical cancellation before using the
naive formula on values with a very large offset.
