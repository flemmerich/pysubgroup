---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Specialized tasks
---

# Specialized tasks and connections

The same condition language and search interface can support problems beyond
“find an unusual label share.” Two examples are target-free pattern mining and
the search for a good rule in a boosting objective.

## Target-free pattern mining

With no designated outcome, `FITarget` stores only the dataset size.
`CountQF` ranks descriptions by the number of rows they cover:

```{code-cell} python
import numpy as np
import pandas as pd
import pysubgroup as sd

data = pd.DataFrame({"bread": [True, True, True, False, True, False],
    "milk": [True, True, False, True, True, False],
    "fruit": [False, True, True, True, False, True]})
conditions = [sd.Condition(column, "eq", True) for column in data.columns]
target = sd.FITarget().fit(data)
task = sd.SubgroupDiscoveryTask(data, target, conditions, sd.CountQF(),
    depth=3, result_set_size=6, filters=[sd.MinSizeFilter(2)])
results = sd.DFS().execute(task)
sd.to_dataframe(results, data, target)
```

Manual positive-item conditions avoid filling the result list with absence
patterns if only present items matter. `generate_conditions` would also offer
conditions for false values.

`AreaQF` multiplies coverage by description length:

```{code-cell} python
from dataclasses import replace

area_results = sd.DFS().execute(replace(task, qf=sd.AreaQF()))
sd.to_dataframe(area_results, data, target, ["size", "depth"])
```

It can rank longer, moderately frequent descriptions ahead of shorter ones.
It has no optimistic estimate in this interface and needs `depth=` for manual
evaluation. The searches and result table helper supply depth automatically.
The randomization significance helpers do not test `AreaQF`.

## Gradient targets for rule boosting

In second-order boosting, a row contributes a gradient $g_i$ and Hessian $h_i$
of the loss at the current prediction. A candidate subgroup defines a rule
whose optimal weight is

$$w_G=-\frac{\sum_{i\in G}g_i}{\lambda+\sum_{i\in G}h_i}.$$

The improvement of the quadratic objective, in the package's normalization, is

$$q(G)=\frac{\left(\sum_{i\in G}g_i\right)^2}
 {2N\left(\lambda+\sum_{i\in G}h_i\right)}.$$

`GradientBoostingQF` implements this objective with the tight subset bound of
{cite:t}`boley2021`. It finds a rule useful for the **current boosting step**,
not necessarily a subgroup with a high raw target mean.

For logistic loss at initial predicted probability one half,
$g_i=0.5-y_i$ and $h_i=0.25$:

```{code-cell} python
from sklearn.datasets import load_iris

X, labels = load_iris(return_X_y=True, as_frame=True)
y = (labels == 2).to_numpy(dtype=float)
boost_data = X.assign(gradient=0.5 - y, hessian=np.full(len(y), 0.25))
gradient_target = sd.GradientTarget().fit(boost_data, "gradient", "hessian")
gradient_conditions = sd.generate_conditions(X, n_bins=4)
gradient_qf = sd.GradientBoostingQF(reg_lambda=1)
gradient_task = sd.SubgroupDiscoveryTask(boost_data, gradient_target,
    gradient_conditions, gradient_qf, depth=2, result_set_size=4,
    filters=[sd.MinSizeFilter(10)])
gradient_results = sd.DFS().execute(gradient_task)
rows = []
for quality, rule in gradient_results:
    mask = sd.utils.subgroup_masks([rule], boost_data)[0]
    G = gradient_target.gradients_[mask].sum()
    H = gradient_target.hessians_[mask].sum()
    rows.append({"rule": str(rule), "objective_improvement": quality,
                 "optimal_weight": -G / (1 + H), "size": mask.sum()})
pd.DataFrame(rows)
```

Both positive and negative weights can improve the objective. Gradients and
Hessians must be finite, Hessians nonnegative, and regularization nonnegative.
The bound uses rows ordered by gradient/Hessian ratio, so this measure stays
scalar rather than using the standard count/sum batching interface.

A complete learner must update predictions, recompute derivatives, and repeat
the search. This example performs one rule search only.

## Discovered rules and predictive rules

A discovered `Rule` has no consequent. A learned predictive rule can have a
consequent, and a predictive ruleset may use ordering or a default. Do not
interpret an overlapping subgroup list as a ready-made classifier without a
separately specified prediction procedure.

**Experiment.** Increase `reg_lambda` while keeping derivatives fixed. How do
optimal weights and the preferred rule change? Relate the result to the
denominator of the objective, rather than to an arbitrary score rescaling.
