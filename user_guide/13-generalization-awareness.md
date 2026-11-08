---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Generalization awareness
---

# Does another condition really add information?

Suppose `plan=monthly AND channel=web` has a high cancellation rate, but
`plan=monthly` already has the same rate. Is the extra condition informative?
Generalization-aware quality compares a description with **all descriptions
obtained by removing conditions**, including the empty description
{cite:p}`lemmerich2014`.

## A transparent redundancy example

```{code-cell} python
import pandas as pd
import pysubgroup as sd

data = pd.DataFrame({
    "plan": ["monthly"] * 8 + ["annual"] * 8,
    "channel": ["web", "store"] * 8,
    "cancelled": [1, 1, 1, 1, 1, 1, 0, 0] + [0] * 8,
})
target = sd.BinaryTarget().fit(data, "cancelled", 1)
parent = sd.Rule((sd.Condition("plan", "eq", "monthly"),))
child = sd.Rule((*parent.conditions, sd.Condition("channel", "eq", "web")))
ga = sd.GeneralizationAwareStandardQF(a=0.5)
sd.to_dataframe([parent, child], data, target, ["size", "target_share"],
    measures={"ordinary": sd.StandardQF(0.5), "generalization_aware": ga})
```

Both monthly descriptions have target share $6/8=3/4$ or $3/4$, respectively.
The more specific one is enriched relative to the full dataset but adds no
rate improvement over the monthly parent. Its generalization-aware rate score
is zero. A useful table can therefore distinguish “associated with the target”
from “adds an improvement over simpler explanations.”

## Two forms of generalization awareness

For a subgroup $G$, let $H\prec G$ denote descriptions obtained by removing at
least one condition. The empty description is included.

The generic wrapper subtracts the best **quality** of a generalization:

$$q_{GA}(G)=q(G)-\max_{H\prec G}q(H).$$

The binary statistic-based measure instead subtracts the best **target share**
inside its size weighting:

$$r_a(G)=(n_G/N)^a\left(p_G-\max_{H\prec G}p_H\right).$$

These are different measures. Likewise, the numeric version compares subgroup
mean with the largest generalization mean and uses $n_G^a$. The specialized
binary and numeric measures search increases; there is no direction argument.

Generalizations here only remove conditions. Replacing `age>40` with `age>30`
does not count as a generalization for this API, even if it covers more rows.
Thus the score depends on the **description**, not just the covered rows.

## Use a search that supplies the reference

```{code-cell} python
conditions = sd.generate_conditions(data[["plan", "channel"]])
task = sd.SubgroupDiscoveryTask(data, target, conditions, ga,
    depth=2, result_set_size=6, min_quality=0,
    filters=[sd.MinSizeFilter(2)])
results = sd.GeneralizationAwareDFS().execute(task)
sd.to_dataframe(results, data, target,
    measures={"ordinary": sd.StandardQF(0.5), "GA": ga})
```

`GeneralizationAwareApriori` propagates references level by level.
`GeneralizationAwareDFS` visits the tree in reverse sibling order so that the
necessary generalizations are processed first. Standard DFS and other ordinary
searches do not provide these references and reject these measures.

For one manually specified description, the helper enumerates its generalizations:

```{code-cell} python
mask = sd.utils.subgroup_masks([child], data)[0]
reference = sd.generalization_reference(ga, child, data, target)
print("Largest generalization share:", reference)
print("Child score:", ga.evaluate(mask, target, reference=reference))
```

There are $2^d-1$ proper generalizations for a description with $d$ conditions.
The searches maintain references incrementally rather than independently
enumerating every subset for every evaluated candidate.

## Generalization awareness, diversity, and validation

These address separate questions:

- Generalization awareness: does this condition combination improve on simpler
  descriptions?
- Diversity: does this result repeat another selected result's rows or theme?
- Validation: is the observed statistic unusual on separate data under a
  specified null model?

For classifier auditing, `GeneralizationAwareQF(ROCAUCQF(...))` subtracts the
best generalization's weighted quality. When testing such results with the
significance helpers, the tested statistic is the underlying `qf.value`, not
the search-dependent generalization-aware subtraction.

The searches offer `difference_based="auto"`, `"on"`, or `"off"`. The auto
setting uses difference-based bounds with negated conditions or overlapping
numeric intervals. This is an optimization choice; it should not redefine the
measure or change the attainable quality values.

**Experiment.** Change only the outcomes for monthly web customers. At what
point does the two-condition description gain a positive generalization-aware
score? Remember to compare it with both one-condition generalizations.
