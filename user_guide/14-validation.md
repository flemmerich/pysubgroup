---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Validation after search
---

# Validation after searching many subgroups

A discovery algorithm deliberately selects descriptions with high observed
quality. Even noise can contain an extreme-looking winner if enough
descriptions are tried. A holdout procedure separates **choosing descriptions**
from **testing their statistics** {cite:p}`webb2007,siegl2025`.

## First split, then choose numeric cuts

For a strict discovery/validation separation, generate data-dependent candidate
thresholds on discovery rows only. The validation rows should not help select
cuts, descriptions, weights, or reporting thresholds.

```{code-cell} python
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import pysubgroup as sd

X, y = load_iris(return_X_y=True, as_frame=True)
data = X.assign(species=y)
discovery, validation = sd.holdout_split(data, validation_size=0.5,
                                        stratify="species", random_state=21)
conditions = sd.generate_conditions(discovery.drop(columns="species"), n_bins=3)
search_target = sd.BinaryTarget().fit(discovery, "species", 2)
qf = sd.WRAccQF()
task = sd.SubgroupDiscoveryTask(discovery, search_target, conditions, qf,
    depth=2, result_set_size=5, min_quality=0, filters=[sd.MinSizeFilter(8)])
candidates = sd.DFS().execute(task)
validation_target = search_target.refit(validation)
print("Discovery rows:", len(discovery), "validation rows:", len(validation))
```

`refit` creates a new target with the same fitting specification. Reusing the
discovery target on validation rows would mix up positional arrays and
reference statistics.

## What the randomization test asks

For each candidate, the test finds its cover on the validation data. It then
draws random subsets **without replacement within each draw**, all with the
same size as that validation cover. The statistic is evaluated on these subsets
using the validation population as reference.

The null model asks whether the observed rows have an unusually high statistic
compared with random sets of the same size. This is not a new search on shuffled
data and not a test of whether every individual condition is independently useful.

```{code-cell} python
p_values = sd.randomization_p_values(candidates, validation, validation_target, qf,
    n_samples=999, random_state=3, plus_one=True)
adjusted = sd.adjust_p_values(p_values, method="by")
validation_table = sd.to_dataframe([rule for _, rule in candidates], validation,
    validation_target, ["size", "target_share"], measures={"validation_WRAcc": qf})
validation_table.assign(p_value=p_values, adjusted_p_value=adjusted,
                        significant=adjusted <= 0.05)
```

The example explicitly chooses `plus_one=True`. The package default follows
SubROC's $b/R$: if $b$ of $R$ subsets score at least as high, $p=b/R$.
That can be zero. The alternative $(b+1)/(R+1)$ prevents zero Monte Carlo
p-values {cite:p}`phipson2010`. Its minimum is $1/(R+1)$, so the number of
draws affects the resolution available after multiple-testing correction.

Keep the test setup fixed before interpreting validation outcomes. Seeded
randomization makes the example reproducible; it does not remove Monte Carlo
uncertainty.

## Multiple testing

`adjust_p_values` supports:

| Method | Intended control |
| --- | --- |
| `"by"` (default) | Benjamini–Yekutieli false discovery rate under arbitrary dependency |
| `"bh"` | Benjamini–Hochberg false discovery rate under its dependency assumptions |
| `"bonferroni"` | Family-wise error through Bonferroni adjustment |
| `"holm"` | Step-down family-wise error adjustment |
| `"none"` | No multiple-testing correction |

Subgroup tests overlap and are dependent. BY is the package's default for that
reason {cite:p}`benjamini2001`. Correction applies to the fixed candidate family
tested on the holdout data. Repeatedly revising the search after inspecting
validation results changes the statistical procedure.

## Stratification changes the null model

`stratify_target=True` also preserves the subgroup's class counts. This is useful
for comparing classifier performance at fixed class balance. For WRAcc, however,
size and class counts completely determine the score: all randomized scores
then equal the observed score and the p-value is one.

```{code-cell} python
fixed_balance_p = sd.randomization_p_values(candidates, validation, validation_target, qf,
    n_samples=49, stratify_target=True, random_state=3)
print("WRAcc with fixed size and class counts:", fixed_balance_p)
```

The default preserves only size, deliberately differing from SubROC's
class-balanced null. `stratify=` can preserve counts across user-selected
columns; with target stratification, these form combined strata. More constraints
mean a different scientific question, not automatically a better test.

Undefined validation statistics receive p-value one. For example, ROC AUC is
undefined if a validation subgroup contains only one class. A group that was
well supported in discovery can still be too small in validation.

## The combined helper

`holdout_search` performs splitting, target refitting, search, testing,
correction, and selection in one call. To keep thresholds independent of the
validation rows here, use externally specified cuts rather than cuts fitted on
the whole input dataset:

```{code-cell} python
fixed_conditions = [sd.Condition("petal length (cm)", "gt", cut) for cut in [2, 4, 5]]
fixed_conditions += [sd.Condition("petal width (cm)", "gt", cut) for cut in [0.5, 1, 1.5]]
whole_target = sd.BinaryTarget().fit(data, "species", 2)
whole_task = sd.SubgroupDiscoveryTask(data, whole_target, fixed_conditions, qf,
    depth=2, result_set_size=5, min_quality=0, filters=[sd.MinSizeFilter(8)])
heldout = sd.holdout_search(sd.DFS(), whole_task, k=3, validation_size=0.5,
    split_stratify="species", n_samples=999, plus_one=True, random_state=21)
heldout.to_dataframe(["size", "target_share"], candidates=True)
```

The helper keeps the provided search space; it does **not** regenerate cuts after
splitting. Its table combines discovery statistics, `validation ...` columns,
p-values, and the significance decision. `candidates=True` shows every tested
candidate; the default table contains only the final accepted results.

For a generalization-aware measure, the tested statistic is `qf.value`, since
the reference used during discovery depends on the searched description lattice.
`AreaQF` is not supported by these significance tests.

**Experiment.** Compare unstratified and class-stratified randomization for a
ROC AUC measure. Explain which variation each null model allows and why this
distinction differs from the WRAcc example above.
