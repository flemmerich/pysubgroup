---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Practical performance
---

# Making searches practical

The largest speed improvement often comes from a better-specified task.
Doubling the number of numeric cut points or increasing depth can create more
work than an accelerator saves. Optimize the search problem before the backend.

## Control the combinatorics

If feature $j$ contributes $c_j$ conditions, the number of standard-search
descriptions at depth $d$ is bounded by the sum of products of $c_j$ over feature
sets of sizes one through $d$. This grows rapidly with both depth and candidate
count. Constraints and quality bounds may reduce the explored fraction, but
they do not remove the need to choose a meaningful language.

Start with relevant descriptors, moderate discretization, a shallow depth, and
a useful minimum size. Then examine whether more detail changes the findings.

## Measure on the actual task

This small example compares scalar and batched evaluation. The printed times
are measurements on the current machine, not portable benchmark claims.

```{code-cell} python
from dataclasses import replace
from time import perf_counter
from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import pysubgroup as sd

X, y = load_diabetes(return_X_y=True, as_frame=True)
X = X[["age", "sex", "bmi", "bp"]]
data = X.assign(response=y)
target = sd.NumericTarget().fit(data, "response")
conditions = sd.generate_conditions(X, n_bins=4, numeric_candidates="thresholds")
task = sd.SubgroupDiscoveryTask(data, target, conditions,
    sd.StandardQFNumeric(0.5, estimator="combined"), depth=2,
    result_set_size=5, filters=[sd.MinSizeFilter(30)])
sd.DFS().execute(task)  # warm up optional compiled kernels
rows, qualities = [], []
for batched in [False, True]:
    start = perf_counter()
    found = sd.DFS().execute(replace(task, batched=batched))
    rows.append({"batched": batched, "seconds": perf_counter() - start})
    qualities.append([q for q, _ in found])
np.testing.assert_allclose(qualities[0], qualities[1])
pd.DataFrame(rows)
```

The assertion checks the scientific result as well as the timing. Floating-point
sum order can cause small rounding differences; exact tied descriptions may
differ at a cutoff.

## CPU batching, numba, and GPU execution

| Option | Meaning |
| --- | --- |
| `batched=None` | Use batching when algorithm, measure, and pruning filters support it |
| `batched=True` | Require batching; raise an error if unsupported |
| `batched=False` | Evaluate one subgroup at a time |
| `device="cpu"` | Default CPU execution |
| `device="cuda"` | Use CuPy on a compatible CUDA GPU; requires batching |

DFS, best-first, OPUS, DpSubgroup, Apriori, and beam search support standard
batched child evaluation. Generalization-aware DFS supports compatible measures.
SD-Map, VLSD, NumBSD, and GP-growth use their own scalar/tree/ordered machinery
and do not support `batched=True` or CUDA through that interface.

Binary and multiclass measures are generally batchable. Mean-based numeric
measures with simple summary bounds are batchable; median/order estimates, KS,
and SubROC measures are not. Moment-based EMM measures can batch, while logistic
and likelihood fits remain scalar. A custom nonbatchable pruning filter can
disable batching even if the measure supports it.

Optional installation commands, to run in your chosen environment:

```console
pip install "pysubgroup[numba]"
pip install "pysubgroup[gpu]"
```

The GPU extra targets the supported CUDA/CuPy build; a GPU does not accelerate
unsupported measures. Device transfer and startup can dominate small tasks.
Numba similarly has compilation overhead. The examples work on the NumPy path;
`PYSUBGROUP_DISABLE_NUMBA=1` explicitly selects it.

## Count evaluations, not just seconds

```{code-cell} python
counted = sd.count_calls(sd.StandardQFNumeric(0.5, estimator="combined"))
sd.OPUS().execute(replace(task, qf=counted))
print("Method calls:", counted.call_counts.calls)
print("Subgroups scored:", counted.call_counts.subgroups)
```

A batched call can score many subgroups, so call counts and evaluated-subgroup
counts are different quantities. Compare both when diagnosing pruning or batching.
`count_calls` returns a counting copy of the quality function.

## Coverage storage

```{code-cell} python
storage = []
for representation in ["dense", "packed"]:
    coverage = sd.Encoder(conditions, representation=representation).fit_transform(X)
    storage.append({"representation": representation, "bytes": coverage.storage.nbytes})
pd.DataFrame(storage)
```

Packed storage is roughly eight times smaller for raw Boolean coverage, aside
from padding. The encoder's representation is a separate interface from a
task's device and batching choices; constructing a packed `Coverage` object does
not configure a discovery task to use it.

## Diagnose the bottleneck

Many candidates with little pruning suggest revisiting bounds, depth, or
constraints. Large level-wise frontiers suggest a memory problem rather than
slow arithmetic. Model fits per candidate suggest a fitting bottleneck. A
tighter numeric bound can reduce evaluations but cost more per evaluation.
Choose optimizations using measured task behavior, not an algorithm's name.

**Experiment.** Compare a two-level `Apriori` search with `DFS` and `OPUS` on
this small task. Then change only the numeric estimator. Separate changes in
evaluation count from changes in time per evaluation.
