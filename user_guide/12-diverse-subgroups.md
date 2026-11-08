---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Diverse subgroup sets
---

# From a top-k list to a diverse subgroup set

Top-k search optimizes individual scores. If one relationship is especially
strong, many small variations can fill the list. A diverse result set aims to
communicate several themes or to cover different rows. Diversity must therefore
be defined as carefully as individual quality {cite:p}`vanleeuwen2012`.

## Post-process a larger candidate list

```{code-cell} python
from dataclasses import replace
from sklearn.datasets import load_iris
import pandas as pd
import pysubgroup as sd

X, y = load_iris(return_X_y=True, as_frame=True)
data = X.assign(species=y)
target = sd.BinaryTarget().fit(data, "species", 2)
conditions = sd.generate_conditions(X, n_bins=4, numeric_candidates="thresholds")
task = sd.SubgroupDiscoveryTask(data, target, conditions, sd.WRAccQF(),
    depth=2, result_set_size=30, min_quality=0, filters=[sd.MinSizeFilter(15)])
candidates = sd.DFS().execute(task)
selected = sd.overlap_filter(candidates, data, max_similarity=0.6)[:5]
sd.to_dataframe(selected, data, target)
```

`overlap_filter` visits candidates in their given order and accepts one only if
its similarity to every previously accepted group is at most the threshold.
The default similarity is Jaccard. `similarity="overlap"` uses intersection
divided by the smaller size, and equals one whenever one cover contains the other.

Two other helpers offer different selection criteria:

```{code-cell} python
different_attributes = sd.unique_attributes(candidates)[:5]
large_enriched = sd.statistic_filter(candidates, data, target,
    minimum={"size": 25, "lift": 1.5})[:5]
print("Different feature sets:", len(different_attributes))
print("Count and lift filter:", len(large_enriched))
```

`unique_attributes` keeps the first description per set of descriptor features,
regardless of their values. `statistic_filter` uses inclusive bounds on reported
statistics. None of these operations searches for replacements outside the
candidate list. Start with a larger pool if the filtered list is too short.

## Diversity during search: DSSD

DSSD uses diverse beam selection, pools promising descriptions across depths,
removes redundant conditions through dominance pruning, and selects a final
diverse set. Since its search is heuristic, diversity can change which branches
are explored, not merely the final presentation.

```{code-cell} python
dssd_task = replace(task, result_set_size=5)
diverse_results = sd.DSSD(selection="cover", beam_width=20, pool_size=150,
                           n_bins=None).execute(dssd_task)
sd.to_dataframe(diverse_results, data, target)
```

`n_bins=None` keeps the supplied numeric conditions. By default, DSSD computes
equal-frequency cuts inside the subgroup being refined, with `n_bins=6`.
Its operator also allows multiple negated conditions, or multiple threshold
conditions, on one feature. Consequently, it can explore descriptions outside
the standard searches' one-condition-per-feature language.

## Selection strategies

| `selection` | Diversity idea |
| --- | --- |
| `"top_k"` | Ordinary beam ranking by quality |
| `"description"` | Avoid almost-identical descriptions with exactly equal quality |
| `"var_description"` | Limit repeated attribute use |
| `"cover"` | Discount rows already covered by selected groups |
| `"var_cover"` | Cover-based selection with a stopping fraction |
| `"compression"` | Select groups that improve a discrete distribution code |
| `"var_compression"` | Stop when no candidate shortens the encoding |

Cover-based selection multiplies quality by an average row discount. With
`alpha=0.9`, a row already covered $c$ times receives factor $0.9^c$. These
strategies select only positive-quality groups, so results or beams can be
smaller than the requested size. Variable-size strategies are also capped by
the final `result_set_size`.

Compression strategies require binary, multiclass, or nominal model attributes.
Their connection to [WKL](08-multiple-targets-and-wkl.md) is distributional:
describing rows with a subgroup-specific code can save bits relative to the
global code. The last selected subgroup covering a row determines its code in
the package's compression selection. WKG/Krimp is not implemented.

## Measure the set, not just its best member

```{code-cell} python
rows = []
for name, found in [("top-k", candidates[:5]), ("post-filtered", selected), ("DSSD cover", diverse_results)]:
    covers = sd.subgroup_covers(found, data)
    rows.append({"selection": name, "subgroups": len(found),
        "cover_redundancy": sd.cover_redundancy(covers),
        "joint_entropy_bits": sd.joint_entropy(covers),
        "distinct_rows": covers.any(axis=0).sum()})
pd.DataFrame(rows)
```

Cover redundancy summarizes uneven/repeated coverage, with lower values
indicating less redundancy under its definition. Joint entropy describes the
diversity of membership combinations, in bits. Neither replaces the original
quality or a domain assessment. Compare sets with their number of subgroups,
individual effects, and distinct coverage visible.

**Experiment.** Change the overlap cutoff from 0.9 to 0.3 and compare the number
of retained groups. Can a stricter diversity requirement reduce the best
remaining overall coverage even when it reduces repeated themes?
