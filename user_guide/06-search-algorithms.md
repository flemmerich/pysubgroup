---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Search algorithms
---

# How search works, and which algorithm to choose

With many descriptor attributes, enumerating all conjunctions becomes expensive.
Search algorithms organize this enumeration and try to exclude branches that
cannot improve the result. Exhaustive and heuristic methods differ in what
they guarantee about unvisited descriptions.

## Exhaustive versus heuristic search

**Exhaustive algorithms guarantee the optimal top-k quality values** within the
configured search space, depth, constraints, and quality function. They need not
evaluate every description: safe pruning can rule out branches without changing
the answer.

**For the same task, all compatible exhaustive algorithms produce the same
optimal quality values; they differ in runtime, not in solution quality.**
Choosing between them is therefore a computational decision. Their exploration
order, pruning effectiveness, and evaluation machinery affect runtime; memory
requirements can differ too. Equal-quality descriptions at the cutoff may differ
because a branch whose bound equals the current threshold can be pruned.

**Heuristic algorithms do not guarantee the optimal result.** They may discard
a branch that contains a better subgroup. Their search settings can therefore
affect both runtime and the quality of the returned results. In this package,
`BeamSearch` and `DSSD` are heuristic; all the other searches listed below are
exhaustive in their supported settings.

The comparison always assumes the **same task and description language**. A
numeric measure, a generalization-aware measure, and a diversity objective ask
different questions; matching guarantees do not make their scores comparable.

## A search tree

Starting at the empty description, a search adds a condition at each step:

```text
whole dataset
├── plan=monthly
│   ├── plan=monthly AND contacts>2
│   └── plan=monthly AND tenure≤12
├── contacts>2
│   └── contacts>2 AND tenure≤12
└── tenure≤12
```

Each conjunction is generated once under a canonical condition ordering.
Adding a condition can only remove rows, although it may increase the quality.
Pruning by the current quality alone is therefore generally unsafe.

## Optimistic estimates

An optimistic estimate $oe(G)$ bounds the quality of every permitted refinement
$H$ below $G$:

$$q(H)\le oe(G).$$

It may consider row subsets that no legal description can express. This makes
the estimate optimistic: it solves an easier problem than the actual search.

For positive-direction WRAcc, if $G$ contains $t>0$ positives, the best arbitrary
subset retains those positives and discards every negative. Its bound is

$$oe(G)=\frac{t}{N}(1-p).$$

For example, take $N=100$ and $p=0.2$. A branch with only two remaining positives
has bound $0.016$. Once the current top-k threshold is $0.03$, no refinement of
that branch can enter the results, even if it becomes perfectly pure.

```{code-cell} python
import numpy as np
import pandas as pd
import pysubgroup as sd

data = pd.DataFrame({"positive": [1] * 20 + [0] * 80})
target = sd.BinaryTarget().fit(data, "positive", 1)
mask = np.zeros(100, dtype=bool)
mask[[0, 1, 20, 21, 22]] = True
qf = sd.WRAccQF()
print("Parent quality:", qf.evaluate(mask, target))
print("Refinement bound:", qf.optimistic_estimate(mask, target))
```

Bounds can be looser than the best legal refinement. A tighter bound may prune
more, but also cost more to compute. This trade-off is particularly important
for [numeric targets](07-numeric-targets.md).

## General-purpose algorithms

| Algorithm | Guarantee | Exploration idea | Scientific source | Practical consideration |
| --- | --- | --- | --- | --- |
| `DFS` | **Exhaustive** | Depth-first with bounds | Lemmerich (2014), Chapter 3: subgroup-search strategy reference {cite:p}`lemmerich2014` | Straightforward exhaustive baseline |
| `BestFirstSearch` | **Exhaustive** | Expand the best estimated open node | Lemmerich (2014), Chapter 3: subgroup-search strategy reference {cite:p}`lemmerich2014` | Can find strong candidates early; stores an open frontier |
| `OPUS` | **Exhaustive** | Reorder active operators and remove pruned operators from siblings | Webb (1995), OPUS_o, Figure 10 {cite:p}`webb1995` | Branch-local operator management |
| `DpSubgroup` | **Exhaustive** | Score children before descending; choose exploration ordering | Grosskreutz et al. (2008), Algorithm 1 and Section 3 {cite:p}`grosskreutz2008` | Look-ahead and ordering influence when the threshold rises |
| `Apriori` | **Exhaustive** | Level-wise generation using promising sub-conjunctions | Agrawal and Srikant (1994), original Apriori algorithm {cite:p}`agrawal1994` | Broad levels can consume substantial memory |
| `SDMap` | **Exhaustive** | FP-growth conditional trees | Atzmueller and Puppe (2006), *SD-Map—A Fast Algorithm for Exhaustive Subgroup Discovery* {cite:p}`atzmueller2006` | Useful alternative organization of the same fixed candidates |
| `SDMapStar` | **Exhaustive** | FP-growth with bounds, header pruning, and estimate ordering | Atzmueller and Lemmerich (2009); Lemmerich et al. (2016), Section 5.3.1 {cite:p}`atzmueller2009,lemmerich2016` | Can reduce conditional-tree work |
| `VLSD` | **Exhaustive** | Vertical lists and equivalence classes | Lopez-Martinez-Carrasco et al. (2023), Algorithms 1–3 {cite:p}`lopez2023` | Scalar CPU evaluation |
| `BeamSearch` | **Heuristic** | Retain only a beam of promising descriptions per level | Duivesteijn et al. (2016), Algorithm 1: a subgroup/EMM beam-search reference {cite:p}`duivesteijn2016` | Can discard a branch containing a later optimum |

DFS, best-first search, and beam search are general search strategies with no
single subgroup-discovery invention paper. Their entries cite scientific
treatments of the strategy in this setting, rather than claim those works
invented the general method. The package's baseline DFS, best-first, Apriori,
and beam-search implementations were adapted from pysubgroup 0.x; its scientific
package reference is Lemmerich and Becker (2018) {cite:p}`lemmerich2018`.

The original Apriori paper concerns frequent itemsets. Here its level-wise
candidate-generation principle is adapted to subgroup discovery, using
constraints and optimistic quality bounds. Likewise, the beam-search reference
illustrates the strategy; this package's general `BeamSearch` uses the task's
fixed candidate conditions, rather than the reference's EMM-specific numeric
refinement behavior.

## Algorithms for specific settings

Specialized search does not mean approximate search. NumBSD, GP-growth, and the
generalization-aware searches preserve exhaustive guarantees for their supported
tasks:

| Algorithm | Guarantee | Setting and exploration idea | Scientific source | Further reading |
| --- | --- | --- | --- | --- |
| `NumBSD` | **Exhaustive** | Numeric targets with supported measures; look-ahead DFS over target-ordered bitsets, using ordering-based bounds | Lemmerich, Atzmueller, and Puppe (2016), Algorithm 1 {cite:p}`lemmerich2016` | [Numeric targets](07-numeric-targets.md) |
| `GPGrowth` (GP-growth) | **Exhaustive** | Pattern-tree search with aggregated valuation bases, particularly useful for EMM; uses condensed statistics when possible and row sets otherwise | Lemmerich, Becker, and Atzmueller (2012), *Generic Pattern Trees for Exhaustive Exceptional Model Mining* {cite:p}`lemmerich2012` | [Exceptional Model Mining](09-exceptional-model-mining.md) |
| `GeneralizationAwareApriori` | **Exhaustive** | Generalization-aware measures; level-wise search propagating the largest generalization value | Lemmerich (2014), Section 6.3 {cite:p}`lemmerich2014` | [Generalization awareness](13-generalization-awareness.md) |
| `GeneralizationAwareDFS` | **Exhaustive** | Generalization-aware measures; reverse DFS processes the necessary generalizations first and retains their reference values | Lemmerich (2014), Algorithm 4, Section 3.2.1, applied to the measures of Chapter 6 {cite:p}`lemmerich2014` | [Generalization awareness](13-generalization-awareness.md) |
| `DSSD` | **Heuristic** | Diverse subgroup sets; diverse beam selection, a candidate pool, dominance pruning, and final set selection | van Leeuwen and Knobbe (2012), *Diverse Subgroup Set Discovery*, Algorithm 1 {cite:p}`vanleeuwen2012` | [Diverse subgroup sets](12-diverse-subgroups.md) |

GP-growth is not restricted to EMM: its automatic valuation selection also works
with other compatible measures. It changes how subgroup statistics are stored
and aggregated, not the measure being optimized. For a supported ordinary task,
it therefore agrees in optimal quality with other exhaustive searches.

Generalization-aware measures require a reference computed from simpler
descriptions. `GeneralizationAwareApriori` and `GeneralizationAwareDFS` supply
that reference; ordinary `Apriori` and `DFS` do not. The two dedicated searches
find the same optimal quality values for the same generalization-aware task.
For the DFS variant, the dissertation proposes reverse DFS for these measures
but does not evaluate that combination; the package adds the per-node reference
tables and batching described in the generalization-awareness chapter.

DSSD is heuristic even with `selection="top_k"`. Its default on-the-fly numeric
cuts and its allowance for certain repeated conditions on one feature can also
make its search space different from the standard searches' language. Distinguish
this language difference from the absence of an exhaustive guarantee.

## Compare searches on the same task

```{code-cell} python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True, as_frame=True)
data = X.assign(species=y)
target = sd.BinaryTarget().fit(data, "species", 2)
conditions = sd.generate_conditions(X, n_bins=3)
task = sd.SubgroupDiscoveryTask(data, target, conditions, sd.WRAccQF(),
                                result_set_size=5, depth=2,
                                filters=[sd.MinSizeFilter(10)])
rows = []
for algorithm in [sd.DFS(), sd.OPUS(), sd.Apriori(), sd.SDMapStar(), sd.BeamSearch(beam_width=5)]:
    found = algorithm.execute(task)
    rows.append({"algorithm": type(algorithm).__name__,
                 "qualities": [round(q, 6) for q, _ in found]})
pd.DataFrame(rows)
```

In this example, `BeamSearch` happens to return the same quality values as the
other algorithms. **That agreement is a coincidence of this dataset, search
space, and beam width, not a guarantee of beam search.** On another task it may
miss a better subgroup.

The **first four algorithms—`DFS`, `OPUS`, `Apriori`, and `SDMapStar`—must agree
on the optimal quality values**, because they all perform exhaustive search on
the same task. Tied descriptions at the cutoff may differ across algorithms:
an estimate equal to the current threshold can be pruned. Keep candidate order
fixed when checking reproducibility.

If a measure has no optimistic estimate, exhaustive search can still enumerate
the space, but cannot prune by quality. Minimum-size pruning may still help.
For model-fitting measures, start with a small space or a beam and assess the
computational cost.

## A practical starting rule

Begin with `DFS`, a shallow depth, and a useful minimum size. Compare `OPUS` or
another exhaustive method on your actual task if search cost becomes material.
Use beam search when a bounded heuristic exploration is acceptable; increasing
beam width improves coverage but does not create an exhaustive guarantee.

The beam width must be at least `task.result_set_size` in this implementation.
Use `beam_width_adaptive=True` to set it to the requested result count.

**Experiment.** Set both `beam_width=1` and `result_set_size=1`, and increase the depth. Can beam search match
the exhaustive best score? Explain why agreement in one experiment cannot
establish a guarantee for other data or quality functions.
