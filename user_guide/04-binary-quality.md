---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Binary quality functions
---

# What makes a subgroup interesting? Binary targets

A binary target asks whether a row has a selected label. The definition of
interestingness determines whether the search favors a few extreme rows or a
larger pattern with a more moderate effect. Supervised descriptive rule mining
offers many related measures {cite:p}`kralj2009,lavrac2004`.

## Counts first

Let $N$ be the dataset size, $P$ the number of positives, $n$ the subgroup size,
and $t$ its positives. Write $p=P/N$ and $p_G=t/n$.

| Statistic | Definition | Question |
| --- | --- | --- |
| Relative size / coverage | $n/N$ | What fraction of all rows belongs? |
| Target share | $t/n$ | How common is the positive label within the subgroup? |
| Support | $t/N$ | What fraction of all rows is both covered and positive? |
| Recall | $t/P$ | What fraction of positives is covered? |
| Lift | $p_G/p$ | How many times the dataset rate is the subgroup rate? |
| WRAcc / leverage | $(n/N)(p_G-p)$ | How much excess positive mass is covered? |

“Support” has different uses across pattern-mining communities. State whether
you mean subgroup size or positive joint support when reporting an analysis.

## StandardQF and the size exponent

The package's standard family is

$$q_a(G)=(n/N)^a(p_G-p),\qquad 0\le a\le1.$$

At $a=0$, only the rate difference matters. At $a=1$, the score is WRAcc.
Intermediate exponents balance the two more gently. `SimpleBinomialQF()` is
`StandardQF(0.5)`; it is a ranking measure, not a post-search significance test.

We can reproduce the candidates from the introduction before performing a
search. Here the counts are hypothetical candidate summaries.

```{code-cell} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysubgroup as sd

candidates = pd.DataFrame({"size": [5, 30, 60], "positives": [5, 15, 18]}, index=["A", "B", "C"])
candidates["share"] = candidates.positives / candidates["size"]
for a in [0, 0.5, 1]:
    candidates[f"a={a}"] = (candidates["size"] / 100) ** a * (candidates.share - 0.2)
candidates
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(6, 4))
exponents = np.linspace(0, 1, 101)
for name, row in candidates.iterrows():
    scores = (row["size"] / 100) ** exponents * (row["share"] - 0.2)
    ax.plot(exponents, scores, label=name)
ax.set(xlabel="Size exponent a", ylabel="Quality", title="The preferred subgroup can change")
ax.legend()
plt.show()
```

The curves cross because the candidates trade size for deviation. This is a
substantive modeling choice, not just a tuning parameter for faster execution.

## Evaluate a concrete description

```{code-cell} python
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True, as_frame=True)
data = X.assign(species=y)
target = sd.BinaryTarget().fit(data, "species", 2)
rule = sd.Rule((sd.Condition("petal length (cm)", "gt", 5),))
mask = sd.utils.subgroup_masks([rule], data)[0]
print("WRAcc:", sd.WRAccQF().evaluate(mask, target))
sd.to_dataframe([rule], data, target, statistics="all",
                measures={"WRAcc": sd.WRAccQF(), "share_difference": sd.LiftQF()})
```

Despite its name, **`LiftQF()` returns the target-share difference**,
`StandardQF(0)`, rather than the ratio $p_G/p$. For a fixed dataset with $p>0$,
the two give the same ordering. The `lift` statistic in a result table is the
ratio. This distinction matters when reporting numerical values.

## Search direction

- `direction="positive"` rewards a rate above the dataset rate.
- `direction="negative"` rewards a rate below it.
- `direction="both"` rewards the absolute rate difference.

```{code-cell} python
pd.Series({direction: sd.WRAccQF(direction=direction).evaluate(mask, target)
           for direction in ["positive", "negative", "both"]})
```

Scores in the opposite direction can be negative. With `min_quality=0`, they
will not be returned. For an empty subgroup, these standard measures return
`nan`; there is no meaningful within-group rate.

## Choosing a measure

Start with WRAcc when both reach and enrichment matter. Use a smaller exponent
when a small, strongly enriched group is valuable, and enforce a minimum size
appropriate to the application. Inspect the subgroup counts whatever the score.

`ChiSquaredQF()` scores a binary subgroup-versus-complement contingency table.
Its test statistic rewards evidence of association and is unbounded in the
search API. `stat="p"` returns a p-value, which is **smaller** for stronger
evidence; the search always maximizes, so do not use raw p-values as if larger
meant more interesting. A test score computed on the discovery data also does
not adjust for searching many candidate descriptions.

**Experiment.** Search Iris with exponents 0, 0.5, and 1, holding the search space
and minimum size fixed. Explain changes using both size and target share, rather
than comparing quality numbers across differently defined measures.
