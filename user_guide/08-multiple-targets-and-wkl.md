---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Multiple targets and WKL
---

# Class distributions, multiple targets, and WKL

There are several distinct ways for a target to be multivariate. A nominal
class distribution, a vector of numeric means, and several categorical model
attributes require different representations and measures.

## One categorical attribute with several classes

Instead of choosing one Iris species as positive, ask **where the species
distribution differs from its population distribution**:

```{code-cell} python
from sklearn.datasets import load_iris
import pandas as pd
import pysubgroup as sd

X, y = load_iris(return_X_y=True, as_frame=True)
data = X.assign(species=y)
target = sd.MulticlassTarget().fit(data, "species")
conditions = sd.generate_conditions(X, n_bins=3)
task = sd.SubgroupDiscoveryTask(data, target, conditions, sd.ClassDistributionL1QF(),
                                result_set_size=5, depth=2,
                                filters=[sd.MinSizeFilter(15)])
results = sd.DFS().execute(task)
sd.to_dataframe(results, data, target, ["size", "class_shares", "majority_class", "majority_share"])
```

Let $\boldsymbol p_G$ and $\boldsymbol p_D$ be the subgroup and dataset class
shares. Available choices include:

| Measure | Quality |
| --- | --- |
| `SplitQF` | $n\sum_c(p_{G,c}-p_{D,c})^2$ |
| `MulticlassChiSquaredQF` | $n\sum_c(p_{G,c}-p_{D,c})^2/p_{D,c}$ |
| `GiniQF` | $n/(N-n)\sum_c(p_{G,c}-p_{D,c})^2$ |
| `ClassDistributionL1QF` | $(n/N)\sum_c|p_{G,c}-p_{D,c}|$ |
| `MaxConfidenceQF` | $\max_c p_{G,c}$ |

The first three measures and their bounds follow the DpSubgroup report
{cite:p}`grosskreutz2008`; the L1 measure connects to election-distribution
analysis {cite:p}`grosskreutz2010`. `MaxConfidenceQF` favors purity, rather than
explicitly subtracting the dataset distribution. Missing class labels are
rejected by `MulticlassTarget`.

## Several numeric target attributes

Suppose both petal length and width are outcomes of interest. Exclude both
from the descriptor frame and use the sepal measurements as descriptions:

```{code-cell} python
model_columns = ["petal length (cm)", "petal width (cm)"]
vector_target = sd.VectorTarget().fit(data, *model_columns)
vector_conditions = sd.generate_conditions(data[["sepal length (cm)", "sepal width (cm)"]], n_bins=4)
vector_task = sd.SubgroupDiscoveryTask(data, vector_target, vector_conditions,
    sd.MeanVectorL1QF(), result_set_size=5, depth=2,
    filters=[sd.MinSizeFilter(15)])
vector_results = sd.DFS().execute(vector_task)
sd.to_dataframe(vector_results, data, vector_target)
```

`MeanVectorL1QF` computes

$$q_{dst}(G)=\frac{n}{N}\left\|\boldsymbol\mu_G-\boldsymbol\mu_D\right\|_1.$$

It sums absolute coordinate shifts. **Units matter:** a coordinate measured in
thousands can dominate one measured in fractions. Choose units or preprocessing
that reflect the scientific question. The measure does not directly capture
changes in correlation with unchanged means; use EMM for that.

The package derives tight optimistic estimates for the L1 measures. Their exact
computation grows exponentially in the number of classes/dimensions; above
`max_exact_classes` or `max_exact_dimensions`, a looser component-wise bound is
used. These switches affect pruning, not the quality definition.

## Several nominal model attributes: weighted KL divergence

`NominalTarget` stores multiple categorical attributes. The WKL quality of
{cite:t}`vanleeuwen2012` compares each attribute's marginal distribution with
its dataset distribution:

$$
WKL(G)=\frac{n}{N}\sum_j\sum_v
  P_G(M_j=v)\log_2\frac{P_G(M_j=v)}{P_D(M_j=v)}.
$$

A zero subgroup probability contributes zero. A category observed in a subgroup
also occurs in the containing dataset, so its dataset probability is positive.
WKL is nonnegative and uses bits. Its size weight balances divergence against
the amount of data represented.

Here we discretize the two petal targets, while retaining sepal descriptors:

```{code-cell} python
nominal_data = data.assign(
    length_band=pd.cut(data["petal length (cm)"], [-float("inf"), 2, 5, float("inf")]),
    width_band=pd.cut(data["petal width (cm)"], [-float("inf"), 0.6, 1.7, float("inf")]),
)
nominal_target = sd.NominalTarget().fit(nominal_data, "length_band", "width_band")
nominal_task = sd.SubgroupDiscoveryTask(nominal_data, nominal_target, vector_conditions,
    sd.WeightedKLQF(), result_set_size=5, depth=2,
    filters=[sd.MinSizeFilter(15)])
nominal_results = sd.DFS().execute(nominal_task)
sd.to_dataframe(nominal_results, nominal_data, nominal_target)
```

`WeightedKLQF` also accepts binary and multiclass targets. For multiple nominal
attributes it sums **marginal** divergences: this is not the divergence of their
full joint distribution. A pure association change can leave every marginal
unchanged and receive zero WKL. The association models in the next chapter
address relationships between attributes.

Missing nominal model values count as a category of their own. WKL currently
has no optimistic estimate, so exhaustive searches cannot prune by this quality.
It also supplies the distributional coding interpretation used by
[compression-based DSSD](12-diverse-subgroups.md).

**Experiment.** Duplicate one nominal model attribute under a second name and
fit both as targets. Why does this double WKL? What does that tell you about
choosing correlated or redundant model attributes?
