---
kernelspec:
  name: python3
  display_name: Python 3
short_title: Classifier auditing
---

# Auditing a classifier with subgroup discovery

A classifier can have good overall performance and poor performance within a
describable population. SubROC searches for exceptional ranking performance
using labels and prediction scores {cite:p}`siegl2025`.

## Train once, then search separate rows

We fit a simple model on part of scikit-learn's breast-cancer dataset and audit
its scores on different rows. Only six measurements are used as descriptors to
keep the demonstration small.

```{code-cell} python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pysubgroup as sd
from pysubgroup import visualization as viz

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X = X.iloc[:, :6]
X_train, X_audit, y_train, y_audit = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=13)
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
model.fit(X_train, y_train)
data = X_audit.assign(label=y_audit, score=model.predict_proba(X_audit)[:, 1])
target = sd.SoftClassifierTarget().fit(data, "label", "score", positive_value=1)
conditions = sd.generate_conditions(X_audit, n_bins=3)
qf = sd.ROCAUCQF(subgroup_class_balance_weight=1, subgroup_size_weight=0.5)
task = sd.SubgroupDiscoveryTask(data, target, conditions, qf,
    depth=2, result_set_size=4, min_quality=0,
    filters=[sd.MinSizeFilter(35), sd.MinPositivesFilter(5)])
results = sd.DFS().execute(task)
sd.to_dataframe(results, data, target, ["size", "positives", "target_share", "roc_auc"])
```

`positive_value` defines which label the score ranks upward. Check its meaning:
the loader encodes benign as 1. Reversing the score-label pairing would reverse
the performance interpretation. Both label and score columns are target
attributes and must stay outside subgroup descriptions.

## What the measures detect

| Measure | Positive quality means |
| --- | --- |
| `ROCAUCQF` | Subgroup ROC AUC is lower than dataset ROC AUC |
| `PRAUCQF` | Subgroup precision–recall area is lower |
| `ARLQF` | Subgroup average ranking loss is higher |

ROC AUC is the probability of a positive scoring above a negative, counting ties
as half. It requires both labels. PR area here uses trapezoidal integration,
including the initial precision of one; it is not scikit-learn's average
precision. ARL is the average **number** of negatives ranked above a positive,
so it depends on the number of negatives as well as their ordering.

For ROC AUC, the weighted score is

$$q(G)=n_G^\alpha cb(G)^\beta\left(AUC(D)-AUC(G)\right),$$

where $cb$ is the minority/majority class-count ratio. The constructor names the
balance weight first ($\beta$) and size weight second ($\alpha$); keyword
arguments make this unambiguous. A subgroup with an undefined metric scores
`-inf` and is not reported. `MinPositivesFilter` alone does not ensure negatives.

## Read the curves and sample sizes

```{code-cell} python
if results:
    viz.plot_performance_curves(results, data, target, curve="roc")
    plt.show()
```

The dataset curve provides the reference; subgroup curves show local ranking.
Their shapes may be jagged because some groups contain few positive-negative
pairs. A lower AUC does not automatically imply lower accuracy at the operating
threshold, worse calibration, or a causal explanation for performance loss.
Those are different audit questions.

## Choice of weights affects discovery and pruning

Without weights, a tiny group with a single reversed pair can be very extreme.
Size and balance weights favor effects supported by more observations and both
classes. The implemented bound is useful without weighting or with
$0<\alpha\le\beta$. Unsupported weight combinations can still be scored, but
their optimistic estimate may be infinite and provide no pruning.

## Exploration is not confirmation

These audit rows were not used to fit the classifier, but they **were used to
choose the subgroup descriptions**. Their subgroup performance estimates are
still selected. The [validation chapter](14-validation.md) introduces independent
subgroup validation, and the [complete audit workflow](16-validated-audit.md)
keeps training, discovery, and validation separate.

**Experiment.** Repeat with `ROCAUCQF()` and with `PRAUCQF(1, 0.5)`. Are the
same descriptions preferred? Inspect class balance and both curves before
interpreting disagreement as a contradiction.
