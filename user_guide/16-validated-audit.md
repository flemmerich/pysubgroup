---
kernelspec:
  name: python3
  display_name: Python 3
short_title: A validated model audit
---

# A validated model audit

We now keep **training, subgroup discovery, and subgroup validation** separate.
The classifier is fitted only on training rows. Candidate descriptions and
numeric cut points are chosen only on discovery rows. The fixed candidates
are then evaluated on validation rows.

The breast-cancer dataset is small, making this an executable teaching example.
A result list with no significant subgroup is a valid outcome. The workflow
does not guarantee a detectable model weakness.

## 1. Reserve the three roles

```{code-cell} python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pysubgroup as sd
from pysubgroup import visualization as viz

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X = X.iloc[:, :6]
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X, y, test_size=0.6, stratify=y, random_state=31)
X_search, X_validation, y_search, y_validation = train_test_split(
    X_remaining, y_remaining, test_size=0.5, stratify=y_remaining, random_state=32)
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000))
model.fit(X_train, y_train)
search = X_search.assign(label=y_search, score=model.predict_proba(X_search)[:, 1])
validation = X_validation.assign(label=y_validation, score=model.predict_proba(X_validation)[:, 1])
print("Training, discovery, validation:", len(X_train), len(search), len(validation))
assert set(X_train.index).isdisjoint(search.index)
assert set(X_train.index).isdisjoint(validation.index)
assert set(search.index).isdisjoint(validation.index)
```

The scaler is fitted inside the training pipeline, so audit rows do not set its
mean or scale. Here the positive score corresponds to label 1, benign in this
loader. Prediction scores are computed after fitting and are not descriptor
attributes.

## 2. Define the search before viewing validation outcomes

We search a generalization-aware version of the weighted ROC AUC decrease.
A small custom pruning filter ensures at least five negatives, complementing
the built-in minimum-positive filter. Negative count is anti-monotone because
a refinement cannot gain negatives.

```{code-cell} python
conditions = sd.generate_conditions(X_search, n_bins=3, numeric_candidates="thresholds")
search_target = sd.SoftClassifierTarget().fit(search, "label", "score", 1)
base_qf = sd.ROCAUCQF(subgroup_class_balance_weight=1, subgroup_size_weight=0.5)
qf = sd.GeneralizationAwareQF(base_qf)
min_negatives = sd.CustomFilter(
    lambda mask, target, conditions: np.count_nonzero(mask & ~target.positives_) >= 5,
    anti_monotone=True,
)
task = sd.SubgroupDiscoveryTask(search, search_target, conditions, qf,
    result_set_size=5, depth=2, min_quality=0,
    filters=[sd.MinSizeFilter(25), sd.MinPositivesFilter(5), min_negatives])
candidates = sd.GeneralizationAwareDFS().execute(task)
sd.to_dataframe(candidates, search, search_target,
    ["size", "positives", "target_share", "roc_auc"], measures={"weighted_AUC_drop": base_qf})
```

The generalization-aware quality rewards a description only to the extent that
it improves on its generalizations' weighted performance-drop scores. The
table also includes the base score so that these quantities are not confused.

## 3. Test the fixed candidates

For this audit, class-stratified randomization asks whether the ranking is worse
than in random validation subsets with the same size and class balance. The
weighted size/balance factor is constant across each candidate's null subsets.

```{code-cell} python
validation_target = search_target.refit(validation)
p_values = sd.randomization_p_values(candidates, validation, validation_target, qf,
    n_samples=999, stratify_target=True, random_state=33, plus_one=True)
adjusted = sd.adjust_p_values(p_values, method="by")
validation_table = sd.to_dataframe([rule for _, rule in candidates], validation,
    validation_target, ["size", "positives", "target_share", "roc_auc"],
    measures={"validation_weighted_AUC_drop": base_qf})
report = validation_table.assign(p_value=p_values, adjusted_p_value=adjusted,
                                significant=adjusted <= 0.05)
report
```

For the wrapper, the significance helper tests the underlying value (here the
weighted AUC decrease), not the discovery-dependent generalization subtraction.
An undefined validation AUC yields p-value one. This can happen even after
discovery-count constraints, because validation covers contain different rows.

The 999 draws make a quick demonstration. A more precise study should choose
the Monte Carlo budget before inspecting outcomes and account for the resolution
of the selected p-value convention and correction.

## 4. Visualize and report accepted results

```{code-cell} python
accepted = [candidate for candidate, p in zip(candidates, adjusted, strict=True) if p <= 0.05]
print(f"Tested candidates: {len(candidates)}; accepted after BY: {len(accepted)}")
if accepted:
    viz.plot_performance_curves(accepted, validation, validation_target)
    plt.show()
else:
    print("No subgroup passed this predefined validation procedure.")
```

If none passes, do not silently relax the threshold or rerun discovery until
something does. The report can still document candidate descriptions and both
discovery and validation effects as exploratory observations, labeled with
their validation outcomes.

For an accepted subgroup, describe the counts of both classes, discovery and
validation ROC AUC, reference AUC, quality definition, null model, number of
draws, p-value convention, and correction. A local AUC drop does not identify a
cause; it identifies a population in which ranking warrants investigation.

## Design choices to revisit for real data

Random row splits assume a suitable sampling structure. If deployment involves
new patients, later time periods, or unseen sites, use a split that reflects
that structure and select appropriate randomization strata. The helpers implement
row-based operations; the analyst must choose a design matching the question.

**Experiment.** Repeat the full procedure with `PRAUCQF`, deciding the new setup
before reading its validation results. Compare which aspect of performance
the two audits investigate, particularly when class balance changes.
