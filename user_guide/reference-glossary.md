# Glossary and notation

## Core vocabulary

**Condition**
: A predicate on one descriptor attribute, such as `age > 40`.

**Description / intension**
: The logical expression defining membership. In the standard searches it is
  a nonempty conjunction of supplied conditions on distinct features.

**Cover / extension**
: The set of rows matching a description. Two descriptions may have the same
  cover while receiving different generalization-aware scores.

**Subgroup**
: Depending on context, the description, its cover, or both. This book states
  which aspect matters when the distinction affects the analysis.

**Coverage / relative size**
: The proportion of dataset rows matching a description. An encoder's
  `Coverage` object instead denotes a representation of condition masks.

**Target**
: The outcome, distribution, or model behavior under investigation. In the API,
  a fitted object storing positional per-row information and reference statistics.

**Descriptor**
: An attribute permitted to define subgroup membership, as distinct from model
  or target attributes.

**Quality function**
: The definition used to rank candidates, always maximized by the search.
  Its numerical scale depends on the measure and target units.

**Support**
: Often the number/fraction of matching rows in frequent pattern mining. In
  binary result tables, `support` is the fraction both covered and positive.

**Target share**
: The positive-label fraction within a binary subgroup.

**Lift**
: Subgroup target share divided by dataset target share. The API's `LiftQF`
  instead scores the share *difference*, which is order-equivalent when the
  reference positive share is fixed and nonzero.

**Refinement / specialization**
: A description made more specific by adding conditions. Its cover cannot grow.

**Generalization**
: A less specific description. For generalization-aware measures here, it is
  obtained only by removing conditions; interval widening is not included.

**Empty description**
: A conjunction with no conditions, covering the full dataset. It is used as
  the reference generalization but is not returned by standard discovery searches.

**Optimistic estimate**
: An upper bound on the quality attainable by a refinement, allowing a branch
  to be pruned without visiting every descendant. It is not an estimate of
  the subgroup's expected population effect.

**Anti-monotone constraint**
: Once it fails, it also fails for every refinement. Minimum covered count is
  an example; minimum target share is not.

**Top-k**
: Up to the $k$ highest-scoring eligible descriptions. The number returned is
  not the number evaluated.

**Beam**
: A bounded set of descriptions retained for further refinement by a heuristic
  search. A discarded branch can contain the optimum.

**Complement**
: Dataset rows outside the subgroup. Unlike the dataset reference, it does not
  contain the subgroup itself.

**Diversity**
: A property of a selected set, defined through descriptions, covered rows,
  attribute use, or compression rather than only individual quality.

**Generalization awareness**
: Scoring a description's improvement over simpler descriptions. It does not
  imply low overlap with other selected descriptions or independent validation.

**Exceptional Model Mining (EMM)**
: Discovery where interestingness concerns a fitted model or relationship among
  target attributes, rather than only a single target's level.

**Valuation basis**
: An aggregatable summary of rows from which a model or quality can be computed,
  such as count, sum, and cross-product statistics used by GP-growth.

**Holdout validation**
: Testing fixed discovery candidates on data not used to choose them.

**Randomization null model**
: The distribution of the statistic over randomized row sets constrained as
  specified, such as equal size and optionally equal class/stratum counts.

**Adjusted p-value**
: A p-value transformed for a specified multiple-testing procedure over the
  tested family. It is not the probability that the subgroup is false.

## Notation used in the book

| Symbol | Meaning |
| --- | --- |
| $D$, $N=|D|$ | Reference dataset and its size |
| $G$, $n=|G|$ | Subgroup cover and its size |
| $C=D\setminus G$ | Subgroup complement |
| $P$, $t$ | Positive counts in dataset and subgroup |
| $p=P/N$, $p_G=t/n$ | Positive shares |
| $\bar y_D$, $\bar y_G$ | Numeric target means |
| $\boldsymbol p$, $\boldsymbol\mu$ | Class-share vector and numeric mean vector |
| $q(G)$, $oe(G)$ | Quality and optimistic estimate |
| $a$ | Size exponent in a standard quality function |
| $k$, $k'$ | Final result limit and larger candidate pool for validation |
| $R$, $b$ | Random draws and number scoring at least as high as observed |
| $\alpha$ | Significance level, or size weight in the SubROC formula when explicitly stated |
| $\beta$ | Class-balance exponent in SubROC |

The same letters can have different local uses in publications. Constructor
keyword arguments are preferable when a paper's notation and parameter order
can be confused.
