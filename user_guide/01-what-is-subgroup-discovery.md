---
short_title: What is subgroup discovery?
---

# What is subgroup discovery?

Suppose a service team knows that 20% of its customers cancel a subscription.
An average describes the population, but it leaves an important question open:
**which describable groups have a substantially different cancellation rate?**
Subgroup discovery searches for answers such as

> Customers on a monthly contract who contacted support more than twice.

The answer has two parts: a **description**, which a person can read, and a
**set of rows**, which can be examined and compared with a reference population.
The discovery process ranks descriptions by a chosen definition of interestingness.
This belongs to supervised descriptive pattern mining {cite:p}`kralj2009`.

## Descriptions and covered rows

A condition is a test on one attribute: `contract=monthly` or `contacts > 2`.
A subgroup description combines conditions with **and**. A row belongs to the
subgroup when it satisfies every condition. The set of matching rows is its
*extension* or *cover*.

The target is the behavior we want to study: cancellation in this example.
Descriptor attributes determine membership; target attributes determine what
is interesting about the members. Keeping those roles separate prevents a
tautology such as discovering that `cancelled=yes` predicts cancellation.

Two descriptions can cover the same rows. Conversely, two subgroups can share
many rows without having similar descriptions. These distinctions become
important when discussing [diversity](12-diverse-subgroups.md) and
[generalization awareness](13-generalization-awareness.md).

## A small example: interesting is not the same as extreme

Imagine a dataset with 100 customers, of whom 20 cancel:

| Candidate | Rows | Cancellations | Cancellation rate | Increase over 20% |
| --- | ---: | ---: | ---: | ---: |
| A | 5 | 5 | 100% | 80 percentage points |
| B | 30 | 15 | 50% | 30 percentage points |
| C | 60 | 18 | 30% | 10 percentage points |

All three are unusual in the same direction. A has the most extreme rate; B
describes a larger group with a strong deviation; C covers most of the
population but has a modest deviation.

A quality function, also called an **interestingness measure**, makes the
trade-off explicit. There are many possible measures; **weighted relative
accuracy (WRAcc) is just one example**. It multiplies the rate increase by the
subgroup's population share:

$$
q(G)=\frac{|G|}{|D|}\left(p_G-p_D\right).
$$

The scores are 0.04, 0.09, and 0.06. WRAcc prefers B. A measure that ignores
size would prefer A. Neither decision can be made sensibly without considering
the analysis question. The [binary-target chapter](04-binary-quality.md)
develops this choice in detail.

## What a subgroup discovery result means

A typical result is a ranked list of descriptions with scores. The descriptions
can overlap: one customer may satisfy several. The list need not cover every
row, and the target rate need not be high in every returned group—one can search
for decreases or other forms of deviation as well.

The scope of a result depends on the **search space**. If only three thresholds
for age are provided, the search cannot return every possible age interval.
An exhaustive algorithm finds the best allowed descriptions, not the best
descriptions in an unrestricted language.

High quality is an exploratory finding. It does not establish that the
conditions cause the outcome. Searching many descriptions can also produce
strong-looking results by chance. [Validation](14-validation.md) addresses the
selection process using separate data and a specified null model.

## Related methods

| Method | Main aim | Relationship to subgroup discovery |
| --- | --- | --- |
| Classification | Predict a label for new rows | A subgroup list describes local behavior; it is not itself a complete classifier |
| Clustering | Group rows by similarity | A subgroup is defined through explicit conditions and a target of interest; groups may overlap |
| Association rules | Find co-occurring items, often as antecedent → consequent | Many binary subgroup measures are closely related, but subgroup discovery fixes the target concept |
| Frequent pattern mining | Find common combinations | This is a target-free special case available through `FITarget` |
| Exceptional Model Mining | Find subgroups with unusual fitted models | Extends the target from a value or distribution to a relationship or model |

Subgroup discovery is especially useful when a domain expert can act on a
small collection of understandable local patterns. The algorithm explores
combinations; the analyst chooses the question, language, and interpretation.

Next, [run a first discovery](02-first-discovery.md) on a sample of the Adult
census-income dataset.
