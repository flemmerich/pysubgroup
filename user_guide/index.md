# Subgroup Discovery with pysubgroup

**A practical guide to finding, understanding, and validating interesting subgroups.**

Subgroup discovery looks for interpretable descriptions of parts of a dataset
whose target behavior is unusual. A description might combine two conditions;
its rows might have an unusually high rate of an outcome, a different numeric
distribution, or an exceptional relationship between variables.

This book connects the ideas of subgroup discovery to executable analyses
with `pysubgroup`: formulating a question, defining candidate descriptions,
choosing a quality function, searching, and interpreting and validating the
results.

## Who this guide is for

The intended reader knows basic Python and pandas. The guide introduces
subgroup discovery from the beginning and develops the relevant concepts alongside
the code.

## Contents

:::{toc}
:context: project
:depth: 2
:::

## Choose a reading path

**New to subgroup discovery?** Start with [the idea](01-what-is-subgroup-discovery.md)
and [a complete first search](02-first-discovery.md), then work through Part II.
You will learn to choose descriptions, quality functions, constraints, and search
algorithms before moving to more complex targets.

**Have a specific target?** Go to [numeric outcomes](07-numeric-targets.md),
[multiple targets and WKL](08-multiple-targets-and-wkl.md),
[Exceptional Model Mining](09-exceptional-model-mining.md), or
[classifier auditing](10-classifier-auditing.md).

**Already have results?** Learn to [read and visualize them](11-results-and-visualization.md),
[choose a diverse set](12-diverse-subgroups.md),
[compare with simpler descriptions](13-generalization-awareness.md), and
[validate fixed candidates](14-validation.md).

**Want a complete analysis?** Follow the [Adult exploration](15-exploratory-workflow.md)
or the [validated model audit](16-validated-audit.md).

**Extending or tuning the package?** Read about [performance](17-performance.md),
[custom quality functions](18-custom-quality-functions.md),
[custom targets and models](19-custom-extensions.md), and
[specialized tasks](20-specialized-tasks.md).

For quick lookup, use the [method-selection tables](reference-choosing-methods.md),
[API reference](reference-api.md), [glossary](reference-glossary.md), and
[troubleshooting guide](reference-troubleshooting.md).

## Running the examples

Each executable chapter starts with its own imports and data setup. Run its cells
in order in a fresh Python kernel. Examples require `pysubgroup[plot]`. The first
search and the longer Adult workflow download Adult from OpenML on first use
and share a book-local data cache. Most other chapters use offline scikit-learn
datasets or synthetic data. Randomized examples use explicit seeds.

The book uses small tasks suitable for a laptop. Plots and tables are computed
from the examples, rather than pasted from an unrelated run. Most chapters end
with an experiment to help connect the code with the underlying question.
