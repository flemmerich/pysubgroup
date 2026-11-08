# Changelog

## 2.0.0

A complete rewrite of pysubgroup. The import name stays `pysubgroup`, but the API
is not compatible with version 0.x. **Existing code needs to be migrated, or
pinned to the old version:**

```console
pip install "pysubgroup<2"
```

The old version is also available in the git branch `legacy`, where its own
changelog continues.

### Migrating from 0.x

- Targets are created and then fitted: `sd.BinaryTarget().fit(data, "class", ">50K")`.
- Search spaces are lists of immutable `Condition` objects instead of selectors,
  and subgroup descriptions are `Rule` conjunctions. Disjunctions are not part of
  the discovery language.
- Searches return a list of `(quality, rule)` pairs, best first. Use
  `sd.to_dataframe(result, data, target)` for a result table.
- Numeric intervals are `(lower, upper]` instead of `[lower, upper)`.
- Missing values match only `is_missing`; in particular, they never match `ne`.
- Row masks are positional and independent of DataFrame index labels.
- `LiftQF` scores the share difference, while the `lift` column of result tables is
  a ratio.
- Bundled example datasets were removed; use scikit-learn's loaders such as
  `fetch_openml`.
- Requires Python 3.11 to 3.14 and NumPy 2.

See "Moving from pysubgroup 0.x" in the
[user guide](https://flemmerich.github.io/pysubgroup/reference-troubleshooting) for
details.

### New

- Batched evaluation of candidates on packed bitsets and optional numba kernels.
- Search algorithms: OPUS, DpSubgroup, SD-Map, SD-Map*, VLSD, NumBSD, DSSD, and
  generalization-aware search, in addition to DFS, best-first, Apriori, GP-growth,
  and beam search.
- Multiclass, nominal, and vector-valued targets; exceptional model mining in
  `pysubgroup.emm`; significance filtering; and new visualizations.
