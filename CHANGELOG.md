# Changelog
## [0.6.2] - 2019-31-10
### Changed
- **SubgroupDescription** has been replaced with **Conjunction**
- Selector __.covers__ function returns a numpy array instead of a pandas Series (speedup on dense data)
- quality functions have a different interface
  - calculate_constant_statistics(self, task) caches necessary precomputation
  - calculate_statistics(self, subgroup, data=None) returns a namedtuple with necessary statistics
  - evaluate(self, subgroup, statistics=None) computes quality from provided statistics
  - optimistic_estimate(self, subgroup, statistics=None) computes optimistic estimate from provided statistics
- 

### Added
- Conjunction (replaces SubgroupDescription)
- Disjunction
- representations (given a dataset selectors are queried once and thereafter the representations are used)
- SimpleSearch algorithm
- DFS (Depth first search) using a representation for StandardQF
- tests
  - tests for selector classes (NominalSelector, NumericSelector)
    - \_\_eq\_\_
    - \_\_lt\_\_
    - \_\_hash\_\_ similarity
    - uniqueness of selectors
    - cover function for NominalSelector
  - tests for Conjunction, Disjuntion
    - \_\_eq\_\_
    - \_\_lt\_\_
    - \_\_hash\_\_ similarity
  - tests for algorithms with nominal target concept on the creditg dataset (three different cases)
    - Apriori
    - SimpleDFS
    - BeamSearch
    - DFS_bitset
    - DFS_set
    - DFS_numpy_sets
    - SimpleSearch

### Improvements
- Apriori algorithm now runs significantly faster due to precomputing and usage of list comprehension