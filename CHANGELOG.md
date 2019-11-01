# Changelog
## [0.6.2] - 2019-31-10
### Changed
- **SubgroupDescription** has been replaced with **Conjunction**
- Selector _.covers_ function returns a numpy array instead of a pandas Series (speedup on dense data)
- Conjunction _.selectors_ is renamed to Conjunction.*\_selectors*
- quality functions have a different interface
  - calculate_constant_statistics(self, task) caches necessary precomputation
  - calculate_statistics(self, subgroup, data=None) returns a namedtuple with necessary statistics
  - evaluate(self, subgroup, statistics=None) computes quality from provided statistics
  - optimistic_estimate(self, subgroup, statistics=None) computes optimistic estimate from provided statistics
- 

### Added
- Conjunction (replaces SubgroupDescription)
- Disjunction
- DNF (Disjunctive Normal Form)
- representations (given a dataset selectors are queried only once)
  - BitsetRepresentation
  - SetRepresentation
  - NumpySetRepresentation
- SimpleSearch algorithm
- DFS (Depth first search) using a representation for StandardQF
- tests
  - access to datasets for testing is provided through pysubgroup.tests.DataSets class
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
    - cover
  - tests for algorithms with nominal target concept on the creditg dataset (StandardQF(1) + NominalSearchSpace, StandardQF(1)+Nominal&Numeric Searchspace, StandardQF(0.5)+Nominal&Numeric Searchspace)
    - Apriori
    - SimpleDFS
    - BeamSearch
    - DFS_bitset
    - DFS_set
    - DFS_numpy_sets
    - SimpleSearch
  - tests for algorithms with numeric target concept (StandardQFNumeric)
    - Apriori
    - SimpleDFS
    - DFSNumeric
  - tests for algorithm with fi target (CountQF)
    - Apriori
    - DFS
  - tests for algorithms to find the best Disjunctions
    - Apriori
    - Generalising BFS

### Improvements
- Apriori algorithm now runs significantly faster due to precomputing and usage of list comprehension