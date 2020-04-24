# Changelog
## [0.6.2.1] - 2020-04-24

This update prepares pysubgroup for a better future. To do so we had to break backwards compatibility. Many of the classes that you know and love have been renamed so as to make their purpose more clear.
### Changed:
- SubgroupDescription is now called Conjunction
- NominalTarget is now called BinaryTarget
- algorithms now return a SubgroupDiscoveryResult object
- the structure of quality functions changed (see documentation for more info)

### Added
 - pysubgroup now has a bunch of tests
 - some algorithms and quality functions support numba for just in time compilation
 - ModelTarget
 - gp-growth
 - 3 types of Representations (bitset, set, numpy-set)
 - Refinement operator
 - Disjunction
 - New algorithms


## [0.6.2.1] - 2019-20-11
### Added
- Apriori now has the option to disable numba using the use_numba flag
- SimpleSrach now has a progressbar (enabled via the show_progress=True flag)
- The number of quality function evaluations can now be tracked using the CountCallsInterestingMeasure as a wrapper
- StandardQfNumeric now offers three different options to calculate the optimistic estimate
  - 'sum' (default) sums the values larger then the dataset mean (cf. Lemmerich 2014 p. 81 top)
  - 'average' uses the maximum target values as estimate (cf. Lemmerich 2014 p. 82 center)
  - 'order' uses ordering based bounds (cf. Lemmerich 2014 p. 89 bottom)


### Bugfix
- Apriori now calculates the constant statistics before using representation
- DFS now properly works with any quality function

### Improvements
- Apriori now reuses the compiled numba function
- Nominal target now uses subgroup.size to access the size of a subgroup representation
- StaticSpecializationOperator now avoids checking refinements of the same attribute
- test_algorithms_numeric now checks more algorithms

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