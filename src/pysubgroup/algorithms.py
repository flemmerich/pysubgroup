"""
Created on 29.04.2016

@author: lemmerfn
"""

import copy
import warnings
from collections import Counter, defaultdict, namedtuple
from heapq import heappop, heappush
from itertools import chain, combinations
from math import factorial

import numpy as np

import pysubgroup as ps


class SubgroupDiscoveryTask:
    """
    Encapsulates all parameters required to perform standard subgroup discovery.
    """

    def __init__(
        self,
        data,
        target,
        search_space,
        qf,
        result_set_size=10,
        depth=3,
        min_quality=float("-inf"),
        constraints=None,
    ):
        """
        Initializes a new SubgroupDiscoveryTask.

        Parameters:
            data: The dataset to be analyzed.
            target: The target concept for subgroup discovery.
            search_space: The search space of possible selectors.
            qf: The quality function to evaluate subgroups.
            result_set_size: The maximum number of subgroups to return.
            depth: The maximum depth (length) of the subgroups.
            min_quality: The minimal quality threshold for subgroups.
            constraints: A list of constraints to be satisfied by subgroups.
        """
        self.data = data
        self.target = target
        self.search_space = search_space
        self.qf = qf
        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality
        if constraints is None:
            constraints = []
        self.constraints = constraints
        self.constraints_monotone = [
            constr for constr in constraints if constr.is_monotone
        ]
        self.constraints_other = [
            constr for constr in constraints if not constr.is_monotone
        ]


def constraints_satisfied(constraints, subgroup, statistics=None, data=None):
    """
    Checks if all constraints are satisfied for a given subgroup.

    Parameters:
        constraints: A list of constraints to check.
        subgroup: The subgroup to be evaluated.
        statistics: Precomputed statistics for the subgroup (optional).
        data: The dataset to be analyzed (optional).

    Returns:
        True if all constraints are satisfied, False otherwise.
    """
    return all(
        constr.is_satisfied(subgroup, statistics, data) for constr in constraints
    )


try:  # pragma: no cover
    from numba import (  # pylint: disable=import-error, import-outside-toplevel
        int32,
        int64,
        njit,
    )

    @njit([(int32[:, :], int64[:])], cache=True)
    def getNewCandidates(candidates, hashes):  # pragma: no cover
        """
        Generates new candidate pairs for the next level using Numba for acceleration.

        Parameters:
            candidates: A 2D numpy array of candidate selector IDs.
            hashes: A 1D numpy array of hash values for the candidates.

        Returns:
            A list of tuples, each containing indices of candidate pairs to be combined.
        """
        result = []
        for i in range(len(candidates) - 1):
            for j in range(i + 1, len(candidates)):
                if hashes[i] == hashes[j]:
                    if np.all(candidates[i, :-1] == candidates[j, :-1]):
                        result.append((i, j))
        return result

except ImportError:  # pragma: no cover
    pass


class Apriori:
    """
    Implementation of the Apriori algorithm for subgroup discovery.

    This class provides methods to perform level-wise search for subgroups
    using the Apriori algorithm.
    """

    def __init__(
        self, representation_type=None, combination_name="Conjunction", use_numba=True
    ):
        """
        Initializes the Apriori algorithm.

        Parameters:
            representation_type: The representation type to use for subgroups
                (default is BitSetRepresentation).
            combination_name: The name of the combination method
                (e.g., "Conjunction" or "Disjunction").
            use_numba: Whether to use Numba for performance optimization.
        """
        self.combination_name = combination_name

        if representation_type is None:
            representation_type = ps.BitSetRepresentation
        self.representation_type = representation_type
        self.use_vectorization = True
        self.optimistic_estimate_name = "optimistic_estimate"
        self.next_level = self.get_next_level
        self.compiled_func = None
        if use_numba:  # pragma: no cover
            try:
                import numba  # pylint: disable=unused-import, import-outside-toplevel # noqa: F401, E501

                self.next_level = self.get_next_level_numba
                print("Apriori: Using numba for speedup")
            except ImportError:
                pass

    def get_next_level_candidates(self, task, result, next_level_candidates):
        """
        Evaluates candidates at the current level and filters promising ones for
        the next level.

        Parameters:
            task: The subgroup discovery task.
            result: The current list of discovered subgroups.
            next_level_candidates: List of subgroups to be evaluated at the current
            level.

        Returns:
            A list of promising candidates (selectors) for the next level.
        """
        promising_candidates = []
        optimistic_estimate_function = getattr(task.qf, self.optimistic_estimate_name)
        for sg in next_level_candidates:
            statistics = task.qf.calculate_statistics(sg, task.target, task.data)
            ps.add_if_required(
                result,
                sg,
                task.qf.evaluate(sg, task.target, task.data, statistics),
                task,
                statistics=statistics,
            )
            optimistic_estimate = optimistic_estimate_function(
                sg, task.target, task.data, statistics
            )

            if optimistic_estimate >= ps.minimum_required_quality(
                result, task
            ) and ps.constraints_satisfied(
                task.constraints_monotone, sg, statistics, task.data
            ):
                promising_candidates.append((optimistic_estimate, sg.selectors))
        min_quality = ps.minimum_required_quality(result, task)
        promising_candidates = [
            selectors
            for estimate, selectors in promising_candidates
            if estimate > min_quality
        ]
        return promising_candidates

    def get_next_level_candidates_vectorized(self, task, result, next_level_candidates):
        """
        Vectorized evaluation of candidates at the current level to filter promising
        ones for the next level.

        Parameters:
            task: The subgroup discovery task.
            result: The current list of discovered subgroups.
            next_level_candidates: List of subgroups to be evaluated at the current
            level.

        Returns:
            A list of promising candidates (selectors) for the next level.
        """
        promising_candidates = []
        statistics = []
        optimistic_estimate_function = getattr(task.qf, self.optimistic_estimate_name)
        next_level_candidates = list(next_level_candidates)
        if len(next_level_candidates) == 0:
            return []
        for sg in next_level_candidates:
            statistics.append(task.qf.calculate_statistics(sg, task.target, task.data))
        tpl_class = statistics[0].__class__
        vec_statistics = tpl_class._make(np.array(tpl) for tpl in zip(*statistics))
        qualities = task.qf.evaluate(
            slice(None), task.target, task.data, vec_statistics
        )
        optimistic_estimates = optimistic_estimate_function(
            None, None, None, vec_statistics
        )

        for sg, quality, stats in zip(next_level_candidates, qualities, statistics):
            ps.add_if_required(result, sg, quality, task, statistics=stats)

        min_quality = ps.minimum_required_quality(result, task)
        for sg, optimistic_estimate in zip(next_level_candidates, optimistic_estimates):
            if optimistic_estimate >= min_quality:
                promising_candidates.append(sg.selectors)
        return promising_candidates

    def get_next_level_numba(self, promising_candidates):  # pragma: no cover
        """
        Generates the next level of candidates using Numba for acceleration.

        Parameters:
            promising_candidates: A list of promising candidate selectors.

        Returns:
            A list of new candidate selectors for the next level.
        """
        if not hasattr(self, "compiled_func") or self.compiled_func is None:
            self.compiled_func = getNewCandidates

        # Map selectors to unique IDs
        all_selectors = Counter(chain.from_iterable(promising_candidates))
        all_selectors_ids = {selector: i for i, selector in enumerate(all_selectors)}
        promising_candidates_selector_ids = [
            tuple(all_selectors_ids[sel] for sel in selectors)
            for selectors in promising_candidates
        ]
        shape1 = len(promising_candidates_selector_ids)
        if shape1 == 0:
            return []
        shape2 = len(promising_candidates_selector_ids[0])
        arr = np.array(promising_candidates_selector_ids, dtype=np.int32).reshape(
            shape1, shape2
        )

        print(len(arr))
        hashes = np.array(
            [hash(tuple(x[:-1])) for x in promising_candidates_selector_ids],
            dtype=np.int64,
        )
        print(len(arr), arr.dtype, hashes.dtype)
        candidates_int = self.compiled_func(arr, hashes)
        return [
            (*promising_candidates[i], promising_candidates[j][-1])
            for i, j in candidates_int
        ]

    def get_next_level(self, promising_candidates):
        """
        Generates the next level of candidates based on the current promising
        candidates.

        Parameters:
            promising_candidates: A list of promising candidate selectors.

        Returns:
            A list of new candidate selectors for the next level.
        """
        by_prefix_dict = defaultdict(list)
        for sg in promising_candidates:
            by_prefix_dict[tuple(sg[:-1])].append(sg[-1])
        return [
            prefix + real_suffix
            for prefix, suffixes in by_prefix_dict.items()
            for real_suffix in combinations(sorted(suffixes), 2)
        ]

    def execute(self, task):
        """
        Executes the Apriori algorithm on the given task.

        Parameters:
            task: The subgroup discovery task to be executed.

        Returns:
            A SubgroupDiscoveryResult containing the discovered subgroups.
        """
        if not isinstance(
            task.qf, ps.BoundedInterestingnessMeasure
        ):  # pragma: no cover
            warnings.warn(
                "Quality function is unbounded, long runtime expected", RuntimeWarning
            )

        task.qf.calculate_constant_statistics(task.data, task.target)

        with self.representation_type(task.data, task.search_space) as representation:
            combine_selectors = getattr(representation.__class__, self.combination_name)
            result = []
            # Initialize the first level candidates
            next_level_candidates = []
            for sel in task.search_space:
                sg = combine_selectors([sel])
                if ps.constraints_satisfied(
                    task.constraints_monotone, sg, None, task.data
                ):
                    next_level_candidates.append(sg)

            # Level-wise search
            depth = 1
            while next_level_candidates:
                # Evaluate subgroups from the last level
                if self.use_vectorization:
                    promising_candidates = self.get_next_level_candidates_vectorized(
                        task, result, next_level_candidates
                    )
                else:
                    promising_candidates = self.get_next_level_candidates(
                        task, result, next_level_candidates
                    )
                if len(promising_candidates) == 0:
                    break

                if depth == task.depth:
                    break

                next_level_candidates_no_pruning = self.next_level(promising_candidates)

                # Select selectors and build subgroups for which all subsets are in the
                # set of promising candidates
                curr_depth = depth  # Need copy of depth for lazy evaluation
                set_promising_candidates = set(tuple(p) for p in promising_candidates)
                next_level_candidates = (
                    combine_selectors(selectors)
                    for selectors in next_level_candidates_no_pruning
                    if all(
                        (subset in set_promising_candidates)
                        for subset in combinations(selectors, curr_depth)
                    )
                )

                depth = depth + 1

        result = ps.prepare_subgroup_discovery_result(result, task)
        return ps.SubgroupDiscoveryResult(result, task)


class BestFirstSearch:
    """
    Implements the Best-First Search algorithm for subgroup discovery.
    """

    def execute(self, task):
        """
        Executes the Best-First Search algorithm on the given task.

        Parameters:
            task: The subgroup discovery task to be executed.

        Returns:
            A SubgroupDiscoveryResult containing the discovered subgroups.
        """
        result = []
        queue = [(float("-inf"), ps.Conjunction([]))]
        operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task.data, task.target)
        while queue:
            q, old_description = heappop(queue)
            q = -q
            if not q > ps.minimum_required_quality(result, task):
                break
            for candidate_description in operator.refinements(old_description):
                sg = candidate_description
                statistics = task.qf.calculate_statistics(sg, task.target, task.data)
                ps.add_if_required(
                    result,
                    sg,
                    task.qf.evaluate(sg, task.target, task.data, statistics),
                    task,
                    statistics=statistics,
                )
                if len(candidate_description) < task.depth:
                    if hasattr(task.qf, "optimistic_estimate"):
                        optimistic_estimate = task.qf.optimistic_estimate(
                            sg, task.target, task.data, statistics
                        )
                    else:
                        optimistic_estimate = np.inf

                    # Compute refinements and fill the queue
                    if optimistic_estimate >= ps.minimum_required_quality(result, task):
                        if ps.constraints_satisfied(
                            task.constraints_monotone,
                            candidate_description,
                            statistics,
                            task.data,
                        ):
                            heappush(
                                queue, (-optimistic_estimate, candidate_description)
                            )

        result = ps.prepare_subgroup_discovery_result(result, task)
        return ps.SubgroupDiscoveryResult(result, task)


class GeneralisingBFS:  # pragma: no cover
    """
    Implements a Generalizing Best-First Search algorithm for subgroup discovery.
    """

    def __init__(self):
        self.alpha = 1.10
        self.discarded = [0, 0, 0, 0, 0, 0, 0]
        self.refined = [0, 0, 0, 0, 0, 0, 0]

    def execute(self, task):
        """
        Executes the Generalizing Best-First Search algorithm on the given task.

        Parameters:
            task: The subgroup discovery task to be executed.

        Returns:
            A SubgroupDiscoveryResult containing the discovered subgroups.
        """
        result = []
        queue = []
        operator = ps.StaticGeneralizationOperator(task.search_space)
        # Initialize the first level
        for sel in task.search_space:
            queue.append((float("-inf"), ps.Disjunction([sel])))
        task.qf.calculate_constant_statistics(task.data, task.target)

        while queue:
            q, candidate_description = heappop(queue)
            q = -q
            if q < ps.minimum_required_quality(result, task):
                break

            sg = candidate_description
            statistics = task.qf.calculate_statistics(sg, task.target, task.data)
            quality = task.qf.evaluate(sg, task.target, task.data, statistics)
            ps.add_if_required(result, sg, quality, task, statistics=statistics)

            qual = ps.minimum_required_quality(result, task)

            if (quality, sg) in result:
                new_queue = []
                for q_tmp, c_tmp in queue:
                    if (-q_tmp) > qual:
                        heappush(new_queue, (q_tmp, c_tmp))
                queue = new_queue
            optimistic_estimate = task.qf.optimistic_estimate(
                sg, task.target, task.data, statistics
            )
            # Compute refinements and fill the queue
            if len(candidate_description) < task.depth and (
                optimistic_estimate / self.alpha ** (len(candidate_description) + 1)
            ) >= ps.minimum_required_quality(result, task):
                self.refined[len(candidate_description)] += 1
                for new_description in operator.refinements(candidate_description):
                    heappush(queue, (-optimistic_estimate, new_description))
            else:
                self.discarded[len(candidate_description)] += 1

        result.sort(key=lambda x: x[0], reverse=True)
        print("discarded " + str(self.discarded))
        return ps.SubgroupDiscoveryResult(result, task)


class BeamSearch:
    """
    Implements the Beam Search algorithm for subgroup discovery.
    """

    def __init__(self, beam_width=20, beam_width_adaptive=False):
        """
        Initializes the Beam Search algorithm.

        Parameters:
            beam_width: Width of the beam (number of candidates to keep at each level).
            beam_width_adaptive: Whether to adapt the beam width to the result set size.
        """
        self.beam_width = beam_width
        self.beam_width_adaptive = beam_width_adaptive

    def execute(self, task):
        """
        Executes the Beam Search algorithm on the given task.

        Parameters:
            task: The subgroup discovery task to be executed.

        Returns:
            A SubgroupDiscoveryResult containing the discovered subgroups.
        """
        # Adapt beam width to the result set size if desired
        beam_width = self.beam_width
        if self.beam_width_adaptive:
            beam_width = task.result_set_size

        # Check if beam size is too small for result set
        if beam_width < task.result_set_size:
            raise RuntimeError(
                "Beam width in the beam search algorithm "
                "is smaller than the result set size!"
            )

        task.qf.calculate_constant_statistics(task.data, task.target)

        # Initialize
        beam = [
            (
                0,
                ps.Conjunction([]),
                task.qf.calculate_statistics(slice(None), task.target, task.data),
            )
        ]
        previous_beam = None

        depth = 0
        while beam != previous_beam and depth < task.depth:
            previous_beam = beam.copy()
            for _, last_sg, _ in previous_beam:
                if getattr(last_sg, "visited", False):
                    continue
                setattr(last_sg, "visited", True)
                for sel in task.search_space:
                    # Create a clone
                    if sel in last_sg.selectors:
                        continue
                    sg = ps.Conjunction(last_sg.selectors + (sel,))
                    statistics = task.qf.calculate_statistics(
                        sg, task.target, task.data
                    )
                    quality = task.qf.evaluate(sg, task.target, task.data, statistics)
                    ps.add_if_required(
                        beam,
                        sg,
                        quality,
                        task,
                        check_for_duplicates=True,
                        statistics=statistics,
                        explicit_result_set_size=beam_width,
                    )
            depth += 1

        # Trim the beam to the result set size
        while len(beam) > task.result_set_size:
            heappop(beam)

        result = beam
        result = ps.prepare_subgroup_discovery_result(result, task)
        return ps.SubgroupDiscoveryResult(result, task)


class SimpleSearch:
    """
    Implements a simple exhaustive search algorithm for subgroup discovery.
    """

    def __init__(self, show_progress=True):
        """
        Initializes the Simple Search algorithm.

        Parameters:
            show_progress: Whether to display a progress bar during the search.
        """
        self.show_progress = show_progress

    def execute(self, task):
        """
        Executes the Simple Search algorithm on the given task.

        Parameters:
            task: The subgroup discovery task to be executed.

        Returns:
            A SubgroupDiscoveryResult containing the discovered subgroups.
        """
        task.qf.calculate_constant_statistics(task.data, task.target)
        result = []
        all_selectors = chain.from_iterable(
            combinations(task.search_space, r) for r in range(1, task.depth + 1)
        )
        if self.show_progress:
            try:
                from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

                def binomial(x, y):
                    try:
                        binom = factorial(x) // factorial(y) // factorial(x - y)
                    except ValueError:  # pragma: no cover
                        binom = 0
                    return binom

                total = sum(
                    binomial(len(task.search_space), k)
                    for k in range(1, task.depth + 1)
                )
                all_selectors = tqdm(all_selectors, total=total)
            except ImportError:  # pragma: no cover
                warnings.warn(
                    "tqdm not installed but show_progress=True", ImportWarning
                )
        for selectors in all_selectors:
            sg = ps.Conjunction(selectors)
            statistics = task.qf.calculate_statistics(sg, task.target, task.data)
            quality = task.qf.evaluate(sg, task.target, task.data, statistics)
            ps.add_if_required(result, sg, quality, task, statistics=statistics)
        result = ps.prepare_subgroup_discovery_result(result, task)
        return ps.SubgroupDiscoveryResult(result, task)


class SimpleDFS:
    """
    Implements a simple Depth-First Search algorithm for subgroup discovery.
    It is the most elementary (and thus probably slow) algorithm implementation.
    """

    def execute(self, task, use_optimistic_estimates=True):
        """
        Executes the Simple DFS algorithm on the given task.

        Parameters:
            task: The subgroup discovery task to be executed.
            use_optimistic_estimates: Whether to use optimistic estimates for pruning.

        Returns:
            A SubgroupDiscoveryResult containing the discovered subgroups.
        """
        task.qf.calculate_constant_statistics(task.data, task.target)
        result = self.search_internal(
            task, [], task.search_space, [], use_optimistic_estimates
        )
        result = ps.prepare_subgroup_discovery_result(result, task)
        return ps.SubgroupDiscoveryResult(result, task)

    def search_internal(
        self, task, prefix, modification_set, result, use_optimistic_estimates
    ):
        """
        Recursively searches for subgroups in a depth-first manner.

        Parameters:
            task: The subgroup discovery task.
            prefix: The current list of selectors in the subgroup description.
            modification_set: The remaining selectors to consider.
            result: The current list of discovered subgroups.
            use_optimistic_estimates: Whether to use optimistic estimates for pruning.

        Returns:
            The updated list of discovered subgroups.
        """
        sg = ps.Conjunction(copy.copy(prefix))

        statistics = task.qf.calculate_statistics(sg, task.target, task.data)
        if (
            use_optimistic_estimates
            and len(prefix) < task.depth
            and isinstance(task.qf, ps.BoundedInterestingnessMeasure)
        ):
            optimistic_estimate = task.qf.optimistic_estimate(
                sg, task.target, task.data, statistics
            )
            if not optimistic_estimate > ps.minimum_required_quality(result, task):
                return result

        quality = task.qf.evaluate(sg, task.target, task.data, statistics)
        ps.add_if_required(result, sg, quality, task, statistics=statistics)
        if not ps.constraints_satisfied(
            task.constraints_monotone, sg, statistics=statistics, data=task.data
        ):
            return result
        if len(prefix) < task.depth:
            new_modification_set = copy.copy(modification_set)
            for sel in modification_set:
                prefix.append(sel)
                new_modification_set.pop(0)
                self.search_internal(
                    task, prefix, new_modification_set, result, use_optimistic_estimates
                )
                # Remove the selector again
                prefix.pop(-1)
        return result


class DFS:
    """
    Depth-first search with look-ahead for a provided data structure.
    """

    def __init__(self, apply_representation=None):
        """
        Initializes the DFS algorithm.

        Parameters:
            apply_representation: The representation type to use for subgroups.
        """
        self.target_bitset = None
        if apply_representation is None:
            apply_representation = ps.BitSetRepresentation
        self.apply_representation = apply_representation
        self.operator = None
        self.params_tpl = namedtuple(
            "StandardQF_parameters", ("size_sg", "positives_count")
        )

    def execute(self, task):
        """
        Executes the DFS algorithm on the given task.

        Parameters:
            task: The subgroup discovery task to be executed.

        Returns:
            A SubgroupDiscoveryResult containing the discovered subgroups.
        """
        self.operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task.data, task.target)
        result = []
        with self.apply_representation(task.data, task.search_space) as representation:
            self.search_internal(task, result, representation.Conjunction([]))
        result = ps.prepare_subgroup_discovery_result(result, task)
        return ps.SubgroupDiscoveryResult(result, task)

    def search_internal(self, task, result, sg):
        """
        Recursively searches for subgroups in a depth-first manner.

        Parameters:
            task: The subgroup discovery task.
            result: The current list of discovered subgroups.
            sg: The current subgroup being evaluated.
        """
        statistics = task.qf.calculate_statistics(sg, task.target, task.data)
        if not constraints_satisfied(
            task.constraints_monotone, sg, statistics, task.data
        ):
            return
        optimistic_estimate = task.qf.optimistic_estimate(
            sg, task.target, task.data, statistics
        )
        if not optimistic_estimate > ps.minimum_required_quality(result, task):
            return
        quality = task.qf.evaluate(sg, task.target, task.data, statistics)
        ps.add_if_required(result, sg, quality, task, statistics=statistics)

        if sg.depth < task.depth:
            for new_sg in self.operator.refinements(sg):
                self.search_internal(task, result, new_sg)


class DFSNumeric:
    """
    Implements a specialized DFS algorithm for numeric quality functions.
    """

    tpl = namedtuple("size_mean_parameters", ("size_sg", "mean"))

    def __init__(self):
        self.pop_size = 0
        self.f = None
        self.target_values = None
        self.bitsets = {}
        self.num_calls = 0
        self.evaluate = None

    def execute(self, task):
        """
        Executes the DFSNumeric algorithm on the given task.

        Parameters:
            task: The subgroup discovery task to be executed.

        Returns:
            A SubgroupDiscoveryResult containing the discovered subgroups.
        """
        if not isinstance(task.qf, ps.StandardQFNumeric):
            raise RuntimeError(
                "BSD_numeric so far is only implemented for StandardQFNumeric"
            )
        self.pop_size = len(task.data)
        sorted_data = task.data.sort_values(
            task.target.get_attributes()[0], ascending=False
        )

        # Generate target values
        self.target_values = sorted_data[task.target.get_attributes()[0]].to_numpy()

        task.qf.calculate_constant_statistics(task.data, task.target)

        # Generate selector bitsets
        self.bitsets = {}
        for sel in task.search_space:
            # Generate bitset
            self.bitsets[sel] = sel.covers(sorted_data)
        result = self.search_internal(
            task, [], task.search_space, [], np.ones(len(sorted_data), dtype=bool)
        )
        result = ps.prepare_subgroup_discovery_result(result, task)
        return ps.SubgroupDiscoveryResult(result, task)

    def search_internal(self, task, prefix, modification_set, result, bitset):
        """
        Recursively searches in a dfs-manner for numeric quality functions.

        Parameters:
            task: The subgroup discovery task.
            prefix: The current list of selectors in the subgroup description.
            modification_set: The remaining selectors to consider.
            result: The current list of discovered subgroups.
            bitset: The current bitset representing the subgroup.

        Returns:
            The updated list of discovered subgroups.
        """
        self.num_calls += 1
        sg_size = bitset.sum()
        if sg_size == 0:
            return result
        target_values_sg = self.target_values[bitset]

        target_values_cs = np.cumsum(target_values_sg, dtype=np.float64)

        sizes = np.arange(1, len(target_values_cs) + 1)
        mean_values_cs = target_values_cs / sizes
        tpl = DFSNumeric.tpl(sizes, mean_values_cs)
        qualities = task.qf.evaluate(None, None, None, tpl)
        optimistic_estimate = np.max(qualities)

        if optimistic_estimate <= ps.minimum_required_quality(result, task):
            return result

        sg = ps.Conjunction(copy.copy(prefix))

        quality = qualities[-1]
        ps.add_if_required(result, sg, quality, task)

        if len(prefix) < task.depth:
            new_modification_set = copy.copy(modification_set)
            for sel in modification_set:
                prefix.append(sel)
                new_bitset = bitset & self.bitsets[sel]
                new_modification_set.pop(0)
                self.search_internal(
                    task, prefix, new_modification_set, result, new_bitset
                )
                # Remove the selector again
                prefix.pop(-1)
        return result
