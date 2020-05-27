'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
from math import factorial
from itertools import combinations, chain
from heapq import heappush, heappop
from collections import Counter, namedtuple
import warnings
import numpy as np
import pysubgroup as ps


class SubgroupDiscoveryTask:
    '''
    Capsulates all parameters required to perform standard subgroup discovery
    '''

    def __init__(self, data, target, search_space, qf, result_set_size=10, depth=3, min_quality=0, constraints=None):
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
        self.constraints_monotone = [constr for constr in constraints if constr.is_monotone]
        self.constraints_other = [constr for constr in constraints if not constr.is_monotone]


def constraints_satisfied(constraints, subgroup, statistics=None, data=None):
    return all(constr.is_satisfied(subgroup, statistics, data) for constr in constraints)



class Apriori:
    def __init__(self, representation_type=None, combination_name='Conjunction', use_numba=True):
        self.combination_name = combination_name

        if representation_type is None:
            representation_type = ps.BitSetRepresentation
        self.representation_type = representation_type
        self.use_vectorization = True
        self.use_repruning = False
        self.optimistic_estimate_name = 'optimistic_estimate'
        self.next_level = self.get_next_level
        self.compiled_func = None
        if use_numba:
            try:
                import numba # pylint: disable=unused-import, import-outside-toplevel
                self.next_level = self.get_next_level_numba
                print('Apriori: Using numba for speedup')
            except ImportError:
                pass


    def get_next_level_candidates(self, task, result, next_level_candidates):
        promising_candidates = []
        optimistic_estimate_function = getattr(task.qf, self.optimistic_estimate_name)
        for sg in next_level_candidates:
            statistics = task.qf.calculate_statistics(sg, task.target, task.data)
            ps.add_if_required(result, sg, task.qf.evaluate(sg, statistics, task.target, task.data), task, statistics=statistics)
            optimistic_estimate = optimistic_estimate_function(sg, task.target, task.data, statistics)

            if optimistic_estimate >= ps.minimum_required_quality(result, task):
                if ps.constraints_hold(task.constraints_monotone, sg, statistics, task.data):
                    promising_candidates.append((optimistic_estimate, sg.selectors))
        min_quality = ps.minimum_required_quality(result, task)
        promising_candidates = [selectors for estimate, selectors in promising_candidates if estimate > min_quality]
        return promising_candidates


    def get_next_level_candidates_vectorized(self, task, result, next_level_candidates):
        promising_candidates = []
        statistics = []
        optimistic_estimate_function = getattr(task.qf, self.optimistic_estimate_name)
        for sg in next_level_candidates:
            statistics.append(task.qf.calculate_statistics(sg, task.target, task.data))
        tpl_class = statistics[0].__class__
        vec_statistics = tpl_class._make(np.array(tpl) for tpl in zip(*statistics))
        qualities = task.qf.evaluate(None, task.target, task.data, vec_statistics)
        optimistic_estimates = optimistic_estimate_function(None, None, None, vec_statistics)

        for sg, quality, stats in zip(next_level_candidates, qualities, statistics):
            ps.add_if_required(result, sg, quality, task, statistics=stats)

        min_quality = ps.minimum_required_quality(result, task)
        for sg, optimistic_estimate in zip(next_level_candidates, optimistic_estimates):
            if optimistic_estimate >= min_quality:
                promising_candidates.append(sg.selectors)
        return promising_candidates

    def reprune_lower_levels(self, promising_candidates, depth):
        for k in range(1, depth):
            promising_candidates_k = (combinations(selectors, k) for selectors in promising_candidates)
            combination_counter = Counter(chain.from_iterable(promising_candidates_k))
            d = depth + 1 - k
            unpromising_combinations = set(frozenset(sel) for sel, count in combination_counter.items() if count < d)
            promising_candidates = list(selectors for selectors in promising_candidates
                                        if all(frozenset(comb) not in unpromising_combinations for comb in combinations(selectors, k)))
        return promising_candidates

    def get_next_level_numba(self, promising_candidates):
        from numba import jit # pylint: disable=import-error, import-outside-toplevel
        if not hasattr(self, 'compiled_func') or self.compiled_func is None:
            @jit
            def getNewCandidates(l, hashes):
                result = []
                for i in range(len(l)-1):
                    for j in range(i + 1, len(l)):
                        if hashes[i] == hashes[j]:
                            if np.all(l[i, :-1] == l[j, :-1]):
                                result.append((i, j))
                return result
            self.compiled_func = getNewCandidates

        all_selectors = Counter(chain.from_iterable(promising_candidates))
        d = {selector:i for i, selector in enumerate(all_selectors)}
        l = [tuple(d[sel] for sel in selectors) for selectors in promising_candidates]
        arr = np.array(l, dtype=int)

        print(len(arr))
        hashes = np.array([hash(tuple(x[:-1])) for x in l], dtype=np.int64)
        candidates_int = self.compiled_func(arr, hashes)
        return list((*promising_candidates[i], promising_candidates[j][-1])  for i, j in candidates_int)

    def get_next_level(self, promising_candidates):
        precomputed_list = list((tuple(sg), sg[-1], hash(tuple(sg[:-1])), tuple(sg[:-1])) for sg in promising_candidates)
        return list((*sg1, new_selector) for (sg1, _, hash_l, selectors_l), (_, new_selector, hash_r, selectors_r) in combinations(precomputed_list, 2)
                    if (hash_l == hash_r) and (selectors_l == selectors_r))

    def execute(self, task):
        if not isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            raise RuntimeWarning("Quality function is unbounded, long runtime expected")

        task.qf.calculate_constant_statistics(task.data, task.target)

        with self.representation_type(task.data, task.search_space) as representation:
            combine_selectors = getattr(representation.__class__, self.combination_name)
            result = []
            # init the first level
            next_level_candidates = []
            for sel in task.search_space:
                next_level_candidates.append(combine_selectors([sel]))

            # level-wise search
            depth = 1
            while next_level_candidates:
                # check sgs from the last level
                if self.use_vectorization:
                    promising_candidates = self.get_next_level_candidates_vectorized(task, result, next_level_candidates)
                else:
                    promising_candidates = self.get_next_level_candidates(task, result, next_level_candidates)

                if depth == task.depth:
                    break

                if self.use_repruning:
                    promising_candidates = self.reprune_lower_levels(promising_candidates, depth)

                next_level_candidates_no_pruning = self.next_level(promising_candidates)

                # select those selectors and build a subgroup from them
                #   for which all subsets of length depth (=candidate length -1) are in the set of promising candidates
                set_promising_candidates = set(tuple(p) for p in promising_candidates)
                next_level_candidates = [combine_selectors(selectors) for selectors in next_level_candidates_no_pruning
                                         if all((subset in set_promising_candidates) for subset in combinations(selectors, depth))]
                depth = depth + 1

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)


class BestFirstSearch:
    def execute(self, task):
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
                ps.add_if_required(result, sg, task.qf.evaluate(sg, task.target, task.data, statistics), task, statistics=statistics)
                if len(candidate_description) < task.depth:
                    optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)

                    # compute refinements and fill the queue
                    if optimistic_estimate >= ps.minimum_required_quality(result, task):
                        if ps.constraints_satisfied(task.constraints_monotone, candidate_description, statistics, task.data):
                            heappush(queue, (-optimistic_estimate, candidate_description))

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)


class GeneralisingBFS:
    def __init__(self):
        self.alpha = 1.10
        self.discarded = [0, 0, 0, 0, 0, 0, 0]
        self.refined = [0, 0, 0, 0, 0, 0, 0]

    def execute(self, task):
        result = []
        queue = []
        operator = ps.StaticGeneralizationOperator(task.search_space)
        # init the first level
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
            quality = task.qf.evaluate(sg, statistics)
            ps.add_if_required(result, sg, quality, task, statistics=statistics)

            qual = ps.minimum_required_quality(result, task)

            if (quality, sg) in result:
                new_queue = []
                for q_tmp, c_tmp in queue:
                    if (-q_tmp) > qual:
                        heappush(new_queue, (q_tmp, c_tmp))
                queue = new_queue
            optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)
            # else:
            #    ps.add_if_required(result, sg, task.qf.evaluate_from_dataset(task.data, sg), task)
            #    optimistic_estimate = task.qf.optimistic_generalisation_from_dataset(task.data, sg) if qf_is_bounded else float("inf")

            # compute refinements and fill the queue
            if len(candidate_description) < task.depth and (optimistic_estimate / self.alpha ** (len(candidate_description) + 1)) >= ps.minimum_required_quality(result, task):
                # print(qual)
                # print(optimistic_estimate)
                self.refined[len(candidate_description)] += 1
                # print(str(candidate_description))
                for new_description in operator.refinements(candidate_description):
                    heappush(queue, (-optimistic_estimate, new_description))
            else:
                self.discarded[len(candidate_description)] += 1

        result.sort(key=lambda x: x[0], reverse=True)
        for qual, sg in result:
            print("{} {}".format(qual, sg))
        print("discarded " + str(self.discarded))
        return ps.SubgroupDiscoveryResult(result, task)


class BeamSearch:
    '''
    Implements the BeamSearch algorithm. Its a basic implementation
    '''

    def __init__(self, beam_width=20, beam_width_adaptive=False):
        self.beam_width = beam_width
        self.beam_width_adaptive = beam_width_adaptive

    def execute(self, task):
        # adapt beam width to the result set size if desired
        if self.beam_width_adaptive:
            self.beam_width = task.result_set_size

        # check if beam size is to small for result set
        if self.beam_width < task.result_set_size:
            raise RuntimeError('Beam width in the beam search algorithm is smaller than the result set size!')

        task.qf.calculate_constant_statistics(task.data, task.target)

        # init
        beam = [(0, ps.Conjunction([]), task.qf.calculate_statistics(slice(None), task.target, task.data))]
        last_beam = None

        depth = 0
        while beam != last_beam and depth < task.depth:
            last_beam = beam.copy()
            for (_, last_sg, _) in last_beam:
                if not getattr(last_sg, 'visited', False):
                    setattr(last_sg, 'visited', True)
                    for sel in task.search_space:
                        # create a clone
                        new_selectors = list(last_sg.selectors)
                        if sel not in new_selectors:
                            new_selectors.append(sel)
                            sg = ps.Conjunction(new_selectors)
                            statistics = task.qf.calculate_statistics(sg, task.target, task.data)
                            quality = task.qf.evaluate(sg, task.target, task.data, statistics)
                            ps.add_if_required(beam, sg, quality, task, check_for_duplicates=True, statistics=statistics)
            depth += 1
# TODO make sure there is no bug here
        result = beam[:task.result_set_size]
        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)


class SimpleSearch:
    def __init__(self, show_progress=True):
        self.show_progress = show_progress
    def execute(self, task):
        task.qf.calculate_constant_statistics(task.data, task.target)
        result = []
        all_selectors = chain.from_iterable(combinations(task.search_space, r) for r in range(1, task.depth + 1))
        if self.show_progress:
            try:
                from tqdm import tqdm   # pylint: disable=import-outside-toplevel
                def binomial(x, y):
                    try:
                        binom = factorial(x) // factorial(y) // factorial(x - y)
                    except ValueError:
                        binom = 0
                    return binom
                total = sum(binomial(len(task.search_space), k) for k in range(1, task.depth + 1))
                all_selectors = tqdm(all_selectors, total=total)
            except ImportError:
                pass
        for selectors in all_selectors:
            sg = ps.Conjunction(selectors)
            statistics = task.qf.calculate_statistics(sg, task.target, task.data)
            quality = task.qf.evaluate(sg, task.target, task.data, statistics)
            ps.add_if_required(result, sg, quality, task, statistics=statistics)
        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)


class SimpleDFS:
    def execute(self, task, use_optimistic_estimates=True):
        task.qf.calculate_constant_statistics(task.data, task.target)
        result = self.search_internal(task, [], task.search_space, [], use_optimistic_estimates)
        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)

    def search_internal(self, task, prefix, modification_set, result, use_optimistic_estimates):
        sg = ps.Conjunction(copy.copy(prefix))

        statistics = task.qf.calculate_statistics(sg, task.target, task.data)
        if use_optimistic_estimates and len(prefix) < task.depth and isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)
            if not optimistic_estimate > ps.minimum_required_quality(result, task):
                return result

        quality = task.qf.evaluate(sg, task.target, task.data, statistics)
        ps.add_if_required(result, sg, quality, task, statistics=statistics)
        if not ps.constraints_satisfied(task.constraints_monotone, sg, statistics=statistics, data=task.data):
            return
        if len(prefix) < task.depth:
            new_modification_set = copy.copy(modification_set)
            for sel in modification_set:
                prefix.append(sel)
                new_modification_set.pop(0)
                self.search_internal(task, prefix, new_modification_set, result, use_optimistic_estimates)
                # remove the sel again
                prefix.pop(-1)
        return result


class DFS:
    """
    Implementation of a depth-first-search with look-ahead using a provided datastructure.
    """

    def __init__(self, apply_representation):
        self.target_bitset = None
        self.apply_representation = apply_representation
        self.operator = None
        self.params_tpl = namedtuple('StandardQF_parameters', ('size_sg', 'positives_count'))

    def execute(self, task):
        self.operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task.data, task.target)
        result = []
        with self.apply_representation(task.data, task.search_space) as representation:
            self.search_internal(task, result, representation.Conjunction([]))
        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)

    def search_internal(self, task, result, sg):
        statistics = task.qf.calculate_statistics(sg, task.target, task.data)
        if not constraints_satisfied(task.constraints_monotone, sg, statistics, task.data):
            return
        optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)
        if not optimistic_estimate > ps.minimum_required_quality(result, task):
            return
        quality = task.qf.evaluate(sg, task.target, task.data, statistics)
        ps.add_if_required(result, sg, quality, task, statistics=statistics)

        if sg.depth < task.depth:
            for new_sg in self.operator.refinements(sg):
                self.search_internal(task, result, new_sg)


class DFSNumeric:
    tpl = namedtuple('size_mean_parameters', ('size_sg', 'mean'))
    def __init__(self):
        self.pop_size = 0
        self.f = None
        self.target_values = None
        self.bitsets = {}
        self.num_calls = 0

    def execute(self, task):
        if not isinstance(task.qf, ps.StandardQFNumeric):
            warnings.warn("BSD_numeric so far is only implemented for StandardQFNumeric")
        self.pop_size = len(task.data)
        sorted_data = task.data.sort_values(task.target.get_attributes(), ascending=False)

        # generate target bitset
        self.target_values = sorted_data[task.target.get_attributes()[0]].to_numpy()

        task.qf.calculate_constant_statistics(task.data, task.target)

        # generate selector bitsets
        self.bitsets = {}
        for sel in task.search_space:
            # generate bitset
            self.bitsets[sel] = sel.covers(sorted_data)
        result = self.search_internal(task, [], task.search_space, [], np.ones(len(sorted_data), dtype=bool))
        result.sort(key=lambda x: x[0], reverse=True)

        return ps.SubgroupDiscoveryResult(result, task)

    def search_internal(self, task, prefix, modification_set, result, bitset):
        self.num_calls += 1
        sg_size = bitset.sum()
        if sg_size == 0:
            return result
        target_values_sg = self.target_values[bitset]

        target_values_cs = np.cumsum(target_values_sg)
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
                self.search_internal(task, prefix, new_modification_set, result, new_bitset)
                # remove the sel again
                prefix.pop(-1)
        return result
