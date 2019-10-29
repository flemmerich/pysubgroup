'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
import functools
from itertools import combinations, chain
from heapq import heappush, heappop
from collections import Counter, namedtuple

import numpy as np
import pysubgroup as ps


class SubgroupDiscoveryTask:
    '''
    Capsulates all parameters required to perform standard subgroup discovery
    '''

    def __init__(self, data, target, search_space, qf, result_set_size=10, depth=3, min_quality=0, weighting_attribute=None):
        self.data = data
        self.target = target
        self.search_space = search_space
        self.qf = qf
        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality
        self.weighting_attribute = weighting_attribute


class Apriori:
    def __init__(self, apply_representation=None, combination=None):
        if combination is None:
            combination = ps.RepresentationConjunction
        self.combination = combination
        if apply_representation is None:
            apply_representation = ps.BitSetRepresentation
        self.apply_representation = apply_representation
    def execute(self, task):
        if not isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            raise RuntimeWarning("Quality function is unbounded, long runtime expected")
        with self.apply_representation(task.data):
            result = []
            task.qf.calculate_constant_statistics(task)
            # init the first level
            next_level_candidates = []
            for sel in task.search_space:
                next_level_candidates.append(ps.Subgroup(task.target, self.combination([sel])))

            # level-wise search
            depth = 1
            while next_level_candidates:
                # check sgs from the last level
                promising_candidates = []
                for sg in next_level_candidates:
                    statistics = task.qf.calculate_statistics(sg, task.data)
                    ps.add_if_required(result, sg, task.qf.evaluate(sg, statistics), task)
                    optimistic_estimate = task.qf.optimistic_estimate(sg, statistics)

                    if optimistic_estimate >= ps.minimum_required_quality(result, task):
                        promising_candidates.append(list(sg.subgroup_description._selectors))

                if depth == task.depth:
                    break
                set_promising_candidates = set(frozenset(p) for p in promising_candidates)
                # print(len(promising_candidates))
                # 
                # for k in range(1, depth):
                #     promising_candidates_k = (combinations(selectors, k) for selectors in promising_candidates)
                #     tmp = Counter(chain.from_iterable(promising_candidates_k))
                #     d = depth + 1 - k
                #     tmp2 = set(frozenset(sel) for sel, count in tmp.items() if count < d)
                #     promising_candidates = list(selectors for selectors in promising_candidates if all(frozenset(comb) not in tmp2 for comb in combinations(selectors, k)))
                #     print(len(tmp2))
                # print()
                combine = self.combination
                #promising_candidates = list(selectors for selectors in promising_candidates if all(sel not in tmp2 for sel in selectors))
                l = list((sg, [sg[-1]], hash(tuple(sg[:-1])), sg[:-1]) for sg in promising_candidates)
                next_level_candidates_no_pruning = (sg1 + n_r for (sg1, _, hash_l, l_l), (_, n_r, hash_r, l_r) in combinations(l, 2)
                                                    if (hash_l == hash_r) and (l_l == l_r))
                #next_level_candidates_no_pruning = (sg1 + [sg2[-1]] for sg1, sg2 in combinations(promising_candidates, 2) if sg1[:-1] == sg2[:-1])
                # select those selectors and build a subgroup from them
                #   for which all subsets of length depth (=candidate length -1) are in the set of promising candidates
                
                next_level_candidates = [ps.Subgroup(task.target, combine(selectors)) for selectors in next_level_candidates_no_pruning
                                        if all((frozenset(subset) in set_promising_candidates) for subset in combinations(selectors, depth))]
                depth = depth + 1

        result.sort(key=lambda x: x[0], reverse=True)
        return result


class BestFirstSearch:
    def execute(self, task):
        result = []
        queue = []
        operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task)
        # init the first level
        for sel in task.search_space:
            queue.append((float("-inf"), ps.Conjunction([sel])))
        
        while queue:
            q, candidate_description = heappop(queue)
            q = -q
            if q < ps.minimum_required_quality(result, task):
                break
            sg = ps.Subgroup(task.target, candidate_description)
            statistics = task.qf.calculate_statistics(sg, task.data)
            ps.add_if_required(result, sg, task.qf.evaluate(sg, statistics), task)
            optimistic_estimate = task.qf.optimistic_estimate(sg, statistics)


            # compute refinements and fill the queue
            if len(candidate_description) < task.depth and optimistic_estimate >= ps.minimum_required_quality(result, task):
                #print(ps.minimum_required_quality(result, task))
                for new_description in operator.refinements(candidate_description):
                    heappush(queue, (-optimistic_estimate, new_description))

        result.sort(key=lambda x: x[0], reverse=True)
        return result


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
        task.qf.calculate_constant_statistics(task)

        while queue:
            q, candidate_description = heappop(queue)
            q = -q
            if q < ps.minimum_required_quality(result, task):
                break

            sg = ps.Subgroup(task.target, candidate_description)
            statistics = task.qf.calculate_statistics(sg, task.data)
            quality = task.qf.evaluate(sg, statistics)
            ps.add_if_required(result, sg, quality, task)
            

            qual = ps.minimum_required_quality(result, task)

            if (quality, sg) in result:
                #print(qual)
                # print(queue)
                new_queue = []
                for q_tmp, c_tmp in queue:
                    if (-q_tmp) > qual:
                        heappush(new_queue, (q_tmp, c_tmp))
                queue = new_queue
                # print(queue)
            optimistic_estimate = task.qf.optimistic_estimate(sg, statistics)
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
            print("{} {}".format(qual, sg.subgroup_description))
        print("discarded " + str(self.discarded))
        return result


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

        task.qf.calculate_constant_statistics(task)

        # init
        beam = [(0, ps.Subgroup(task.target, ps.Conjunction([])))]
        last_beam = None

        depth = 0
        while beam != last_beam and depth < task.depth:
            last_beam = beam.copy()
            for (_, last_sg) in last_beam:
                if not getattr(last_sg, 'visited', False):
                    setattr(last_sg, 'visited', True)
                    for sel in task.search_space:
                        # create a clone
                        new_selectors = list(last_sg.subgroup_description._selectors)
                        if sel not in new_selectors:
                            new_selectors.append(sel)
                            sg = ps.Subgroup(task.target, ps.Conjunction(new_selectors))
                            quality = task.qf.evaluate(sg, task.data)
                            ps.add_if_required(beam, sg, quality, task, check_for_duplicates=True)
            depth += 1

        result = beam[:task.result_set_size]
        result.sort(key=lambda x: x[0], reverse=True)
        return result


class SimpleDFS:
    def execute(self, task, use_optimistic_estimates=True):
        task.qf.calculate_constant_statistics(task)
        result = self.search_internal(task, [], task.search_space, [], use_optimistic_estimates)
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def search_internal(self, task, prefix, modification_set, result, use_optimistic_estimates):
        sg = ps.Subgroup(task.target, ps.Conjunction(copy.copy(prefix)))
        
        if use_optimistic_estimates and len(prefix) < task.depth and isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            optimistic_estimate = task.qf.optimistic_estimate_from_dataset(task.data, sg)
            if optimistic_estimate <= ps.minimum_required_quality(result, task):
                return result


        quality = task.qf.evaluate(sg, task.data)
        ps.add_if_required(result, sg, quality, task)

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
        self.params_tpl = namedtuple('StandardQF_parameters', ('size', 'positives_count'))

    def execute(self, task):
        self.target_bitset = task.target.covers(task.data)
        self.operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task)
        result = []
        with self.apply_representation(task.data):
            self.search_internal(task, result, ps.RepresentationConjunction([]))
        result.sort(key=lambda x: x[0], reverse=True)
        result = [(quality, ps.Subgroup(task, sgd)) for quality, sgd in result]
        return result

    def search_internal(self, task, result, sg):
        params = self.params_tpl(sg.size, np.count_nonzero(self.target_bitset[sg]))

        optimistic_estimate = task.qf.optimistic_estimate(sg, params)
        if optimistic_estimate <= ps.minimum_required_quality(result, task):
            return
        quality = task.qf.evaluate(sg, params)
        ps.add_if_required(result, sg, quality, task)

        if len(sg) < task.depth:
            for new_sg in self.operator.refinements(sg):
                self.search_internal(task, result, new_sg)


class DFSNumeric():
    def __init__(self):
        self.pop_size = 0
        self.f = None
        self.target_values = None
        self.bitsets = {}

    def execute(self, task):
        if not isinstance(task.qf, ps.StandardQFNumeric):
            raise NotImplementedError("BSD_numeric so far is only implemented for StandardQFNumeric")
        self.pop_size = len(task.data)
        sorted_data = task.data.sort_values(task.target.get_attributes(), ascending=False)

        # generate target bitset
        self.target_values = sorted_data[task.target.get_attributes()[0]].values

        f = functools.partial(task.qf.evaluate_from_statistics, len(sorted_data), self.target_values.mean())
        self.f = np.vectorize(f)

        # generate selector bitsets
        self.bitsets = {}
        for sel in task.search_space:
            # generate bitset
            self.bitsets[sel] = sel.covers(sorted_data)
        result = self.search_internal(task, [], task.search_space, [], np.ones(len(sorted_data), dtype=bool))
        result.sort(key=lambda x: x[0], reverse=True)

        return result

    def search_internal(self, task, prefix, modification_set, result, bitset):
        sg_size = bitset.sum()
        if sg_size == 0:
            return result
        target_values_sg = self.target_values[bitset]

        target_values_cs = np.cumsum(target_values_sg)
        mean_values_cs = target_values_cs / (np.arange(len(target_values_cs)) + 1)
        qualities = self.f(np.arange(len(target_values_cs)) + 1, mean_values_cs)
        optimistic_estimate = np.max(qualities)

        if optimistic_estimate <= ps.minimum_required_quality(result, task):
            return result

        sg = ps.Subgroup(task.target, ps.Conjunction(copy.copy(prefix)))

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
