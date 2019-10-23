'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
import functools
from itertools import combinations
from heapq import heappush, heappop

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
    def execute(self, task):
        measure_statistics_based = hasattr(task.qf, 'optimistic_estimate_from_statistics')
        if not isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            raise RuntimeWarning("Quality function is unbounded, long runtime expected")
        result = []

        # init the first level
        next_level_candidates = []
        for sel in task.search_space:
            next_level_candidates.append(ps.Subgroup(task.target, [sel]))

        # level-wise search
        depth = 1
        while next_level_candidates:
            # check sgs from the last level
            promising_candidates = []
            for sg in next_level_candidates:
                if measure_statistics_based:
                    statistics = sg.get_base_statistics(task.data)
                    ps.add_if_required(result, sg, task.qf.evaluate_from_statistics(*statistics), task)
                    optimistic_estimate = task.qf.optimistic_estimate_from_statistics(*statistics) if isinstance(task.qf, ps.BoundedInterestingnessMeasure) else float("inf")
                else:
                    ps.add_if_required(result, sg, task.qf.evaluate_from_dataset(task.data, sg), task)
                    optimistic_estimate = task.qf.optimistic_estimate_from_dataset(task.data, sg) if isinstance(task.qf, ps.BoundedInterestingnessMeasure) else float("inf")

                if optimistic_estimate >= ps.minimum_required_quality(result, task):
                    promising_candidates.append(list(sg.subgroup_description.selectors))

            if depth == task.depth:
                break
            print(len(promising_candidates))
            set_promising_candidates=set(frozenset(p) for p in promising_candidates)
            # generate candidates next level
            
            next_level_candidates_no_pruning=(sg1+[sg2[-1]] for sg1, sg2 in combinations(promising_candidates,2) if sg1[:-1] == sg2[:-1])
            next_level_candidates = [ps.Subgroup(task.target, selectors) for selectors in next_level_candidates_no_pruning  
                                     if all( (frozenset(g) in set_promising_candidates) for g in combinations(selectors, depth))]           
            depth = depth + 1

        result.sort(key=lambda x: x[0], reverse=True)
        return result


class BestFirstSearch:
    def execute(self, task):
        result = []
        queue = []
        operator=ps.StaticSpecializationOperator(task.search_space)
        measure_statistics_based = hasattr(task.qf, 'optimistic_estimate_from_statistics')
        qf_is_bounded = isinstance(task.qf, ps.BoundedInterestingnessMeasure)
        # init the first level
        for sel in task.search_space:
            queue.append((float("-inf"), ps.SubgroupDescription([sel])))

        while queue:
            q, candidate_description = heappop(queue)
            q = -q
            if q < ps.minimum_required_quality(result, task):
                break

            sg = ps.Subgroup(task.target, candidate_description)

            if measure_statistics_based:
                statistics = sg.get_base_statistics(task.data)
                ps.add_if_required(result, sg, task.qf.evaluate_from_statistics(*statistics), task)
                optimistic_estimate = task.qf.optimistic_estimate_from_statistics(*statistics) if qf_is_bounded else float("inf")
            else:
                ps.add_if_required(result, sg, task.qf.evaluate_from_dataset(task.data, sg), task)
                optimistic_estimate = task.qf.optimistic_estimate_from_dataset(task.data, sg) if qf_is_bounded else float("inf")

            # compute refinements and fill the queue
            if len(candidate_description) < task.depth and optimistic_estimate >= ps.minimum_required_quality(result, task):
                for new_description in operator.refinements(candidate_description):
                    heappush(queue, (-optimistic_estimate, new_description))
        result.sort(key=lambda x: x[0], reverse=True)
        return result


class GeneralisingBFS:
    def __init__(self):
        self.alpha=1.10
        self.discarded=[0,0,0,0,0,0,0]
        self.refined=[0,0,0,0,0,0,0]

    def execute(self, task):
        result = []
        queue = []
        operator=ps.StaticGeneralizationOperator(task.search_space)
        # init the first level
        for sel in task.search_space:
            queue.append((float("-inf"), ps.Disjunction([sel])))

        while queue:
            q, candidate_description = heappop(queue)
            q = -q
            if q < ps.minimum_required_quality(result, task):
                break

            sg = ps.Subgroup(task.target, candidate_description)

            #if measure_statistics_based:
            statistics = sg.get_base_statistics(task.data)
            quality=task.qf.evaluate_from_statistics(*statistics)
            ps.add_if_required(result, sg, quality, task)

            
            qual=ps.minimum_required_quality(result, task)

            
            if (quality, sg) in result:
                print(qual)
                #print(queue)
                new_queue=[]
                for q_tmp, c_tmp in queue:
                    if (-q_tmp) > qual:
                        heappush(new_queue,(q_tmp,c_tmp))
                queue=new_queue
                #print(queue)
            optimistic_estimate = task.qf.optimistic_generalisation_from_statistics(*statistics)# if qf_is_bounded else float("inf")
            #else:
            #    ps.add_if_required(result, sg, task.qf.evaluate_from_dataset(task.data, sg), task)
            #    optimistic_estimate = task.qf.optimistic_generalisation_from_dataset(task.data, sg) if qf_is_bounded else float("inf")

            # compute refinements and fill the queue
            if len(candidate_description) < task.depth and (optimistic_estimate / self.alpha ** (len(candidate_description)+1)) >= ps.minimum_required_quality(result, task):
                #print(qual)
                #print(optimistic_estimate)
                self.refined[len(candidate_description)]+=1
                #print(str(candidate_description))
                for new_description in operator.refinements(candidate_description):
                    heappush(queue, (-optimistic_estimate, new_description))
            else:
                self.discarded[len(candidate_description)]+=1
        
        result.sort(key=lambda x: x[0], reverse=True)
        for qual,sg in result:
            print("{} {}".format(qual,sg.subgroup_description))
        print("discarded "+str(self.discarded))
        return result


class BeamSearch:
    '''
    Implements the BeamSearch algorithm. Its a basic implementation without any optimization, i.e., refinements get tested multiple times.
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

        # init
        beam = [(0, ps.Subgroup(task.target, []))]
        last_beam = None

        depth = 0
        while beam != last_beam and depth < task.depth:
            last_beam = beam.copy()
            for (_, last_sg) in last_beam:
                if getattr(last_sg,'visited',False) == False:
                    setattr(last_sg,'visited',True)
                    for sel in task.search_space:
                        # create a clone
                        new_selectors = list(last_sg.subgroup_description.selectors)
                        if sel not in new_selectors:
                            new_selectors.append(sel)
                            sg = ps.Subgroup(task.target, new_selectors)
                            quality = task.qf.evaluate_from_dataset(task.data, sg)
                            ps.add_if_required(beam, sg, quality, task, check_for_duplicates=True)
            depth += 1

        result = beam[:task.result_set_size]
        result.sort(key=lambda x: x[0], reverse=True)
        return result


class SimpleDFS:
    def execute(self, task, use_optimistic_estimates=True):
        result = self.search_internal(task, [], task.search_space, [], use_optimistic_estimates)
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def search_internal(self, task, prefix, modification_set, result, use_optimistic_estimates):
        sg = ps.Subgroup(task.target, ps.SubgroupDescription(copy.copy(prefix)))

        if use_optimistic_estimates and len(prefix) < task.depth and isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            optimistic_estimate = task.qf.optimistic_estimate_from_dataset(task.data, sg)
            if optimistic_estimate <= ps.minimum_required_quality(result, task):
                return result

        if task.qf.supports_weights():
            quality = task.qf.evaluate_from_dataset(task.data, sg, task.weighting_attribute)
        else:
            quality = task.qf.evaluate_from_dataset(task.data, sg)
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

    def __init__(self, structure):
        self.pop_positives = 0
        self.pop_size = 0
        self.target_bitset = None
        self.structure=structure

    def execute(self, task):
        self.pop_size = len(task.data)
        self.target_bitset = task.target.covers(task.data)
        self.pop_positives = self.target_bitset.sum()
        with self.structure(task.data):
            result = self.search_internal(task, task.search_space, [], ps.SubgroupDescription([]))
        result.sort(key=lambda x: x[0], reverse=True)
        result=[(quality,ps.Subgroup(task,ps.SubgroupDescription(sG.selectors))) for quality, sG in result]
        return result

    def search_internal(self, task, modification_set, result, sg):

        sg_size = sg.size
        sg_positive_count = self.target_bitset[sg].sum()

        optimistic_estimate = task.qf.optimistic_estimate_from_statistics(self.pop_size, self.pop_positives, sg_size,
                                                                          sg_positive_count)
        if optimistic_estimate <= ps.minimum_required_quality(result, task):
            return result
        quality = task.qf.evaluate_from_statistics(self.pop_size, self.pop_positives, sg_size, sg_positive_count)
        ps.add_if_required(result, sg, quality, task)

        if len(sg) < task.depth:
            new_modification_set = copy.copy(modification_set)
            for sel in modification_set:
                new_sg=copy.copy(sg)
                new_sg.append_and(sel)
                new_modification_set.pop(0)
                self.search_internal(task, new_modification_set, result, new_sg)
        return result


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

        sg = ps.Subgroup(task.target, copy.copy(prefix))

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
