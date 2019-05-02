'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
import functools
from heapq import heappush, heappop
from itertools import islice

import numpy as np
import pysubgroup as ps


class SubgroupDiscoveryTask(object):
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


class Apriori(object):
    def execute(self, task):
        measure_statistics_based = hasattr(task.qf, 'optimistic_estimate_from_statistics')
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
                
                # optimistic_estimate = task.qf.optimistic_estimate_from_dataset(task.data, sg)
                # if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")
                # quality = task.qf.evaluate_from_dataset(task.data, sg)
                # ut.add_if_required (result, sg, quality, task)
                if optimistic_estimate >= ps.minimum_required_quality(result, task):
                    promising_candidates.append(sg.subgroup_description.selectors)
            
            if depth == task.depth:
                break
            
            # generate candidates next level
            next_level_candidates = []
            for i, sg1 in enumerate(promising_candidates):
                for j, sg2 in enumerate (promising_candidates):
                    if i < j and sg1 [:-1] == sg2[:-1]:
                        candidate = list(sg1) + [sg2[-1]]
                        # check if ALL generalizations are contained in promising_candidates
                        generalization_descriptions = [[x for x in candidate if x != sel] for sel in candidate]
                        if all(g in promising_candidates for g in generalization_descriptions):
                            next_level_candidates.append(ps.Subgroup(task.target, candidate))
            depth = depth + 1
        
        result.sort(key=lambda x: x[0], reverse=True) 
        return result


class BestFirstSearch (object):
    def execute(self, task):
        result = []
        queue = []
        measure_statistics_based = hasattr(task.qf, 'optimistic_estimate_from_statistics')

        # init the first level
        for sel in task.search_space:
            queue.append((float("-inf"), [sel]))
        
        while queue:
            q, candidate_description = heappop(queue)
            q = -q
            if q < ps.minimum_required_quality(result, task):
                break
            
            sg = ps.Subgroup(task.target, candidate_description)
            
            if measure_statistics_based:
                statistics = sg.get_base_statistics(task.data)
                ps.add_if_required(result, sg, task.qf.evaluate_from_statistics(*statistics), task)
                optimistic_estimate = task.qf.optimistic_estimate_from_statistics(*statistics) if isinstance(task.qf, ps.BoundedInterestingnessMeasure) else float("inf")
            else:
                ps.add_if_required(result, sg, task.qf.evaluate_from_dataset(task.data, sg), task)
                optimistic_estimate = task.qf.optimistic_estimate_from_dataset(task.data, sg) if isinstance(task.qf, ps.BoundedInterestingnessMeasure) else float("inf")
            
            # compute refinements and fill the queue
            if len (candidate_description) < task.depth and optimistic_estimate >= ps.minimum_required_quality(result, task):
                # iterate over all selectors that are behind the last selector contained in the evaluated candidate
                # according to the initial order
                index_of_last_selector = min(task.search_space.index(candidate_description[-1]), len(task.search_space) - 1)
                
                for sel in islice(task.search_space, index_of_last_selector + 1, None):
                    new_description = candidate_description + [sel]
                    heappush(queue, (-optimistic_estimate, new_description))
        result.sort(key=lambda x: x[0], reverse=True) 
        return result


class BeamSearch(object):
    '''
    Implements the BeamSearch algorithm. Its a basic implementation without any optimization, i.e., refinements get tested multiple times.
    '''
    def __init__(self, beam_width=20, beam_width_adaptive=False):
        self.beam_width = beam_width
        self.beam_width_adaptive = beam_width_adaptive
    
    def execute (self, task):
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
                for sel in task.search_space:
                    # create a clone
                    new_selectors = list(last_sg.subgroup_description.selectors)
                    if not sel in new_selectors:
                        new_selectors.append(sel)
                        sg = ps.Subgroup(task.target, new_selectors)
                        quality = task.qf.evaluate_from_dataset (task.data, sg)
                        ps.add_if_required(beam, sg, quality, task, check_for_duplicates=True)
            depth += 1

        result = beam[:task.result_set_size]
        result.sort(key=lambda x: x[0], reverse=True) 
        return result


class SimpleDFS(object):
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


class BSD (object):
    """
    Implementation of the BSD algorithm for binary targets. See
    Lemmerich, Florian, Mathias Rohlfs, and Martin Atzmueller. "Fast Discovery of Relevant Subgroup Patterns.",
    FLAIRS Conference. 2010.
    """
    def execute(self, task):
        self.pop_size = len(task.data)
        self.target_bitset = task.target.covers(task.data)
        self.pop_positives = self.target_bitset.sum()
        self.bitsets = {}
        for sel in task.search_space:
            self.bitsets[sel] = sel.covers(task.data).values

        result = self.search_internal(task, [], task.search_space, [], np.ones(self.pop_size, dtype=bool))
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def search_internal(self, task, prefix, modification_set, result, bitset):

        sg_size = bitset.sum()
        positive_instances = np.logical_and(bitset, self.target_bitset)
        sg_positive_count = positive_instances.sum()

        optimistic_estimate = task.qf.optimistic_estimate_from_statistics(self.pop_size, self.pop_positives, sg_size,
                                                                          sg_positive_count)
        if optimistic_estimate <= ps.minimum_required_quality(result, task):
            return result

        sg = ps.Subgroup(task.target, copy.copy(prefix))

        quality = task.qf.evaluate_from_statistics(self.pop_size, self.pop_positives, sg_size, sg_positive_count)
        ps.add_if_required(result, sg, quality, task)

        if len(prefix) < task.depth:
            new_modification_set = copy.copy(modification_set)
            for sel in modification_set:
                prefix.append(sel)
                newBitset = np.logical_and(bitset, self.bitsets[sel])
                new_modification_set.pop(0)
                self.search_internal(task, prefix, new_modification_set, result, newBitset)
                # remove the sel again
                prefix.pop(-1)
        return result


class TID_SD (object):
    """
    Implementation of a depth-first-search with look-ahead using vertical ID lists as data structure.
    """

    def execute(self, task, use_sets=False):
        self.popSize = len(task.data)

        # generate target bitset
        x = task.target.covers(task.data)
        if use_sets:
            self.targetBitset = set (x.nonzero()[0])
        else:
            self.targetBitset = list(x.nonzero()[0])


        self.popPositives = len(self.targetBitset)

        # generate selector bitsets
        self.bitsets = {}
        for sel in task.search_space:
            # generate data structure
            x = task.target.covers(task.data)
            if use_sets:
                sel_bitset = set (x.nonzero()[0])
            else:
                sel_bitset = list(x.nonzero()[0])
            self.bitsets[sel] = sel_bitset
        if use_sets:
            result = self.search_internal(task, [], task.search_space, [], set(range(self.popSize)), use_sets)
        else:
            result = self.search_internal(task, [], task.search_space, [], list(range(self.popSize)), use_sets)
        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def search_internal(self, task, prefix, modificationSet, result, bitset, use_sets):

        sgSize = len(bitset)
        if use_sets:
            positiveInstances = bitset & self.targetBitset
        else:
            positiveInstances = ps.intersect_of_ordered_list(bitset, self.targetBitset)
        sgPositiveCount = len(positiveInstances)

        optimisticEstimate = task.qf.optimistic_estimate_from_statistics(self.popSize, self.popPositives, sgSize,
                                                                      sgPositiveCount)
        if (optimisticEstimate <= ps.minimum_required_quality(result, task)):
            return result

        sg = ps.Subgroup(task.target, copy.copy(prefix))

        quality = task.qf.evaluate_from_statistics(self.popSize, self.popPositives, sgSize, sgPositiveCount)
        ps.add_if_required(result, sg, quality, task)

        if (len(prefix) < task.depth):
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                if use_sets:
                    newBitset = bitset & self.bitsets[sel]
                else:
                    newBitset = ps.intersect_of_ordered_list(bitset, self.bitsets[sel])
                newModificationSet.pop(0)
                self.search_internal(task, prefix, newModificationSet, result, newBitset, use_sets)
                # remove the sel again
                prefix.pop(-1)
        return result


class DFSNumeric(object):
    def execute(self, task):
        if not isinstance (task.qf, ps.StandardQFNumeric):
            NotImplemented("BSD_numeric so far is only implemented for StandardQFNumeric")
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
            self.bitsets[sel] = sel.covers(sorted_data).values
        result = self.search_internal(task, [], task.search_space, [], np.ones(len(sorted_data), dtype=bool))
        result.sort(key=lambda x: x[0], reverse=True)

        return result

    def search_internal(self, task, prefix, modification_set, result, bitset):
        sg_size = bitset.sum()
        if sg_size == 0:
            return
        target_values_sg = self.target_values[bitset]

        target_values_cs = np.cumsum(target_values_sg)
        mean_values_cs = target_values_cs / (np.arange(len(target_values_cs)) + 1)
        qualities = self.f (np.arange(len(target_values_cs)) + 1, mean_values_cs)
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
