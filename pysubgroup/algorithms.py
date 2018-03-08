'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
from heapq import heappush, heappop
from itertools import islice

from pysubgroup.subgroup import Subgroup, SubgroupDescription
import pysubgroup.utils as ut
import pysubgroup.measures as m



class SubgroupDiscoveryTask(object):
    '''
    Capsulates all parameters required to perform standard subgroup discovery 
    '''
    def __init__(self, data, target, searchSpace, qf, resultSetSize=10, depth=3, minQuality=0, weightingAttribute=None):
            self.data = data
            self.target = target
            self.searchSpace = searchSpace
            self.qf = qf
            self.resultSetSize = resultSetSize
            self.depth = depth
            self.minQuality = minQuality
            self.weightingAttribute = weightingAttribute


class Apriori(object):
    def execute(self, task):
        measure_statistics_based = hasattr(task.qf, 'optimisticEstimateFromStatistics')
        result = []
        
        # init the first level
        next_level_candidates = []
        for sel in task.searchSpace:
            next_level_candidates.append (Subgroup(task.target, [sel]))
            
        # level-wise search
        depth = 1
        while (next_level_candidates):
            # check sgs from the last level
            promising_candidates = []
            for sg in next_level_candidates:
                if (measure_statistics_based):
                    statistics = sg.get_base_statistics(task.data)
                    ut.addIfRequired (result, sg, task.qf.evaluateFromStatistics (*statistics), task)
                    optimistic_estimate = task.qf.optimisticEstimateFromStatistics (*statistics) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")
                else:
                    ut.addIfRequired (result, sg, task.qf.evaluateFromDataset(task.data, sg), task)
                    optimistic_estimate = task.qf.optimisticEstimateFromDataset(task.data, sg) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")
                
                # optimistic_estimate = task.qf.optimisticEstimateFromDataset(task.data, sg) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf") 
                # quality = task.qf.evaluateFromDataset(task.data, sg)
                # ut.addIfRequired (result, sg, quality, task)
                if (optimistic_estimate >= ut.minimumRequiredQuality(result, task)):
                    promising_candidates.append(sg.subgroupDescription.selectors)
            
            if (depth == task.depth):
                break
            
            # generate candidates next level
            next_level_candidates = []
            for i, sg1 in enumerate(promising_candidates):
                for j, sg2 in enumerate (promising_candidates):
                    if (i < j and sg1 [:-1] == sg2 [:-1]):
                        candidate = list(sg1) + [sg2[-1]]
                        # check if ALL generalizations are contained in promising_candidates
                        generalization_descriptions = [[x for x in candidate if x != sel] for sel in candidate]
                        if all (g in promising_candidates for g in generalization_descriptions):
                            next_level_candidates.append(Subgroup (task.target, candidate))
            depth = depth + 1
        
        result.sort(key=lambda x: x[0], reverse=True) 
        return result

class BestFirstSearch (object):
    def execute(self, task):
        result = []
        queue = []
        measure_statistics_based = hasattr(task.qf, 'optimisticEstimateFromStatistics')
        
        
        # init the first level
        for sel in task.searchSpace:
            queue.append ((float("-inf"), [sel]))
        
        while (queue):
            q, candidate_description = heappop(queue)
            q = -q
            if (q) < ut.minimumRequiredQuality(result, task):
                break
            
            sg = Subgroup (task.target, candidate_description)
            
            if (measure_statistics_based):
                statistics = sg.get_base_statistics(task.data)
                ut.addIfRequired (result, sg, task.qf.evaluateFromStatistics (*statistics), task)
                optimistic_estimate = task.qf.optimisticEstimateFromStatistics (*statistics) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")
            else:
                ut.addIfRequired (result, sg, task.qf.evaluateFromDataset(task.data, sg), task)
                optimistic_estimate = task.qf.optimisticEstimateFromDataset(task.data, sg) if isinstance(task.qf, m.BoundedInterestingnessMeasure) else float("inf")
            
            # compute refinements and fill the queue
            if (len (candidate_description) < task.depth and optimistic_estimate >= ut.minimumRequiredQuality(result, task)):
                # iterate over all selectors that are behind the last selector contained in the evaluated candidate according to the initial order
                index_of_last_selector = min (task.searchSpace.index(candidate_description[-1]), len (task.searchSpace) - 1)
                
                for sel in islice(task.searchSpace, index_of_last_selector + 1, None):
                    new_description = candidate_description + [sel]
                    heappush(queue, (-optimistic_estimate, new_description))
        result.sort(key=lambda x: x[0], reverse=True) 
        return result
    
class BeamSearch(object):
    '''
    Implements the BeamSearch algorithm. Its a basic implementation without any optimization, i.e., refinements get tested multiple times.
    '''
    def __init__(self, beamWidth=20, beamWidthAdaptive=False):
        self.beamWidth = beamWidth
        self.beamWidthAdaptive = beamWidthAdaptive
    
    def execute (self, task):
        # adapt beam width to the result set size if desired
        if self.beamWidthAdaptive:
            self.beamWidth = task.resultSetSize
        
        # check if beam size is to small for result set
        if (self.beamWidth < task.resultSetSize):
            raise RuntimeError('Beam width in the beam search algorithm is smaller than the result set size!')
        
        
        # init
        beam = [(0, Subgroup(task.target, []))]
        last_beam = None
        
        depth = 0
        while (beam != last_beam and depth < task.depth):
            last_beam = beam.copy()
            for (_, last_sg) in last_beam:
                for sel in task.searchSpace:
                    # create a clone
                    new_selectors = list(last_sg.subgroupDescription.selectors)
                    if not sel in new_selectors:
                        new_selectors.append(sel)
                        sg = Subgroup(task.target, new_selectors)
                        quality = task.qf.evaluateFromDataset (task.data, sg)
                        ut.addIfRequired(beam, sg, quality, task, check_for_duplicates=True)
            depth += 1

        result = beam [:task.resultSetSize]
        result.sort(key=lambda x: x[0], reverse=True) 
        return result
        
class SimpleDFS(object):
    def execute (self, task, useOptimisticEstimates=True):
        result = self.searchInternal(task, [], task.searchSpace, [], useOptimisticEstimates)
        result.sort(key=lambda x: x[0], reverse=True)
        return result
    
    def searchInternal(self, task, prefix, modificationSet, result, useOptimisticEstimates):
        sg = Subgroup(task.target, SubgroupDescription(copy.copy(prefix)))
        
        optimisticEstimate = float("inf")
        if useOptimisticEstimates and len(prefix) < task.depth and isinstance(task.qf, m.BoundedInterestingnessMeasure):
            optimisticEstimate = task.qf.optimisticEstimateFromDataset(task.data, sg)
            if (optimisticEstimate <= ut.minimumRequiredQuality(result, task)):
                return result
        
        if task.qf.supportsWeights():
            quality = task.qf.evaluateFromDataset(task.data, sg, task.weightingAttribute)
        else: 
            quality = task.qf.evaluateFromDataset(task.data, sg)
        ut.addIfRequired (result, sg, quality, task)
     
        if (len(prefix) < task.depth):
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                newModificationSet.pop(0)
                self.searchInternal(task, prefix, newModificationSet, result, useOptimisticEstimates)
                # remove the sel again
                prefix.pop(-1)
        return result
