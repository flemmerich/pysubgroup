'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
from heapq import heappush

from pysubgroup import measures
from pysubgroup.subgroup import Subgroup, SubgroupDescription
import pysubgroup.utils as ut


class SubgroupDiscoveryTask(object):
    def __init__(self, data, target, searchSpace, qf=measures.WRAccQF(), resultSetSize=10, depth=3, minQuality=0, weightingAttribute=None):
        self.data = data
        self.target = target
        self.searchSpace = searchSpace
        self.qf = qf
        self.resultSetSize = resultSetSize
        self.depth = depth
        self.minQuality = minQuality
        self.weightingAttribute = weightingAttribute

class BeamSearch(object):
    '''
    Implements the BeamSearch algorithm
    '''
    def __init__(self, beamWidth, beamWidthAdaptive=False):
        self.beamWidth = beamWidth
        self.beamWidthAdaptive = beamWidthAdaptive
    
    def execute (self, task):
        if self.beamWidthAdaptive:
            self.beamWidth = task.resultSetSize
            
        beam = []
        
        for sel in task.searchSpace:
            sg = Subgroup(task.target, [sel])
            quality = task.qf.evaluateFromDataset (task.data, sg)
            heappush (beam, (quality, sg))
        print (beam)
        
class SimpleDFS(object):
    
    def execute (self, task, useOptimisticEstimates=True):
        result = self.searchInternal(task, [], task.searchSpace, [], useOptimisticEstimates)
        result.sort(key=lambda x: x[0], reverse=True)
        return result
    
    def searchInternal(self, task, prefix, modificationSet, result, useOptimisticEstimates):
        sg = Subgroup(task.target, SubgroupDescription(copy.copy(prefix)))
        
        optimisticEstimate = float("inf")
        if useOptimisticEstimates and len(prefix) < task.depth and isinstance(task.qf, measures.BoundedInterestingnessMeasure):
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
