'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
import pysubgroup.InterestingnessMeasures
import pysubgroup.SGDUtils
from pysubgroup.Subgroup import Subgroup

class SimpleDFS(object):
    
    def execute (self, task):
        return self.searchInternal(task, [], task.searchSpace, [])
    
    def searchInternal(self, task, prefix, modificationSet, result):
        sg = Subgroup(task.target, copy.copy(prefix))
        
        optimisticEstimate = float("inf")
        if isinstance(task.qf, pysubgroup.InterestingnessMeasures.BoundedInterestingnessMeasure):
            optimisticEstimate = task.qf.optimisticEstimateFromDataset(task.data, sg)
        if (optimisticEstimate <= pysubgroup.SGDUtils.minimumRequiredQuality(result, task)):
            return result
        
        if task.qf.supportsWeights():
            quality = task.qf.evaluateFromDataset(task.data, sg, task.weightingAttribute)
        else: 
            quality = task.qf.evaluateFromDataset(task.data, sg)
        pysubgroup.SGDUtils.addIfRequired (result, sg, quality, task)
     
        if (len(prefix) < task.depth):
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                newModificationSet.pop(0)
                self.searchInternal(task, prefix, newModificationSet, result)
                # remove the sel again
                prefix.pop(-1)
        

        result.sort(key=lambda x: x[0])
        return result
