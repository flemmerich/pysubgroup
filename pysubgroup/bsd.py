'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
import pysubgroup.utils as ut
from pysubgroup.subgroup import Subgroup
from bitarray import bitarray

class BSD_Bitarray(object):
    
    def execute (self, task):
        self.popSize = len(task.data)
        
        # generate target bitset
        self.targetBitset = bitarray(self.popSize)
        for i, inst in enumerate(task.data):
            self.targetBitset[i] = task.target.covers(inst)
        self.popPositives = self.targetBitset.count()

        # generate selector bitsets
        self.bitsets = {}
        for sel in task.searchSpace:
            # generate bitset
            selBitset = bitarray(self.popSize)
            for i, inst in enumerate(task.data):
                selBitset[i] = sel.covers(inst)
            self.bitsets [sel] = selBitset
        result = self.searchInternal(task, [], task.searchSpace, [], self.popSize * bitarray('1'))
        return result.sort(key=lambda x: x[0], reverse=True)
    
    
    def searchInternal(self, task, prefix, modificationSet, result, bitset):
        sg = Subgroup(task.target, copy.copy(prefix))
        
        sgSize = bitset.count()
        positiveInstances = bitset & self.targetBitset
        sgPositiveCount = positiveInstances.count()
         
        optimisticEstimate = task.qf.optimisticEstimateFromStatistics (self.popSize, self.popPositives, sgSize, sgPositiveCount)
        if (optimisticEstimate <= ut.minimumRequiredQuality(result, task)):
            return result
        
        quality = task.qf.evaluateFromStatistics (self.popSize, self.popPositives, sgSize, sgPositiveCount) 
        ut.addIfRequired (result, sg, quality, task)
     
        if (len(prefix) < task.depth):
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                newBitset = bitset & self.bitsets [sel]
                newModificationSet.pop(0)
                self.searchInternal(task, prefix, newModificationSet, result, newBitset)
                # remove the sel again
                prefix.pop(-1)
        return result
