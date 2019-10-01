'''
Created on 29.04.2016

@author: lemmerfn
'''
import copy
import pysubgroup as ps
from bitarray import bitarray

class BSD_Bitarray(object):
    
    def execute (self, task):
        self.popSize = len(task.data)
        
        # generate target bitset
        self.targetBitset = bitarray(self.popSize)
        for index, row in task.data.iterrows():
            self.targetBitset[index] = task.target.covers(row)
        self.popPositives = self.targetBitset.count()

        # generate selector bitsets
        self.bitsets = {}
        for sel in task.search_space:
            # generate bitset
            selBitset = bitarray(self.popSize)
            for index, row in task.data.iterrows():
                selBitset[index] = sel.covers(row)
            self.bitsets[sel] = selBitset
        result = self.search_internal(task, [], task.search_space, [], self.popSize * bitarray('1'))
        result.sort(key=lambda x: x[0], reverse=True)
        return result
    
    
    def search_internal(self, task, prefix, modificationSet, result, bitset):
        sg = ps.Subgroup(task.target, copy.copy(prefix))
        
        sgSize = bitset.count()
        positiveInstances = bitset & self.targetBitset
        sgPositiveCount = positiveInstances.count()
         
        optimisticEstimate = task.qf.optimistic_estimate_from_statistics (self.popSize, self.popPositives, sgSize, sgPositiveCount)
        if (optimisticEstimate <= ps.minimum_required_quality(result, task)):
            return result
        
        quality = task.qf.evaluate_from_statistics(self.popSize, self.popPositives, sgSize, sgPositiveCount)
        ps.add_if_required (result, sg, quality, task)
     
        if len(prefix) < task.depth:
            newModificationSet = copy.copy(modificationSet)
            for sel in modificationSet:
                prefix.append(sel)
                newBitset = bitset & self.bitsets [sel]
                newModificationSet.pop(0)
                self.search_internal(task, prefix, newModificationSet, result, newBitset)
                # remove the sel again
                prefix.pop(-1)
        return result
