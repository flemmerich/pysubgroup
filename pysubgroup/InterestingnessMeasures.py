'''
Created on 28.04.2016

@author: lemmerfn
'''
from __future__ import division

class AbstractInterestingnessMeasure(object):
    def optimisticEstimateFromDataset(self, data, subgroup):
        return float("inf")
    
    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return float("inf")
        
class BoundedInterestingnessMeasure:
    pass

class StandardQF (AbstractInterestingnessMeasure, BoundedInterestingnessMeasure):
    @staticmethod     
    def standardQF (a, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        if (instancesSubgroup == 0):
            return 0
        p_subgroup = positivesSubgroup / instancesSubgroup
        p_dataset = positivesDataset / instancesDataset
        return (instancesSubgroup / instancesDataset) ** a * (p_subgroup - p_dataset)
    
    def __init__(self, a):
        self.a = a

    def evaluateFromDataset(self, data, subgroup):
        return self.evaluateFromStatistics (*extractStatisticsFromDataset(data, subgroup))
    
    def optimisticEstimateFromDataset(self, data, subgroup):
        return self.optimisticEstimateFromStatistics (*extractStatisticsFromDataset(data, subgroup))

    def evaluateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return StandardQF.standardQF (self.a, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)

    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return StandardQF.standardQF (self.a, instancesDataset, positivesDataset, positivesSubgroup, positivesSubgroup)


class WRAccQF (StandardQF):
    def __init__(self):
        self.a = 1.0
        
class RelativeGainQF (StandardQF):
    def __init__(self):
        self.a = 0.0
        
class SimpleBinomial(StandardQF):
    def __init__(self):
        self.a = 0.5

def extractStatisticsFromDataset (data, subgroup): 
    instancesDataset = positivesDataset = instancesSubgroup = positivesSubgroup = 0
    for i in data:
        positive = subgroup.target.covers(i)
        instancesDataset += 1
        positivesDataset += positive  # +1 if True
        if (subgroup.subgroupDescription.covers(i)):
            instancesSubgroup += 1
            positivesSubgroup += positive
    return (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)
