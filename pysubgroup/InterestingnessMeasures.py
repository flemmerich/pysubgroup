'''
Created on 28.04.2016

@author: lemmerfn
'''
from pysubgroup import SGDUtils

class AbstractInterestingnessMeasure(object):
    def optimisticEstimateFromDataset(self, data, subgroup):
        return float("inf")
    
    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return float("inf")
    
    def supportsWeights(self):
        return False
        
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

    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        return self.evaluateFromStatistics (*SGDUtils.extractStatisticsFromDataset(data, subgroup, weightingAttribute))
    
    def optimisticEstimateFromDataset(self, data, subgroup):
        return self.optimisticEstimateFromStatistics (*SGDUtils.extractStatisticsFromDataset(data, subgroup))

    def evaluateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return StandardQF.standardQF (self.a, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)

    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return StandardQF.standardQF (self.a, instancesDataset, positivesDataset, positivesSubgroup, positivesSubgroup)

    def supportsWeights(self):
        return True

class WRAccQF (StandardQF):
    def __init__(self):
        self.a = 1.0
        
class RelativeGainQF (StandardQF):
    def __init__(self):
        self.a = 0.0
        
class SimpleBinomial(StandardQF):
    def __init__(self):
        self.a = 0.5
