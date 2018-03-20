'''
Created on 29.09.2017

@author: lemmerfn
'''
from functools import total_ordering
from pysubgroup.measures import AbstractInterestingnessMeasure, \
    BoundedInterestingnessMeasure

@total_ordering
class FITarget(object):
    def __init__(self):
        pass
        
    def __repr__(self):
        return "T: Frequent Itemsets"
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def getAttributes(self):
        return []
    
    def get_base_statistics (self, data, subgroup, weightingAttribute=None): 
        if (weightingAttribute == None):
            sg_instances = subgroup.subgroupDescription.covers(data)
            return sg_instances.sum()  
        else:
            raise NotImplemented("Attribute weights with numeric targets are not yet implemented.")
        
    def calculateStatistics (self, subgroup, data, weightingAttribute=None):
        if weightingAttribute != None:
            raise NotImplemented("Attribute weights with numeric targets are not yet implemented.")
        sg_instances = subgroup.subgroupDescription.covers(data)
        
        subgroup.statistics['size_sg'] = len(sg_instances)
        subgroup.statistics['size_dataset'] = len (data)
        
class CountQF (AbstractInterestingnessMeasure, BoundedInterestingnessMeasure):
    def __init__(self):
        pass
    
    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        return subgroup.subgroupDescription.covers(data).sum()
    
    def optimisticEstimateFromDataset(self, data, subgroup):
        return subgroup.subgroupDescription.covers(data).sum()
        
    def evaluateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return instancesSubgroup
    
    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return instancesSubgroup

    def isApplicable(self, subgroup):
        return isinstance(subgroup.target, FITarget)

    def supportsWeights(self):
        return False


class AreaQF(AbstractInterestingnessMeasure):
    def __init__(self):
        pass

    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        return len(subgroup.subgroupDescription) * subgroup.subgroupDescription.covers(data).sum()

    def isApplicable(self, subgroup):
        return isinstance(subgroup.target, FITarget)

    def supportsWeights(self):
        return False
