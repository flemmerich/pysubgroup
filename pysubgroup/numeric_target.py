'''
Created on 29.09.2017

@author: lemmerfn
'''
import numpy as np
import pysubgroup as ps
from functools import total_ordering
from pysubgroup.measures import AbstractInterestingnessMeasure, \
    BoundedInterestingnessMeasure
from pysubgroup.utils import conditional_invert



@total_ordering
class NumericTarget(object):
    def __init__(self, target_variable):
        self.target_variable = target_variable
        
    def __repr__(self):
        return "T: " + str(self.target_variable)
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def getAttributes(self):
        return [self.target_variable]
    
    def get_base_statistics (self, data, subgroup, weightingAttribute=None): 
        if (weightingAttribute is None):
            sg_instances = subgroup.subgroupDescription.covers(data)
            all_target_values = data[self.target_variable]
            sg_target_values = all_target_values[sg_instances]
            instances_dataset = len(data)
            instances_subgroup = np.sum(sg_instances)
            mean_sg = np.mean (sg_target_values)
            mean_dataset = np.mean (all_target_values)
            return (instances_dataset, mean_dataset, instances_subgroup, mean_sg)  
        else:
            raise NotImplemented("Attribute weights with numeric targets are not yet implemented.")
        
    def calculateStatistics (self, subgroup, data, weightingAttribute=None):
        if weightingAttribute is not None:
            raise NotImplemented("Attribute weights with numeric targets are not yet implemented.")
        sg_instances = subgroup.subgroupDescription.covers(data)
        all_target_values = data[self.target_variable]
        sg_target_values = all_target_values[sg_instances]
        subgroup.statistics['size_sg'] = len(sg_target_values)
        subgroup.statistics['size_dataset'] = len (data)
        subgroup.statistics['mean_sg'] = np.mean(sg_target_values)
        subgroup.statistics['mean_dataset'] = np.mean(all_target_values)
        subgroup.statistics['std_sg'] = np.std(sg_target_values)
        subgroup.statistics['std_dataset'] = np.std(all_target_values)
        subgroup.statistics['median_sg'] = np.median(sg_target_values)
        subgroup.statistics['median_dataset'] = np.median(all_target_values)
        subgroup.statistics['max_sg'] = np.max(sg_target_values)
        subgroup.statistics['max_dataset'] = np.max(all_target_values)
        subgroup.statistics['min_sg'] = np.min(sg_target_values)
        subgroup.statistics['min_dataset'] = np.min(all_target_values)
        subgroup.statistics['mean_lift'] = subgroup.statistics['mean_sg'] / subgroup.statistics['mean_dataset']
        subgroup.statistics['median_lift'] = subgroup.statistics['median_sg'] / subgroup.statistics['median_dataset']
        
class StandardQF_numeric (AbstractInterestingnessMeasure, BoundedInterestingnessMeasure):
    
    @staticmethod     
    def standardQF_numeric (a, instances_dataset, mean_dataset, instances_subgroup, mean_sg):
        if (instances_subgroup == 0):
            return 0
        return (instances_subgroup / instances_dataset) ** a * (mean_sg - mean_dataset)
        
    def __init__(self, a, invert=False):
        self.a = a
        self.invert = invert
        
    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        if not self.isApplicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return ps.conditional_invert(self.evaluateFromStatistics (*subgroup.get_base_statistics(data, weightingAttribute)), self.invert)

    
    def optimisticEstimateFromDataset(self, data, subgroup):
        if not self.isApplicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        sg_instances = subgroup.subgroupDescription.covers(data)
        all_target_values = data[subgroup.target.target_variable]
        mean_dataset = np.mean(all_target_values)
        sg_target_values = all_target_values[sg_instances]
        target_values_larger_than_mean = sg_target_values [sg_target_values > mean_dataset]
        return ps.conditional_invert(np.sum(target_values_larger_than_mean) - (len (target_values_larger_than_mean) * mean_dataset), self.invert)
        

    def evaluateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return StandardQF_numeric.standardQF_numeric (self.a, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)
    
    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return float("inf")

    def isApplicable(self, subgroup):
        return isinstance(subgroup.target, NumericTarget)

    def supportsWeights(self):
        return False

class GAStandardQF_numeric (AbstractInterestingnessMeasure):    
    def __init__(self, a, invert=False):
        self.a = a
        self.invert = invert
        
    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        (instances_dataset, _, instances_subgroup, mean_sg) = subgroup.get_base_statistics(data, weightingAttribute)
        if ((instances_subgroup == 0) or (instances_dataset == instances_subgroup)):
            return 0
        maxMean = getMaxGeneralizationMean(data, subgroup, weightingAttribute)
        relativeSize = (instances_subgroup / instances_dataset)
        return conditional_invert (relativeSize ** self.a * (mean_sg - maxMean), self.invert)

    def supportsWeights(self):
        return True

    def isApplicable(self, subgroup):
        return isinstance(subgroup.target, NumericTarget)
    
def getMaxGeneralizationMean(data, subgroup, weightingAttribute=None):
    selectors = subgroup.subgroupDescription.selectors
    generalizations = ps.powerset(selectors)
    maxMean = 0
    for sels in generalizations:
        sg = ps.Subgroup(subgroup.target, ps.SubgroupDescription(list(sels)))
        mean_sg = sg.get_base_statistics(data, weightingAttribute) [3]
        maxMean = max(maxMean, mean_sg)
    return maxMean