'''
Created on 28.04.2016

@author: lemmerfn
'''
from pysubgroup import SGDUtils
import numpy as np

class SubgroupDescription(object):
    def __init__(self, selectors):
        if isinstance(selectors, list):
            self.selectors = selectors
        else:
            self.selectors = [selectors]
    
    def __repr__(self):
        result = "{"
        for sel in self.selectors:
            result += str(sel)
        result = result [:-1]
        return result + "}"
    
    def covers(self, instance):
        if (not self.selectors):
            return np.full((1, len(instance)), True, dtype=bool)
        return np.all([sel.covers(instance) for sel in self.selectors], axis=0)
    
    def count (self, data):
        return sum(1 for x in data if self.covers(x)) 


class NominalTarget(object):
    def __init__(self, targetSelector):
        self.targetSelector = targetSelector
        
    def __repr__(self):
        return "T: " + str(self.targetSelector)
    
    def covers(self, instance):
        return self.targetSelector.covers(instance)
    
    def getAttributes(self):
        return [self.targetSelector.getAttributeName()]


class Subgroup(object):
    def __init__(self, target, subgroupDescription):
        # If its already a NominalTarget object, we are fine, otherwise we create a new one
        if (isinstance(target, NominalTarget)):
            self.target = target
        else:
            self.target = NominalTarget(target)
        
        # If its already a SubgroupDescription object, we are fine, otherwise we create a new one
        if (isinstance(subgroupDescription, SubgroupDescription)):
            self.subgroupDescription = subgroupDescription
        else:
            self.subgroupDescription = SubgroupDescription(subgroupDescription)
            
        # initialize empty cache for statistics
        self.statistics = {} 
    
    def __repr__(self):
        return "<<" + str(self.target) + "; D: " + str(self.subgroupDescription) + ">>"
    
    def covers(self, instance):
        return self.subgroupDescription.covers(instance)
    
    def count(self, data):
        return sum(1 for x in data if self.covers(x)) 
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def calculateStatistics (self, data, weightingAttribute=None):
        (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = SGDUtils.extractStatisticsFromDataset(data, self)
        self.statistics['size_sg'] = instancesSubgroup
        self.statistics['size_dataset'] = instancesDataset
        self.statistics['positives_sg'] = positivesSubgroup
        self.statistics['positives_dataset'] = positivesDataset
        self.statistics['target_share_sg'] = positivesSubgroup / instancesSubgroup
        self.statistics['target_share_dataset'] = positivesDataset / instancesDataset
        
        if (weightingAttribute != None):
            (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = SGDUtils.extractStatisticsFromDataset(data, self, weightingAttribute)
        self.statistics['size_sg_weighted'] = instancesSubgroup
        self.statistics['size_dataset_weighted'] = instancesDataset
        self.statistics['positives_sg_weighted'] = positivesSubgroup
        self.statistics['positives_dataset_weighted'] = positivesDataset
        self.statistics['target_share_sg_weighted'] = positivesSubgroup / instancesSubgroup
        self.statistics['target_share_dataset_weighted'] = positivesDataset / instancesDataset
    
            
            
