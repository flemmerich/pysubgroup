'''
Created on 29.09.2017

@author: lemmerfn
'''
import numpy as np
import scipy.stats
from functools import total_ordering
from pysubgroup.measures import AbstractInterestingnessMeasure, \
    BoundedInterestingnessMeasure
import pysubgroup.utils as ut
from pysubgroup.subgroup import SubgroupDescription, Subgroup, NominalSelector

@total_ordering
class NominalTarget(object):
    
    def __init__(self, target_attribute=None, target_value=None, target_selector=None):
        """
        Creates a new target for the boolean model class (classic subgroup discovery). 
        If target_attribute and target_value are given, the target_selector is computed using attribute and value
        """
        if target_attribute != None and target_value != None:
            if target_selector is not None:
                raise BaseException("NominalTarget is to be constructed EITHER by a selector OR by attribute/value pair")
            target_selector = NominalSelector(target_attribute, target_value)
        if target_selector is None:
            raise BaseException("No target selector given")
        self.targetSelector = target_selector
        
    def __repr__(self):
        return "T: " + str(self.targetSelector)
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def covers(self, instance):
        return self.targetSelector.covers(instance)
    
    def getAttributes(self):
        return [self.targetSelector.getAttributeName()]

    def get_base_statistics (self, data, subgroup, weightingAttribute=None): 
        if (weightingAttribute == None):
            sgInstances = subgroup.subgroupDescription.covers(data)
            positives = subgroup.target.covers(data)
            instancesSubgroup = np.sum(sgInstances)
            positivesDataset = np.sum(positives)
            instancesDataset = len(data)
            positivesSubgroup = np.sum(np.logical_and(sgInstances, positives))
            return (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)  
        else:
            weights = data[weightingAttribute]
            sgInstances = subgroup.subgroupDescription.covers(data)
            positives = subgroup.target.covers(data)                         
    
            instancesDataset = np.sum(weights)
            instancesSubgroup = np.sum(np.dot(sgInstances, weights))
            positivesDataset = np.sum(np.dot(positives, weights))
            positivesSubgroup = np.sum(np.dot(np.logical_and(sgInstances, positives), weights))
            return (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)
    
    def calculateStatistics (self, subgroup, data, weightingAttribute=None):
        (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = self.get_base_statistics (data, subgroup, weightingAttribute)
        subgroup.statistics['size_sg'] = instancesSubgroup
        subgroup.statistics['size_dataset'] = instancesDataset
        subgroup.statistics['positives_sg'] = positivesSubgroup
        subgroup.statistics['positives_dataset'] = positivesDataset
        
        subgroup.statistics['size_complement'] = instancesDataset - instancesSubgroup
        subgroup.statistics['relative_size_sg'] = instancesSubgroup / instancesDataset
        subgroup.statistics['relative_size_complement'] = (instancesDataset - instancesSubgroup) / instancesDataset
        subgroup.statistics['coverage_sg'] = positivesSubgroup / positivesDataset
        subgroup.statistics['coverage_complement'] = (positivesDataset - positivesSubgroup) / positivesDataset
        subgroup.statistics['target_share_sg'] = positivesSubgroup / instancesSubgroup
        subgroup.statistics['target_share_complement'] = (positivesDataset - positivesSubgroup) / (instancesDataset - instancesSubgroup)
        subgroup.statistics['target_share_dataset'] = positivesDataset / instancesDataset
        subgroup.statistics['lift'] = (positivesSubgroup / instancesSubgroup) / (positivesDataset / instancesDataset)
        
        if (weightingAttribute != None):
            (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = subgroup.get_base_statistics(data, self, weightingAttribute)
        subgroup.statistics['size_sg_weighted'] = instancesSubgroup
        subgroup.statistics['size_dataset_weighted'] = instancesDataset
        subgroup.statistics['positives_sg_weighted'] = positivesSubgroup
        subgroup.statistics['positives_dataset_weighted'] = positivesDataset
        
        subgroup.statistics['size_complement_weighted'] = instancesDataset - instancesSubgroup
        subgroup.statistics['relative_size_sg_weighted'] = instancesSubgroup / instancesDataset
        subgroup.statistics['relative_size_complement_weighted'] = (instancesDataset - instancesSubgroup) / instancesDataset
        subgroup.statistics['coverage_sg_weighted'] = positivesSubgroup / positivesDataset
        subgroup.statistics['coverage_complement_weighted'] = (positivesDataset - positivesSubgroup) / positivesDataset
        subgroup.statistics['target_share_sg_weighted'] = positivesSubgroup / instancesSubgroup
        subgroup.statistics['target_share_complement_weighted'] = (positivesDataset - positivesSubgroup) / (instancesDataset - instancesSubgroup)
        subgroup.statistics['target_share_dataset_weighted'] = positivesDataset / instancesDataset
        subgroup.statistics['lift_weighted'] = (positivesSubgroup / instancesSubgroup) / (positivesDataset / instancesDataset)
        
    
class ChiSquaredQF (AbstractInterestingnessMeasure):
    @staticmethod     
    def chiSquaredQF (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup, min_instances=5, bidirect=True, direction_positive=True):
        if (instancesSubgroup < min_instances) or ((instancesDataset - instancesSubgroup) < min_instances):
            return float("-inf")
        p_subgroup = positivesSubgroup / instancesSubgroup
        p_dataset = positivesDataset / instancesDataset
        positivesComplement = positivesDataset - positivesSubgroup

        # instancesComplement = instancesDataset - instancesSubgroup
        negativesSubgroup = instancesSubgroup - positivesSubgroup
        negativesDataset = instancesDataset - positivesDataset
        negativesComplement = negativesDataset - negativesSubgroup
        
        # observed = [positivesSubgroup, positivesComplement,negativesSubgroup, negativesComplement]
        #
        # if round(positivesSubgroup) < 0 or round(positivesComplement) < 0 or round(negativesSubgroup) <0 or round (negativesComplement) < 0:
        #    print ("XXXXX")
        val = scipy.stats.chi2_contingency([[round(positivesSubgroup), round(positivesComplement)],
                                            [round(negativesSubgroup), round(negativesComplement)]], correction=False)[0]
        if bidirect:
            return val
        elif direction_positive and p_subgroup > p_dataset:
            return val
        elif not direction_positive and p_subgroup < p_dataset:
            return val
        return -val
    
    @staticmethod     
    def chiSquaredQFWeighted (subgroup, data, weightingAttribute, effectiveSampleSize=0, min_instances=5,):
        (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = subgroup.get_base_statistics(data, weightingAttribute)
        if (instancesSubgroup < min_instances) or ((instancesDataset - instancesSubgroup) < 5):
            return float("inf")
        if effectiveSampleSize == 0:
            effectiveSampleSize = ut.effective_sample_size(data[weightingAttribute])
        # p_subgroup = positivesSubgroup / instancesSubgroup
        # p_dataset = positivesDataset / instancesDataset

        negativesSubgroup = instancesSubgroup - positivesSubgroup
        negativesDataset = instancesDataset - positivesDataset
        positivesComplement = positivesDataset - positivesSubgroup
        negativesComplement = negativesDataset - negativesSubgroup
        val = scipy.stats.chi2_contingency([[positivesSubgroup, positivesComplement],
                                            [negativesSubgroup, negativesComplement]], correction=True) [0]
        return scipy.stats.chi2.sf(val * effectiveSampleSize / instancesDataset, 1)
    
    def __init__(self, direction='bidirect', min_instances=5):
        if (direction == 'bidirect'):
            self.bidirect = True
            self.direction_positive = True
        if (direction == 'positive'):
            self.bidirect = False
            self.direction_positive = True
        if (direction == 'negative'):
            self.bidirect = False
            self.direction_positive = False
        self.min_instances = min_instances

    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        if not self.isApplicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        if weightingAttribute == None:
            result = self.evaluateFromStatistics (*subgroup.get_base_statistics(data))
        else:
            (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = subgroup.get_base_statistics(data, weightingAttribute)
            weights = data[weightingAttribute]
            base = self.evaluateFromStatistics (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)           
            result = base * ut.effective_sample_size(weights) / instancesDataset
        return result
    
    def evaluateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return ChiSquaredQF.chiSquaredQF (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup, self.min_instances, self.bidirect, self.direction_positive)

    def supportsWeights(self):
        return True
    
    def isApplicable(self, subgroup):
        return isinstance(subgroup.target, NominalTarget)
    
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
        if not self.isApplicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return self.evaluateFromStatistics (*subgroup.get_base_statistics(data, weightingAttribute))
    
    def optimisticEstimateFromDataset(self, data, subgroup, weightingAttribute=None):
        if not self.isApplicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return self.optimisticEstimateFromStatistics (*subgroup.get_base_statistics(data, weightingAttribute))

    def evaluateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return StandardQF.standardQF (self.a, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)
    
    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return StandardQF.standardQF (self.a, instancesDataset, positivesDataset, positivesSubgroup, positivesSubgroup)

    def supportsWeights(self):
        return True
    
    def isApplicable(self, subgroup):
        return isinstance(subgroup.target, NominalTarget)

class WRAccQF (StandardQF):
    def __init__(self, a):
        super().__init__(a)
        self.a = 1.0
        
class LiftQF (StandardQF):
    def __init__(self, a):
        super().__init__(a)
        self.a = 0.0
        
class SimpleBinomial(StandardQF):
    def __init__(self, a):
        super().__init__(a)
        self.a = 0.5
        
#####
# GeneralizationAware Interestingness Measures
#####
class GAStandardQF (AbstractInterestingnessMeasure):    
    def __init__(self, a):
        self.a = a

    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        (instancesDataset, _, instancesSubgroup, positivesSubgroup) = subgroup.get_base_statistics(data, weightingAttribute)
        if ((instancesSubgroup == 0) or (instancesDataset == instancesSubgroup)):
            return 0
        p_subgroup = positivesSubgroup / instancesSubgroup
        maxTargetShare = getMaxGeneralizationTargetShare(data, subgroup, weightingAttribute)
        relativeSize = (instancesSubgroup / instancesDataset)
        return relativeSize ** self.a * (p_subgroup - maxTargetShare)

    def supportsWeights(self):
        return True

    def isApplicable(self, subgroup):
        return isinstance(subgroup.target, NominalTarget)


def getMaxGeneralizationTargetShare(data, subgroup, weightingAttribute=None):
    selectors = subgroup.subgroupDescription.selectors
    generalizations = ut.powerset(selectors)
    maxTargetShare = 0
    for sels in generalizations:
        sgd = SubgroupDescription(list(sels))
        sg = Subgroup(subgroup.target, sgd)
        (_, _, instancesSubgroup, positivesSubgroup) = sg.get_base_statistics(data, weightingAttribute)
        targetShare = positivesSubgroup / instancesSubgroup
        maxTargetShare = max(maxTargetShare, targetShare)
    return maxTargetShare




