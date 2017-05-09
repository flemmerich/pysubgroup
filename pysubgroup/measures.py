'''
Created on 28.04.2016

@author: lemmerfn
'''
import itertools

from scipy.stats import chi2
import scipy.stats

import numpy as np
import pysubgroup.utils as ut
from pysubgroup.subgroup import Subgroup, SubgroupDescription


class AbstractInterestingnessMeasure(object):
    def optimisticEstimateFromDataset(self, data, subgroup):
        return float("inf")
    
    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return float("inf")
    
    def supportsWeights(self):
        return False
        
class BoundedInterestingnessMeasure:
    pass

class ChiSquaredQF (AbstractInterestingnessMeasure):
    @staticmethod     
    def chiSquaredQF (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup, min_instances=5, bidirect=True, direction_positive=True):
        if (instancesSubgroup < min_instances) or ((instancesDataset - instancesSubgroup) < 5):
            return float("-inf")
        p_subgroup = positivesSubgroup / instancesSubgroup
        p_dataset = positivesDataset / instancesDataset
        positivesComplement = positivesDataset - positivesSubgroup

        # instancesComplement = instancesDataset - instancesSubgroup
        negativesSubgroup = instancesSubgroup - positivesSubgroup
        negativesDataset = instancesDataset - positivesDataset
        negativesComplement = negativesDataset - negativesSubgroup
        val = scipy.stats.chi2_contingency([[positivesSubgroup, positivesComplement],
                                            [negativesSubgroup, negativesComplement]], correction=False)[0]
        if bidirect:
            return val
        elif direction_positive and p_subgroup > p_dataset:
            return val
        elif not direction_positive and p_subgroup < p_dataset:
            return val
        return -val
    
    @staticmethod     
    def chiSquaredQFWeighted (subgroup, data, weightingAttribute, effectiveSampleSize=0, min_instances=5,):
        (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = ut.extractStatisticsFromDataset(data, subgroup, weightingAttribute)
        if (instancesSubgroup < min_instances) or ((instancesDataset - instancesSubgroup) < 5):
            return float("inf")
        if effectiveSampleSize == 0:
            effectiveSampleSize = effective_sample_size(data[weightingAttribute])
        # p_subgroup = positivesSubgroup / instancesSubgroup
        # p_dataset = positivesDataset / instancesDataset

        negativesSubgroup = instancesSubgroup - positivesSubgroup
        negativesDataset = instancesDataset - positivesDataset
        positivesComplement = positivesDataset - positivesSubgroup
        negativesComplement = negativesDataset - negativesSubgroup
        val = scipy.stats.chi2_contingency([[positivesSubgroup, positivesComplement],
                                            [negativesSubgroup, negativesComplement]], correction=True) [0]
        return chi2.sf(val * effectiveSampleSize / instancesDataset, 1)
    
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
        if weightingAttribute == None:
            result = self.evaluateFromStatistics (*ut.extractStatisticsFromDataset(data, subgroup, None))
        else:
            (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = ut.extractStatisticsFromDataset(data, subgroup, weightingAttribute)
            weights = data[weightingAttribute]
            base = self.evaluateFromStatistics (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)           
            result = base * effective_sample_size(weights) / instancesDataset
        return result
    
    def evaluateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return ChiSquaredQF.chiSquaredQF (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup, self.min_instances, self.bidirect, self.direction_positive)

    def supportsWeights(self):
        return True

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
        return self.evaluateFromStatistics (*ut.extractStatisticsFromDataset(data, subgroup, weightingAttribute))
    
    def optimisticEstimateFromDataset(self, data, subgroup):
        return self.optimisticEstimateFromStatistics (*ut.extractStatisticsFromDataset(data, subgroup))

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
def effective_sample_size(weights):
    return sum(weights) ** 2 / sum(weights ** 2)


#####
# GeneralizationAware Interestingness Measures
#####

class GAStandardQF (AbstractInterestingnessMeasure):    
    def __init__(self, a):
        self.a = a

    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = ut.extractStatisticsFromDataset(data, subgroup, weightingAttribute)
        if ((instancesSubgroup == 0) or (instancesDataset == instancesSubgroup)):
            return 0
        p_subgroup = positivesSubgroup / instancesSubgroup
        maxTargetShare = getMaxGeneralizationTargetShare(data, subgroup, weightingAttribute)
        relativeSize = (instancesSubgroup / instancesDataset)
        return instancesSubgroup ** self.a * (p_subgroup - maxTargetShare)

    def supportsWeights(self):
        return True

# from https://docs.python.org/3/library/itertools.html#recipes
def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)))
    
def getMaxGeneralizationTargetShare(data, subgroup, weightingAttribute=None):
    selectors = subgroup.subgroupDescription.selectors
    generalizations = powerset(selectors)
    maxTargetShare = 0
    for sels in generalizations:
        sgd = SubgroupDescription(list(sels))
        sg = Subgroup(subgroup.target, sgd)
        (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = ut.extractStatisticsFromDataset(data, sg, weightingAttribute)
        targetShare = positivesSubgroup / instancesSubgroup
        maxTargetShare = max(maxTargetShare, targetShare)
    return maxTargetShare



##########
# Filter
##########
def uniqueAttributes (resultSet, data):
    result = []
    usedAttributes = []
    for (q, sg) in resultSet:
        atts = sg.subgroupDescription.getAttributes()
        if not atts in usedAttributes or all([ut.isCategoricalAttribute(data, x) for x in atts]):
            result.append ((q, sg))
            usedAttributes.append(atts)
    return result

def minimumStatisticFilter (resultSet, statistic, minimum):
    result = []
    for (q, sg) in resultSet:
        if sg.statistics [statistic] >= minimum:
            result.append ((q, sg))
    return result

def minimumQualityFilter (resultSet, minimum):
    result = []
    for (q, sg) in resultSet:
        if q >= minimum:
            result.append ((q, sg))
    return result

def maximumStatisticFilter (resultSet, statistic, maximum):
    result = []
    for (q, sg) in resultSet:
        if sg.statistics [statistic] <= maximum:
            result.append ((q, sg))
    return result

    
def overlapFilter(resultSet, data, similarity_level=0.9):
    result = []
    resultSGs = []
    for (q, sg) in resultSet:
        if not overlapsList(sg, resultSGs, data, similarity_level):
            resultSGs.append(sg)
            result.append((q, sg))
    return result

def overlapsList (sg, listOfSGs, data, similarity_level=0.9):
    for anotherSG in listOfSGs:
        if overlaps (sg, anotherSG, data, similarity_level):
            return True
    return False

def overlaps (sg, anotherSG, data, similarity_level=0.9):
    coverSG = sg.covers(data)
    coverAnotherSG = anotherSG.covers(data)
    union = np.logical_or (coverSG, coverAnotherSG)
    intercept = np.logical_and (coverSG, coverAnotherSG)
    sim = np.sum(intercept) / np.sum(union)
    return (sim) > similarity_level
