'''
Created on 28.04.2016

@author: lemmerfn
'''
import numpy as np
import pysubgroup.utils as ut

class AbstractInterestingnessMeasure(object):
    def optimisticEstimateFromDataset(self, data, subgroup):
        return float("inf")
    
    def optimisticEstimateFromStatistics (self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return float("inf")
    
    def supportsWeights(self):
        return False
    
    def isApplicable (self):
        return False
        
class BoundedInterestingnessMeasure:
    pass

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
