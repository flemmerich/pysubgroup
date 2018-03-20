'''
Created on 28.04.2016

@author: lemmerfn
'''
import numpy as np
import pysubgroup.utils as ut


class AbstractInterestingnessMeasure(object):
    def optimisticEstimateFromDataset(self, data, subgroup):
        return float("inf")

    def optimisticEstimateFromStatistics(self, instancesDataset, positivesDataset, instancesSubgroup,
                                         positivesSubgroup):
        return float("inf")

    def supportsWeights(self):
        return False

    def isApplicable(self, subgroup):
        return False


class BoundedInterestingnessMeasure:
    pass


class CombinedInterestingnessMeasure(AbstractInterestingnessMeasure, BoundedInterestingnessMeasure):
    def __init__(self, measures, weights=None):
        self.measures = measures
        if weights == None:
            weights = [1] * len(measures)
        self.weights = weights

    def evaluateFromDataset(self, data, subgroup, weightingAttribute=None):
        if not self.isApplicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return np.dot([m.evaluateFromDataset(data, subgroup, weightingAttribute) for m in self.measures], self.weights)

    def optimisticEstimateFromDataset(self, data, subgroup):
        if not self.isApplicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return np.dot([m.optimisticEstimateFromDataset(data, subgroup) for m in self.measures], self.weights)

    def evaluateFromStatistics(self, instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup):
        return np.dot(
            [m.evaluateFromStatistics(instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) for m in
             self.measures], self.weights)

    def optimisticEstimateFromStatistics(self, instancesDataset, positivesDataset, instancesSubgroup,
                                         positivesSubgroup):
        return np.dot(
            [m.evaluateFromStatistics(instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) for m in
             self.measures], self.weights)

    def isApplicable(self, subgroup):
        return all([x.isApplicable(subgroup) for x in self.measures])

    def supportsWeights(self):
        return all([x.supportsWeights() for x in self.measures])


##########
# Filter
##########

def uniqueAttributes(resultSet, data):
    result = []
    usedAttributes = []
    for (q, sg) in resultSet:
        atts = sg.subgroupDescription.getAttributes()
        if not atts in usedAttributes or all([ut.isCategoricalAttribute(data, x) for x in atts]):
            result.append((q, sg))
            usedAttributes.append(atts)
    return result


def minimumStatisticFilter(resultSet, statistic, minimum, data):
    result = []
    for (q, sg) in resultSet:
        if len(sg.statistics) == 0:
            sg.calculateStatistics(data)
        if sg.statistics[statistic] >= minimum:
            result.append((q, sg))
    return result


def minimumQualityFilter(resultSet, minimum):
    result = []
    for (q, sg) in resultSet:
        if q >= minimum:
            result.append((q, sg))
    return result


def maximumStatisticFilter(resultSet, statistic, maximum):
    result = []
    for (q, sg) in resultSet:
        if sg.statistics[statistic] <= maximum:
            result.append((q, sg))
    return result


def overlapFilter(resultSet, data, similarity_level=0.9):
    result = []
    resultSGs = []
    for (q, sg) in resultSet:
        if not overlapsList(sg, resultSGs, data, similarity_level):
            resultSGs.append(sg)
            result.append((q, sg))
    return result


def overlapsList(sg, listOfSGs, data, similarity_level=0.9):
    for anotherSG in listOfSGs:
        if overlaps(sg, anotherSG, data, similarity_level):
            return True
    return False


def overlaps(sg, anotherSG, data, similarity_level=0.9):
    coverSG = sg.covers(data)
    coverAnotherSG = anotherSG.covers(data)
    union = np.logical_or(coverSG, coverAnotherSG)
    intercept = np.logical_and(coverSG, coverAnotherSG)
    sim = np.sum(intercept) / np.sum(union)
    return (sim) > similarity_level
