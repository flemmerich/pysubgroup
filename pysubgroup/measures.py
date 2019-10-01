'''
Created on 28.04.2016

@author: lemmerfn
'''
import numpy as np
import pysubgroup as ps


class AbstractInterestingnessMeasure(object):
    def optimistic_estimate_from_dataset(self, data, subgroup):
        return float("inf")

    def optimistic_estimate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup,
                                            positives_subgroup):
        return float("inf")

    def supports_weights(self):
        return False

    def is_applicable(self, subgroup):
        return False


class BoundedInterestingnessMeasure:
    pass


class CombinedInterestingnessMeasure(AbstractInterestingnessMeasure, BoundedInterestingnessMeasure):
    def __init__(self, measures, weights=None):
        self.measures = measures
        if weights is None:
            weights = [1] * len(measures)
        self.weights = weights

    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
        if not self.is_applicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return np.dot([m.evaluate_from_dataset(data, subgroup, weighting_attribute) for m in self.measures], self.weights)

    def optimistic_estimate_from_dataset(self, data, subgroup):
        if not self.is_applicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return np.dot([m.optimistic_estimate_from_dataset(data, subgroup) for m in self.measures], self.weights)

    def evaluate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return np.dot(
            [m.evaluate_from_statistics(instances_dataset, positives_dataset, instances_subgroup, positives_subgroup)
                for m in self.measures], self.weights)

    def optimistic_estimate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup,
                                         positives_subgroup):
        return np.dot(
            [m.evaluate_from_statistics(instances_dataset, positives_dataset, instances_subgroup, positives_subgroup)
             for m in self.measures], self.weights)

    def is_applicable(self, subgroup):
        return all([x.is_applicable(subgroup) for x in self.measures])

    def supports_weights(self):
        return all([x.supports_weights() for x in self.measures])


##########
# Filter
##########
def unique_attributes(result_set, data):
    result = []
    used_attributes = []
    for (q, sg) in result_set:
        atts = sg.subgroup_description.get_attributes()
        if atts not in used_attributes or all([ps.is_categorical_attribute(data, x) for x in atts]):
            result.append((q, sg))
            used_attributes.append(atts)
    return result


def minimum_statistic_filter(result_set, statistic, minimum, data):
    result = []
    for (q, sg) in result_set:
        if len(sg.statistics) == 0:
            sg.calculate_statistics(data)
        if sg.statistics[statistic] >= minimum:
            result.append((q, sg))
    return result


def minimum_quality_filter(result_set, minimum):
    result = []
    for (q, sg) in result_set:
        if q >= minimum:
            result.append((q, sg))
    return result


def maximum_statistic_filter(result_set, statistic, maximum):
    result = []
    for (q, sg) in result_set:
        if sg.statistics[statistic] <= maximum:
            result.append((q, sg))
    return result


def overlap_filter(result_set, data, similarity_level=0.9):
    result = []
    result_sgs = []
    for (q, sg) in result_set:
        if not overlaps_list(sg, result_sgs, data, similarity_level):
            result_sgs.append(sg)
            result.append((q, sg))
    return result


def overlaps_list(sg, list_of_sgs, data, similarity_level=0.9):
    for anotherSG in list_of_sgs:
        if ps.overlap(sg, anotherSG, data) > similarity_level:
            return True
    return False
