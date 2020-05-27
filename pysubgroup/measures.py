'''
Created on 28.04.2016

@author: lemmerfn
'''
from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import combinations
import numpy as np
import pysubgroup as ps


class AbstractInterestingnessMeasure(ABC):

    # pylint: disable=no-member
    def ensure_statistics(self, subgroup, target, data, statistics=None):
        if not self.has_constant_statistics:
            self.calculate_constant_statistics(data, target)
        if any(not hasattr(statistics, attr) for attr in self.required_stat_attrs):
            if getattr(subgroup, 'statistics', False):
                return subgroup.statistics
            else:
                return self.calculate_statistics(subgroup, target, data, statistics)
        return statistics
    # pylint: enable=no-member
    #def optimistic_estimate_from_dataset(self, data, subgroup, weighting_attribute=None): #pylint: disable=unused-argument
    #    return float("inf")


class BoundedInterestingnessMeasure(AbstractInterestingnessMeasure):
    pass
    #@abstractmethod
    #def optimistic_estimate_from_dataset(self, data, subgroup, weighting_attribute=None):
    #    pass



#####
# FIX ME: This is currently not working anymore
#####
class CombinedInterestingnessMeasure(BoundedInterestingnessMeasure):
    def __init__(self, measures, weights=None):
        self.measures = measures

        if weights is None:
            weights = [1] * len(measures)
        assert len(weights) == len(measures)
        self.weights = weights

    def calculate_constant_statistics(self, data, target):
        pass

    def calculate_statistics(self, subgroup, target, data, cached_statistics=None):
        pass

    def evaluate(self, subgroup, target, data, statistics=None):
        #FIX USE of constant statistics
        return np.dot([m.evaluate(subgroup, target, data, None) for m in self.measures], self.weights)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        # FIX USE of constant statistics
        return np.dot([m.optimistic_estimate(subgroup, target, data, None) for m in self.measures], self.weights)

    def evaluate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return np.dot([m.evaluate_from_statistics(instances_dataset, positives_dataset, instances_subgroup, positives_subgroup) for m in self.measures], self.weights)

    #def optimistic_estimate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
    #    return np.dot(
    #        [m.evaluate_from_statistics(instances_dataset, positives_dataset, instances_subgroup, positives_subgroup) for m in self.measures],
    #        self.weights)


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


class CountCallsInterestingMeasure(BoundedInterestingnessMeasure):
    def __init__(self, qf):
        self.qf = qf
        self.calls = 0

    def calculate_statistics(self, sg, target, data, statistics=None):
        self.calls += 1
        return self.qf.calculate_statistics(sg, target, data, statistics)

    def __getattr__(self, name):
        return getattr(self.qf, name)

    def __hasattr__(self, name):
        return hasattr(self.qf, name)


#####
# GeneralizationAware Interestingness Measures
#####
class GeneralizationAwareQF(AbstractInterestingnessMeasure):
    ga_tuple = namedtuple('ga_tuple', ['subgroup_quality', 'generalisation_quality'])
    def __init__(self, qf):
        self.qf = qf

        # this cache maps the representation of descriptions to tuples
        # the first entry is the quality and the second one is
        # the largest quality of all its predessors
        self.cache = {}
        self.has_constant_statistics = False
        self.required_stat_attrs = ['subgroup_quality', 'generalisation_quality']
        self.q0 = 0

    def calculate_constant_statistics(self, data, target):
        self.cache = {}
        self.qf.calculate_constant_statistics(data, target)
        self.q0 = self.qf.evaluate(slice(None), target, data)
        self.has_constant_statistics = self.qf.has_constant_statistics

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        sg_repr = repr(subgroup)
        if sg_repr in self.cache:
            return GeneralizationAwareQF.ga_tuple(*self.cache[sg_repr])
        else:
            (q_sg, q_prev) = self.get_qual_and_previous_qual(subgroup, target, data)
            self.cache[sg_repr] = (q_sg, q_prev)
            return GeneralizationAwareQF.ga_tuple(q_sg, q_prev)

    def get_qual_and_previous_qual(self, subgroup, target, data):
        q_subgroup = self.qf.evaluate(subgroup, target, data)
        max_q = 0
        selectors = subgroup.selectors
        if len(selectors) > 0:
            # compute quality of all generalizations
            generalizations = combinations(selectors, len(selectors)-1)

            for sels in generalizations:
                sgd = ps.Conjunction(list(sels))
                (q_sg, q_prev) = self.calculate_statistics(sgd, target, data)
                max_q = max(max_q, q_sg, q_prev)
        return (q_subgroup, max_q)

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.subgroup_quality - statistics.generalisation_quality


#####
# GeneralizationAware Interestingness Measures
#####
class GeneralizationAwareQF_stats(AbstractInterestingnessMeasure):
    ga_tuple = namedtuple('ga_stats_tuple', ['subgroup_stats', 'generalisation_stats'])
    def __init__(self, qf):
        self.qf = qf

        # this cache maps the representation of descriptions to tuples
        # the first entry is the quality and the second one is
        # the largest quality of all its predecessors
        self.cache = {}
        self.has_constant_statistics = False
        self.required_stat_attrs = GeneralizationAwareQF_stats.ga_tuple._fields
        self.stats0 = None

    def calculate_constant_statistics(self, data, target):
        self.cache = {}
        self.qf.calculate_constant_statistics(data, target)
        self.stats0 = self.qf.calculate_statistics(slice(None), target, data)
        self.has_constant_statistics = self.qf.has_constant_statistics

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        sg_repr = repr(subgroup)
        if sg_repr in self.cache:
            return GeneralizationAwareQF_stats.ga_tuple(*self.cache[sg_repr])
        else:
            (stats_sg, stats_prev) = self.get_stats_and_previous_stats(subgroup, target, data)
            self.cache[sg_repr] = (stats_sg, stats_prev)
            return GeneralizationAwareQF_stats.ga_tuple(stats_sg, stats_prev)

    def get_stats_and_previous_stats(self, subgroup, target, data):
        stats_subgroup = self.qf.calculate_statistics(subgroup, target, data)
        max_stats = self.stats0
        selectors = subgroup.selectors
        if len(selectors) > 0:
            # compute quality of all generalizations
            generalizations = combinations(selectors, len(selectors)-1)

            for sels in generalizations:
                sgd = ps.Conjunction(list(sels))
                (stats_sg, stats_prev) = self.calculate_statistics(sgd, target, data)
                max_stats = self.get_max(max_stats, stats_sg, stats_prev)
        return (stats_subgroup, max_stats)

    def evaluate(self, subgroup, statistics_or_data=None):
        raise NotImplementedError

    def get_max(self, *args):
        raise NotImplementedError