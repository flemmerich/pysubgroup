'''
Created on 29.09.2017

@author: lemmerfn
'''
from collections import namedtuple
from functools import total_ordering
import numpy as np
import pysubgroup as ps


@total_ordering
class NumericTarget:
    def __init__(self, target_variable):
        self.target_variable = target_variable

    def __repr__(self):
        return "T: " + str(self.target_variable)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return [self.target_variable]

    def get_base_statistics(self, data, subgroup, weighting_attribute=None):
        if weighting_attribute is None:
            sg_instances = subgroup.subgroup_description.covers(data)
            all_target_values = data[self.target_variable]
            sg_target_values = all_target_values[sg_instances]
            instances_dataset = len(data)
            instances_subgroup = np.sum(sg_instances)
            mean_sg = np.mean(sg_target_values)
            mean_dataset = np.mean(all_target_values)
            return (instances_dataset, mean_dataset, instances_subgroup, mean_sg)
        else:
            raise NotImplementedError("Attribute weights with numeric targets are not yet implemented.")

    def calculate_statistics(self, subgroup, data, weighting_attribute=None):
        if weighting_attribute is not None:
            raise NotImplementedError("Attribute weights with numeric targets are not yet implemented.")
        sg_instances = subgroup.subgroup_description.covers(data)
        all_target_values = data[self.target_variable]
        sg_target_values = all_target_values[sg_instances]
        subgroup.statistics['size_sg'] = len(sg_target_values)
        subgroup.statistics['size_dataset'] = len(data)
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


class StandardQFNumeric(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('StandardQFNumeric_parameters' , ('size' , 'mean', 'size_greater_mean','sum_greater_mean'))
    @staticmethod
    def standard_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
        return instances_subgroup ** a * (mean_sg - mean_dataset)

    def __init__(self, a, invert=False):
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size', 'mean')
        self.datatset = None
        self.positives = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, task):
        if not self.is_applicable(task):
            raise BaseException("Quality measure cannot be used for this target class")
        data = task.data
        self.all_target_values = data[task.target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        self.indices_greater_mean = np.nonzero(self.all_target_values > target_mean)[0]
        self.target_values_greater_mean = self.all_target_values[self.indices_greater_mean]
        self.dataset = StandardQFNumeric.tpl(len(data), target_mean, 0, 0)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, statistics=None):
        statistics = self.ensure_statistics(subgroup, statistics)
        dataset = self.dataset
        return StandardQFNumeric.standard_qf_numeric(self.a, dataset.size, dataset.mean, statistics.size, statistics.mean)

    def calculate_statistics(self, subgroup, data=None):
        if hasattr(subgroup, "representation"):
            cover_arr = np.array(subgroup)
        else:
            cover_arr = subgroup.covers(data)
        sg_size=np.count_nonzero(cover_arr)
        sg_mean=np.array([0])
        #size_greater_mean=0
        #sum_greater_mean=np.array([0])
        if sg_size > 0:
            sg_mean=np.mean(self.all_target_values[cover_arr])
        larger_than_mean = self.target_values_greater_mean[cover_arr[self.indices_greater_mean]]
        size_greater_mean = len(larger_than_mean)
        sum_greater_mean = np.sum(larger_than_mean)

        return StandardQFNumeric.tpl(sg_size, sg_mean, size_greater_mean, sum_greater_mean)


    def optimistic_estimate(self, subgroup, statistics=None):
        statistics = self.ensure_statistics(subgroup, statistics)
        dataset = self.dataset
        a = statistics.sum_greater_mean
        b = statistics.size_greater_mean
        sg_mean=np.divide(a, b, out=np.zeros_like(a), where=b!=0) # deal with the case where b==0
        return StandardQFNumeric.standard_qf_numeric(self.a, dataset.size, dataset.mean, statistics.size_greater_mean, sg_mean )

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, NumericTarget)

    def supports_weights(self):
        return False


class GAStandardQFNumeric(ps.AbstractInterestingnessMeasure):
    def __init__(self, a, invert=False):
        self.a = a
        self.invert = invert

    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
        (instances_dataset, _, instances_subgroup, mean_sg) = subgroup.get_base_statistics(data, weighting_attribute)
        if instances_subgroup in (0, instances_dataset):
            return 0
        max_mean = get_max_generalization_mean(data, subgroup, weighting_attribute)
        relative_size = (instances_subgroup / instances_dataset)
        return ps.conditional_invert(relative_size ** self.a * (mean_sg - max_mean), self.invert)

    def supports_weights(self):
        return True

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, NumericTarget)


def get_max_generalization_mean(data, subgroup, weighting_attribute=None):
    selectors = subgroup.subgroup_description.selectors
    generalizations = ps.powerset(selectors)
    max_mean = 0
    for sels in generalizations:
        sg = ps.Subgroup(subgroup.target, ps.Conjunction(list(sels)))
        mean_sg = sg.get_base_statistics(data, weighting_attribute)[3]
        max_mean = max(max_mean, mean_sg)
    return max_mean
