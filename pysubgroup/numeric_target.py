'''
Created on 29.09.2017

@author: lemmerfn
'''
import warnings
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
    tpl = namedtuple('StandardQFNumeric_parameters', ('size', 'mean', 'estimate'))
    @staticmethod
    def standard_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
        return instances_subgroup ** a * (mean_sg - mean_dataset)

    def __init__(self, a, invert=False, estimator='sum'):
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size', 'mean')
        self.dataset = None
        self.all_target_values = None
        self.has_constant_statistics = False
        if estimator == 'sum':
            self.estimator = StandardQFNumeric.Summation_Estimator(self)
        elif estimator == 'average':
            self.estimator = StandardQFNumeric.Average_Estimator(self)
        elif estimator == 'order':
            self.estimator = StandardQFNumeric.Ordering_Estimator(self)
        else:
            self.estimator = estimator
            warnings.warn('estimator is not one of the following: ' + str(['sum', 'average', 'order']))

    def calculate_constant_statistics(self, task):
        if not self.is_applicable(task):
            raise BaseException("Quality measure cannot be used for this target class")
        data = self.estimator.get_data(task)
        self.all_target_values = data[task.target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        data_size = len(data)
        self.dataset = StandardQFNumeric.tpl(data_size, target_mean, None)
        self.estimator.calculate_constant_statistics(task)
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
        sg_size = np.count_nonzero(cover_arr)
        sg_mean = np.array([0])
        sg_target_values = 0
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            sg_mean = np.mean(sg_target_values)
            estimate = self.estimator.get_estimate(subgroup, sg_size, sg_mean, cover_arr, sg_target_values)
        else:
            estimate = float('-inf')
        return StandardQFNumeric.tpl(sg_size, sg_mean, estimate)


    def optimistic_estimate(self, subgroup, statistics=None):
        statistics = self.ensure_statistics(subgroup, statistics)
        return statistics.estimate

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, NumericTarget)

    def supports_weights(self):
        return False

    class Summation_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self.target_values_greater_mean = None

        def get_data(self, task):
            return task.data

        def calculate_constant_statistics(self, task):
            self.indices_greater_mean = np.nonzero(self.qf.all_target_values > self.qf.dataset.mean)[0]
            self.target_values_greater_mean = self.qf.all_target_values[self.indices_greater_mean]

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, _):
            larger_than_mean = self.target_values_greater_mean[cover_arr[self.indices_greater_mean]]
            size_greater_mean = len(larger_than_mean)
            sum_greater_mean = np.sum(larger_than_mean)

            return sum_greater_mean - size_greater_mean * self.qf.dataset.mean



    class Average_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self.target_values_greater_mean = None

        def get_data(self, task):
            return task.data

        def calculate_constant_statistics(self, task):
            self.indices_greater_mean = np.nonzero(self.qf.all_target_values > self.qf.dataset.mean)[0]
            self.target_values_greater_mean = self.qf.all_target_values[self.indices_greater_mean]

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, _):
            larger_than_mean = self.target_values_greater_mean[cover_arr[self.indices_greater_mean]]
            size_greater_mean = len(larger_than_mean)
            max_greater_mean = np.sum(larger_than_mean)

            return size_greater_mean ** self.qf.a * (max_greater_mean - self.qf.dataset.mean)



    class Ordering_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self._get_estimate = self.get_estimate_numpy
            self.use_numba = True
            self.numba_in_place = False

        def get_data(self, task):
            task.data = task.data.sort_values(task.target.get_attributes(), ascending=False)
            return task.data

        def calculate_constant_statistics(self, task):
            if self.use_numba and not self.numba_in_place:
                try:
                    from numba import njit # pylint: disable=unused-import
                    print('StandardQf_Numeric: Using numba for speedup')
                except ImportError:
                    return
                @njit
                def estimate_numba(values_sg, a, mean_dataset):
                    n = 1
                    sum_values = 0
                    max_value = -10 ** 10
                    for val in values_sg:
                        sum_values += val
                        mean_sg = sum_values / n
                        quality = n ** a * (mean_sg - mean_dataset)
                        if quality > max_value:
                            max_value = quality
                        n += 1
                    return max_value
                self._get_estimate = estimate_numba
                self.numba_in_place = True

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, target_values_sg):
            if self.numba_in_place:
                return self._get_estimate(target_values_sg, self.qf.a, self.qf.dataset.mean)
            else:
                return self._get_estimate(target_values_sg, self.qf.a, self.qf.dataset.mean)

        def get_estimate_numpy(self, values_sg, a, mean_dataset):
            target_values_cs = np.cumsum(values_sg)
            sizes = np.arange(1, len(target_values_cs) + 1)
            mean_values = target_values_cs / sizes
            stats = StandardQFNumeric.tpl(sizes, mean_values, mean_dataset)
            qualities = self.qf.evaluate(None, stats)
            optimistic_estimate = np.max(qualities)
            return optimistic_estimate





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
