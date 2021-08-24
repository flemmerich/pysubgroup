'''
Created on 29.09.2017

@author: lemmerfn
'''
import numbers
from collections import namedtuple
from functools import total_ordering
import numpy as np
import pysubgroup as ps


@total_ordering
class NumericTarget:

    statistic_types = (
        'size_sg', 'size_dataset', 'mean_sg', 'mean_dataset', 'std_sg', 'std_dataset', 'median_sg', 'median_dataset',
        'max_sg', 'max_dataset', 'min_sg', 'min_dataset', 'mean_lift', 'median_lift')

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

    def get_base_statistics(self, subgroup, data):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(data), data)
        all_target_values = data[self.target_variable]
        sg_target_values = all_target_values[cover_arr]
        instances_dataset = len(data)
        instances_subgroup = size_sg
        mean_sg = np.mean(sg_target_values)
        mean_dataset = np.mean(all_target_values)
        return (instances_dataset, mean_dataset, instances_subgroup, mean_sg)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        if cached_statistics is None or not isinstance(cached_statistics, dict):
            statistics = dict()
        elif all(k in cached_statistics for k in NumericTarget.statistic_types):
            return cached_statistics
        else:
            statistics = cached_statistics

        cover_arr, _ = ps.get_cover_array_and_size(subgroup, len(data), data)
        all_target_values = data[self.target_variable].to_numpy()
        sg_target_values = all_target_values[cover_arr]

        statistics['size_sg'] = len(sg_target_values)
        statistics['size_dataset'] = len(data)
        statistics['mean_sg'] = np.mean(sg_target_values)
        statistics['mean_dataset'] = np.mean(all_target_values)
        statistics['std_sg'] = np.std(sg_target_values)
        statistics['std_dataset'] = np.std(all_target_values)
        statistics['median_sg'] = np.median(sg_target_values)
        statistics['median_dataset'] = np.median(all_target_values)
        statistics['max_sg'] = np.max(sg_target_values)
        statistics['max_dataset'] = np.max(all_target_values)
        statistics['min_sg'] = np.min(sg_target_values)
        statistics['min_dataset'] = np.min(all_target_values)
        statistics['mean_lift'] = statistics['mean_sg'] / statistics['mean_dataset']
        statistics['median_lift'] = statistics['median_sg'] / statistics['median_dataset']
        return statistics


class StandardQFNumeric(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('StandardQFNumeric_parameters', ('size_sg', 'mean', 'estimate'))
    @staticmethod
    def standard_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
        return instances_subgroup ** a * (mean_sg - mean_dataset)

    def __init__(self, a, invert=False, estimator='sum'):
        if not isinstance(a, numbers.Number):
            raise ValueError(f'a is not a number. Received a={a}')
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size_sg', 'mean')
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        if estimator == 'sum':
            self.estimator = StandardQFNumeric.Summation_Estimator(self)
        elif estimator == 'average':
            self.estimator = StandardQFNumeric.Average_Estimator(self)
        elif estimator == 'order':
            self.estimator = StandardQFNumeric.Ordering_Estimator(self)
        else:
            raise ValueError('estimator is not one of the following: ' + str(['sum', 'average', 'order']))

    def calculate_constant_statistics(self, data, target):
        data = self.estimator.get_data(data, target)
        self.all_target_values = data[target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        data_size = len(data)
        self.dataset_statistics = StandardQFNumeric.tpl(data_size, target_mean, None)
        self.estimator.calculate_constant_statistics(data, target)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQFNumeric.standard_qf_numeric(self.a, dataset.size_sg, dataset.mean, statistics.size_sg, statistics.mean)

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_values), data)
        sg_mean = np.array([0])
        sg_target_values = 0
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            sg_mean = np.mean(sg_target_values)
            estimate = self.estimator.get_estimate(subgroup, sg_size, sg_mean, cover_arr, sg_target_values)
        else:
            estimate = float('-inf')
        return StandardQFNumeric.tpl(sg_size, sg_mean, estimate)


    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate


    class Summation_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self.target_values_greater_mean = None

        def get_data(self, data, target):
            return data

        def calculate_constant_statistics(self, data, target):  # pylint: disable=unused-argument
            self.indices_greater_mean = self.qf.all_target_values > self.qf.dataset_statistics.mean
            self.target_values_greater_mean = self.qf.all_target_values#[self.indices_greater_mean]

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, _):  # pylint: disable=unused-argument
            larger_than_mean = self.target_values_greater_mean[cover_arr][self.indices_greater_mean[cover_arr]]
            size_greater_mean = len(larger_than_mean)
            sum_greater_mean = np.sum(larger_than_mean)

            return sum_greater_mean - size_greater_mean * self.qf.dataset_statistics.mean

    class Average_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self.target_values_greater_mean = None

        def get_data(self, data, target):
            return data

        def calculate_constant_statistics(self, data, target): # pylint: disable=unused-argument
            self.indices_greater_mean = self.qf.all_target_values > self.qf.dataset_statistics.mean
            self.target_values_greater_mean = self.qf.all_target_values

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, _): # pylint: disable=unused-argument
            larger_than_mean = self.target_values_greater_mean[cover_arr][self.indices_greater_mean[cover_arr]]
            size_greater_mean = len(larger_than_mean)
            max_greater_mean = np.sum(larger_than_mean)

            return size_greater_mean ** self.qf.a * (max_greater_mean - self.qf.dataset_statistics.mean)



    class Ordering_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self._get_estimate = self.get_estimate_numpy
            self.use_numba = True
            self.numba_in_place = False

        def get_data(self, data, target):
            data.sort_values(target.get_attributes(), ascending=False, inplace=True)
            return data

        def calculate_constant_statistics(self, data, target):
            if self.use_numba and not self.numba_in_place:
                try:
                    from numba import njit # pylint: disable=unused-import, import-outside-toplevel
                    #print('StandardQf_Numeric: Using numba for speedup')
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

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, target_values_sg):  # pylint: disable=unused-argument
            if self.numba_in_place:
                return self._get_estimate(target_values_sg, self.qf.a, self.qf.dataset_statistics.mean)
            else:
                return self._get_estimate(target_values_sg, self.qf.a, self.qf.dataset_statistics.mean)

        def get_estimate_numpy(self, values_sg, _, mean_dataset):
            target_values_cs = np.cumsum(values_sg)
            sizes = np.arange(1, len(target_values_cs) + 1)
            mean_values = target_values_cs / sizes
            stats = StandardQFNumeric.tpl(sizes, mean_values, mean_dataset)
            qualities = self.qf.evaluate(None, None, None, stats)
            optimistic_estimate = np.max(qualities)
            return optimistic_estimate



class StandardQFNumericMedian(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('StandardQFNumericMedian_parameters', ('size_sg', 'median', 'estimate'))
    @staticmethod
    def standard_qf_numeric(a, _, median_dataset, instances_subgroup, median_sg):
        return instances_subgroup ** a * (median_sg - median_dataset)

    def __init__(self, a, invert=False, estimator='sum'):
        if not isinstance(a, numbers.Number):
            raise ValueError(f'a is not a number. Received a={a}')
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size_sg', 'median')
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        if estimator == 'sum':
            self.estimator = StandardQFNumericMedian.Summation_Estimator(self)
        elif estimator == 'average':
            self.estimator = StandardQFNumericMedian.Average_Estimator(self)
        elif estimator == 'order':
            self.estimator = StandardQFNumericMedian.Ordering_Estimator(self)
        else:
            raise ValueError('estimator is not one of the following: ' + str(['sum', 'average', 'order']))

    def calculate_constant_statistics(self, data, target):
        data = self.estimator.get_data(data, target)
        self.all_target_values = data[target.target_variable].to_numpy()
        target_median = np.median(self.all_target_values)
        data_size = len(data)
        self.dataset_statistics = StandardQFNumericMedian.tpl(data_size, target_median, None)
        self.estimator.calculate_constant_statistics(data, target)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQFNumericMedian.standard_qf_numeric(self.a, dataset.size_sg, dataset.median, statistics.size_sg, statistics.median)

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_values), data)
        sg_median = np.array([0])
        sg_target_values = 0
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            sg_median = np.median(sg_target_values)
            estimate = self.estimator.get_estimate(subgroup, sg_size, sg_median, cover_arr, sg_target_values)
        else:
            estimate = float('-inf')
        return StandardQFNumericMedian.tpl(sg_size, sg_median, estimate)


    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate


    class Summation_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_median = None
            self.target_values_greater_median = None

        def get_data(self, data, target):
            return data

        def calculate_constant_statistics(self, data, target):  # pylint: disable=unused-argument
            self.indices_greater_median = self.qf.all_target_values > self.qf.dataset_statistics.median
            self.target_values_greater_median = self.qf.all_target_values#[self.indices_greater_median]

        def get_estimate(self, subgroup, sg_size, sg_median, cover_arr, _):  # pylint: disable=unused-argument
            larger_than_median = self.target_values_greater_median[cover_arr][self.indices_greater_median[cover_arr]]
            size_greater_median = len(larger_than_median)
            sum_greater_median = np.sum(larger_than_median)

            return sum_greater_median - size_greater_median * self.qf.dataset_statistics.median

    class Average_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self.target_values_greater_mean = None

        def get_data(self, data, target):
            return data

        def calculate_constant_statistics(self, data, target): # pylint: disable=unused-argument
            self.indices_greater_mean = self.qf.all_target_values > self.qf.dataset_statistics.mean
            self.target_values_greater_mean = self.qf.all_target_values

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, _): # pylint: disable=unused-argument
            larger_than_mean = self.target_values_greater_mean[cover_arr][self.indices_greater_mean[cover_arr]]
            size_greater_mean = len(larger_than_mean)
            max_greater_mean = np.sum(larger_than_mean)

            return size_greater_mean ** self.qf.a * (max_greater_mean - self.qf.dataset_statistics.mean)



    class Ordering_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self._get_estimate = self.get_estimate_numpy
            self.use_numba = True
            self.numba_in_place = False

        def get_data(self, data, target):
            data.sort_values(target.get_attributes(), ascending=False, inplace=True)
            return data

        def calculate_constant_statistics(self, data, target):
            if self.use_numba and not self.numba_in_place:
                try:
                    from numba import njit # pylint: disable=unused-import, import-outside-toplevel
                    #print('StandardQf_Numeric: Using numba for speedup')
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

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, target_values_sg):  # pylint: disable=unused-argument
            if self.numba_in_place:
                return self._get_estimate(target_values_sg, self.qf.a, self.qf.dataset_statistics.mean)
            else:
                return self._get_estimate(target_values_sg, self.qf.a, self.qf.dataset_statistics.mean)

        def get_estimate_numpy(self, values_sg, _, mean_dataset):
            target_values_cs = np.cumsum(values_sg)
            sizes = np.arange(1, len(target_values_cs) + 1)
            mean_values = target_values_cs / sizes
            stats = StandardQFNumericMedian.tpl(sizes, mean_values, mean_dataset)
            qualities = self.qf.evaluate(None, None, None, stats)
            optimistic_estimate = np.max(qualities)
            return optimistic_estimate


class StandardQFNumericTscore(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('StandardQFNumericTscore_parameters', ('size_sg', 'mean', 'std' , 'estimate'))
    @staticmethod
    def standard_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg, std_sg):
        if std_sg == 0 :
            return 0
        else :
            return (instances_subgroup ** 0.5 * (mean_sg - mean_dataset)) / std_sg 

    def __init__(self, a, invert=False, estimator='sum'):
        if not isinstance(a, numbers.Number):
            raise ValueError(f'a is not a number. Received a={a}')
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size_sg', 'mean', 'std')
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        if estimator == 'sum':
            self.estimator = StandardQFNumericTscore.Summation_Estimator(self)
        elif estimator == 'average':
            self.estimator = StandardQFNumericTscore.Average_Estimator(self)
        elif estimator == 'order':
            self.estimator = StandardQFNumericTscore.Ordering_Estimator(self)
        else:
            raise ValueError('estimator is not one of the following: ' + str(['sum', 'average', 'order']))

    def calculate_constant_statistics(self, data, target):
        data = self.estimator.get_data(data, target)
        self.all_target_values = data[target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        target_std = np.std(self.all_target_values)
        data_size = len(data)
        self.dataset_statistics = StandardQFNumericTscore.tpl(data_size, target_mean, target_std, None)
        self.estimator.calculate_constant_statistics(data, target)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQFNumericTscore.standard_qf_numeric(self.a, dataset.size_sg, dataset.mean, statistics.size_sg, statistics.mean, statistics.std)

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_values), data)
        sg_mean = np.array([0])
        sg_std = np.array([0])
        sg_target_values = 0
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            sg_mean = np.mean(sg_target_values)
            sg_std = np.std(sg_target_values)
            estimate = self.estimator.get_estimate(subgroup, sg_size, sg_mean, cover_arr, sg_target_values)
        else:
            estimate = float('-inf')
        return StandardQFNumericTscore.tpl(sg_size, sg_mean, sg_std, estimate)


    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate


    class Summation_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self.target_values_greater_mean = None

        def get_data(self, data, target):
            return data

        def calculate_constant_statistics(self, data, target):  # pylint: disable=unused-argument
            self.indices_greater_mean = self.qf.all_target_values > self.qf.dataset_statistics.mean
            self.target_values_greater_mean = self.qf.all_target_values#[self.indices_greater_mean]

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, _):  # pylint: disable=unused-argument
            larger_than_mean = self.target_values_greater_mean[cover_arr][self.indices_greater_mean[cover_arr]]
            size_greater_mean = len(larger_than_mean)
            sum_greater_mean = np.sum(larger_than_mean)

            return sum_greater_mean - size_greater_mean * self.qf.dataset_statistics.mean

    class Average_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self.target_values_greater_mean = None

        def get_data(self, data, target):
            return data

        def calculate_constant_statistics(self, data, target): # pylint: disable=unused-argument
            self.indices_greater_mean = self.qf.all_target_values > self.qf.dataset_statistics.mean
            self.target_values_greater_mean = self.qf.all_target_values

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, _): # pylint: disable=unused-argument
            larger_than_mean = self.target_values_greater_mean[cover_arr][self.indices_greater_mean[cover_arr]]
            size_greater_mean = len(larger_than_mean)
            max_greater_mean = np.sum(larger_than_mean)

            return size_greater_mean ** self.qf.a * (max_greater_mean - self.qf.dataset_statistics.mean)



    class Ordering_Estimator:
        def __init__(self, qf):
            self.qf = qf
            self.indices_greater_mean = None
            self._get_estimate = self.get_estimate_numpy
            self.use_numba = True
            self.numba_in_place = False

        def get_data(self, data, target):
            data.sort_values(target.get_attributes(), ascending=False, inplace=True)
            return data

        def calculate_constant_statistics(self, data, target):
            if self.use_numba and not self.numba_in_place:
                try:
                    from numba import njit # pylint: disable=unused-import, import-outside-toplevel
                    #print('StandardQf_Numeric: Using numba for speedup')
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

        def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, target_values_sg):  # pylint: disable=unused-argument
            if self.numba_in_place:
                return self._get_estimate(target_values_sg, self.qf.a, self.qf.dataset_statistics.mean)
            else:
                return self._get_estimate(target_values_sg, self.qf.a, self.qf.dataset_statistics.mean)

        def get_estimate_numpy(self, values_sg, _, mean_dataset):
            target_values_cs = np.cumsum(values_sg)
            sizes = np.arange(1, len(target_values_cs) + 1)
            mean_values = target_values_cs / sizes
            stats = StandardQFNumericTscore.tpl(sizes, mean_values, mean_dataset)
            qualities = self.qf.evaluate(None, None, None, stats)
            optimistic_estimate = np.max(qualities)
            return optimistic_estimate


# TODO Update to new format
#class GAStandardQFNumeric(ps.AbstractInterestingnessMeasure):
#    def __init__(self, a, invert=False):
#        self.a = a
#        self.invert = invert
#
#    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
#        (instances_dataset, _, instances_subgroup, mean_sg) = subgroup.get_base_statistics(data, weighting_attribute)
#        if instances_subgroup in (0, instances_dataset):
#            return 0
#        max_mean = get_max_generalization_mean(data, subgroup, weighting_attribute)
#        relative_size = (instances_subgroup / instances_dataset)
#        return ps.conditional_invert(relative_size ** self.a * (mean_sg - max_mean), self.invert)

#    def supports_weights(self):
#        return True

#    def is_applicable(self, subgroup):
#        return isinstance(subgroup.target, NumericTarget)


#def get_max_generalization_mean(data, subgroup, weighting_attribute=None):
#    selectors = subgroup.subgroup_description.selectors
#    generalizations = ps.powerset(selectors)
#    max_mean = 0
#    for sels in generalizations:
#        sg = ps.Subgroup(subgroup.target, ps.Conjunction(list(sels)))
#        mean_sg = sg.get_base_statistics(data, weighting_attribute)[3]
#        max_mean = max(max_mean, mean_sg)
#    return max_mean