import numpy as np
from functools import total_ordering
from scipy.stats import pearsonr
from scipy.stats import norm
import warnings

from .measures import CorrelationModelMeasure

warnings.filterwarnings('ignore')


@total_ordering
class ComplexTarget(object):
    def __init__(self, target_variable_set):
        if not isinstance(target_variable_set, tuple):
            raise TypeError('Target should be a tuple of two numeric variables')

        self.target_variable_set = target_variable_set

    def __repr__(self):
        return "T: " + str(self.target_variable_set)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return [var for var in self.target_variable_set]

    def get_corr_statistics(self, data, subgroup):
        sg_instances = subgroup.subgroup_description.covers(data)
        instances_dataset_len = len(data)
        instances_sg_len = np.sum(sg_instances)
        complement_sg_len = instances_dataset_len - instances_sg_len

        all_target_values_x1 = data[self.target_variable_set[0]]
        all_target_values_x2 = data[self.target_variable_set[1]]
        sg_target_values_x1 = all_target_values_x1[sg_instances]
        sg_target_values_x2 = all_target_values_x2[sg_instances]
        sg_complement_x1 = all_target_values_x1[~all_target_values_x1.index.isin(sg_instances)]
        sg_complement_x2 = all_target_values_x2[~all_target_values_x2.index.isin(sg_instances)]

        # mean prediciton change
        dataset_mean_x2 = np.mean(all_target_values_x2)
        sg_mean_x2 = np.mean(sg_target_values_x2)

        # pearson correlation
        try:
            sg_corr = pearsonr(sg_target_values_x1, sg_target_values_x2)[0]
            dataset_corr = pearsonr(all_target_values_x1, all_target_values_x2)[0]
            sg_complement_corr = pearsonr(sg_complement_x1, sg_complement_x2)[0]
        except ValueError:
            sg_corr = dataset_corr = sg_complement_corr = 0

        return (instances_dataset_len, dataset_corr,
                instances_sg_len, sg_corr,
                complement_sg_len, sg_complement_corr,
                dataset_mean_x2, sg_mean_x2)

    def calculate_corr_statistics(self, data, subgroup):
        args = self.get_corr_statistics(data, subgroup)
        instances_dataset_len = args[0]
        dataset_corr = args[1]
        instances_sg_len = args[2]
        sg_corr = args[3]
        complement_sg_len = args[4]
        sg_complement_corr = args[5]
        dataset_mean_pred = args[6]
        sg_mean_pred = args[7]

        subgroup.statistics['sg_size'] = instances_sg_len
        subgroup.statistics['dataset_size'] = instances_dataset_len
        subgroup.statistics['complement_sg_size'] = complement_sg_len
        subgroup.statistics['sg_corr'] = sg_corr
        subgroup.statistics['dataset_corr'] = dataset_corr
        subgroup.statistics['complement_sg_corr'] = sg_complement_corr
        subgroup.statistics['dataset_mean_pred'] = dataset_mean_pred
        subgroup.statistics['sg_mean_pred'] = sg_mean_pred
        try:
            subgroup.statistics['corr_lift'] = subgroup.statistics['sg_corr'] / subgroup.statistics['dataset_corr']
        except ZeroDivisionError:
            subgroup.statistics['corr_lift'] = 0
        return subgroup.statistics


class CorrelationQF(CorrelationModelMeasure):

    def __init__(self, measure='entropy'):
        """
        correlation model quality measurement
        Parameters:
        measure: str
                'abs_diff': absolute difference of corr between sg and it's complement subgroup
                'entropy': considering the entropy of two subgroups H(p)*abs_diff
                'significance_test': quality defined by 1-p_value
        """
        self.measure = measure

    def corr_qf(self, **statistics):
        complement_sg_size = statistics['complement_sg_size']
        complement_sg_corr = statistics['complement_sg_corr']
        sg_size = statistics['sg_size']
        sg_corr = statistics['sg_corr']
        if sg_size == 0 or np.isnan(sg_corr):
            return 0
        if self.measure == 'abs_diff':
            return round(np.abs(sg_corr - complement_sg_corr), 4)
        elif self.measure == 'entropy':
            n = sg_size / (sg_size + complement_sg_size)
            n_bar = 1 - n
            entropy = -(np.log2(n) * n + np.log2(n_bar) * n_bar)
            return round(entropy * np.abs(sg_corr - complement_sg_corr), 4)
        elif self.measure == 'significance_test':
            z = 0.5 * np.log((1 + sg_corr) / (1 - sg_corr))
            z_bar = 0.5 * np.log((1 + complement_sg_corr) / (1 - complement_sg_corr))
            if sg_size > 25 and complement_sg_size > 25:
                denominator = np.sqrt(1 / (sg_size - 3) + 1 / (complement_sg_size - 3))
                z_score = (z - z_bar) / denominator
                p_value = norm.sf(abs(z_score)) * 2  # twosided
                print('quality: ', 1 - p_value)
                return 1 - p_value
            else:
                return 0
        else:
            raise BaseException("measurement is constrained to "
                                "['abs_diff', 'entropy', 'significance_test']")

    def evaluate_from_dataset(self, data, subgroup):
        if not self.is_applicable(subgroup):
            raise BaseException('Correlation model Quality measure can not be used')
        statistics = subgroup.calculate_corr_statistics(data)
        return self.evaluate_from_statistics(**statistics)

    def evaluate_from_statistics(self, **statistics):
        return self.corr_qf(**statistics)

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, ComplexTarget)
