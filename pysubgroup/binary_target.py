'''
Created on 29.09.2017

@author: lemmerfn
'''
from collections import namedtuple
from functools import total_ordering
import numpy as np
import scipy.stats

import pysubgroup as ps

from pysubgroup.subgroup_description import EqualitySelector


@total_ordering
class BinaryTarget:

    statistic_types = ('size_sg', 'size_dataset', 'positives_sg', 'positives_dataset', 'size_complement',
                      'relative_size_sg', 'relative_size_complement', 'coverage_sg', 'coverage_complement',
                      'target_share_sg', 'target_share_complement', 'target_share_dataset', 'lift')

    def __init__(self, target_attribute=None, target_value=None, target_selector=None):
        """
        Creates a new target for the boolean model class (classic subgroup discovery).
        If target_attribute and target_value are given, the target_selector is computed using attribute and value
        """
        if target_attribute is not None and target_value is not None:
            if target_selector is not None:
                raise BaseException("BinaryTarget is to be constructed EITHER by a selector OR by attribute/value pair")
            target_selector = EqualitySelector(target_attribute, target_value)
        if target_selector is None:
            raise BaseException("No target selector given")
        self.target_selector = target_selector

    def __repr__(self):
        return "T: " + str(self.target_selector)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def covers(self, instance):
        return self.target_selector.covers(instance)

    def get_attributes(self):
        return [self.target_selector.get_attribute_name()]

    def get_base_statistics(self, subgroup, data):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(data), data)
        positives = self.covers(data)
        instances_subgroup = size_sg
        positives_dataset = np.sum(positives)
        instances_dataset = len(data)
        positives_subgroup = np.sum(positives[cover_arr])
        return instances_dataset, positives_dataset, instances_subgroup, positives_subgroup

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        if cached_statistics is None or not isinstance(cached_statistics, dict):
            statistics = dict()
        elif all(k in cached_statistics for k in BinaryTarget.statistic_types):
            return cached_statistics
        else:
            statistics = cached_statistics

        (instances_dataset, positives_dataset, instances_subgroup, positives_subgroup) = \
            self.get_base_statistics(subgroup, data)
        statistics['size_sg'] = instances_subgroup
        statistics['size_dataset'] = instances_dataset
        statistics['positives_sg'] = positives_subgroup
        statistics['positives_dataset'] = positives_dataset
        statistics['size_complement'] = instances_dataset - instances_subgroup
        statistics['relative_size_sg'] = instances_subgroup / instances_dataset
        statistics['relative_size_complement'] = (instances_dataset - instances_subgroup) / instances_dataset
        statistics['coverage_sg'] = positives_subgroup / positives_dataset
        statistics['coverage_complement'] = (positives_dataset - positives_subgroup) / positives_dataset
        statistics['target_share_sg'] = positives_subgroup / instances_subgroup
        statistics['target_share_complement'] = (positives_dataset - positives_subgroup) / (instances_dataset - instances_subgroup)
        statistics['target_share_dataset'] = positives_dataset / instances_dataset
        statistics['lift'] = statistics['target_share_sg'] / statistics['target_share_dataset']
        return statistics


class SimplePositivesQF(ps.AbstractInterestingnessMeasure):  # pylint: disable=abstract-method
    tpl = namedtuple('PositivesQF_parameters', ('size_sg', 'positives_count'))

    def __init__(self):
        self.dataset_statistics = None
        self.positives = None
        self.has_constant_statistics = False
        self.required_stat_attrs = ('size_sg', 'positives_count')

    def calculate_constant_statistics(self, data, target):
        assert isinstance(target, BinaryTarget)
        self.positives = target.covers(data)
        self.dataset_statistics = SimplePositivesQF.tpl(len(data), np.sum(self.positives))
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(self.positives), data)
        return SimplePositivesQF.tpl(size_sg, np.count_nonzero(self.positives[cover_arr]))

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return (statistics.positives_count / dataset.positives_count)


        

# TODO Make ChiSquared useful for real nominal data not just binary
# TODO Introduce Enum for direction
# TODO Maybe it is possible to give a optimistic estimate for ChiSquared
class ChiSquaredQF(SimplePositivesQF):
    """
    ChiSquaredQF which test for statistical independence of a subgroup against it's complement

    ...

    """

    @staticmethod
    def chi_squared_qf(instances_dataset, positives_dataset, instances_subgroup, positives_subgroup, min_instances=5, bidirect=True, direction_positive=True, index=0):
        """
        Performs chi2 test of statistical independence

        Test whether a subgroup is statistically independent from it's complement (see scipy.stats.chi2_contingency).


        Parameters
        ----------
        instances_dataset, positives_dataset, instances_subgroup, positives_subgroup : int
            counts of subgroup and dataset
        min_instances : int, optional
            number of required instances, if less -inf is returned for that subgroup
        bidirect : bool, optional
            If true both directions are considered interesting else direction_positive decides which direction is interesting
        direction_positive: bool, optional
            Only used if bidirect=False; specifies whether you are interested in positive (True) or negative deviations
        index : {0, 1}, optional
            decides whether the test statistic (0) or the p-value (1) should be used
        """

        if (instances_subgroup < min_instances) or ((instances_dataset - instances_subgroup) < min_instances):
            return float("-inf")

        negatives_subgroup = instances_subgroup - positives_subgroup # pylint: disable=bad-whitespace
        negatives_dataset = instances_dataset - positives_dataset # pylint: disable=bad-whitespace
        negatives_complement = negatives_dataset - negatives_subgroup
        positives_complement = positives_dataset - positives_subgroup

        val = scipy.stats.chi2_contingency([[positives_subgroup, positives_complement],
                                            [negatives_subgroup, negatives_complement]], correction=False)[index]
        if bidirect:
            return val
        p_subgroup = positives_subgroup / instances_subgroup
        p_dataset = positives_dataset / instances_dataset
        if direction_positive and p_subgroup > p_dataset:
            return val
        elif not direction_positive and p_subgroup < p_dataset:
            return val
        return -val

    @staticmethod
    def chi_squared_qf_weighted(subgroup, data, weighting_attribute, effective_sample_size=0, min_instances=5, ):
        (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = subgroup.get_base_statistics(data, weighting_attribute)
        if (instancesSubgroup < min_instances) or ((instancesDataset - instancesSubgroup) < 5):
            return float("inf")
        if effective_sample_size == 0:
            effective_sample_size = ps.effective_sample_size(data[weighting_attribute])
        # p_subgroup = positivesSubgroup / instancesSubgroup
        # p_dataset = positivesDataset / instancesDataset

        negatives_subgroup = instancesSubgroup - positivesSubgroup
        negatives_dataset = instancesDataset - positivesDataset
        positives_complement = positivesDataset - positivesSubgroup
        negatives_complement = negatives_dataset - negatives_subgroup
        val = scipy.stats.chi2_contingency([[positivesSubgroup, positives_complement],
                                            [negatives_subgroup, negatives_complement]], correction=True)[0]
        return scipy.stats.chi2.sf(val * effective_sample_size / instancesDataset, 1)

    def __init__(self, direction='both', min_instances=5, stat='chi2'):
        """
        Parameters
        ----------
        direction : {'both', 'positive', 'negative'}
            direction of deviation that is of interest
        min_instances : int, optional
            number of required instances, if less -inf is returned for that subgroup
        stat : {'chi2', 'p'}
            whether to report the test statistic or the p-value (see scipy.stats.chi2_contingency)
        """
        if direction == 'both':
            self.bidirect = True
            self.direction_positive = True
        if direction == 'positive':
            self.bidirect = False
            self.direction_positive = True
        if direction == 'negative':
            self.bidirect = False
            self.direction_positive = False
        self.min_instances = min_instances
        self.index = {'chi2' : 0, 'p': 1}[stat]
        super().__init__()

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return ChiSquaredQF.chi_squared_qf(dataset.size_sg, dataset.positives_count, statistics.size_sg, statistics.positives_count, self.min_instances, self.bidirect, self.direction_positive, self.index)


class StandardQF(SimplePositivesQF, ps.BoundedInterestingnessMeasure):
    """
    StandardQF which weights the relative size against the difference in averages

    The StandardQF is a general form of quality function which for different values of a is order equivalen to
    many popular quality measures.

    Attributes
    ----------
    a : float
        used as an exponent to scale the relative size to the difference in averages

    """

    @staticmethod
    def standard_qf(a, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        if not hasattr(instances_subgroup, '__array_interface__') and (instances_subgroup == 0):
            return np.nan
        p_subgroup = np.divide(positives_subgroup, instances_subgroup)
        #if instances_subgroup == 0:
        #    return 0
        #p_subgroup = positives_subgroup / instances_subgroup
        p_dataset = positives_dataset / instances_dataset
        return (instances_subgroup / instances_dataset) ** a * (p_subgroup - p_dataset)

    def __init__(self, a):
        """
        Parameters
        ----------
        a : float
            exponent to trade-off the relative size with the difference in means
        """
        self.a = a
        super().__init__()

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQF.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.size_sg, statistics.positives_count)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQF.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.positives_count, statistics.positives_count)

    def optimistic_generalisation(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        pos_remaining = dataset.positives_count - statistics.positives_count
        return StandardQF.standard_qf(self.a, dataset.size_sg, dataset.positives_count, statistics.size_sg + pos_remaining, dataset.positives_count)


class LiftQF(StandardQF):
    """
    Lift Quality Function

    LiftQF is a StandardQF with a=0.
    Thus it treats the difference in ratios as the quality without caring about the relative size of a subgroup.
    """

    def __init__(self):
        """
        """

        super().__init__(0.0)



# TODO add true binomial quality function as in https://opus.bibliothek.uni-wuerzburg.de/opus4-wuerzburg/frontdoor/index/index/docId/1786
class SimpleBinomialQF(StandardQF):
    """
    Simple Binomial Quality Function

    SimpleBinomialQF is a StandardQF with a=0.5.
    It is an order equivalent approximation of the full binomial test if the subgroup size is much smaller than the size of the entire dataset.
    """

    def __init__(self):
        """
        """

        super().__init__(0.5)


class WRAccQF(StandardQF):
    """
    Weighted Relative Accuracy Quality Function

    WRAccQF is a StandardQF with a=1.
    It is order equivalent to the difference in the observed and expected number of positive instances.
    """

    def __init__(self):
        """
        """

        super().__init__(1.0)


#####
# GeneralizationAware Interestingness Measures
#####
class GeneralizationAware_StandardQF(ps.GeneralizationAwareQF_stats):
    def __init__(self, a):
        super().__init__(StandardQF(0))
        self.a = a

    def get_max(self, *args):
        max_ratio = 0.0
        max_stats = None
        for stat in args:
            if stat.size_sg > 0:
                ratio = stat.positives_count / stat.size_sg
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_stats = stat
        return max_stats

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        sg_stats = statistics.subgroup_stats
        general_stats = statistics.generalisation_stats
        if sg_stats.size_sg == 0 or general_stats.size_sg == 0:
            return np.nan

        sg_ratio = sg_stats.positives_count / sg_stats.size_sg
        general_ratio = general_stats.positives_count / general_stats.size_sg
        return (sg_stats.size_sg / self.stats0.size_sg) ** self.a * (sg_ratio - general_ratio)
