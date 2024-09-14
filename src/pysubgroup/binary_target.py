"""
Created on 29.09.2017

@author: lemmerfn
"""
from collections import namedtuple
from functools import total_ordering

import numpy as np

from pysubgroup.measures import (
    AbstractInterestingnessMeasure,
    BoundedInterestingnessMeasure,
    GeneralizationAwareQF_stats,
)

from .subgroup_description import EqualitySelector, get_cover_array_and_size
from .utils import BaseTarget, derive_effective_sample_size


@total_ordering
class BinaryTarget(BaseTarget):
    statistic_types = (
        "size_sg",
        "size_dataset",
        "positives_sg",
        "positives_dataset",
        "size_complement",
        "relative_size_sg",
        "relative_size_complement",
        "coverage_sg",
        "coverage_complement",
        "target_share_sg",
        "target_share_complement",
        "target_share_dataset",
        "lift",
    )

    def __init__(self, target_attribute=None, target_value=None, target_selector=None):
        """
        Creates a new target for the boolean model class (classic subgroup discovery).
        If target_attribute and target_value are given, the target_selector is computed
        using attribute and value
        """
        if target_attribute is not None and target_value is not None:
            if target_selector is not None:
                raise ValueError(
                    "BinaryTarget is to be constructed"
                    "EITHER by a selector OR by attribute/value pair"
                )
            target_selector = EqualitySelector(target_attribute, target_value)
        if target_selector is None:
            raise ValueError("No target selector given")
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
        return (self.target_selector.attribute_name,)

    def get_base_statistics(self, subgroup, data):
        cover_arr, size_sg = get_cover_array_and_size(subgroup, len(data), data)
        positives = self.covers(data)
        instances_subgroup = size_sg
        positives_dataset = np.sum(positives)
        instances_dataset = len(data)
        positives_subgroup = np.sum(positives[cover_arr])
        return (
            instances_dataset,
            positives_dataset,
            instances_subgroup,
            positives_subgroup,
        )

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        if self.all_statistics_present(cached_statistics):
            return cached_statistics

        (
            instances_dataset,
            positives_dataset,
            instances_subgroup,
            positives_subgroup,
        ) = self.get_base_statistics(subgroup, data)
        statistics = {}
        statistics["size_sg"] = instances_subgroup
        statistics["size_dataset"] = instances_dataset
        statistics["positives_sg"] = positives_subgroup
        statistics["positives_dataset"] = positives_dataset
        statistics["size_complement"] = instances_dataset - instances_subgroup
        statistics["relative_size_sg"] = instances_subgroup / instances_dataset
        statistics["relative_size_complement"] = (
            instances_dataset - instances_subgroup
        ) / instances_dataset
        statistics["coverage_sg"] = positives_subgroup / positives_dataset
        statistics["coverage_complement"] = (
            positives_dataset - positives_subgroup
        ) / positives_dataset
        statistics["target_share_sg"] = positives_subgroup / instances_subgroup
        if instances_dataset == instances_subgroup:
            statistics["target_share_complement"] = float("nan")
        else:
            statistics["target_share_complement"] = (
                positives_dataset - positives_subgroup
            ) / (instances_dataset - instances_subgroup)
        statistics["target_share_dataset"] = positives_dataset / instances_dataset
        statistics["lift"] = (
            statistics["target_share_sg"] / statistics["target_share_dataset"]
        )
        return statistics


class SimplePositivesQF(
    AbstractInterestingnessMeasure
):  # pylint: disable=abstract-method
    tpl = namedtuple("PositivesQF_parameters", ("size_sg", "positives_count"))

    def __init__(self):
        self.dataset_statistics = None
        self.positives = None
        self.has_constant_statistics = False
        self.required_stat_attrs = ("size_sg", "positives_count")

    def calculate_constant_statistics(self, data, target):
        assert isinstance(target, BinaryTarget)
        self.positives = target.covers(data)
        self.dataset_statistics = SimplePositivesQF.tpl(
            len(data), np.sum(self.positives)
        )
        self.has_constant_statistics = True

    def calculate_statistics(
        self, subgroup, target, data, statistics=None
    ):  # pylint: disable=unused-argument
        cover_arr, size_sg = get_cover_array_and_size(
            subgroup, len(self.positives), data
        )
        return SimplePositivesQF.tpl(
            size_sg, np.count_nonzero(self.positives[cover_arr])
        )

    # <<< GpGrowth >>>
    def gp_get_stats(self, row_index):
        return np.array([1, self.positives[row_index]], dtype=int)

    def gp_get_null_vector(self):
        return np.zeros(2)

    def gp_merge(self, left, right):
        left += right

    def gp_get_params(self, _cover_arr, v):
        return SimplePositivesQF.tpl(v[0], v[1])

    def gp_to_str(self, stats):
        return " ".join(map(str, stats))

    def gp_size_sg(self, stats):
        return stats[0]

    @property
    def gp_requires_cover_arr(self):
        return False


# TODO Make ChiSquared useful for real nominal data not just binary
#      Introduce Enum for direction
#      Maybe it is possible to give a optimistic estimate for ChiSquared
class ChiSquaredQF(SimplePositivesQF):  # pragma: no cover
    """
    ChiSquaredQF which test for statistical independence
    of a subgroup against it's complement

    ...

    """

    @staticmethod
    def chi_squared_qf(
        instances_dataset,
        positives_dataset,
        instances_subgroup,
        positives_subgroup,
        min_instances=5,
        bidirect=True,
        direction_positive=True,
        index=0,
    ):
        """
        Performs chi2 test of statistical independence

        Test whether a subgroup is statistically independent
        from it's complement (see scipy.stats.chi2_contingency).


        Parameters
        ----------
        instances_dataset,
                positives_dataset,
                instances_subgroup,
                positives_subgroup : int
            counts of subgroup and dataset
        min_instances : int, optional
            number of required instances, if less -inf is returned for that subgroup
        bidirect : bool, optional
            If true both directions are considered interesting
            else direction_positive decides which direction is interesting
        direction_positive: bool, optional
            Only used if bidirect=False; specifies whether you are interested
            in positive (True) or negative deviations
        index : {0, 1}, optional
            decides whether the test statistic (0) or the p-value (1) should be used
        """
        import scipy.stats  # pylint:disable=import-outside-toplevel

        if (instances_subgroup < min_instances) or (
            (instances_dataset - instances_subgroup) < min_instances
        ):
            return float("-inf")

        negatives_subgroup = instances_subgroup - positives_subgroup
        negatives_dataset = instances_dataset - positives_dataset
        negatives_complement = negatives_dataset - negatives_subgroup
        positives_complement = positives_dataset - positives_subgroup

        val = scipy.stats.chi2_contingency(
            [
                [positives_subgroup, positives_complement],
                [negatives_subgroup, negatives_complement],
            ],
            correction=False,
        )[index]
        if bidirect:
            return val
        p_subgroup = positives_subgroup / instances_subgroup
        p_dataset = positives_dataset / instances_dataset
        if direction_positive and p_subgroup > p_dataset:
            return val
        if not direction_positive and p_subgroup < p_dataset:
            return val
        return -val

    @staticmethod
    def chi_squared_qf_weighted(
        subgroup,
        data,
        weighting_attribute,
        effective_sample_size=0,
        min_instances=5,
    ):
        import scipy.stats  # pylint:disable=import-outside-toplevel

        (
            instancesDataset,
            positivesDataset,
            instancesSubgroup,
            positivesSubgroup,
        ) = subgroup.get_base_statistics(data, weighting_attribute)
        if (instancesSubgroup < min_instances) or (
            (instancesDataset - instancesSubgroup) < 5
        ):
            return float("inf")
        if effective_sample_size == 0:
            effective_sample_size = derive_effective_sample_size(
                data[weighting_attribute]
            )
        # p_subgroup = positivesSubgroup / instancesSubgroup
        # p_dataset = positivesDataset / instancesDataset

        negatives_subgroup = instancesSubgroup - positivesSubgroup
        negatives_dataset = instancesDataset - positivesDataset
        positives_complement = positivesDataset - positivesSubgroup
        negatives_complement = negatives_dataset - negatives_subgroup
        val = scipy.stats.chi2_contingency(
            [
                [positivesSubgroup, positives_complement],
                [negatives_subgroup, negatives_complement],
            ],
            correction=True,
        )[0]
        return scipy.stats.chi2.sf(val * effective_sample_size / instancesDataset, 1)

    def __init__(self, direction="both", min_instances=5, stat="chi2"):
        """
        Parameters
        ----------
        direction : {'both', 'positive', 'negative'}
            direction of deviation that is of interest
        min_instances : int, optional
            number of required instances, if less -inf is returned for that subgroup
        stat : {'chi2', 'p'}
            whether to report the test statistic
            or the p-value (see scipy.stats.chi2_contingency)
        """
        if direction == "both":
            self.bidirect = True
            self.direction_positive = True
        if direction == "positive":
            self.bidirect = False
            self.direction_positive = True
        if direction == "negative":
            self.bidirect = False
            self.direction_positive = False
        self.min_instances = min_instances
        self.index = {"chi2": 0, "p": 1}[stat]
        super().__init__()

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return ChiSquaredQF.chi_squared_qf(
            dataset.size_sg,
            dataset.positives_count,
            statistics.size_sg,
            statistics.positives_count,
            self.min_instances,
            self.bidirect,
            self.direction_positive,
            self.index,
        )


class StandardQF(SimplePositivesQF, BoundedInterestingnessMeasure):
    """
    StandardQF which weights the relative size against the difference in averages

    The StandardQF is a general form of quality function
    which for different values of a is order equivalen to
    many popular quality measures.

    Attributes
    ----------
    a : float
        used as an exponent to scale the relative size to the difference in averages

    """

    @staticmethod
    def standard_qf(
        a, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup
    ):
        if not hasattr(instances_subgroup, "__array_interface__") and (
            instances_subgroup == 0
        ):
            return np.nan
        p_subgroup = np.divide(positives_subgroup, instances_subgroup)
        # if instances_subgroup == 0:
        #    return 0
        # p_subgroup = positives_subgroup / instances_subgroup
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
        return StandardQF.standard_qf(
            self.a,
            dataset.size_sg,
            dataset.positives_count,
            statistics.size_sg,
            statistics.positives_count,
        )

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQF.standard_qf(
            self.a,
            dataset.size_sg,
            dataset.positives_count,
            statistics.positives_count,
            statistics.positives_count,
        )

    def optimistic_generalisation(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        pos_remaining = dataset.positives_count - statistics.positives_count
        return StandardQF.standard_qf(
            self.a,
            dataset.size_sg,
            dataset.positives_count,
            statistics.size_sg + pos_remaining,
            dataset.positives_count,
        )


class LiftQF(StandardQF):
    """
    Lift Quality Function

    LiftQF is a StandardQF with a=0.
    Thus it treats the difference in ratios as the quality
    without caring about the relative size of a subgroup.
    """

    def __init__(self):
        """ """

        super().__init__(0.0)


# TODO add true binomial quality function as in
# https://opus.bibliothek.uni-wuerzburg.de/opus4-wuerzburg/frontdoor/index/index/docId/1786 # noqa: E501
class SimpleBinomialQF(StandardQF):
    """
    Simple Binomial Quality Function

    SimpleBinomialQF is a StandardQF with a=0.5.
    It is an order equivalent approximation of the full binomial test
    if the subgroup size is much smaller than the size of the entire dataset.
    """

    def __init__(self):
        """ """

        super().__init__(0.5)


class WRAccQF(StandardQF):
    """
    Weighted Relative Accuracy Quality Function

    WRAccQF is a StandardQF with a=1.
    It is order equivalent to the difference in the observed
    and expected number of positive instances.
    """

    def __init__(self):
        """ """

        super().__init__(1.0)


#####
# GeneralizationAware Interestingness Measures
#####
class GeneralizationAware_StandardQF(
    GeneralizationAwareQF_stats, BoundedInterestingnessMeasure
):
    ga_sQF_agg_tuple = namedtuple(
        "ga_sQF_agg_tuple", ["max_p", "min_delta_negatives", "min_negatives"]
    )

    def __init__(self, a, optimistic_estimate_strategy="default"):
        super().__init__(StandardQF(a))
        if optimistic_estimate_strategy in ("default", "difference"):
            self.optimistic_estimate = self.difference_based_optimistic_estimate
            self.aggregate_statistics = self.difference_based_agg_function
            self.read_p = self.difference_based_read_p
        elif optimistic_estimate_strategy == "max":
            self.optimistic_estimate = self.max_based_optimistic_estimate
            self.aggregate_statistics = self.max_based_aggregate_statistics
            self.read_p = self.max_based_read_p
        else:
            raise ValueError(
                "optimistic_estimate_strategy should be one of "
                "('default', 'max', 'difference')"
            )
        self.a = a

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        sg_stats = statistics.subgroup_stats
        if sg_stats.size_sg == 0:
            return np.nan

        general_stats = statistics.generalisation_stats
        sg_ratio = sg_stats.positives_count / sg_stats.size_sg
        return (sg_stats.size_sg / self.stats0.size_sg) ** self.a * (
            sg_ratio - self.read_p(general_stats)
        )

    def max_based_aggregate_statistics(self, stats_subgroup, list_of_pairs):
        if len(list_of_pairs) == 0:
            return stats_subgroup
        max_ratio = -100
        max_stats = None
        for pair in list_of_pairs:
            ratio = -np.inf
            for agg_stat in pair:
                if agg_stat.size_sg == 0:  # pragma: no cover
                    continue
                ratio = agg_stat.positives_count / agg_stat.size_sg
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_stats = agg_stat

        return max_stats

    def max_based_optimistic_estimate(self, subgroup, target, data, statistics=None):
        """
        Computes the oe as the hypothetical subgroup containing only positive instances
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        sg_stats = statistics.subgroup_stats
        general_stats = statistics.generalisation_stats
        if sg_stats.size_sg == 0 or general_stats.size_sg == 0:
            return np.nan

        general_ratio = general_stats.positives_count / general_stats.size_sg
        return (sg_stats.positives_count / self.stats0.size_sg) ** self.a * (
            1 - general_ratio
        )

    def max_based_read_p(self, agg_tuple):
        return agg_tuple.positives_count / agg_tuple.size_sg

    def difference_based_optimistic_estimate(self, subgroup, target, data, statistics):
        sg_stats, agg_stats = self.ensure_statistics(subgroup, target, data, statistics)
        if np.isposinf(agg_stats.min_delta_negatives):
            return np.inf
        delta_n = agg_stats.min_delta_negatives
        size_dataset = self.qf.dataset_statistics.size_sg
        tau_diff = 0
        if self.qf.a == 0:
            pos = 1
            # return delta_n /(1 + delta_n)
        elif self.qf.a == 1.0:
            pos = sg_stats.positives_count
            # return pos / size_dataset * delta_n /(pos + delta_n)
        else:
            a = self.qf.a
            p_hat = min(np.ceil(a * delta_n / (1 - a)), sg_stats.positives_count)
            pos = p_hat
            # return (p_hat / size_dataset) ** a * delta_n /(p_hat+delta_n)
        tau_diff = pos / (pos + delta_n)
        if sg_stats.size_sg > 0:
            tau_sg = sg_stats.positives_count / sg_stats.size_sg
        else:
            tau_sg = -1
        tau_max = max(tau_diff, tau_sg, agg_stats.max_p)
        return (sg_stats.positives_count / size_dataset) ** self.a * (1 - tau_max)

    def difference_based_agg_function(self, stats_subgroup, list_of_pairs):
        """
        list_of_pairs is a list of (stats, agg_tuple) for all the generalizations
        """

        def get_negatives_count(sg_stats):
            return sg_stats.size_sg - sg_stats.positives_count

        def get_percentage_positives(sg_stats):
            if sg_stats.size_sg == 0:
                return np.nan
            return sg_stats.positives_count / sg_stats.size_sg

        if len(list_of_pairs) == 0:  # empty pattern
            return GeneralizationAware_StandardQF.ga_sQF_agg_tuple(
                get_percentage_positives(stats_subgroup), np.infty, np.infty
            )

        subgroup_negatives = stats_subgroup.size_sg - stats_subgroup.positives_count
        min_immediate_generalizations_negatives = min(
            get_negatives_count(x.subgroup_stats) for x in list_of_pairs
        )
        min_immediate_generalizations_delta_negatives = min(
            x.generalisation_stats.min_delta_negatives for x in list_of_pairs
        )
        max_percentage_positives = max(
            max(
                get_percentage_positives(x.subgroup_stats), x.generalisation_stats.max_p
            )
            for x in list_of_pairs
        )

        sg_delta_negatives = (
            min_immediate_generalizations_negatives - subgroup_negatives
        )
        min_delta_negatives = min(
            sg_delta_negatives, min_immediate_generalizations_delta_negatives
        )
        return GeneralizationAware_StandardQF.ga_sQF_agg_tuple(
            max_percentage_positives, min_delta_negatives, sg_delta_negatives
        )

    def difference_based_read_p(self, agg_tuple):
        return agg_tuple.max_p
