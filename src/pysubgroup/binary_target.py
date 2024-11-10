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
    """Binary target for classic subgroup discovery with boolean targets.

    Stores the target attribute and value, and computes various statistics related to
    the target within a subgroup.
    """

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
        Initialize a BinaryTarget instance.

        Creates a new target for the boolean model class (classic subgroup discovery).
        If target_attribute and target_value are given, the target_selector is computed
        using the attribute and value.

        Parameters:
            target_attribute (str, optional): The name of the target attribute.
            target_value (any, optional): The value of the target attribute.
            target_selector (Selector, optional): A predefined target selector.

        Raises:
            ValueError: If both target_selector and target_attribute/target_value
                        are provided, or if none are provided.
        """
        if target_attribute is not None and target_value is not None:
            if target_selector is not None:
                raise ValueError(
                    "BinaryTarget is to be constructed "
                    "EITHER by a selector OR by attribute/value pair"
                )
            target_selector = EqualitySelector(target_attribute, target_value)
        if target_selector is None:
            raise ValueError("No target selector given")
        self.target_selector = target_selector

    def __repr__(self):
        """String representation of the BinaryTarget."""
        return "T: " + str(self.target_selector)

    def __eq__(self, other):
        """Check equality based on the instance dictionary."""
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        """Define less-than comparison for sorting purposes."""
        return str(self) < str(other)

    def covers(self, instance):
        """Determine whether the target selector covers the given instance.

        Parameters:
            instance (pandas DataFrame): The data instance to check.

        Returns:
            numpy.ndarray: Boolean array indicating coverage.
        """
        return self.target_selector.covers(instance)

    def get_attributes(self):
        """Get the attribute names used in the target.

        Returns:
            tuple: A tuple containing the attribute name.
        """
        return (self.target_selector.attribute_name,)

    def get_base_statistics(self, subgroup, data):
        """Compute basic statistics for the target within the subgroup and dataset.

        Parameters:
            subgroup: The subgroup for which to compute statistics.
            data (pandas DataFrame): The dataset.

        Returns:
            tuple: Contains instances_dataset, positives_dataset,
                   instances_subgroup, positives_subgroup.
        """
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
        """Calculate various statistics for the subgroup.

        Parameters:
            subgroup: The subgroup for which to calculate statistics.
            data (pandas DataFrame): The dataset.
            cached_statistics (dict, optional): Previously computed statistics.

        Returns:
            dict: A dictionary containing various statistical measures.
        """
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
    """Quality function for binary targets based on positive instances."""

    tpl = namedtuple("PositivesQF_parameters", ("size_sg", "positives_count"))

    def __init__(self):
        """Initialize the SimplePositivesQF."""
        self.dataset_statistics = None
        self.positives = None
        self.has_constant_statistics = False
        self.required_stat_attrs = ("size_sg", "positives_count")

    def calculate_constant_statistics(self, data, target):
        """Calculate statistics that remain constant for the dataset.

        Parameters:
            data (pandas DataFrame): The dataset.
            target (BinaryTarget): The target definition.

        Raises:
            AssertionError: If the target is not an instance of BinaryTarget.
        """
        assert isinstance(target, BinaryTarget)
        self.positives = target.covers(data)
        self.dataset_statistics = SimplePositivesQF.tpl(
            len(data), np.sum(self.positives)
        )
        self.has_constant_statistics = True

    def calculate_statistics(
        self, subgroup, target, data, statistics=None
    ):  # pylint: disable=unused-argument
        """Calculate statistics specific to the subgroup.

        Parameters:
            subgroup: The subgroup for which to calculate statistics.
            target (BinaryTarget): The target definition.
            data (pandas DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            namedtuple: Contains size_sg and positives_count for the subgroup.
        """
        cover_arr, size_sg = get_cover_array_and_size(
            subgroup, len(self.positives), data
        )
        return SimplePositivesQF.tpl(
            size_sg, np.count_nonzero(self.positives[cover_arr])
        )

    # <<< GpGrowth >>>
    def gp_get_stats(self, row_index):
        """Get statistics for a single row (used in GP-Growth algorithms).

        Parameters:
            row_index (int): The index of the row.

        Returns:
            numpy.ndarray: Array containing [1, positives[row_index]].
        """
        return np.array([1, self.positives[row_index]], dtype=int)

    def gp_get_null_vector(self):
        """Get a null vector for initialization in GP-Growth algorithms.

        Returns:
            numpy.ndarray: Zero-initialized array of size 2.
        """
        return np.zeros(2)

    def gp_merge(self, left, right):
        """Merge two statistics vectors by summing them.

        Parameters:
            left (numpy.ndarray): Left statistics vector.
            right (numpy.ndarray): Right statistics vector.
        """
        left += right

    def gp_get_params(self, _cover_arr, v):
        """Extract parameters from the statistics vector.

        Parameters:
            _cover_arr: Unused parameter.
            v (numpy.ndarray): Statistics vector.

        Returns:
            namedtuple: Contains size_sg and positives_count.
        """
        return SimplePositivesQF.tpl(v[0], v[1])

    def gp_to_str(self, stats):
        """Convert statistics to a string representation.

        Parameters:
            stats (numpy.ndarray): Statistics vector.

        Returns:
            str: String representation of the statistics.
        """
        return " ".join(map(str, stats))

    def gp_size_sg(self, stats):
        """Get the size of the subgroup from the statistics.

        Parameters:
            stats (numpy.ndarray): Statistics vector.

        Returns:
            int: Size of the subgroup.
        """
        return stats[0]

    @property
    def gp_requires_cover_arr(self):
        """Indicate whether the GP-Growth algorithm requires a cover array.

        Returns:
            bool: False, since cover array is not required.
        """
        return False


# TODO Make ChiSquared useful for real nominal data not just binary
#      Introduce Enum for direction
#      Maybe it is possible to give an optimistic estimate for ChiSquared
class ChiSquaredQF(SimplePositivesQF):  # pragma: no cover
    """
    ChiSquaredQF tests for statistical independence
    of a subgroup against its complement.

    Calculates the chi-squared statistic or p-value to measure the
    significance of the difference between the subgroup and the dataset.
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
        Perform chi-squared test of statistical independence.

        Tests whether a subgroup is statistically independent
        from its complement (see scipy.stats.chi2_contingency).

        Parameters:
            instances_dataset (int): Total number of instances in the dataset.
            positives_dataset (int): Total number of positive instances in the dataset.
            instances_subgroup (int): Number of instances in the subgroup.
            positives_subgroup (int): Number of positive instances in the subgroup.
            min_instances (int, optional): Minimum required instances;
                                           return -inf if less.
            bidirect (bool, optional): If True, both directions are considered
                                       interesting.
            direction_positive (bool, optional): If bidirect is False, specifies the
                                                 direction.
            index (int, optional): Whether to return statistic (0) or p-value (1).

        Returns:
            float: Chi-squared statistic or p-value, depending on the index parameter.
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
        """Perform chi-squared test for weighted data.

        Parameters:
            subgroup: The subgroup for which to calculate the statistic.
            data (pandas DataFrame): The dataset.
            weighting_attribute (str): The attribute used for weighting.
            effective_sample_size (int, optional): Effective sample size.
            min_instances (int, optional): Minimum required instances.

        Returns:
            float: The p-value from the chi-squared test.
        """
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
        Initialize the ChiSquaredQF.

        Parameters:
            direction (str, optional): Direction of deviation of interest
                                       ('both', 'positive', 'negative').
            min_instances (int, optional): Minimum required instances;
                                           return -inf if less.
            stat (str, optional): Use test statistic ('chi2') or the p-value ('p')?
        """
        if direction == "both":
            self.bidirect = True
            self.direction_positive = True
        elif direction == "positive":
            self.bidirect = False
            self.direction_positive = True
        elif direction == "negative":
            self.bidirect = False
            self.direction_positive = False
        else:
            raise ValueError(
                "Invalid direction; must be 'both', 'positive', or 'negative'"
            )
        self.min_instances = min_instances
        self.index = {"chi2": 0, "p": 1}[stat]
        super().__init__()

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup using the chi-squared test.

        Parameters:
            subgroup: The subgroup to evaluate.
            target (BinaryTarget): The target definition.
            data (pandas DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            float: The chi-squared statistic or p-value.
        """
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
    StandardQF which weights the relative size against the difference in averages.

    The StandardQF is a general form of quality function
    which for different values of 'a' is order equivalent to
    many popular quality measures.
    """

    @staticmethod
    def standard_qf(
        a, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup
    ):
        """Compute the standard quality function.

        Parameters:
            a (float): Exponent to trade-off the relative size with difference in means.
            instances_dataset (int): Total number of instances in the dataset.
            positives_dataset (int): Total number of positive instances in the dataset.
            instances_subgroup (int): Number of instances in the subgroup.
            positives_subgroup (int): Number of positive instances in the subgroup.

        Returns:
            float: The computed quality value.
        """
        if not hasattr(instances_subgroup, "__array_interface__") and (
            instances_subgroup == 0
        ):
            return np.nan
        p_subgroup = np.divide(positives_subgroup, instances_subgroup)
        p_dataset = positives_dataset / instances_dataset
        return (instances_subgroup / instances_dataset) ** a * (p_subgroup - p_dataset)

    def __init__(self, a):
        """
        Initialize the StandardQF.

        Parameters:
            a (float): Exponent to trade-off the relative size with the difference in
                       means.
        """
        self.a = a
        super().__init__()

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup using the standard quality function.

        Parameters:
            subgroup: The subgroup to evaluate.
            target (BinaryTarget): The target definition.
            data (pandas DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            float: The computed quality value.
        """
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
        """Compute the optimistic estimate of the quality function.

        Parameters:
            subgroup: The subgroup for which to compute the optimistic estimate.
            target (BinaryTarget): The target definition.
            data (pandas DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            float: The optimistic estimate of the quality value.
        """
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
        """Compute the optimistic generalization of the quality function.

        Parameters:
            subgroup: The subgroup for which to compute the optimistic generalization.
            target (BinaryTarget): The target definition.
            data (pandas DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            float: The optimistic generalization of the quality value.
        """
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
    Lift Quality Function.

    LiftQF is a StandardQF with a=0.
    Thus it treats the difference in ratios as the quality
    without caring about the relative size of a subgroup.
    """

    def __init__(self):
        """Initialize the LiftQF."""
        super().__init__(0.0)


# TODO add true binomial quality function as in
# https://opus.bibliothek.uni-wuerzburg.de/opus4-wuerzburg/frontdoor/index/index/docId/1786
class SimpleBinomialQF(StandardQF):
    """
    Simple Binomial Quality Function.

    SimpleBinomialQF is a StandardQF with a=0.5.
    It is an order-equivalent approximation of the full binomial test
    if the subgroup size is much smaller than the size of the entire dataset.
    """

    def __init__(self):
        """Initialize the SimpleBinomialQF."""
        super().__init__(0.5)


class WRAccQF(StandardQF):
    """
    Weighted Relative Accuracy Quality Function.

    WRAccQF is a StandardQF with a=1.
    It is order-equivalent to the difference in the observed
    and expected number of positive instances.
    """

    def __init__(self):
        """Initialize the WRAccQF."""
        super().__init__(1.0)


#####
# Generalization-Aware Interestingness Measures
#####
class GeneralizationAware_StandardQF(
    GeneralizationAwareQF_stats, BoundedInterestingnessMeasure
):
    """Generalization-Aware Standard Quality Function.

    Extends the StandardQF to consider generalizations during subgroup discovery,
    providing methods for optimistic estimates and aggregate statistics.
    """

    ga_sQF_agg_tuple = namedtuple(
        "ga_sQF_agg_tuple", ["max_p", "min_delta_negatives", "min_negatives"]
    )

    def __init__(self, a, optimistic_estimate_strategy="default"):
        """
        Initialize the GeneralizationAware_StandardQF.

        Parameters:
            a (float): Exponent to trade-off the relative size
                       with the difference in means.
            optimistic_estimate_strategy (str, optional): Strategy for optimistic
                                                          estimates.
        """
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
        """Evaluate the quality of the subgroup considering generalizations.

        Parameters:
            subgroup: The subgroup to evaluate.
            target (BinaryTarget): The target definition.
            data (pandas DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            float: The computed quality value.
        """
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
        """Aggregate statistics using the maximum-based strategy.

        Parameters:
            stats_subgroup: Statistics of the current subgroup.
            list_of_pairs: List of (stats, agg_tuple) for all generalizations.

        Returns:
            The aggregated statistics.
        """
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
        """Compute the optimistic estimate using the maximum-based strategy.

        Parameters:
            subgroup: The subgroup for which to compute the estimate.
            target (BinaryTarget): The target definition.
            data (pandas DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            float: The optimistic estimate of the quality value.
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
        """Read the p-value from the aggregate tuple using the maximum-based strategy.

        Parameters:
            agg_tuple: The aggregate statistics tuple.

        Returns:
            float: The ratio of positives in the aggregate statistics.
        """
        return agg_tuple.positives_count / agg_tuple.size_sg

    def difference_based_optimistic_estimate(self, subgroup, target, data, statistics):
        """Compute the optimistic estimate using the difference-based strategy.

        Parameters:
            subgroup: The subgroup for which to compute the estimate.
            target (BinaryTarget): The target definition.
            data (pandas DataFrame): The dataset.
            statistics (any): Current statistics.

        Returns:
            float: The optimistic estimate of the quality value.
        """
        sg_stats, agg_stats = self.ensure_statistics(subgroup, target, data, statistics)
        if np.isposinf(agg_stats.min_delta_negatives):
            return np.inf
        delta_n = agg_stats.min_delta_negatives
        size_dataset = self.qf.dataset_statistics.size_sg
        if self.qf.a == 0:
            pos = 1
        elif self.qf.a == 1.0:
            pos = sg_stats.positives_count
        else:
            a = self.qf.a
            p_hat = min(np.ceil(a * delta_n / (1 - a)), sg_stats.positives_count)
            pos = p_hat
        tau_diff = pos / (pos + delta_n)
        if sg_stats.size_sg > 0:
            tau_sg = sg_stats.positives_count / sg_stats.size_sg
        else:
            tau_sg = -1
        tau_max = max(tau_diff, tau_sg, agg_stats.max_p)
        return (sg_stats.positives_count / size_dataset) ** self.a * (1 - tau_max)

    def difference_based_agg_function(self, stats_subgroup, list_of_pairs):
        """Aggregate statistics using the difference-based strategy.

        Parameters:
            stats_subgroup: Statistics of the current subgroup.
            list_of_pairs: List of (stats, agg_tuple) for all generalizations.

        Returns:
            namedtuple: Aggregate statistics tuple.
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
        """
        Read the p-value from the aggregate tuple using the difference-based strategy.

        Parameters:
            agg_tuple: The aggregate statistics tuple.

        Returns:
            float: The maximum percentage of positives.
        """
        return agg_tuple.max_p
