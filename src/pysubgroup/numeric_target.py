"""
This module defines the NumericTarget and associated quality functions for
subgroup discovery when the target variable is numeric.
"""

import numbers
from collections import namedtuple
from functools import total_ordering

import numpy as np

import pysubgroup as ps


@total_ordering
class NumericTarget:
    """Target class for numeric variables in subgroup discovery.

    Represents a target where the variable of interest is numeric,
    and computes statistics such as mean, median, standard deviation within subgroups.
    """

    statistic_types = (
        "size_sg",
        "size_dataset",
        "mean_sg",
        "mean_dataset",
        "std_sg",
        "std_dataset",
        "median_sg",
        "median_dataset",
        "max_sg",
        "max_dataset",
        "min_sg",
        "min_dataset",
        "mean_lift",
        "median_lift",
    )

    def __init__(self, target_variable):
        """Initialize the NumericTarget with the specified target variable.

        Parameters:
            target_variable (str): The name of the numeric target variable.
        """
        self.target_variable = target_variable

    def __repr__(self):
        """String representation of the NumericTarget."""
        return "T: " + str(self.target_variable)

    def __eq__(self, other):
        """Check equality based on the instance dictionary."""
        return self.__dict__ == other.__dict__  # pragma: no cover

    def __lt__(self, other):
        """Define less-than comparison for sorting purposes."""
        return str(self) < str(other)  # pragma: no cover

    def get_attributes(self):
        """Get a list of attribute names used by the target.

        Returns:
            list: A list containing the target variable name.
        """
        return [self.target_variable]

    def get_base_statistics(self, subgroup, data):
        """Compute basic statistics for the subgroup and dataset.

        Parameters:
            subgroup: The subgroup for which to compute statistics.
            data (pandas.DataFrame): The dataset.

        Returns:
            tuple: (instances_dataset, mean_dataset, instances_subgroup, mean_sg)
        """
        cover_arr, size_sg = ps.get_cover_array_and_size(subgroup, len(data), data)
        all_target_values = data[self.target_variable]
        sg_target_values = all_target_values[cover_arr]
        instances_dataset = len(data)
        instances_subgroup = size_sg
        mean_sg = np.mean(sg_target_values)
        mean_dataset = np.mean(all_target_values)
        return (instances_dataset, mean_dataset, instances_subgroup, mean_sg)

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        """Calculate various statistics for the subgroup and dataset.

        Parameters:
            subgroup: The subgroup for which to calculate statistics.
            data (pandas.DataFrame): The dataset.
            cached_statistics (dict, optional): Previously computed statistics.

        Returns:
            dict: A dictionary containing statistical measures.
        """
        if cached_statistics is None or not isinstance(cached_statistics, dict):
            statistics = {}
        elif all(k in cached_statistics for k in NumericTarget.statistic_types):
            return cached_statistics
        else:
            statistics = cached_statistics

        cover_arr, _ = ps.get_cover_array_and_size(subgroup, len(data), data)
        all_target_values = data[self.target_variable].to_numpy()
        sg_target_values = all_target_values[cover_arr]

        statistics["size_sg"] = len(sg_target_values)
        statistics["size_dataset"] = len(data)
        statistics["mean_sg"] = np.mean(sg_target_values)
        statistics["mean_dataset"] = np.mean(all_target_values)
        statistics["std_sg"] = np.std(sg_target_values)
        statistics["std_dataset"] = np.std(all_target_values)
        statistics["median_sg"] = np.median(sg_target_values)
        statistics["median_dataset"] = np.median(all_target_values)
        statistics["max_sg"] = np.max(sg_target_values)
        statistics["max_dataset"] = np.max(all_target_values)
        statistics["min_sg"] = np.min(sg_target_values)
        statistics["min_dataset"] = np.min(all_target_values)
        statistics["mean_lift"] = statistics["mean_sg"] / statistics["mean_dataset"]
        statistics["median_lift"] = (
            statistics["median_sg"] / statistics["median_dataset"]
        )
        return statistics


def read_median(tpl):
    """Extract the median value from a namedtuple.

    Parameters:
        tpl (namedtuple): A namedtuple containing a 'median' field.

    Returns:
        float: The median value.
    """
    return tpl.median


def read_mean(tpl):
    """Extract the mean value from a namedtuple.

    Parameters:
        tpl (namedtuple): A namedtuple containing a 'mean' field.

    Returns:
        float: The mean value.
    """
    return tpl.mean


def calc_sorted_median(arr):
    """Calculate the median of a sorted array.

    Parameters:
        arr (numpy.ndarray): A sorted array.

    Returns:
        float: The median value.
    """
    half = (len(arr) - 1) // 2
    if len(arr) % 2 == 0:
        return (arr[half] + arr[half + 1]) / 2
    else:
        return arr[half]


class StandardQFNumeric(ps.BoundedInterestingnessMeasure):
    """Standard Quality Function for numeric targets.

    This quality function computes interestingness of subgroups based on
    the difference between subgroup mean (or median) and dataset mean (or median),
    weighted by the size of the subgroup raised to the power of 'a'.

    Attributes:
        a (float): Exponent to trade off between subgroup size and difference in means.
        invert (bool): Whether to invert the quality function (not used currently).
        estimator (str): Strategy for optimistic estimation ('sum', 'max', 'order').
        centroid (str): Central tendency measure ('mean', 'median', 'sorted_median').
    """

    tpl = namedtuple("StandardQFNumeric_parameters", ("size_sg", "mean", "estimate"))
    mean_tpl = tpl
    median_tpl = namedtuple(
        "StandardQFNumeric_median_parameters", ("size_sg", "median", "estimate")
    )

    @staticmethod
    def standard_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
        """Compute the standard quality function for numeric targets.

        Parameters:
            a (float): Exponent for weighting the subgroup size.
            _ : Unused parameter (size of dataset).
            mean_dataset (float): Mean of the target variable in the dataset.
            instances_subgroup (int): Number of instances in the subgroup.
            mean_sg (float): Mean of the target variable in the subgroup.

        Returns:
            float: The computed quality value.
        """
        return instances_subgroup**a * (mean_sg - mean_dataset)

    def __init__(self, a, invert=False, estimator="default", centroid="mean"):
        """Initialize the StandardQFNumeric.

        Parameters:
            a (float): Exponent for weighting the subgroup size.
            invert (bool): Whether to invert the quality function (not used currently).
            estimator (str): Strategy for optimistic estimation ('sum', 'max', 'order').
            centroid (str): Central tendency measure to use. Can be one of
                            ('mean', 'median', 'sorted_median').

        Raises:
            ValueError: If 'a' is not a number.
            ValueError: If 'centroid' is not one of 'mean', 'median', 'sorted_median'.
            ValueError: If 'estimator' is invalid.
        """
        if not isinstance(a, numbers.Number):
            raise ValueError(f"a is not a number. Received a={a}")
        self.a = a
        self.invert = invert

        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False

        if centroid == "median":
            if estimator == "default":
                estimator = "max"
            assert estimator in (
                "max",
                "order",
            ), "For median only estimator = max or order are possible"
            self.required_stat_attrs = ("size_sg", "median")
            self.agg = np.median
            self.tpl = StandardQFNumeric.median_tpl
            self.read_centroid = read_median
        elif centroid == "sorted_median":
            if estimator == "default":
                estimator = "max"
            assert estimator in (
                "max",
                "order",
            ), "For median only estimator = max or order are possible"
            self.required_stat_attrs = ("size_sg", "median")
            self.agg = calc_sorted_median
            self.tpl = StandardQFNumeric.median_tpl
            self.read_centroid = read_median
        elif centroid == "mean":
            if estimator == "default":
                estimator = "sum"
            self.required_stat_attrs = ("size_sg", "mean")
            self.agg = np.mean
            self.tpl = StandardQFNumeric.mean_tpl
            self.read_centroid = read_mean
        else:
            raise ValueError(
                f"centroid was {centroid} which is not in (median, sorted_median, mean)"
            )

        if estimator == "sum":
            self.estimator = StandardQFNumeric.Summation_Estimator(self)
        elif estimator == "max":
            self.estimator = StandardQFNumeric.Max_Estimator(self)
        elif estimator == "average":
            self.estimator = StandardQFNumeric.Max_Estimator(self)
        elif estimator == "order":
            if centroid == "mean":
                self.estimator = StandardQFNumeric.MeanOrdering_Estimator(self)
            else:
                raise NotImplementedError(
                    "Order estimation is not implemented for median qf"
                )
        else:
            raise ValueError(
                "estimator is not one of the following: "
                + str(["sum", "average", "order"])
            )

    def calculate_constant_statistics(self, data, target):
        """Calculate statistics that remain constant for the dataset.

        Parameters:
            data (pandas.DataFrame): The dataset.
            target (NumericTarget): The target definition.
        """
        data = self.estimator.get_data(data, target)
        self.all_target_values = data[target.target_variable].to_numpy()
        target_centroid = self.agg(self.all_target_values)
        data_size = len(data)
        self.dataset_statistics = self.tpl(data_size, target_centroid, None)
        self.estimator.calculate_constant_statistics(data, target)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup using the standard quality function.

        Parameters:
            subgroup: The subgroup to evaluate.
            target (NumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Previously computed statistics.

        Returns:
            float: The computed quality value.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQFNumeric.standard_qf_numeric(
            self.a,
            dataset.size_sg,
            self.read_centroid(dataset),
            statistics.size_sg,
            self.read_centroid(statistics),
        )

    def calculate_statistics(
        self, subgroup, target, data, statistics=None
    ):  # pylint: disable=unused-argument
        """Calculate statistics specific to the subgroup.

        Parameters:
            subgroup: The subgroup for which to calculate statistics.
            target (NumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            namedtuple: Contains size_sg, mean or median, and estimate.
        """
        cover_arr, sg_size = ps.get_cover_array_and_size(
            subgroup, len(self.all_target_values), data
        )
        sg_centroid = 0
        sg_target_values = 0
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            sg_centroid = self.agg(sg_target_values)
            estimate = self.estimator.get_estimate(
                subgroup, sg_size, sg_centroid, cover_arr, sg_target_values
            )
        else:
            estimate = float("-inf")
        return self.tpl(sg_size, sg_centroid, estimate)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """Compute the optimistic estimate of the quality function.

        Parameters:
            subgroup: The subgroup for which to compute the optimistic estimate.
            target (NumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Previously computed statistics.

        Returns:
            float: The optimistic estimate of the quality value.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate

    class Summation_Estimator:
        r"""Estimator for optimistic estimate using summation strategy.

        This estimator calculates the optimistic estimate as a hypothetical subgroup
        which contains only instances with value greater than the dataset mean and
        is of maximal size.

        From Florian Lemmerich's Dissertation [section 4.2.2.1, Theorem 2 (page 81)]:

        .. math::
            oe(sg) = \sum_{x \in sg, T(x)>0} (T(sg) - \mu_0)
        """

        def __init__(self, qf):
            """Initialize the Summation_Estimator.

            Parameters:
                qf (StandardQFNumeric): Reference to the quality function instance.
            """
            self.qf = qf
            self.indices_greater_centroid = None
            self.target_values_greater_centroid = None

        def get_data(self, data, target):  # pylint: disable=unused-argument
            """Prepare data for estimation (no changes for this estimator).

            Parameters:
                data (pandas.DataFrame): The dataset.
                target (NumericTarget): The target definition.

            Returns:
                pandas.DataFrame: The unmodified dataset.
            """
            return data

        def calculate_constant_statistics(
            self, data, target
        ):  # pylint: disable=unused-argument
            """Calculate constant statistics needed for estimation.

            Parameters:
                data (pandas.DataFrame): The dataset.
                target (NumericTarget): The target definition.
            """
            self.indices_greater_centroid = (
                self.qf.all_target_values
                > self.qf.read_centroid(self.qf.dataset_statistics)
            )
            self.target_values_greater_centroid = self.qf.all_target_values

        def get_estimate(
            self, subgroup, sg_size, sg_centroid, cover_arr, _
        ):  # pylint: disable=unused-argument
            """Compute the optimistic estimate for the subgroup.

            Parameters:
                subgroup: The subgroup description.
                sg_size (int): Size of the subgroup.
                sg_centroid (float): Mean or median of the subgroup.
                cover_arr (numpy.ndarray): Boolean array indicating subgroup instances.
                _ : Unused parameter.

            Returns:
                float: The optimistic estimate.
            """
            larger_than_centroid = self.target_values_greater_centroid[cover_arr][
                self.indices_greater_centroid[cover_arr]
            ]
            size_greater_centroid = len(larger_than_centroid)
            sum_greater_centroid = np.sum(larger_than_centroid)

            return sum_greater_centroid - size_greater_centroid * self.qf.read_centroid(
                self.qf.dataset_statistics
            )

    class Max_Estimator:
        r"""Estimator for optimistic estimate using maximum value strategy.

        This estimator calculates the optimistic estimate based on the maximum value
        greater than the dataset centroid.

        From Florian Lemmerich's Dissertation [section 4.2.2.1, Theorem 4 (page 82)]:

        .. math::
            oe(sg) = n_{>\mu_0}^a (T^{\max}(sg) - \mu_0)
        """

        def __init__(self, qf):
            """Initialize the Max_Estimator.

            Parameters:
                qf (StandardQFNumeric): Reference to the quality function instance.
            """
            self.qf = qf
            self.indices_greater_centroid = None
            self.target_values_greater_centroid = None

        def get_data(self, data, target):  # pylint: disable=unused-argument
            """Prepare data for estimation (no changes for this estimator).

            Parameters:
                data (pandas.DataFrame): The dataset.
                target (NumericTarget): The target definition.

            Returns:
                pandas.DataFrame: The unmodified dataset.
            """
            return data

        def calculate_constant_statistics(
            self, data, target
        ):  # pylint: disable=unused-argument
            """Calculate constant statistics needed for estimation.

            Parameters:
                data (pandas.DataFrame): The dataset.
                target (NumericTarget): The target definition.
            """
            self.indices_greater_centroid = (
                self.qf.all_target_values
                > self.qf.read_centroid(self.qf.dataset_statistics)
            )
            self.target_values_greater_centroid = self.qf.all_target_values

        def get_estimate(
            self, subgroup, sg_size, sg_centroid, cover_arr, _
        ):  # pylint: disable=unused-argument
            """Compute the optimistic estimate for the subgroup.

            Parameters:
                subgroup: The subgroup description.
                sg_size (int): Size of the subgroup.
                sg_centroid (float): Mean or median of the subgroup.
                cover_arr (numpy.ndarray): Boolean array indicating subgroup instances.
                _ : Unused parameter.

            Returns:
                float: The optimistic estimate.
            """
            larger_than_centroid = self.target_values_greater_centroid[cover_arr][
                self.indices_greater_centroid[cover_arr]
            ]
            size_greater_centroid = len(larger_than_centroid)
            if size_greater_centroid == 0:
                return -np.inf
            max_greater_centroid = np.max(larger_than_centroid)

            return size_greater_centroid**self.qf.a * (
                max_greater_centroid - self.qf.read_centroid(self.qf.dataset_statistics)
            )

    class MeanOrdering_Estimator:
        """Estimator for optimistic estimate using mean ordering strategy.

        This estimator sorts the target values and computes the optimal subgroup by
        considering prefixes of the sorted list.
        """

        def __init__(self, qf):
            """Initialize the MeanOrdering_Estimator.

            Parameters:
                qf (StandardQFNumeric): Reference to the quality function instance.
            """
            self.qf = qf
            self.indices_greater_centroid = None
            self._get_estimate = self.get_estimate_numpy
            self.use_numba = True
            self.numba_in_place = False

        def get_data(self, data, target):
            """Prepare data by sorting according to the target variable.

            Parameters:
                data (pandas.DataFrame): The dataset.
                target (NumericTarget): The target definition.

            Returns:
                pandas.DataFrame: The sorted dataset.
            """
            data.sort_values(target.get_attributes()[0], ascending=False, inplace=True)
            return data

        def calculate_constant_statistics(
            self, data, target
        ):  # pylint: disable=unused-argument
            """Set up the estimation function, possibly using Numba for speed.

            Parameters:
                data (pandas.DataFrame): The dataset.
                target (NumericTarget): The target definition.
            """
            if not self.use_numba or self.numba_in_place:
                return
            try:
                from numba import njit  # pylint: disable=import-outside-toplevel

                # Use Numba for speedup
            except ImportError:  # pragma: no cover
                return

            @njit
            def estimate_numba(values_sg, a, mean_dataset):  # pragma: no cover
                n = 1
                sum_values = 0
                max_value = -(10**10)
                for val in values_sg:
                    sum_values += val
                    mean_sg = sum_values / n
                    quality = n**a * (mean_sg - mean_dataset)
                    if quality > max_value:
                        max_value = quality
                    n += 1
                return max_value

            self._get_estimate = estimate_numba
            self.numba_in_place = True

        def get_estimate(
            self, subgroup, sg_size, sg_mean, cover_arr, target_values_sg
        ):  # pylint: disable=unused-argument
            """Compute the optimistic estimate for the subgroup.

            Parameters:
                subgroup: The subgroup description.
                sg_size (int): Size of the subgroup.
                sg_mean (float): Mean of the subgroup.
                cover_arr (numpy.ndarray): Boolean array indicating subgroup instances.
                target_values_sg (numpy.ndarray): Target values in the subgroup.

            Returns:
                float: The optimistic estimate.
            """
            if self.numba_in_place:
                return self._get_estimate(
                    target_values_sg, self.qf.a, self.qf.dataset_statistics.mean
                )
            else:
                return self._get_estimate(
                    target_values_sg, self.qf.a, self.qf.dataset_statistics.mean
                )

        def get_estimate_numpy(self, values_sg, _, mean_dataset):
            """Compute the optimistic estimate using NumPy.

            Parameters:
                values_sg (numpy.ndarray): Sorted target values in the subgroup.
                _ : Unused parameter.
                mean_dataset (float): Mean of the dataset.

            Returns:
                float: The optimistic estimate.
            """
            target_values_cs = np.cumsum(values_sg)
            sizes = np.arange(1, len(target_values_cs) + 1)
            mean_values = target_values_cs / sizes
            stats = StandardQFNumeric.mean_tpl(sizes, mean_values, mean_dataset)
            qualities = self.qf.evaluate(None, None, None, stats)
            optimistic_estimate = np.max(qualities)
            return optimistic_estimate


class StandardQFNumericMedian(ps.BoundedInterestingnessMeasure):
    """Quality function for numeric targets using median (deprecated).

    Note:
        This class is no longer supported. Use StandardQFNumeric with centroid='median'
        instead.
    """

    tpl = namedtuple(
        "StandardQFNumericMedian_parameters",
        (
            "size_sg",
            "median",
            "estimate",
        ),  # this is here to allow older pickles to be loaded
    )

    def __init__(
        self,
    ):
        """Initialize the StandardQFNumericMedian (raises NotImplementedError)."""
        raise NotImplementedError(
            "StandardQFNumericMedian is no longer supported use "
            "StandardQFNumeric(centroid='median' instead)"
        )  # pragma: no cover


class StandardQFNumericTscore(ps.BoundedInterestingnessMeasure):
    """Quality function for numeric targets using T-score."""

    tpl = namedtuple(
        "StandardQFNumericTscore_parameters", ("size_sg", "mean", "std", "estimate")
    )

    @staticmethod
    def t_score(mean_dataset, instances_subgroup, mean_sg, std_sg):
        """Compute the T-score for the subgroup.

        Parameters:
            mean_dataset (float): Mean of the dataset.
            instances_subgroup (int): Number of instances in the subgroup.
            mean_sg (float): Mean of the subgroup.
            std_sg (float): Standard deviation of the subgroup.

        Returns:
            float: The computed T-score.
        """
        if std_sg == 0:
            return 0
        else:
            return (instances_subgroup**0.5 * (mean_sg - mean_dataset)) / std_sg

    def __init__(self, invert=False):
        """Initialize the StandardQFNumericTscore.

        Parameters:
            invert (bool): Whether to invert the quality function (not used currently).
        """
        self.invert = invert
        self.required_stat_attrs = ("size_sg", "mean", "std")
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, data, target):
        """Calculate statistics that remain constant for the dataset.

        Parameters:
            data (pandas.DataFrame): The dataset.
            target (NumericTarget): The target definition.
        """
        self.all_target_values = data[target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        target_std = np.std(self.all_target_values)
        data_size = len(data)
        self.dataset_statistics = StandardQFNumericTscore.tpl(
            data_size, target_mean, target_std, np.inf
        )
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup using the T-score.

        Parameters:
            subgroup: The subgroup to evaluate.
            target (NumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Previously computed statistics.

        Returns:
            float: The computed T-score.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return StandardQFNumericTscore.t_score(
            dataset.mean,
            statistics.size_sg,
            statistics.mean,
            statistics.std,
        )

    def calculate_statistics(
        self, subgroup, target, data, statistics=None
    ):  # pylint: disable=unused-argument
        """Calculate statistics specific to the subgroup.

        Parameters:
            subgroup: The subgroup for which to calculate statistics.
            target (NumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            namedtuple: Contains size_sg, mean, std, and estimate.
        """
        cover_arr, sg_size = ps.get_cover_array_and_size(
            subgroup, len(self.all_target_values), data
        )
        sg_mean = np.array([0])
        sg_std = np.array([0])
        sg_target_values = 0
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            sg_mean = np.mean(sg_target_values)
            sg_std = np.std(sg_target_values)
            estimate = np.inf
        else:
            estimate = float("-inf")
        return StandardQFNumericTscore.tpl(sg_size, sg_mean, sg_std, estimate)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """Compute the optimistic estimate of the quality function.

        Parameters:
            subgroup: The subgroup for which to compute the optimistic estimate.
            target: The target definition.
            data: The dataset.
            statistics (any, optional): Previously computed statistics.

        Returns:
            float: The optimistic estimate of the quality value.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate


class GeneralizationAware_StandardQFNumeric(ps.GeneralizationAwareQF_stats):
    """Generalization-Aware Standard Quality Function for Numeric Targets.

    Extends StandardQFNumeric to consider generalizations during subgroup discovery,
    providing methods for optimistic estimates and aggregate statistics.
    """

    def __init__(self, a, invert=False, estimator="default", centroid="mean"):
        """Initialize the GeneralizationAware_StandardQFNumeric.

        Parameters:
            a (float): Exponent for weighting the subgroup size.
            invert (bool): Whether to invert the quality function (not used currently).
            estimator (str): Strategy for optimistic estimation.
            centroid (str): Central tendency measure. Can be one of
                            ('mean', 'median', 'sorted_median').
        """
        super().__init__(
            StandardQFNumeric(a, invert=invert, estimator=estimator, centroid=centroid)
        )

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup considering generalizations.

        Parameters:
            subgroup: The subgroup to evaluate.
            target (NumericTarget): The target definition.
            data (pandas.DataFrame): The dataset.
            statistics (any, optional): Previously computed statistics.

        Returns:
            float: The computed quality value.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        sg_stats = statistics.subgroup_stats
        general_stats = statistics.generalisation_stats
        if sg_stats.size_sg == 0:
            return np.nan
        read_centroid = self.qf.read_centroid
        return (sg_stats.size_sg / self.stats0.size_sg) ** self.qf.a * (
            read_centroid(sg_stats) - read_centroid(general_stats)
        )

    def aggregate_statistics(self, stats_subgroup, list_of_pairs):
        """Aggregate statistics from generalizations.

        Parameters:
            stats_subgroup: Statistics of the current subgroup.
            list_of_pairs: List of (stats, agg_stats) tuples from generalizations.

        Returns:
            The aggregated statistics.
        """
        read_centroid = self.qf.read_centroid
        if len(list_of_pairs) == 0:
            return stats_subgroup
        max_centroid = 0.0
        max_stats = None
        for stat, agg_stat in list_of_pairs:
            if stat.size_sg == 0:
                continue
            centroid = max(read_centroid(agg_stat), read_centroid(stat))
            if centroid > max_centroid:
                max_centroid = centroid
                max_stats = stat
        return max_stats
