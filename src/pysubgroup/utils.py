"""
Created on 02.05.2016

@author: lemmerfn
"""
import itertools
from collections.abc import Iterable
from functools import partial
from heapq import heappop, heappush

import numpy as np

import pysubgroup as ps

from .algorithms import SubgroupDiscoveryTask


def str_to_bool(s):
    """
    Converts a string representation of a boolean value to a boolean type.

    Parameters:
        s (str): The string to convert (e.g., 'true', 'False', '1', '0').

    Returns:
        bool: The boolean value represented by the string.

    Raises:
        ValueError: If the string does not represent a valid boolean value.
    """
    s = s.lower()
    if s in ["y", "yes", "t", "true", "on", "1"]:
        return True
    elif s in ["n", "no", "f", "false", "off", "0"]:
        return False

    raise ValueError(f"'{s}' is not a valid string representation of a boolean value")


def minimum_required_quality(result, task):
    """
    Determines the minimum quality required for a subgroup to be considered for
    inclusion in the result set.

    Parameters:
        result (list): The current list of subgroups (heap).
        task (SubgroupDiscoveryTask): The task containing parameters like
        result_set_size and min_quality.

    Returns:
        float: The minimum required quality for a subgroup to be added to the result
        set.
    """
    if len(result) < task.result_set_size:
        return task.min_quality
    else:
        return result[0][0]


def prepare_subgroup_discovery_result(result, task):
    """
    Filters and sorts the result set of subgroups according to the task parameters.

    Parameters:
        result (list): The list of subgroups (heap).
        task (SubgroupDiscoveryTask): The task containing parameters like
                                      result_set_size and min_quality.

    Returns:
        list: The filtered and sorted list of subgroups.
    """
    result_filtered = [tpl for tpl in result if tpl[0] > task.min_quality]
    result_filtered.sort(reverse=True)
    result_filtered = result_filtered[: task.result_set_size]
    return result_filtered


def equal_frequency_discretization(
    data, attribute_name, nbins=5, weighting_attribute=None
):
    """
    Discretizes a numerical attribute into bins with approximately equal frequency.

    Parameters:
        data (DataFrame): The dataset containing the attribute to discretize.
        attribute_name (str): The name of the attribute to discretize.
        nbins (int): The number of bins to create.
        weighting_attribute (str, optional): An optional attribute to weight the
                                             instances.

    Returns:
        list: A list of cutpoints defining the bins.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    cutpoints = []
    if weighting_attribute is None:
        cleaned_data = data[attribute_name]
        if isinstance(data[attribute_name].dtype, pd.SparseDtype):
            cleaned_data = data[attribute_name].sparse.sp_values

        cleaned_data = cleaned_data[~np.isnan(cleaned_data)]
        sorted_data = sorted(cleaned_data)
        number_instances = len(sorted_data)
        for i in range(1, nbins):
            position = i * number_instances // nbins
            while True:
                if position >= number_instances:
                    break
                val = sorted_data[position]
                if val not in cutpoints:
                    break
                position += 1
            if val not in cutpoints:
                cutpoints.append(val)
    else:
        cleaned_data = data[[attribute_name, weighting_attribute]]
        cleaned_data = cleaned_data[~np.isnan(cleaned_data[attribute_name])]
        cleaned_data.sort(order=attribute_name)

        overall_weights = cleaned_data[weighting_attribute].sum()
        remaining_weights = overall_weights
        bin_size = overall_weights / nbins
        sum_of_weights = 0
        for row in cleaned_data:
            sum_of_weights += row[weighting_attribute]
            if sum_of_weights > bin_size:
                if not row[attribute_name] in cutpoints:
                    cutpoints.append(row[attribute_name])
                    remaining_weights = remaining_weights - sum_of_weights
                    if remaining_weights < 1.5 * (bin_size):
                        break
                    sum_of_weights = 0
    return cutpoints


def conditional_invert(val, invert):
    """
    Conditionally inverts a value based on a boolean flag.

    Parameters:
        val (float): The value to potentially invert.
        invert (bool): If True, the value is inverted.

    Returns:
        float: The (possibly inverted) value.
    """
    return -2 * (invert - 0.5) * val


def results_df_autoround(df):
    """
    Automatically rounds numerical columns in a DataFrame for better readability.

    Parameters:
        df (DataFrame): The DataFrame containing the results.

    Returns:
        DataFrame: The DataFrame with rounded numerical values.
    """
    return df.round(
        {
            "quality": 3,
            "size_sg": 0,
            "size_dataset": 0,
            "positives_sg": 0,
            "positives_dataset": 0,
            "size_complement": 0,
            "relative_size_sg": 3,
            "relative_size_complement": 3,
            "coverage_sg": 3,
            "coverage_complement": 3,
            "target_share_sg": 3,
            "target_share_complement": 3,
            "target_share_dataset": 3,
            "lift": 3,
            "size_sg_weighted": 1,
            "size_dataset_weighted": 1,
            "positives_sg_weighted": 1,
            "positives_dataset_weighted": 1,
            "size_complement_weighted": 1,
            "relative_size_sg_weighted": 3,
            "relative_size_complement_weighted": 3,
            "coverage_sg_weighted": 3,
            "coverage_complement_weighted": 3,
            "target_share_sg_weighted": 3,
            "target_share_complement_weighted": 3,
            "target_share_dataset_weighted": 3,
            "lift_weighted": 3,
        }
    )


def perc_formatter(x):
    """
    Formats a float as a percentage string with one decimal place.

    Parameters:
        x (float): The value to format.

    Returns:
        str: The formatted percentage string.
    """
    return "{0:.1f}%".format(x * 100)


def float_formatter(x, digits=2):
    """
    Formats a float to a specified number of decimal places.

    Parameters:
        x (float): The value to format.
        digits (int): The number of decimal places.

    Returns:
        str: The formatted string.
    """
    return ("{0:." + str(digits) + "f}").format(x)


def is_categorical_attribute(data, attribute_name):
    """
    Determines if an attribute in the dataset is categorical.

    Parameters:
        data (DataFrame): The dataset.
        attribute_name (str): The name of the attribute.

    Returns:
        bool: True if the attribute is categorical, False otherwise.
    """
    return attribute_name in data.select_dtypes(exclude=["number"]).columns.values


def is_numerical_attribute(data, attribute_name):
    """
    Determines if an attribute in the dataset is numerical.

    Parameters:
        data (DataFrame): The dataset.
        attribute_name (str): The name of the attribute.

    Returns:
        bool: True if the attribute is numerical, False otherwise.
    """
    return attribute_name in data.select_dtypes(include=["number"]).columns.values


def remove_selectors_with_attributes(selector_list, attribute_list):
    """
    Removes selectors that are based on specified attributes.

    Parameters:
        selector_list (list): The list of selectors to filter.
        attribute_list (list): The list of attribute names to remove selectors for.

    Returns:
        list: The filtered list of selectors.
    """
    return [x for x in selector_list if x.attributeName not in attribute_list]


def derive_effective_sample_size(weights):
    """
    Calculates the effective sample size for weighted data.

    Parameters:
        weights (array-like): The weights assigned to the samples.

    Returns:
        float: The effective sample size.
    """
    return sum(weights) ** 2 / sum(weights**2)


def powerset(iterable, max_length=None):
    """
    Generates the power set (all possible combinations) of an iterable up to a maximum
    length.

    Parameters:
        iterable (iterable): The iterable to generate combinations from.
        max_length (int, optional): The maximum length of combinations.

    Returns:
        iterator: An iterator over the power set of the iterable.
    """
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if max_length is None:
        max_length = len(s)
    if max_length < len(s):
        max_length = len(s)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(max_length)
    )


def overlap(sg, another_sg, data):
    """
    Calculates the Jaccard similarity between two subgroups based on their coverage.

    Parameters:
        sg: The first subgroup.
        another_sg: The second subgroup.
        data (DataFrame): The dataset.

    Returns:
        float: The Jaccard similarity between the two subgroups.
    """
    cover_sg = sg.covers(data)
    cover_another_sg = another_sg.covers(data)
    union = np.logical_or(cover_sg, cover_another_sg)
    intercept = np.logical_and(cover_sg, cover_another_sg)
    sim = np.sum(intercept) / np.sum(union)
    return sim


#####
# bitset operations
#####
def to_bits(list_of_ints):
    """
    Converts a list of integers to a bitset represented as an integer.

    Parameters:
        list_of_ints (list): The list of integers to convert.

    Returns:
        int: The bitset represented as an integer.
    """
    v = 0
    for x in list_of_ints:
        v += 1 << x
    return v


def count_bits(bitset_as_int):
    """
    Counts the number of set bits (1s) in a bitset represented as an integer.

    Parameters:
        bitset_as_int (int): The bitset represented as an integer.

    Returns:
        int: The number of set bits.
    """
    c = 0
    while bitset_as_int > 0:
        c += 1
        bitset_as_int &= bitset_as_int - 1
    return c


def find_set_bits(bitset_as_int):
    """
    Finds the indices of set bits in a bitset represented as an integer.

    Parameters:
        bitset_as_int (int): The bitset represented as an integer.

    Yields:
        int: The index of each set bit.
    """
    while bitset_as_int > 0:
        x = bitset_as_int.bit_length() - 1
        yield x
        bitset_as_int = bitset_as_int - (1 << x)


#####
# TID-list operations
#####
def intersect_of_ordered_list(list_1, list_2):
    """
    Computes the intersection of two ordered lists.

    Parameters:
        list_1 (list): The first ordered list.
        list_2 (list): The second ordered list.

    Returns:
        list: The intersection of the two lists.
    """
    result = []
    i = 0
    j = 0
    while i < len(list_1) and j < len(list_2):
        if list_1[i] < list_2[j]:
            i += 1
        elif list_2[j] < list_1[i]:
            j += 1
        else:
            result.append(list_1[i])
            j += 1
            i += 1
    return result


class BaseTarget:
    """
    Base class for defining targets in subgroup discovery.

    Provides a method to check if all required statistics are present.
    """

    def all_statistics_present(self, cached_statistics):
        """
        Checks if all required statistics are present in the cached statistics.

        Parameters:
            cached_statistics (dict): The dictionary of cached statistics.

        Returns:
            bool: True if all required statistics are present, False otherwise.
        """
        # pylint: disable=no-member
        if isinstance(cached_statistics, dict) and all(
            expected_value in cached_statistics
            for expected_value in self.__class__.statistic_types
        ):
            return True
        # pylint: enable=no-member
        return False


class SubgroupDiscoveryResult:
    """
    Represents the result of a subgroup discovery task.

    Contains methods to convert results to different formats.
    """

    def __init__(self, results, task):
        """
        Initializes the SubgroupDiscoveryResult with the results and the task.

        Parameters:
            results (Iterable): An iterable of (quality, subgroup, statistics) tuples.
            task (SubgroupDiscoveryTask): The subgroup discovery task.
        """
        self.task = task
        self.results = results
        assert isinstance(results, Iterable)

    def to_descriptions(self, include_stats=False):
        """
        Converts the results to a list of subgroup descriptions.

        Parameters:
            include_stats (bool): If True, includes statistics in the output.

        Returns:
            list: A list of subgroup descriptions.
        """
        if include_stats:
            return list(self.results)
        else:
            return [(qual, sgd) for qual, sgd, stats in self.results]

    def to_table(
        self, statistics_to_show=None, print_header=True, include_target=False
    ):
        """
        Converts the results to a table format.

        Parameters:
            statistics_to_show (list, optional): The statistics to include in the table.
            print_header (bool): If True, includes a header row.
            include_target (bool): If True, includes the target in the table.

        Returns:
            list: A list of rows representing the table.
        """
        if statistics_to_show is None:
            statistics_to_show = type(self.task.target).statistic_types
        table = []
        if print_header:
            row = ["quality", "subgroup"]
            if include_target:
                row.append("target")
            for stat in statistics_to_show:
                row.append(stat)
            table.append(row)
        for q, sg, stats in self.results:
            stats = self.task.target.calculate_statistics(sg, self.task.data, stats)
            row = [q, sg]
            if include_target:
                row.append(self.task.target)
            for stat in statistics_to_show:
                row.append(stats[stat])
            table.append(row)
        return table

    def to_dataframe(
        self, statistics_to_show=None, autoround=False, include_target=False
    ):
        """
        Converts the results to a pandas DataFrame.

        Parameters:
            statistics_to_show (list, optional): The statistics to include in the
                                                 DataFrame.
            autoround (bool): If True, automatically rounds numerical columns.
            include_target (bool): If True, includes the target in the DataFrame.

        Returns:
            DataFrame: A pandas DataFrame representing the results.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        if statistics_to_show is None:
            statistics_to_show = type(self.task.target).statistic_types
        res = self.to_table(statistics_to_show, True, include_target)
        headers = res.pop(0)
        df = pd.DataFrame(res, columns=headers)
        if autoround:
            df = results_df_autoround(df)
        return df

    def to_latex(self, statistics_to_show=None, escape_underscore=True):
        """
        Converts the results to a LaTeX-formatted table.

        Parameters:
            statistics_to_show (list, optional): The statistics to include in the LaTeX
                                                 table.
            escape_underscore (bool): If True, escapes underscores in strings.

        Returns:
            str: A string containing the LaTeX-formatted table.
        """
        if statistics_to_show is None:
            statistics_to_show = type(self.task.target).statistic_types
        df = self.to_dataframe(statistics_to_show)
        latex = df.to_latex(
            index=False,
            col_space=10,
            formatters={
                "quality": partial(float_formatter, digits=3),
                "size_sg": partial(float_formatter, digits=0),
                "size_dataset": partial(float_formatter, digits=0),
                "positives_sg": partial(float_formatter, digits=0),
                "positives_dataset": partial(float_formatter, digits=0),
                "size_complement": partial(float_formatter, digits=0),
                "relative_size_sg": perc_formatter,
                "relative_size_complement": perc_formatter,
                "coverage_sg": perc_formatter,
                "coverage_complement": perc_formatter,
                "target_share_sg": perc_formatter,
                "target_share_complement": perc_formatter,
                "target_share_dataset": perc_formatter,
                "lift": partial(float_formatter, digits=1),
            },
        )
        latex = latex.replace(" AND ", r" $\wedge$ ")
        if escape_underscore:
            latex = latex.replace("_", r"\_")
        latex = latex.replace(" AND ", r" $\wedge$ ")
        return latex


def add_if_required(
    result,
    sg,
    quality,
    task: SubgroupDiscoveryTask,
    check_for_duplicates=False,
    statistics=None,
    explicit_result_set_size=None,
):
    """
    Adds a subgroup to the result set if it meets the required quality and constraints.

    IMPORTANT:
        Only add/remove subgroups from `result` by using `heappop` and `heappush`
        to ensure order of subgroups by quality.

    Parameters:
        result (list): The current list of subgroups (heap).
        sg: The subgroup to potentially add.
        quality (float): The quality of the subgroup.
        task (SubgroupDiscoveryTask): The task containing parameters and constraints.
        check_for_duplicates (bool): If True, checks for duplicates before adding.
        statistics (optional): Precomputed statistics for the subgroup.
        explicit_result_set_size (int, optional): Overrides the task's result_set_size.

    Returns:
        None
    """
    if explicit_result_set_size is None:
        explicit_result_set_size = task.result_set_size

    if quality >= task.min_quality:
        if not ps.constraints_satisfied(task.constraints, sg, statistics, task.data):
            return
        if check_for_duplicates and (quality, sg, statistics) in result:
            return
        if len(result) < explicit_result_set_size:
            heappush(result, (quality, sg, statistics))
        elif quality > result[0][0]:  # better than worst subgroup
            heappop(result)
            heappush(result, (quality, sg, statistics))
