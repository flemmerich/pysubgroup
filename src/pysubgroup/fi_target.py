"""
Created on 29.09.2017

@author: lemmerfn

This module defines the FITarget and related quality functions for frequent itemset mining
using the pysubgroup package.
"""
from collections import namedtuple
from functools import total_ordering

import pysubgroup as ps


@total_ordering
class FITarget(ps.BaseTarget):
    """Target class for frequent itemset mining.

    Represents the target for mining frequent itemsets,
    extending the BaseTarget class from pysubgroup.
    """

    statistic_types = ("size_sg", "size_dataset")

    def __repr__(self):
        """String representation of the FITarget."""
        return "T: Frequent Itemsets"

    def __eq__(self, other):
        """Check equality based on the instance dictionary."""
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        """Define less-than comparison for sorting purposes."""
        return str(self) < str(other)  # pragma: no cover

    def get_attributes(self):
        """Return an empty list as attributes are not used in FITarget."""
        return []

    def get_base_statistics(self, subgroup, data):
        """Compute the base statistics for the subgroup.

        Parameters:
            subgroup: The subgroup for which to compute statistics.
            data: The dataset.

        Returns:
            int: The size of the subgroup.
        """
        _, size = ps.get_cover_array_and_size(subgroup, len(data), data)
        return size

    def calculate_statistics(self, subgroup_description, data, cached_statistics=None):
        """Calculate statistics for the subgroup.

        Parameters:
            subgroup_description: The description of the subgroup.
            data: The dataset.
            cached_statistics (dict, optional): Previously computed statistics.

        Returns:
            dict: A dictionary containing 'size_sg' and 'size_dataset'.
        """
        if self.all_statistics_present(cached_statistics):
            return cached_statistics

        _, size = ps.get_cover_array_and_size(subgroup_description, len(data), data)
        statistics = {}
        statistics["size_sg"] = size
        statistics["size_dataset"] = len(data)
        return statistics


class SimpleCountQF(ps.AbstractInterestingnessMeasure):
    """Quality function that counts the number of instances in a subgroup.

    Provides basic counting functionality, useful for frequent itemset mining.
    """

    tpl = namedtuple("CountQF_parameters", ("size_sg"))
    gp_requires_cover_arr = False

    def __init__(self):
        """Initialize the SimpleCountQF."""
        self.required_stat_attrs = ("size_sg",)
        self.has_constant_statistics = True
        self.size_dataset = None

    def calculate_constant_statistics(
        self, data, target
    ):  # pylint: disable=unused-argument
        """Calculate statistics that remain constant for the dataset.

        Parameters:
            data: The dataset.
            target: The target definition (unused in this implementation).
        """
        self.size_dataset = len(data)

    def calculate_statistics(
        self, subgroup_description, target, data, statistics=None
    ):  # pylint: disable=unused-argument
        """Calculate statistics specific to the subgroup.

        Parameters:
            subgroup_description: The description of the subgroup.
            target: The target definition (unused in this implementation).
            data: The dataset.
            statistics (any, optional): Unused in this implementation.

        Returns:
            namedtuple: Contains 'size_sg' for the subgroup.
        """
        _, size = ps.get_cover_array_and_size(
            subgroup_description, self.size_dataset, data
        )
        return SimpleCountQF.tpl(size)

    def gp_get_stats(self, _):
        """Get statistics for a single instance (used in GP-Growth algorithms).

        Returns:
            dict: A dictionary with 'size_sg' set to 1.
        """
        return {"size_sg": 1}

    def gp_get_null_vector(self):
        """Get a null vector for initialization in GP-Growth algorithms.

        Returns:
            dict: A dictionary with 'size_sg' set to 0.
        """
        return {"size_sg": 0}

    def gp_merge(self, left, right):
        """Merge two statistics dictionaries by summing 'size_sg'.

        Parameters:
            left (dict): Left statistics dictionary.
            right (dict): Right statistics dictionary.
        """
        left["size_sg"] += right["size_sg"]

    def gp_get_params(self, _cover_arr, v):
        """Extract parameters from the statistics dictionary.

        Parameters:
            _cover_arr: Unused parameter.
            v (dict): Statistics dictionary.

        Returns:
            namedtuple: Contains 'size_sg' from the statistics.
        """
        return SimpleCountQF.tpl(v["size_sg"])

    def gp_to_str(self, stats):
        """Convert statistics to a string representation.

        Parameters:
            stats (dict): Statistics dictionary.

        Returns:
            str: String representation of 'size_sg'.
        """
        return str(stats["size_sg"])

    def gp_size_sg(self, stats):
        """Get the size of the subgroup from the statistics.

        Parameters:
            stats (dict): Statistics dictionary.

        Returns:
            int: Size of the subgroup.
        """
        return stats["size_sg"]


class CountQF(SimpleCountQF, ps.BoundedInterestingnessMeasure):
    """Quality function that evaluates subgroups based on their size.

    Extends SimpleCountQF and BoundedInterestingnessMeasure.
    """

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup.

        Parameters:
            subgroup: The subgroup to evaluate.
            target: The target definition.
            data: The dataset.
            statistics (any, optional): Previously computed statistics.

        Returns:
            int: The size of the subgroup.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.size_sg

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        """Compute the optimistic estimate of the quality function.

        Parameters:
            subgroup: The subgroup for which to compute the optimistic estimate.
            target: The target definition.
            data: The dataset.
            statistics (any, optional): Previously computed statistics.

        Returns:
            int: The size of the subgroup.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.size_sg


class AreaQF(SimpleCountQF):
    """Quality function that evaluates subgroups based on their area.

    The area is computed as the size of the subgroup multiplied by the number of contained items
    """

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the quality of the subgroup.

        Parameters:
            subgroup: The subgroup to evaluate.
            target: The target definition.
            data: The dataset.
            statistics (any, optional): Previously computed statistics.

        Returns:
            int: The area of the subgroup (size_sg * depth).
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.size_sg * subgroup.depth
