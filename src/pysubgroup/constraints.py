import pysubgroup as ps


class MinSupportConstraint:
    """
    A constraint that ensures a subgroup has at least a minimum support.

    Attributes:
        min_support (int): The minimum number of instances that a subgroup must cover.
    """

    def __init__(self, min_support):
        """
        Initializes the MinSupportConstraint with the specified minimum support.

        Parameters:
            min_support (int): The minimum support required for subgroups.
        """
        self.min_support = min_support

    @property
    def is_monotone(self):
        """
        Indicates whether the constraint is monotone.

        Returns:
            bool: True if the constraint is monotone, False otherwise.
        """
        return True

    def is_satisfied(self, subgroup, statistics=None, data=None):
        """
        Checks if the subgroup satisfies the minimum support constraint.

        Parameters:
            subgroup: The subgroup to be evaluated.
            statistics: Precomputed statistics for the subgroup (optional).
            data: The dataset being analyzed (optional).

        Returns:
            bool: True if the subgroup's size is at least the minimum support,
                  False otherwise.
        """
        if hasattr(statistics, "size_sg"):
            return statistics.size_sg >= self.min_support
        if isinstance(statistics, dict) and "size_sg" in statistics:
            return statistics["size_sg"] >= self.min_support
        try:
            return ps.get_size(subgroup, len(data), data) >= self.min_support
        except AttributeError:  # Special case for gp_growth algorithm
            return self.get_size_sg(statistics)

    def gp_prepare(self, qf):
        """
        Prepares the constraint for the GP-Growth algorithm by accessing the size
        function.

        Parameters:
            qf: The quality function used in the GP-Growth algorithm.
        """
        self.get_size_sg = (
            qf.gp_size_sg
        )  # pylint: disable=attribute-defined-outside-init

    def gp_is_satisfied(self, node):
        """
        Checks if a node satisfies the constraint in the GP-Growth algorithm.

        Parameters:
            node: The node to be evaluated.

        Returns:
            bool: True if the node's size is at least the minimum support,
                  False otherwise.
        """
        return self.get_size_sg(node) >= self.min_support


class ContainsValueConstraint:
    """
    A constraint that ensures a subgroup contains in its cover at least one instance that has a specified value for a specified attribute.

    Attributes:
        attribute_name: The attribute that needs to contain the specified value in at least one instance.
        value: The value that needs to be present in the specified attribute in at least one instance.
    """

    def __init__(self, attribute_name, value):
        """
        Initializes the ContainsValueConstraint with the specified value and attribute.

        Parameters:
            attribute_name: The attribute that needs to contain the specified value in at least one instance.
            value: The value that needs to be present in the specified attribute in at least one instance.
        """
        self.attribute_name = attribute_name
        self.value = value

    @property
    def is_monotone(self):
        """
        Indicates whether the constraint is monotone.

        Returns:
            bool: True if the constraint is monotone, False otherwise.
        """
        return True

    def is_satisfied(self, subgroup, statistics=None, data=None):
        """
        Checks if the subgroup satisfies the constraint.

        Parameters:
            subgroup: The subgroup to be evaluated.
            statistics: Precomputed statistics for the subgroup (optional).
            data: The dataset being analyzed (optional).

        Returns:
            bool: True if the subgroup's cover contains at least one instance that has the specified value for the specified attribute (as defined during object construction),
                  False otherwise.
        """
        return sum(data[self.attribute_name][subgroup.representation] == self.value) > 0


class MinUniqueValuesConstraint:
    """
    A constraint that ensures a subgroup contains in its cover a minimum number of unique values for a specified attribute.

    Attributes:
        attribute_name: The attribute that needs to contain at least the specified number of values.
        min_unique_values: The minimum number of unique values that must be present in the attribute in a subgroup cover.
    """

    def __init__(self, attribute_name, min_unique_values):
        """
        Initializes the MinUniqueValuesConstraint with the specified attribute and minimum number of unique values.

        Parameters:
            attribute_name: The attribute that needs to contain at least the specified number of values.
            min_unique_values: The minimum number of unique values that must be present in the attribute in a subgroup cover.
        """
        self.attribute_name = attribute_name
        self.min_unique_values = min_unique_values

    @property
    def is_monotone(self):
        """
        Indicates whether the constraint is monotone.

        Returns:
            bool: True if the constraint is monotone, False otherwise.
        """
        return True

    def is_satisfied(self, subgroup, statistics=None, data=None):
        """
        Checks if the subgroup satisfies the constraint.

        Parameters:
            subgroup: The subgroup to be evaluated.
            statistics: Precomputed statistics for the subgroup (optional).
            data: The dataset being analyzed (optional).

        Returns:
            bool: True if the subgroup's cover contains the minimum number of unique values for the specified attribute (as defined during object construction),
                  False otherwise.
        """
        return (
            data[subgroup.representation][self.attribute_name].nunique()
            >= self.min_unique_values
        )
