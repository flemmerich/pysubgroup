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
