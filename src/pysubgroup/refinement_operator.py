from collections import defaultdict
from itertools import chain


class RefinementOperator:
    """Base class for refinement operators."""

    pass


class StaticSpecializationOperator:
    """Refinement operator for static specialization.

    This operator specializes subgroups by adding selectors in a predefined order,
    ensuring that each attribute is used only once in a subgroup description.
    """

    def __init__(self, selectors):
        """Initialize the StaticSpecializationOperator.

        Parameters:
            selectors: List of selectors to define the search space.
        """
        search_space_dict = defaultdict(list)
        for selector in selectors:
            # Group selectors by their attribute name
            search_space_dict[selector.attribute_name].append(selector)
        self.search_space = list(search_space_dict.values())
        # Map attribute names to their index in the search space
        self.search_space_index = {
            key: i for i, key in enumerate(search_space_dict.keys())
        }

    def refinements(self, subgroup):
        """Generate refinements of the given subgroup.

        Parameters:
            subgroup: The subgroup to refine.

        Returns:
            A generator of refined subgroups.
        """
        if subgroup.depth > 0:
            # Get the index of the attribute of the last selector in the subgroup
            index_of_last = self.search_space_index[
                subgroup._selectors[-1].attribute_name
            ]
            # Generate selectors for attributes that come after the last one used
            new_selectors = chain.from_iterable(self.search_space[index_of_last + 1 :])
        else:
            # If subgroup is empty, use all selectors
            new_selectors = chain.from_iterable(self.search_space)

        return (subgroup & sel for sel in new_selectors)


class StaticGeneralizationOperator:
    """Refinement operator for static generalization.

    This operator generalizes subgroups by adding selectors from a predefined list,
    ensuring that each selector is used in a specific order.
    """

    def __init__(self, selectors):
        """Initialize the StaticGeneralizationOperator.

        Parameters:
            selectors: List of selectors to define the search space.
        """
        self.search_space = selectors

    def refinements(self, sG):
        """Generate refinements of the given subgroup.

        Parameters:
            sG: The subgroup to refine.

        Returns:
            A generator of refined subgroups.
        """
        # Find the index of the last selector used in the subgroup
        index_of_last_selector = min(
            self.search_space.index(sG._selectors[-1]), len(self.search_space) - 1
        )
        # Select the selectors that come after the last used selector
        new_selectors = self.search_space[index_of_last_selector + 1 :]

        return (sG | sel for sel in new_selectors)
