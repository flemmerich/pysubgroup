import pysubgroup as ps
from collections import defaultdict
from itertools import chain
import random
class RefinementOperator:
    pass


class StaticSpecializationOperator:
    def __init__(self, selectors):
        search_space_dict = defaultdict(list)
        for selector in selectors:
            search_space_dict[selector.attribute_name].append(selector)
        self.search_space = list(search_space_dict.values())
        self.search_space_index = {key: i for i, key in enumerate(search_space_dict.keys())}

    def refinements(self, subgroup):
        if subgroup.depth > 0:
            index_of_last = self.search_space_index[subgroup._selectors[-1].attribute_name]
            new_selectors = chain.from_iterable(self.search_space[index_of_last + 1:])
        else:
            new_selectors = chain.from_iterable(self.search_space)

        return (subgroup & sel for sel in new_selectors)


class StaticGeneralizationOperator:
    def __init__(self, selectors):
        self.search_space = selectors

    def refinements(self, sG):
        index_of_last_selector = min(self.search_space.index(sG._selectors[-1]), len(self.search_space) - 1)
        new_selectors = self.search_space[index_of_last_selector + 1:]

        return (sG | sel for sel in new_selectors)
