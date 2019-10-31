import pysubgroup as ps


class RefinementOperator:
    pass


class StaticSpecializationOperator:
    def __init__(self, selectors):
        self.search_space = selectors
        self.search_space_index = {key: i for i, key in enumerate(selectors)}

    def refinements(self, subgroup):
        if len(subgroup) > 0:
            index_of_last_selector = self.search_space_index[subgroup._selectors[-1]]
            new_selectors = self.search_space[index_of_last_selector + 1:]
        else:
            new_selectors = self.search_space

        return (subgroup & sel for sel in new_selectors)


class StaticGeneralizationOperator:
    def __init__(self, selectors):
        self.search_space = selectors

    def refinements(self, sG):
        index_of_last_selector = min(self.search_space.index(sG._selectors[-1]), len(self.search_space) - 1)
        new_selectors = self.search_space[index_of_last_selector + 1:]

        return (sG | sel for sel in new_selectors)
