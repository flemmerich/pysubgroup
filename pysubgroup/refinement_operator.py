import pysubgroup as ps

class RefinementOperator:
    pass

class StaticSpecializationOperator:
    def __init__(self,selectors):
        self.search_space=selectors

    def refinements(self,sG):
        index_of_last_selector = min(self.search_space.index(sG.selectors[-1]), len(self.search_space) - 1)
        new_selectors=self.search_space[index_of_last_selector + 1:]#(sel for sel in self.search_space if sel not in sG.selectors)

        return (ps.SubgroupDescription([*sG.selectors, sel]) for sel in new_selectors)

class StaticGeneralizationOperator:
    def __init__(self,selectors):
        self.search_space=selectors

    def refinements(self,sG):
        index_of_last_selector = min(self.search_space.index(sG._selectors[-1]), len(self.search_space) - 1)
        new_selectors=self.search_space[index_of_last_selector + 1:]#(sel for sel in self.search_space if sel not in sG.selectors)

        return (ps.Disjunction([*sG._selectors, sel]) for sel in new_selectors)
