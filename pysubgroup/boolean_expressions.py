from functools import total_ordering

from pysubgroup import SelectorBase
import numpy as np





@total_ordering
class Conjunction:
    def __init__(self, selectors):
        if isinstance(selectors, (list, tuple)):
            self._selectors = selectors
        else:
            self._selectors = [selectors]

    def covers(self, instance):
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return np.full(len(instance), True, dtype=bool)
        # non-empty description
        return np.all([sel.covers(instance) for sel in self._selectors], axis=0)

    def __len__(self):
        return len(self._selectors)

    def __str__(self, open_brackets="", closing_brackets="", and_term=" AND "):
        if not self._selectors:
            return "Dataset"
        attrs = sorted(str(sel) for sel in self._selectors)
        return "".join((open_brackets, and_term.join(attrs), closing_brackets))

    def __repr__(self):
        if not self._selectors:
            return "True"
        reprs = sorted(repr(sel) for sel in self._selectors)
        return "".join(("(", " and ".join(reprs), ")"))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        return hash(repr(self))

    def append_and(self, selector):
        try:
            self._selectors.extend(selector)
        except TypeError:
            self._selectors.append(selector)

    def append_or(self, selector):
        raise RuntimeError("Or operations are not supported by a pure Conjunction. Consider using DNF.")
    
    def pop_and(self):
        return self._selectors.pop()

    def pop_or(self):
        raise RuntimeError("Or operations are not supported by a pure Conjunction. Consider using DNF.")


@total_ordering
class Disjunction:
    def __init__(self, selectors):
        if isinstance(selectors, (list, tuple)):
            self._selectors = selectors
        else:
            self._selectors = [selectors]

    def covers(self, instance):
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return np.full(len(instance), True, dtype=bool)
        # non-empty description
        return np.any([sel.covers(instance) for sel in self._selectors], axis=0)

    def __len__(self):
        return len(self._selectors)

    def __str__(self, open_brackets="", closing_brackets="", or_term=" OR "):
        if not self._selectors:
            return "Dataset"
        attrs = sorted(str(sel) for sel in self._selectors)
        return "".join((open_brackets, or_term.join(attrs), closing_brackets))

    def __repr__(self):
        if not self._selectors:
            return "True"
        reprs = sorted(repr(sel) for sel in self._selectors)
        return "".join(("(", " or ".join(reprs), ")"))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        return hash(repr(self))

    def append_and(self, selector):
        raise RuntimeError("And operations are not supported by a pure Conjunction. Consider using DNF.")

    def append_or(self, selector):
        try:
            self._selectors.extend(selector)
        except TypeError:
            self._selectors.append(selector)


class DNF(Disjunction):
    def __init__(self, input=None):
        if input is None:
            input=[]
        super().__init__([])
        self.append_or(input)

    def append_or(self,input):
        try:
            conjunctions = [Conjunction(sel) if isinstance(sel,SelectorBase) else sel for sel in input]
        except TypeError:
            if isinstance(input, SelectorBase):
                conjunctions = [Conjunction(input)]
            else:
                conjunctions = [input]
        if all( isinstance(conj, Conjunction) for conj in conjunctions):
            super().append_or(conjunctions)
        else:
            raise ValueError("All inputs must be either conjunctions or selectors")

            

    def append_and(self,input):
        try:
            if not all(isinstance(sel,SelectorBase) for sel in input):
                raise ValueError
            selectors=input
        except TypeError:
            if isinstance(input, Conjunction):
                selectors=input._selectors
            elif isinstance(input, SelectorBase):
                selectors=input
        if len(self._selectors) > 0:
            for conjunction in self._selectors:
                conjunction.append_and(selectors)
        else:
            self._selectors.append(Conjunction(selectors))


    def pop_and(self):
        out_list=[s.pop_and() for s in self._selectors]
        return_val=out_list[0]
        if all(x==return_val for x in out_list):
            return return_val
        else:
            raise RuntimeError("pop_and failed as the result was inconsistent")