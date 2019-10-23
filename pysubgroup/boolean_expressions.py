from functools import total_ordering

from pysubgroup import SelectorBase
import numpy as np





@total_ordering
class Conjunction:
    def __init__(self, selectors):
        try:
            it=iter(selectors)
            self._selectors = list(it)
        except TypeError:
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
        if isinstance(selector, Conjunction):
            self._selectors.extend(selector._selectors)
        else:
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

    def append_and(self, to_append):
        raise RuntimeError("And operations are not supported by a pure Conjunction. Consider using DNF.")

    def append_or(self, to_append):
        try:
            self._selectors.extend(to_append)
        except TypeError:
            self._selectors.append(to_append)


class DNF(Disjunction):
    def __init__(self, selectors=None):
        if selectors is None:
            selectors=[]
        super().__init__([])
        self.append_or(selectors)


    @staticmethod            
    def _ensure_pure_conjunction(to_append):
        if isinstance(to_append, Conjunction):
            return to_append
        elif isinstance(to_append, SelectorBase):
            return Conjunction(to_append)
        else:
            it=iter(to_append)
            if all(isinstance(sel, SelectorBase) for sel in to_append):
                return Conjunction(it)
            else:
                raise ValueError("DNFs only accept an iterable of pure Selectors")


    def append_or(self,to_append):
        try:
            it=iter(to_append)
            conjunctions=[DNF._ensure_pure_conjunction(part) for part in it]
        except TypeError:
            conjunctions=DNF._ensure_pure_conjunction(to_append)
        super().append_or(conjunctions)


    def append_and(self,to_append):
        conj=DNF._ensure_pure_conjunction(to_append)
        if len(self._selectors) > 0:
            for conjunction in self._selectors:
                conjunction.append_and(conj)
        else:
            self._selectors.append(conj)


    def pop_and(self):
        out_list=[s.pop_and() for s in self._selectors]
        return_val=out_list[0]
        if all(x==return_val for x in out_list):
            return return_val
        else:
            raise RuntimeError("pop_and failed as the result was inconsistent")