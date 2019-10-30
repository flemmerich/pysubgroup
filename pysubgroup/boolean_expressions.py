from abc import ABC, abstractmethod
from functools import total_ordering
import copy
import numpy as np
from pysubgroup import SelectorBase


class BooleanExpressionBase(ABC):
    def __or__(self, other):
        tmp = copy.copy(self)
        tmp.append_or(other)
        return tmp

    def __and__(self, other):
        tmp = self.__copy__()
        tmp.append_and(other)
        return tmp

    @abstractmethod
    def append_and(self, to_append):
        pass

    @abstractmethod
    def append_or(self, to_append):
        pass

    @abstractmethod
    def __copy__(self):
        pass

@total_ordering
class Conjunction(BooleanExpressionBase):
    def __init__(self, selectors):
        try:
            it = iter(selectors)
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
        if hasattr(self, "_repr"):
            return self._repr
        else:
            self._repr = self._compute_repr()
            return self._repr

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        if hasattr(self, "_hash"):
            return self._hash
        else:
            self._hash = self._compute_hash()
            return self._hash

    def _compute_representations(self):
        self._repr=self._compute_repr()
        self._hash=self._compute_hash()

    def _compute_repr(self):
        if not self._selectors:
            return "True"
        reprs = sorted(repr(sel) for sel in self._selectors)
        return "".join(("(", " and ".join(reprs), ")"))

    def _compute_hash(self):
        return hash(repr(self))

    def _invalidate_representations(self):
        if hasattr(self, '_repr'):
            delattr(self, '_repr')
        if hasattr(self, '_hash'):
            delattr(self, '_hash')

    def append_and(self, to_append):
        if isinstance(to_append, SelectorBase):
            self._selectors.append(to_append)
        elif isinstance(to_append, Conjunction):
            self._selectors.extend(to_append._selectors)
        else:
            try:
                self._selectors.extend(to_append)
            except TypeError:
                self._selectors.append(to_append)
        self._invalidate_representations()

    def append_or(self, to_append):
        raise RuntimeError("Or operations are not supported by a pure Conjunction. Consider using DNF.")

    def pop_and(self):
        return self._selectors.pop()

    def pop_or(self):
        raise RuntimeError("Or operations are not supported by a pure Conjunction. Consider using DNF.")

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._selectors = list(self._selectors)
        return result
    @property
    def depth(self):
        return len(self._selectors)


@total_ordering
class Disjunction(BooleanExpressionBase):
    def __init__(self, selectors):
        if isinstance(selectors, (list, tuple)):
            self._selectors = selectors
        else:
            self._selectors = [selectors]

    def covers(self, instance):
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return np.full(len(instance), False, dtype=bool)
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

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._selectors = copy.copy(self._selectors)
        return result

class DNF(Disjunction):
    def __init__(self, selectors=None):
        if selectors is None:
            selectors = []
        super().__init__([])
        self.append_or(selectors)

    @staticmethod
    def _ensure_pure_conjunction(to_append):
        if isinstance(to_append, Conjunction):
            return to_append
        elif isinstance(to_append, SelectorBase):
            return Conjunction(to_append)
        else:
            it = iter(to_append)
            if all(isinstance(sel, SelectorBase) for sel in to_append):
                return Conjunction(it)
            else:
                raise ValueError("DNFs only accept an iterable of pure Selectors")

    def append_or(self, to_append):
        try:
            it = iter(to_append)
            conjunctions = [DNF._ensure_pure_conjunction(part) for part in it]
        except TypeError:
            conjunctions = DNF._ensure_pure_conjunction(to_append)
        super().append_or(conjunctions)

    def append_and(self, to_append):
        conj = DNF._ensure_pure_conjunction(to_append)
        if len(self._selectors) > 0:
            for conjunction in self._selectors:
                conjunction.append_and(conj)
        else:
            self._selectors.append(conj)

    def pop_and(self):
        out_list = [s.pop_and() for s in self._selectors]
        return_val = out_list[0]
        if all(x == return_val for x in out_list):
            return return_val
        else:
            raise RuntimeError("pop_and failed as the result was inconsistent")
