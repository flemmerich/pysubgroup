import numpy as np

from pysubgroup.subgroup_description import Conjunction, Disjunction


class RepresentationBase:
    """Base class for different representation strategies.

    Provides methods to patch selectors and manage class-level patches.
    Can be used as a context manager to ensure patches are applied and removed properly.
    """

    def __init__(self, new_conjunction, selectors_to_patch):
        """Initialize the RepresentationBase.

        Parameters:
            new_conjunction: The new Conjunction class to use.
            selectors_to_patch: List of selectors to patch.
        """
        self._new_conjunction = new_conjunction
        self.previous_conjunction = None
        self.selectors_to_patch = selectors_to_patch

    def patch_all_selectors(self):
        """Patch all selectors in the selectors_to_patch list."""
        for sel in self.selectors_to_patch:
            self.patch_selector(sel)

    def patch_selector(self, sel):  # pragma: no cover
        """Patch a single selector.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError()  # pragma: no cover

    def patch_classes(self):
        """Patch the required classes.

        Can be overridden by subclasses to patch class-level attributes or methods.
        """
        pass

    def undo_patch_classes(self):
        """Undo patches applied to classes.

        Can be overridden by subclasses to remove class-level patches.
        """
        pass

    def __enter__(self):
        """Enter the runtime context and apply patches."""
        self.patch_classes()
        self.patch_all_selectors()
        return self

    def __exit__(self, *args):
        """Exit the runtime context and undo patches."""
        self.undo_patch_classes()


class BitSet_Conjunction(Conjunction):
    """Conjunction subclass that uses bitsets for representation.

    Provides efficient computation of the conjunction using numpy boolean arrays.
    """

    n_instances = 0

    def __init__(self, *args, **kwargs):
        """Initialize the BitSet_Conjunction and compute its representation."""
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()

    def compute_representation(self):
        """Compute the bitset representation of the conjunction.

        Returns:
            Numpy boolean array representing the instances covered by the conjunction.
        """
        # empty description ==> return a list of all '1's
        if not self._selectors:
            return np.full(BitSet_Conjunction.n_instances, True, dtype=bool)
        # non-empty description
        return np.all([sel.representation for sel in self._selectors], axis=0)

    @property
    def size_sg(self):
        """Size of the subgroup represented by the conjunction."""
        return np.count_nonzero(self.representation)

    def append_and(self, to_append):
        """Append a selector using logical AND and update the representation.

        Parameters:
            to_append: Selector to append.
        """
        super().append_and(to_append)
        self.representation = np.logical_and(
            self.representation, to_append.representation
        )

    @property
    def __array_interface__(self):
        """Provide the array interface of the representation for compatibility."""
        return self.representation.__array_interface__


class BitSet_Disjunction(Disjunction):
    """Disjunction subclass that uses bitsets for representation.

    Provides efficient computation of the disjunction using numpy boolean arrays.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the BitSet_Disjunction and compute its representation."""
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()

    def compute_representation(self):
        """Compute the bitset representation of the disjunction.

        Returns:
            Numpy boolean array representing the instances covered by the disjunction.
        """
        # empty description ==> return a list of all '0's
        if not self._selectors:
            return np.full(BitSet_Conjunction.n_instances, False, dtype=bool)
        # non-empty description
        return np.any([sel.representation for sel in self._selectors], axis=0)

    @property
    def size_sg(self):
        """Size of the subgroup represented by the disjunction."""
        return np.count_nonzero(self.representation)

    def append_or(self, to_append):
        """Append a selector using logical OR and update the representation.

        Parameters:
            to_append: Selector to append.
        """
        super().append_or(to_append)
        self.representation = np.logical_or(
            self.representation, to_append.representation
        )

    @property
    def __array_interface__(self):
        """Provide the array interface of the representation for compatibility."""
        return self.representation.__array_interface__


class BitSetRepresentation(RepresentationBase):
    """Representation class that uses bitsets for selectors and conjunctions."""

    Conjunction = BitSet_Conjunction
    Disjunction = BitSet_Disjunction

    def __init__(self, df, selectors_to_patch):
        """Initialize the BitSetRepresentation.

        Parameters:
            df: pandas DataFrame containing the data.
            selectors_to_patch: List of selectors to patch.
        """
        self.df = df
        super().__init__(BitSet_Conjunction, selectors_to_patch)

    def patch_selector(self, sel):
        """Patch a selector by computing its bitset representation.

        Parameters:
            sel: Selector to patch.
        """
        sel.representation = sel.covers(self.df)
        sel.size_sg = np.count_nonzero(sel.representation)

    def patch_classes(self):
        """Patch class-level attributes before entering the context."""
        BitSet_Conjunction.n_instances = len(self.df)
        super().patch_classes()


class Set_Conjunction(Conjunction):
    """Conjunction subclass that uses sets for representation."""

    all_set = set()

    def __init__(self, *args, **kwargs):
        """Initialize the Set_Conjunction and compute its representation."""
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()
        self.arr_for_interface = np.array(list(self.representation), dtype=int)

    def compute_representation(self):
        """Compute the set representation of the conjunction.

        Returns:
            Set of indices representing the instances covered by the conjunction.
        """
        # empty description ==> return the set of all indices
        if not self._selectors:
            return Set_Conjunction.all_set
        # non-empty description
        return set.intersection(*[sel.representation for sel in self._selectors])

    @property
    def size_sg(self):
        """Size of the subgroup represented by the conjunction."""
        return len(self.representation)

    def append_and(self, to_append):
        """Append a selector using logical AND and update the representation.

        Parameters:
            to_append: Selector to append.
        """
        super().append_and(to_append)
        self.representation = self.representation.intersection(to_append.representation)
        self.arr_for_interface = np.array(list(self.representation), dtype=int)

    @property
    def __array_interface__(self):
        """Provide the array interface of the representation for compatibility."""
        return self.arr_for_interface.__array_interface__  # pylint: disable=no-member


class SetRepresentation(RepresentationBase):
    """Representation class that uses sets for selectors and conjunctions."""

    Conjunction = Set_Conjunction

    def __init__(self, df, selectors_to_patch):
        """Initialize the SetRepresentation.

        Parameters:
            df: pandas DataFrame containing the data.
            selectors_to_patch: List of selectors to patch.
        """
        self.df = df
        super().__init__(Set_Conjunction, selectors_to_patch)

    def patch_selector(self, sel):
        """Patch a selector by computing its set representation.

        Parameters:
            sel: Selector to patch.
        """
        sel.representation = set(*np.nonzero(sel.covers(self.df)))
        sel.size_sg = len(sel.representation)

    def patch_classes(self):
        """Patch class-level attributes before entering the context."""
        Set_Conjunction.all_set = set(self.df.index)
        super().patch_classes()


class NumpySet_Conjunction(Conjunction):
    """Conjunction subclass that uses numpy arrays for set representation."""

    all_set = None

    def __init__(self, *args, **kwargs):
        """Initialize the NumpySet_Conjunction and compute its representation."""
        super().__init__(*args, **kwargs)
        self.representation = self.compute_representation()

    def compute_representation(self):
        """Compute the numpy array representation of the conjunction.

        Returns:
            Numpy array of indices representing the instances covered by the conjunction.
        """
        # empty description ==> return an array of all indices
        if not self._selectors:
            return NumpySet_Conjunction.all_set
        start = self._selectors[0].representation
        for sel in self._selectors[1:]:
            start = np.intersect1d(start, sel.representation, assume_unique=True)
        return start

    @property
    def size_sg(self):
        """Size of the subgroup represented by the conjunction."""
        return len(self.representation)

    def append_and(self, to_append):
        """Append a selector using logical AND and update the representation.

        Parameters:
            to_append: Selector to append.
        """
        super().append_and(to_append)
        self.representation = np.intersect1d(
            self.representation, to_append.representation, True
        )

    @property
    def __array_interface__(self):
        """Provide the array interface of the representation for compatibility."""
        return self.representation.__array_interface__


class NumpySetRepresentation(RepresentationBase):
    """Representation class that uses numpy arrays for selectors and conjunctions."""

    Conjunction = NumpySet_Conjunction

    def __init__(self, df, selectors_to_patch):
        """Initialize the NumpySetRepresentation.

        Parameters:
            df: pandas DataFrame containing the data.
            selectors_to_patch: List of selectors to patch.
        """
        self.df = df
        super().__init__(NumpySet_Conjunction, selectors_to_patch)

    def patch_selector(self, sel):
        """Patch a selector by computing its numpy array representation.

        Parameters:
            sel: Selector to patch.
        """
        sel.representation = np.nonzero(sel.covers(self.df))[0]
        sel.size_sg = len(sel.representation)

    def patch_classes(self):
        """Patch class-level attributes before entering the context."""
        NumpySet_Conjunction.all_set = np.arange(len(self.df))
        super().patch_classes()
