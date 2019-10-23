import numpy as np
import pysubgroup as ps

class RepresentationBase():
    def patch_all_selectors(self):
        for sel in ps.SelectorBase.__refs__:
            self.patch_selector(sel)


    def patch_selector(self,sel):
        raise NotImplementedError


class BitSet_SubgroupDescription(ps.SubgroupDescription):
    n_instances=0
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.representation=self.compute_representation()
            
    def compute_representation(self):
                # empty description ==> return a list of all '1's
        if not self.selectors:
            return np.full(BitSet_SubgroupDescription.n_instances, True, dtype=bool)
        # non-empty description
        return np.all([sel.representation for sel in self.selectors], axis=0)
    @property
    def size(self):
        return np.sum(self.representation)

    def __copy__(self):
        tmp=super().__copy__()
        tmp.representation=self.representation.copy()
        return tmp
    def append_and(self,to_append):
        self.selectors.append(to_append)
        self.representation=np.logical_and(self.representation, to_append.representation)
    @property
    def __array_interface__(self):
        return self.representation.__array_interface__




class BitSetRepresentation(RepresentationBase):
    def __init__(self,df):
        self.df=df
        self.SD=None

    def patch_selector(self,sel):
        sel.representation=sel.covers(self.df)


    def patch_classes(self):
        BitSet_SubgroupDescription.n_instances=len(self.df)
        self.SD=ps.SubgroupDescription
        ps.SubgroupDescription=BitSet_SubgroupDescription

    def __enter__(self):
        self.patch_all_selectors()
        self.patch_classes()

    def __exit__(self, *args):
        ps.SubgroupDescription=self.SD


class Set_SubgroupDescription(ps.SubgroupDescription):
    all_set=set()
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.representation=self.compute_representation()
            
    def compute_representation(self):
                # empty description ==> return a list of all '1's
        if not self.selectors:
            return Set_SubgroupDescription.all_set
        # non-empty description
        return set.intersection(sel.representation for sel in self.selectors)
    @property
    def size(self):
        return len(self.representation)

    def __copy__(self):
        tmp=super().__copy__()
        tmp.representation=self.representation.copy()
        return tmp
    def append_and(self,selector):
        self.selectors.append(selector)
        self.representation=self.representation.intersection( selector.representation)
    @property
    def __array_interface__(self):
        #print("AAAA")
        #print(self.representation)
        self.arr=np.array(list(self.representation),dtype=int)
        #print("BBB")
        #print(self.arr)
        return self.arr.__array_interface__


class SetRepresentation(RepresentationBase):
    def __init__(self,df):
        self.df=df
        self.SD=None

    def patch_selector(self,sel):
        sel.representation=set(*np.nonzero(sel.covers(self.df)))


    def patch_classes(self):
        Set_SubgroupDescription.all_set=set(self.df.index)
        self.SD=ps.SubgroupDescription
        ps.SubgroupDescription=Set_SubgroupDescription

    def __enter__(self):
        self.patch_all_selectors()
        self.patch_classes()

    def __exit__(self, *args):
        ps.SubgroupDescription=self.SD


class NumpySet_SubgroupDescription(ps.SubgroupDescription):
    all_set=set()
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.representation=self.compute_representation()
            
    def compute_representation(self):
                # empty description ==> return a list of all '1's
        if not self.selectors:
            return NumpySet_SubgroupDescription.all_set
        start=self.selectors[0]
        for sel in self.selectors[1:]:
            start=np.intersect1d(start,sel.representation,True)
        return start
    @property
    def size(self):
        return len(self.representation)

    def __copy__(self):
        tmp=super().__copy__()
        tmp.representation=self.representation.copy()
        return tmp
    def append_and(self,selectors):
        self.selectors.append(selectors)
        self.representation=np.intersect1d(self.representation, selectors.representation,True)
    @property
    def __array_interface__(self):
        return self.representation.__array_interface__


class NumpySetRepresentation(RepresentationBase):
    def __init__(self,df):
        self.df=df
        self.SD=None

    def patch_selector(self,sel):
        sel.representation=np.nonzero(sel.covers(self.df))


    def patch_classes(self):
        NumpySet_SubgroupDescription.all_set=self.df.index.to_numpy()
        self.SD=ps.SubgroupDescription
        ps.SubgroupDescription=NumpySet_SubgroupDescription

    def __enter__(self):
        self.patch_all_selectors()
        self.patch_classes()

    def __exit__(self, *args):
        ps.SubgroupDescription=self.SD