'''
Created on 28.04.2016

@author: lemmerfn
'''
from functools import total_ordering
import numpy as np
import pandas as pd
import pysubgroup as ps

@total_ordering
class SelectorBase:
    def __eq__(self, other):
        if other is None:
            return False
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)


@total_ordering
class SubgroupDescription:
    def __init__(self, selectors):
        if isinstance(selectors, (list, tuple)):
            self.selectors = selectors
        else:
            self.selectors = [selectors]

    def covers(self, instance):
        # empty description ==> return a list of all '1's
        if not self.selectors:
            return np.full(len(instance), True, dtype=bool)
        # non-empty description
        return np.all([sel.covers(instance) for sel in self.selectors], axis=0)

    def __len__(self):
        return len(self.selectors)

    def count(self, data):
        return np.sum(self.covers(data))

    def get_attributes(self):
        return set(x.get_attribute_name() for x in self.selectors)

    def __str__(self, open_brackets="", closing_brackets="", and_term=" AND "):
        if not self.selectors:
            return "Dataset"
        attrs = sorted(str(sel) for sel in self.selectors)
        return "".join((open_brackets, and_term.join(attrs), closing_brackets))

    def __repr__(self):
        if not self.selectors:
            return "True"
        reprs = sorted(repr(sel) for sel in self.selectors)
        return "".join(("(", " and ".join(reprs), ")"))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __lt__(self, other):
        return repr(self) < repr(other)

    def __hash__(self):
        return hash(frozenset(self.selectors))


class NominalSelector(SelectorBase):
    def __init__(self, attribute_name, attribute_value, selector_name=None):
        if attribute_name is None:
            raise TypeError()
        if attribute_value is None:
            raise TypeError()
        self._attribute_name = attribute_name
        self._attribute_value = attribute_value
        self.selector_name = selector_name
        self._is_bool = False
        self.recompute_representations()

    def get_attribute_name(self):
        return self._attribute_name

    def set_attribute_name(self, value):
        self._attribute_name = value
        self.recompute_representations()

    def get_attribute_value(self):
        return self._attribute_value

    def set_attribute_value(self, value):
        self._attribute_value = value
        self.recompute_representations()

    def get_is_bool(self):
        return self._is_bool

    def set_is_bool(self, value):
        self._is_bool = value
        self.recompute_representations()

    attribute_name = property(get_attribute_name, set_attribute_name)
    attribute_value = property(get_attribute_value, set_attribute_value)
    is_bool = property(get_is_bool, set_is_bool)

    def recompute_representations(self):
        if self._is_bool and (str(self.attribute_value) == "True"):
            self._query = self.attribute_name
        elif isinstance(self.attribute_value, (str, bytes)):
            self._query = str(self.attribute_name) + "==" + "'" + str(self.attribute_value) + "'"
        elif np.isnan(self.attribute_value):
            self._query = self.attribute_name + ".isnull()"
        else:
            self._query = str(self.attribute_name) + "==" + str(self.attribute_value)
        if self.selector_name is not None:
            self._string = self.selector_name
        else:
            self._string = self._query
        self._hash_value = self._query.__hash__()

        # return data[self.attributeName] == self.attributeValue

    def __repr__(self):
        return self._query

    def covers(self, data):
        row = data[self.attribute_name].to_numpy()
        if pd.isnull(self.attribute_value):
            return pd.isnull(row)
        return row == self.attribute_value

    def __str__(self, open_brackets="", closing_brackets=""):
        return open_brackets + self._string + closing_brackets

    def __hash__(self):
        return getattr(self, "_hash_value")



class NegatedSelector(SelectorBase):
    def __init__(self, selector):
        self.selector = selector

    def covers(self, data_instance):
        return not self.selector.covers(data_instance)

    def __repr__(self):
        return "(not " + repr(self.selector) + ")"

    def __hash__(self):
        return repr(self).__hash__()

    def __str__(self, open_brackets="", closing_brackets=""):
        return "NOT " + str(self.selector, open_brackets, closing_brackets)

    def get_attribute_name(self):
        return self.selector.attribute_name


# Including the lower bound, excluding the upper_bound
class NumericSelector(SelectorBase):
    def __init__(self, attribute_name, lower_bound, upper_bound, selector_name=None):
        self._attribute_name = attribute_name
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self.selector_name = selector_name
        self.recompute_representations()

    def get_attribute_name(self):
        return self._attribute_name

    def set_attribute_name(self, value):
        self._attribute_name = value
        self.recompute_representations()

    def get_lower_bound(self):
        return self._lower_bound

    def set_lower_bound(self, value):
        self._lower_bound = value
        self.recompute_representations()

    def get_upper_bound(self):
        return self._upper_bound

    def set_upper_bound(self, value):
        self._upper_bound = value
        self.recompute_representations()

    attribute_name = property(get_attribute_name, set_attribute_name)
    lower_bound = property(get_lower_bound, set_lower_bound)
    upper_bound = property(get_upper_bound, set_upper_bound)

    def covers(self, data_instance):
        val = data_instance[self.attribute_name].to_numpy()
        return np.logical_and(val >= self.lower_bound, val < self.upper_bound)

    def __repr__(self):
        return self._query

    def __hash__(self):
        return repr(self).__hash__()

    def __str__(self):
        return self._string

    def recompute_representations(self):
        if self.selector_name is None:
            self._string = self.get_string(rounding_digits=2)
        else:
            self._string = self.selector_name
        self._query = self.get_string(rounding_digits=None)

    def get_string(self, open_brackets="", closing_brackets="", rounding_digits=2):
        if rounding_digits is None:
            formatter = "{}"
        else:
            formatter = "{0:." + str(rounding_digits) + "f}"
        ub = self.upper_bound
        lb = self.lower_bound
        if ub % 1:
            ub = formatter.format(ub)
        if lb % 1:
            lb = formatter.format(lb)

        if self.selector_name is not None:
            repre = self.selector_name
        elif self.lower_bound == float("-inf") and self.upper_bound == float("inf"):
            repre = self.attribute_name + "= anything"
        elif self.lower_bound == float("-inf"):
            repre = self.attribute_name + "<" + str(ub)
        elif self.upper_bound == float("inf"):
            repre = self.attribute_name + ">=" + str(lb)
        else:
            repre = self.attribute_name + ": [" + str(lb) + ":" + str(ub) + "["
        return open_brackets + repre + closing_brackets


@total_ordering
class Subgroup():
    def __init__(self, target, subgroup_description):
        # If its already a NominalTarget object, we are fine, otherwise we create a new one
        # if (isinstance(target, NominalTarget) or isinstance(target, NumericTarget)):
        #    self.target = target
        # else:
        #    self.target = NominalTarget(target)

        # If its already a SubgroupDescription object, we are fine, otherwise we create a new one
        self.target = target
        if isinstance(subgroup_description, SubgroupDescription):
            self.subgroup_description = subgroup_description
        else:
            self.subgroup_description = SubgroupDescription(subgroup_description)

        # initialize empty cache for statistics
        self.statistics = {}

    def __repr__(self):
        return "<<" + repr(self.target) + "; D: " + repr(self.subgroup_description) + ">>"

    def covers(self, instance):
        return self.subgroup_description.covers(instance)

    def count(self, data):
        return np.sum(self.subgroup_description.covers(data))

    def __eq__(self, other):
        if other is None:
            return False
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return repr(self) < repr(other)

    def get_base_statistics(self, data, weighting_attribute=None):
        return self.target.get_base_statistics(data, self, weighting_attribute)

    def calculate_statistics(self, data, weighting_attribute=None):
        self.target.calculate_statistics(self, data, weighting_attribute)


def create_selectors(data, nbins=5, intervals_only=True, ignore=None):
    if ignore is None:
        ignore = []
    sels = create_nominal_selectors(data, ignore)
    sels.extend(create_numeric_selectors(data, nbins, intervals_only, ignore=ignore))
    return sels


def create_nominal_selectors(data, ignore=None):
    if ignore is None:
        ignore = []
    nominal_selectors = []
    # for attr_name in [x for x in data.select_dtypes(exclude=['number']).columns.values if x not in ignore]:
    #    nominal_selectors.extend(create_nominal_selectors_for_attribute(data, attr_name))
    nominal_dtypes = data.select_dtypes(exclude=['number'])
    dtypes = data.dtypes
    # print(dtypes)
    for attr_name in [x for x in nominal_dtypes.columns.values if x not in ignore]:
        nominal_selectors.extend(create_nominal_selectors_for_attribute(data, attr_name, dtypes))
    return nominal_selectors


def create_nominal_selectors_for_attribute(data, attribute_name, dtypes=None):
    nominal_selectors = []
    for val in pd.unique(data[attribute_name]):
        nominal_selectors.append(NominalSelector(attribute_name, val))
    # setting the is_bool flag for selector
    if dtypes is None:
        dtypes = data.dtypes
    if dtypes[attribute_name] == 'bool':
        for s in nominal_selectors:
            s.is_bool = True
    return nominal_selectors


def create_numeric_selectors(data, nbins=5, intervals_only=True, weighting_attribute=None, ignore=None):
    if ignore is None:
        ignore = []
    numeric_selectors = []
    for attr_name in [x for x in data.select_dtypes(include=['number']).columns.values if x not in ignore]:
        numeric_selectors.extend(create_numeric_selector_for_attribute(
            data, attr_name, nbins, intervals_only, weighting_attribute))
    return numeric_selectors


def create_numeric_selector_for_attribute(data, attr_name, nbins=5, intervals_only=True, weighting_attribute=None):
    numeric_selectors = []
    data_not_null = data[data[attr_name].notnull()]

    uniqueValues = np.unique(data_not_null[attr_name])
    if len(data_not_null.index) < len(data.index):
        numeric_selectors.append(NominalSelector(attr_name, np.nan))

    if len(uniqueValues) <= nbins:
        for val in uniqueValues:
            numeric_selectors.append(NominalSelector(attr_name, val))
    else:
        cutpoints = ps.equal_frequency_discretization(data, attr_name, nbins, weighting_attribute)
        if intervals_only:
            old_cutpoint = float("-inf")
            for c in cutpoints:
                numeric_selectors.append(NumericSelector(attr_name, old_cutpoint, c))
                old_cutpoint = c
            numeric_selectors.append(NumericSelector(attr_name, old_cutpoint, float("inf")))
        else:
            for c in cutpoints:
                numeric_selectors.append(NumericSelector(attr_name, c, float("inf")))
                numeric_selectors.append(NumericSelector(attr_name, float("-inf"), c))

    return numeric_selectors


def remove_target_attributes(selectors, target):
    result = []
    for sel in selectors:
        if not sel.get_attribute_name() in target.get_attributes():
            result.append(sel)
    return result
