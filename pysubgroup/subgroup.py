''' 
Created on 28.04.2016

@author: lemmerfn
'''
import numpy as np
import pandas as pd
import pysubgroup as ps
from functools import total_ordering


@total_ordering
class SubgroupDescription(object):
    def __init__(self, selectors):
        if isinstance(selectors, list) or isinstance(selectors, tuple):
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
        return sum(1 for x in data if self.covers(x))
    
    def get_attributes(self):
        return set([x.get_attribute_name() for x in self.selectors])
   
    def to_string(self, open_brackets="", closing_brackets="", and_term=" AND "):
        if not self.selectors:
            return "Dataset"
        result = open_brackets
        for sel in self.selectors:
            result += str(sel) + and_term
        result = result[:-len(and_term)]
        return result + closing_brackets
    
    def __repr__(self):
        return self.to_string()
   
    def __eq__(self, other): 
        return set(self.selectors) == set(other.selectors)
    
    def __lt__(self, other): 
        return str(self) < str(other)


@total_ordering
class NominalSelector:
    def __init__(self, attribute_name, attribute_value, name=None):
        self.attribute_name = attribute_name
        self.attribute_value = attribute_value
        self.selector_name = name

    def covers(self, data):
        return data[self.attribute_name] == self.attribute_value
    
    def to_string(self, open_brackets="", closing_brackets=""):
        if self.selector_name is None:
            return open_brackets + str(self.attribute_name) + "=" + str(self.attribute_value) + closing_brackets
        return open_brackets + self.selector_name + closing_brackets
    
    def __repr__(self):
        return self.to_string()
    
    def __eq__(self, other): 
        if None is other:
            return False
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def __hash__(self): 
        return str(self).__hash__()
    
    def get_attribute_name(self):
        return self.attribute_name


@total_ordering
class NegatedSelector:
    def __init__(self, selector):
        self.selector = selector

    def covers(self, data_instance):
        return not self.selector.covers(data_instance)
    
    def __repr__(self):
        return self.to_string()
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def __hash__(self): 
        return str(self).__hash__()
    
    def to_string(self, open_brackets="", closing_brackets=""):
        return "NOT " + str(self.selector, open_brackets, closing_brackets)
    
    def get_attribute_name(self):
        return self.selector.attribute_name


# Including the lower bound, excluding the upperBound
@total_ordering
class NumericSelector:
    def __init__(self, attribute_name, lower_bound, upper_bound, name=None):
        self.attribute_name = attribute_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.selector_name = name

    def covers(self, data_instance):
        val = data_instance[self.attribute_name]
        return np.logical_and(val >= self.lower_bound, val < self.upper_bound)
    
    def __repr__(self):
        return self.to_string()
    
    def __eq__(self, other): 
        if other is None:
            return False
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def __hash__(self): 
        return str(self).__hash__()

    def to_string(self, open_brackets="", closing_brackets="", rounding_digits=2):
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
    
    def get_attribute_name(self):
        return self.attribute_name


@total_ordering
class Subgroup(object):
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
        return "<<" + str(self.target) + "; D: " + str(self.subgroup_description) + ">>"
    
    def covers(self, instance):
        return self.subgroup_description.covers(instance)
    
    def count(self, data):
        return np.sum(self.subgroup_description.covers(data))

    def __eq__(self, other):
        if other is None:
            return False
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
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
    for attr_name in [x for x in data.select_dtypes(exclude=['number']).columns.values if x not in ignore]:
        nominal_selectors.extend(create_nominal_selectors_for_attribute(data, attr_name))
    return nominal_selectors


def create_nominal_selectors_for_attribute(data, attribute_name):
    nominal_selectors = []
    for val in pd.unique(data[attribute_name]):
        nominal_selectors.append(NominalSelector(attribute_name, val))
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
    unique_values = np.unique(data[attr_name])
    if len(unique_values) <= nbins:
        for val in unique_values:
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
