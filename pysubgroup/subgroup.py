''' 
Created on 28.04.2016

@author: lemmerfn
'''
import numpy as np
import pandas as pd
import pysubgroup.utils as ut
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
        if (not self.selectors):
            return np.full(len(instance), True, dtype=bool)
        # non-empty description
        return np.all([sel.covers(instance) for sel in self.selectors], axis=0)

    def __len__(self):
        return len(self.selectors)

    def count (self, data):
        return sum(1 for x in data if self.covers(x))
    
    def getAttributes(self):
        return set([x.getAttributeName() for x in self.selectors])
   
    def to_string(self, open_brackets="", closing_brackets="", sel_open_bracket="", sel_closing_bracket="", and_term=" AND "):
        if not self.selectors:
            return "Dataset"
        result = open_brackets
        for sel in self.selectors:
            result += str(sel) + and_term
        result = result [:-len(and_term)]
        return result + closing_brackets
    
    def __repr__(self):
        return self.to_string()
   
    def __eq__(self, other): 
        return set(self.selectors) == set(other.selectors)
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
@total_ordering
class NominalSelector:
    def __init__(self, attributeName, attributeValue, name=None):
        self.attributeName = attributeName
        self.attributeValue = attributeValue
        self.selector_name = name

    def covers (self, data):
        return data[self.attributeName] == self.attributeValue
    
    def to_string(self, open_brackets="", closing_brackets=""):
        if self.selector_name == None:
            return open_brackets + str(self.attributeName) + "=" + str(self.attributeValue) + closing_brackets
        return open_brackets + self.selector_name + closing_brackets
    
    def __repr__(self):
        return self.to_string()
    
    def __eq__(self, other): 
        if None == other:
            return False
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def __hash__(self): 
        return str(self).__hash__()
    
    def getAttributeName(self):
        return self.attributeName

@total_ordering
class NegatedSelector:
    def __init__(self, selector):
        self.selector = selector

    def covers (self, dataInstance):
        return not self.selector.covers(dataInstance)
    
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
    
    def getAttributeName(self):
        return self.selector.attributeName


# Including the lower bound, excluding the upperBound
@total_ordering
class NumericSelector:
    def __init__(self, attributeName, lowerBound, upperBound, name=None):
        self.attributeName = attributeName
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.selector_name = name

    def covers (self, dataInstance):
        val = dataInstance[self.attributeName]
        return np.logical_and(val >= self.lowerBound, val < self.upperBound)
    
    def __repr__(self):
        return self.to_string()
    
    def __eq__(self, other): 
        if other == None:
            return False
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def __hash__(self): 
        return str(self).__hash__()

    def to_string(self, open_brackets="", closing_brackets="", roundingDigits=2):
        formatter = "{0:." + str(roundingDigits) + "f}"
        ub = self.upperBound
        lb = self.lowerBound
        if ub % 1:
            ub = formatter.format(ub)
        if lb % 1:
            lb = formatter.format(lb)
        
        if self.selector_name != None:
            repre = self.selector_name
        elif self.lowerBound == float("-inf") and self.upperBound == float("inf"):
            repre = self.attributeName + "= anything"
        elif self.lowerBound == float("-inf"):
            repre = self.attributeName + "<" + str(ub)
        elif self.upperBound == float("inf"):
            repre = self.attributeName + ">=" + str(lb)
        else:
            repre = self.attributeName + ": [" + str(lb) + ":" + str(ub) + "["
        return open_brackets + repre + closing_brackets
    
    def getAttributeName(self):
        return self.attributeName 


@total_ordering
class Subgroup(object):
    def __init__(self, target, subgroupDescription):
        # If its already a NominalTarget object, we are fine, otherwise we create a new one
        # if (isinstance(target, NominalTarget) or isinstance(target, NumericTarget)):
        #    self.target = target
        # else:
        #    self.target = NominalTarget(target)
        
        # If its already a SubgroupDescription object, we are fine, otherwise we create a new one
        self.target = target
        if (isinstance(subgroupDescription, (SubgroupDescription))):
            self.subgroupDescription = subgroupDescription
        else:
            self.subgroupDescription = SubgroupDescription(subgroupDescription)
            
        # initialize empty cache for statistics
        self.statistics = {} 
    
    def __repr__(self):
        return "<<" + str(self.target) + "; D: " + str(self.subgroupDescription) + ">>"
    
    def covers(self, instance):
        return self.subgroupDescription.covers(instance)
    
    def count(self, data):
        return np.sum(self.subgroupDescription.covers(data))

    def __eq__(self, other):
        if other == None:
            return False
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def get_base_statistics (self, data, weightingAttribute=None):
        return self.target.get_base_statistics (data, self, weightingAttribute)
    
    def calculateStatistics (self, data, weightingAttribute=None):
        self.target.calculateStatistics(self, data, weightingAttribute)
        

def createSelectors (data, nbins=5, intervals_only=True, ignore=None):
    if ignore is None:
        ignore = []
    sels = createNominalSelectors(data, ignore)
    sels.extend(createNumericSelectors(data, nbins, intervals_only, ignore=ignore))
    return sels

def createNominalSelectors(data, ignore=None):
    if ignore is None:
        ignore = []
    nominal_selectors = []
    for attr_name in [x for x in data.select_dtypes(exclude=['number']).columns.values if x not in ignore]:
        nominal_selectors.extend(createNominalSelectorsForAttribute(data, attr_name))
    return nominal_selectors

def createNominalSelectorsForAttribute(data, attribute_name):
    nominal_selectors = []
    for val in pd.unique(data[attribute_name]):
        nominal_selectors.append (NominalSelector(attribute_name, val))
    return nominal_selectors        
                    
def createNumericSelectors(data, nbins=5, intervals_only=True, weightingAttribute=None, ignore=None):
    if ignore is None:
        ignore = []
    numeric_selectors = []
    for attr_name in [x for x in data.select_dtypes(include=['number']).columns.values if x not in ignore]:
        numeric_selectors.extend(createNumericSelectorForAttribute(data, attr_name, nbins, intervals_only, weightingAttribute))
    return numeric_selectors

def createNumericSelectorForAttribute(data, attr_name, nbins=5, intervals_only=True, weightingAttribute=None):
            numeric_selectors = []
            uniqueValues = np.unique(data[attr_name])
            if (len(uniqueValues) <= nbins):
                for val in uniqueValues: 
                    numeric_selectors.append(NominalSelector(attr_name, val))
            else: 
                cutpoints = ut.equalFrequencyDiscretization(data, attr_name, nbins, weightingAttribute)
                if intervals_only:
                    old_cutpoint = float ("-inf")
                    for c in cutpoints:
                        numeric_selectors.append(NumericSelector(attr_name, old_cutpoint, c))
                        old_cutpoint = c
                    numeric_selectors.append(NumericSelector(attr_name, old_cutpoint, float("inf")))
                else:
                    for c in cutpoints:
                        numeric_selectors.append(NumericSelector(attr_name, c, float("inf")))
                        numeric_selectors.append(NumericSelector(attr_name, float("-inf"), c))
                
            return numeric_selectors

def removeTargetAttributes (selectors, target):
    result = []
    for sel in selectors:
        if not sel.getAttributeName() in target.getAttributes():
            result.append(sel)
    return result

