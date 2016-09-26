'''
Created on 26.04.2016

@author: lemmerfn
'''
import SGDUtils
import numpy as np
import pandas as pd

def createSelectors (data, nbins=5, intervals_only=True):
    sels = createNominalSelectors(data)
    sels.extend(createNumericSelectors(data, nbins))
    return sels

def createNominalSelectors(data):
    nominal_selectors = []
    for i, attr_name in enumerate(data.dtype.names):
        # # is nominal?
        if (data.dtype[i].type is np.string_) or (data.dtype[i].type is np.object_):
        # if meta.types()[i] == "nominal":
            # this gives a list of attribute values for the attribute with name attr_name
            for val in np.unique(data[attr_name]):
                nominal_selectors.append (NominalSelector(attr_name, val))
    return nominal_selectors

def createNumericSelectors(data, nbins=5, intervals_only=True):
    numeric_selectors = []
    for i, attr_name in enumerate(data.dtype.names):
        if ((data.dtype[i] == 'float64') or (data.dtype[i] == 'float32') or data.dtype[i] == 'int'):
            uniqueValues = np.unique(data[attr_name])
            if (len(uniqueValues) <= nbins):
                for val in uniqueValues:
                    numeric_selectors.append(NominalSelector(attr_name, val))
            else: 
                cutpoints = SGDUtils.equalFrequencyDiscretization(data, attr_name, nbins)
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

class NominalSelector:
    def __init__(self, attributeName, attributeValue):
        self.attributeName = attributeName
        self.attributeValue = attributeValue

    def covers (self, dataInstance):
        return dataInstance[self.attributeName] == self.attributeValue
    
    def __repr__(self):
        return "{" + str(self.attributeName) + "==" + str(self.attributeValue) + "}"
    
    def getAttributeName(self):
        return self.attributeName

class NegatedSelector:
    def __init__(self, selector):
        self.selector = selector

    def covers (self, dataInstance):
        return not self.selector.covers(dataInstance)
    
    def __repr__(self):
        return "NOT " + str(self.selector)
    
    def getAttributeName(self):
        return self.selector.attributeName
    
# Including the lower bound, excluding the upperBound
class NumericSelector:
    def __init__(self, attributeName, lowerBound, upperBound):
        self.attributeName = attributeName
        self.lowerBound = lowerBound
        self.upperBound = upperBound

    def covers (self, dataInstance):
        val = dataInstance[self.attributeName]
        return (val >= self.lowerBound) and (val < self.upperBound)
    
    def __repr__(self):
        if self.lowerBound == float("-inf"):
            return "{" + self.attributeName + ": ]" + str(self.lowerBound) + ":" + str(self.upperBound) + "[}"
        return "{" + self.attributeName + ": [" + str(self.lowerBound) + ":" + str(self.upperBound) + "[}"
    
    def getAttributeName(self):
        return self.attributeName


def removeTargetAttributes (selectors, target):
    result = []
    for sel in selectors:
        if not sel.getAttributeName() in target.getAttributes():
            result.append(sel)
    return result
