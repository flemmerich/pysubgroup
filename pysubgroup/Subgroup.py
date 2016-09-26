'''
Created on 28.04.2016

@author: lemmerfn
'''
class SubgroupDescription(object):
    def __init__(self, selectors):
        self.selectors = selectors
    
    def __repr__(self):
        result = "{"
        for sel in self.selectors:
            result += str(sel)
        result = result [:-1]
        return result + "}"
    
    def covers(self, instance):
        return all(sel.covers(instance) for sel in self.selectors)
    
    def count (self, data):
        return sum(1 for x in data if self.covers(x)) 


class NominalTarget(object):
    def __init__(self, targetSelector):
        self.targetSelector = targetSelector
        
    def __repr__(self):
        return "T: " + str(self.targetSelector)
    
    def covers(self, instance):
        return self.targetSelector.covers(instance)
    
    def getAttributes(self):
        return [self.targetSelector.getAttributeName()]


class Subgroup(object):
    def __init__(self, target, subgroupDescription):
        # If its already a NominalTarget object, we are fine, otherwise we create a new one
        if (isinstance(target, NominalTarget)):
            self.target = target
        else:
            self.target = NominalTarget(target)
        
        # If its already a SubgroupDescription object, we are fine, otherwise we create a new one
        if (isinstance(subgroupDescription, SubgroupDescription)):
            self.subgroupDescription = subgroupDescription
        else:
            self.subgroupDescription = SubgroupDescription(subgroupDescription)
    
    def __repr__(self):
        return "<<" + str(self.target) + "; D: " + str(self.subgroupDescription) + ">>"
    
    def covers(self, instance):
        return all(sel.covers(instance) for sel in self.selectors)
    
    def count(self, data):
        return sum(1 for x in data if self.covers(x)) 
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
