'''
Created on 29.09.2017

@author: lemmerfn
'''
from functools import total_ordering
import pysubgroup as ps

@total_ordering
class FITarget(object):
    def __init__(self):
        pass
        
    def __repr__(self):
        return "T: Frequent Itemsets"
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def get_attributes(self):
        return []
    
    def get_base_statistics (self, data, subgroup, weighting_attribute=None):
        if weighting_attribute is None:
            sg_instances = subgroup.subgroup_description.covers(data)
            return sg_instances.sum()  
        else:
            raise NotImplemented("Attribute weights with numeric targets are not yet implemented.")
        
    def calculate_statistics(self, subgroup, data, weighting_attribute=None):
        if weighting_attribute is not None:
            raise NotImplemented("Attribute weights with numeric targets are not yet implemented.")
        sg_instances = subgroup.subgroup_description.covers(data)
        
        subgroup.statistics['size_sg'] = len(sg_instances)
        subgroup.statistics['size_dataset'] = len(data)


class CountQF (ps.AbstractInterestingnessMeasure, ps.BoundedInterestingnessMeasure):
    def __init__(self):
        pass
    
    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
        return subgroup.subgroup_description.covers(data).sum()
    
    def optimistic_estimate_from_dataset(self, data, subgroup):
        return subgroup.subgroup_description.covers(data).sum()
        
    def evaluate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return instances_subgroup
    
    def optimistic_estimate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return instances_subgroup

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, FITarget)

    def supports_weights(self):
        return False


class AreaQF(ps.AbstractInterestingnessMeasure):
    def __init__(self):
        pass

    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
        return len(subgroup.subgroup_description) * subgroup.subgroup_description.covers(data).sum()

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, FITarget)

    def supports_weights(self):
        return False
