'''
Created on 29.09.2017

@author: lemmerfn
'''
import numpy as np
import pysubgroup as ps
from functools import total_ordering



@total_ordering
class NumericTarget(object):
    def __init__(self, target_variable):
        self.target_variable = target_variable
        
    def __repr__(self):
        return "T: " + str(self.target_variable)
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def get_attributes(self):
        return [self.target_variable]
    
    def get_base_statistics (self, data, subgroup, weighting_attribute=None):
        if weighting_attribute is None:
            sg_instances = subgroup.subgroup_description.covers(data)
            all_target_values = data[self.target_variable]
            sg_target_values = all_target_values[sg_instances]
            instances_dataset = len(data)
            instances_subgroup = np.sum(sg_instances)
            mean_sg = np.mean (sg_target_values)
            mean_dataset = np.mean (all_target_values)
            return (instances_dataset, mean_dataset, instances_subgroup, mean_sg)  
        else:
            raise NotImplemented("Attribute weights with numeric targets are not yet implemented.")
        
    def calculate_statistics (self, subgroup, data, weighting_attribute=None):
        if weighting_attribute is not None:
            raise NotImplemented("Attribute weights with numeric targets are not yet implemented.")
        sg_instances = subgroup.subgroup_description.covers(data)
        all_target_values = data[self.target_variable]
        sg_target_values = all_target_values[sg_instances]
        subgroup.statistics['size_sg'] = len(sg_target_values)
        subgroup.statistics['size_dataset'] = len (data)
        subgroup.statistics['mean_sg'] = np.mean(sg_target_values)
        subgroup.statistics['mean_dataset'] = np.mean(all_target_values)
        subgroup.statistics['std_sg'] = np.std(sg_target_values)
        subgroup.statistics['std_dataset'] = np.std(all_target_values)
        subgroup.statistics['median_sg'] = np.median(sg_target_values)
        subgroup.statistics['median_dataset'] = np.median(all_target_values)
        subgroup.statistics['max_sg'] = np.max(sg_target_values)
        subgroup.statistics['max_dataset'] = np.max(all_target_values)
        subgroup.statistics['min_sg'] = np.min(sg_target_values)
        subgroup.statistics['min_dataset'] = np.min(all_target_values)
        subgroup.statistics['mean_lift'] = subgroup.statistics['mean_sg'] / subgroup.statistics['mean_dataset']
        subgroup.statistics['median_lift'] = subgroup.statistics['median_sg'] / subgroup.statistics['median_dataset']


class StandardQFNumeric(ps.AbstractInterestingnessMeasure, ps.BoundedInterestingnessMeasure):
    
    @staticmethod     
    def standard_qf_numeric (a, instances_dataset, mean_dataset, instances_subgroup, mean_sg):
        if instances_subgroup == 0:
            return 0
        return instances_subgroup ** a * (mean_sg - mean_dataset)
        
    def __init__(self, a, invert=False):
        self.a = a
        self.invert = invert
        
    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None, cache=None):
        if not self.is_applicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return ps.conditional_invert(self.evaluate_from_statistics (*subgroup.get_base_statistics(data, weighting_attribute)), self.invert)

    def optimistic_estimate_from_dataset(self, data, subgroup):
        if not self.is_applicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        all_target_values = data[subgroup.target.target_variable]
        sg_instances = subgroup.subgroup_description.covers(data)
        mean_dataset = np.mean(all_target_values)
        sg_target_values = all_target_values[sg_instances]
        target_values_larger_than_mean = sg_target_values [sg_target_values > mean_dataset]
        return ps.conditional_invert(np.sum(target_values_larger_than_mean) - (len (target_values_larger_than_mean) * mean_dataset), self.invert)

    def evaluate_from_statistics(self, instances_dataset, mean_dataset, instances_subgroup, mean_sg):
        return StandardQFNumeric.standard_qf_numeric (self.a, instances_dataset, mean_dataset, instances_subgroup, mean_sg)
    
    def optimistic_estimate_from_statistics (self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return float("inf")

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, NumericTarget)

    def supports_weights(self):
        return False


class GAStandardQFNumeric(ps.AbstractInterestingnessMeasure):
    def __init__(self, a, invert=False):
        self.a = a
        self.invert = invert
        
    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
        (instances_dataset, _, instances_subgroup, mean_sg) = subgroup.get_base_statistics(data, weighting_attribute)
        if (instances_subgroup == 0) or (instances_dataset == instances_subgroup):
            return 0
        max_mean = get_max_generalization_mean(data, subgroup, weighting_attribute)
        relative_size = (instances_subgroup / instances_dataset)
        return ps.conditional_invert(relative_size ** self.a * (mean_sg - max_mean), self.invert)

    def supports_weights(self):
        return True

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, NumericTarget)


def get_max_generalization_mean(data, subgroup, weighting_attribute=None):
    selectors = subgroup.subgroup_description.selectors
    generalizations = ps.powerset(selectors)
    max_mean = 0
    for sels in generalizations:
        sg = ps.Subgroup(subgroup.target, ps.SubgroupDescription(list(sels)))
        mean_sg = sg.get_base_statistics(data, weighting_attribute)[3]
        max_mean = max(max_mean, mean_sg)
    return max_mean
