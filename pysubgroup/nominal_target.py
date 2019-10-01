'''
Created on 29.09.2017

@author: lemmerfn
'''
import numpy as np
import scipy.stats
from functools import total_ordering

import pysubgroup as ps

from .subgroup import SubgroupDescription, Subgroup, NominalSelector


@total_ordering
class NominalTarget(object):
    
    def __init__(self, target_attribute=None, target_value=None, target_selector=None):
        """
        Creates a new target for the boolean model class (classic subgroup discovery). 
        If target_attribute and target_value are given, the target_selector is computed using attribute and value
        """
        if target_attribute is not None and target_value is not None:
            if target_selector is not None:
                raise BaseException("NominalTarget is to be constructed EITHER by a selector OR by attribute/value pair")
            target_selector = NominalSelector(target_attribute, target_value)
        if target_selector is None:
            raise BaseException("No target selector given")
        self.target_selector = target_selector
        
    def __repr__(self):
        return "T: " + str(self.target_selector)
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__
    
    def __lt__(self, other): 
        return str(self) < str(other)
    
    def covers(self, instance):
        return self.target_selector.covers(instance)
    
    def get_attributes(self):
        return [self.target_selector.get_attribute_name()]

    @staticmethod
    def get_base_statistics(data, subgroup, weighting_attribute=None):

        if weighting_attribute is None:
            sg_instances = subgroup.subgroup_description.covers(data)
            positives = subgroup.target.covers(data)
            instances_subgroup = np.sum(sg_instances)
            positives_dataset = np.sum(positives)
            instances_dataset = len(data)
            positives_subgroup = np.sum(np.logical_and(sg_instances, positives))
            return instances_dataset, positives_dataset, instances_subgroup, positives_subgroup
        else:
            weights = data[weighting_attribute]
            sg_instances = subgroup.subgroup_description.covers(data)
            positives = subgroup.target.covers(data)                         
    
            instances_dataset = np.sum(weights)
            instances_subgroup = np.sum(np.dot(sg_instances, weights))
            positives_dataset = np.sum(np.dot(positives, weights))
            positives_subgroup = np.sum(np.dot(np.logical_and(sg_instances, positives), weights))
            return instances_dataset, positives_dataset, instances_subgroup, positives_subgroup

    @staticmethod
    def calculate_statistics(subgroup, data, weighting_attribute=None):
        (instances_dataset, positives_dataset, instances_subgroup, positives_subgroup) = \
                                                NominalTarget.get_base_statistics(data, subgroup, weighting_attribute)
        subgroup.statistics['size_sg'] = instances_subgroup
        subgroup.statistics['size_dataset'] = instances_dataset
        subgroup.statistics['positives_sg'] = positives_subgroup
        subgroup.statistics['positives_dataset'] = positives_dataset
        
        subgroup.statistics['size_complement'] = instances_dataset - instances_subgroup
        subgroup.statistics['relative_size_sg'] = instances_subgroup / instances_dataset
        subgroup.statistics['relative_size_complement'] = (instances_dataset - instances_subgroup) / instances_dataset
        subgroup.statistics['coverage_sg'] = positives_subgroup / positives_dataset
        subgroup.statistics['coverage_complement'] = (positives_dataset - positives_subgroup) / positives_dataset
        subgroup.statistics['target_share_sg'] = positives_subgroup / instances_subgroup
        subgroup.statistics['target_share_complement'] = (positives_dataset - positives_subgroup) / (instances_dataset - instances_subgroup)
        subgroup.statistics['target_share_dataset'] = positives_dataset / instances_dataset
        subgroup.statistics['lift'] = (positives_subgroup / instances_subgroup) / (positives_dataset / instances_dataset)
        
        if weighting_attribute is not None:
            (instances_dataset, positives_dataset, instances_subgroup, positives_subgroup) = \
                                                NominalTarget.get_base_statistics(subgroup, data, weighting_attribute)
        subgroup.statistics['size_sg_weighted'] = instances_subgroup
        subgroup.statistics['size_dataset_weighted'] = instances_dataset
        subgroup.statistics['positives_sg_weighted'] = positives_subgroup
        subgroup.statistics['positives_dataset_weighted'] = positives_dataset
        
        subgroup.statistics['size_complement_weighted'] = instances_dataset - instances_subgroup
        subgroup.statistics['relative_size_sg_weighted'] = instances_subgroup / instances_dataset
        subgroup.statistics['relative_size_complement_weighted'] = \
                        (instances_dataset - instances_subgroup) / instances_dataset
        subgroup.statistics['coverage_sg_weighted'] = positives_subgroup / positives_dataset
        subgroup.statistics['coverage_complement_weighted'] = (positives_dataset - positives_subgroup) / positives_dataset
        subgroup.statistics['target_share_sg_weighted'] = positives_subgroup / instances_subgroup
        subgroup.statistics['target_share_complement_weighted'] = (positives_dataset - positives_subgroup) / (instances_dataset - instances_subgroup)
        subgroup.statistics['target_share_dataset_weighted'] = positives_dataset / instances_dataset
        subgroup.statistics['lift_weighted'] = (positives_subgroup / instances_subgroup) / (positives_dataset / instances_dataset)
        
    
class ChiSquaredQF (ps.AbstractInterestingnessMeasure):
    @staticmethod     
    def chi_squared_qf(instances_dataset, positives_dataset, instances_subgroup, positives_subgroup, min_instances=5, bidirect=True, direction_positive=True):
        if (instances_subgroup < min_instances) or ((instances_dataset - instances_subgroup) < min_instances):
            return float("-inf")
        p_subgroup = positives_subgroup / instances_subgroup
        p_dataset = positives_dataset / instances_dataset
        positives_complement = positives_dataset - positives_subgroup

        # instancesComplement = instancesDataset - instancesSubgroup
        negatives_subgroup = instances_subgroup - positives_subgroup
        negatives_dataset = instances_dataset - positives_dataset
        negatives_complement = negatives_dataset - negatives_subgroup
        
        # observed = [positivesSubgroup, positives_complement,negatives_subgroup, negatives_complement]
        #
        # if round(positivesSubgroup) < 0 or
        #    round(positives_complement) < 0 or
        #    round(negatives_subgroup) <0 or
        #    round (negatives_complement) < 0:
        #    print ("XXXXX")
        val = scipy.stats.chi2_contingency([[round(positives_subgroup), round(positives_complement)],
                                            [round(negatives_subgroup), round(negatives_complement)]], correction=False)[0]
        if bidirect:
            return val
        elif direction_positive and p_subgroup > p_dataset:
            return val
        elif not direction_positive and p_subgroup < p_dataset:
            return val
        return -val
    
    @staticmethod     
    def chi_squared_qf_weighted (subgroup, data, weighting_attribute, effective_sample_size=0, min_instances=5, ):
        (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = subgroup.get_base_statistics(data, weighting_attribute)
        if (instancesSubgroup < min_instances) or ((instancesDataset - instancesSubgroup) < 5):
            return float("inf")
        if effective_sample_size == 0:
            effective_sample_size = ps.effective_sample_size(data[weighting_attribute])
        # p_subgroup = positivesSubgroup / instancesSubgroup
        # p_dataset = positivesDataset / instancesDataset

        negatives_subgroup = instancesSubgroup - positivesSubgroup
        negatives_dataset = instancesDataset - positivesDataset
        positives_complement = positivesDataset - positivesSubgroup
        negatives_complement = negatives_dataset - negatives_subgroup
        val = scipy.stats.chi2_contingency([[positivesSubgroup, positives_complement],
                                            [negatives_subgroup, negatives_complement]], correction=True) [0]
        return scipy.stats.chi2.sf(val * effective_sample_size / instancesDataset, 1)
    
    def __init__(self, direction='bidirect', min_instances=5):
        if direction == 'bidirect':
            self.bidirect = True
            self.direction_positive = True
        if direction == 'positive':
            self.bidirect = False
            self.direction_positive = True
        if direction == 'negative':
            self.bidirect = False
            self.direction_positive = False
        self.min_instances = min_instances

    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
        if not self.is_applicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        if weighting_attribute is None:
            result = self.evaluate_from_statistics(*subgroup.get_base_statistics(data))
        else:
            (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup) = subgroup.get_base_statistics(data, weighting_attribute)
            weights = data[weighting_attribute]
            base = self.evaluate_from_statistics(instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)
            result = base * ps.effective_sample_size(weights) / instancesDataset
        return result
    
    def evaluate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return ChiSquaredQF.chi_squared_qf(instances_dataset, positives_dataset, instances_subgroup, positives_subgroup, self.min_instances, self.bidirect, self.direction_positive)

    def supports_weights(self):
        return True
    
    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, NominalTarget)


class StandardQF (ps.AbstractInterestingnessMeasure, ps.BoundedInterestingnessMeasure):
    @staticmethod     
    def standard_qf(a, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        if instances_subgroup == 0:
            return 0
        p_subgroup = positives_subgroup / instances_subgroup
        p_dataset = positives_dataset / instances_dataset
        return (instances_subgroup / instances_dataset) ** a * (p_subgroup - p_dataset)
            
    def __init__(self, a):
        self.a = a
    
    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
        if not self.is_applicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return self.evaluate_from_statistics (*subgroup.get_base_statistics(data, weighting_attribute))
    
    def optimistic_estimate_from_dataset(self, data, subgroup, weighting_attribute=None):
        if not self.is_applicable(subgroup):
            raise BaseException("Quality measure cannot be used for this target class")
        return self.optimistic_estimate_from_statistics(*subgroup.get_base_statistics(data, weighting_attribute))

    def evaluate_from_statistics (self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return StandardQF.standard_qf(self.a, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup)
    
    def optimistic_estimate_from_statistics (self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return StandardQF.standard_qf(self.a, instances_dataset, positives_dataset, positives_subgroup, positives_subgroup)

    def supports_weights(self):
        return True
    
    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, NominalTarget)


class WRAccQF (StandardQF):
    def __init__(self, a):
        super().__init__(a)
        self.a = 1.0


class LiftQF (StandardQF):
    def __init__(self, a):
        super().__init__(a)
        self.a = 0.0


class SimpleBinomial(StandardQF):
    def __init__(self, a):
        super().__init__(a)
        self.a = 0.5


#####
# GeneralizationAware Interestingness Measures
#####
class GAStandardQF (ps.AbstractInterestingnessMeasure):
    def __init__(self, a):
        self.a = a

    def evaluate_from_dataset(self, data, subgroup, weighting_attribute=None):
        (instances_dataset, _, instances_subgroup, positives_subgroup) = subgroup.get_base_statistics(data, weighting_attribute)
        if (instances_subgroup == 0) or (instances_dataset == instances_subgroup):
            return 0
        p_subgroup = positives_subgroup / instances_subgroup
        max_target_share = get_max_generalization_target_share(data, subgroup, weighting_attribute)
        relative_size = (instances_subgroup / instances_dataset)
        return relative_size ** self.a * (p_subgroup - max_target_share)

    def supports_weights(self):
        return True

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, NominalTarget)


def get_max_generalization_target_share(data, subgroup, weighting_attribute=None):
    selectors = subgroup.subgroup_description.selectors
    generalizations = ps.powerset(selectors)
    max_target_share = 0
    for sels in generalizations:
        sgd = SubgroupDescription(list(sels))
        sg = Subgroup(subgroup.target, sgd)
        (_, _, instances_subgroup, positives_subgroup) = sg.get_base_statistics(data, weighting_attribute)
        target_share = positives_subgroup / instances_subgroup
        max_target_share = max(max_target_share, target_share)
    return max_target_share
