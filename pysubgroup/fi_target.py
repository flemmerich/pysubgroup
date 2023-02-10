'''
Created on 29.09.2017

@author: lemmerfn
'''
from collections import namedtuple
from functools import total_ordering
import pysubgroup as ps



@total_ordering
class FITarget(ps.BaseTarget):
    statistic_types = ('size_sg', 'size_dataset')

    def __repr__(self):
        return "T: Frequent Itemsets"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return []

    def get_base_statistics(self, subgroup, data):
        _, size = ps.get_cover_array_and_size(subgroup, len(data), data)
        return size

    def calculate_statistics(self, subgroup_description, data, cached_statistics=None):
        if self.all_statistics_present(cached_statistics):
            return cached_statistics

        
        _, size = ps.get_cover_array_and_size(subgroup_description, len(data), data)
        statistics={}
        statistics['size_sg'] = size
        statistics['size_dataset'] = len(data)
        return statistics


class SimpleCountQF(ps.AbstractInterestingnessMeasure):
    tpl = namedtuple('CountQF_parameters', ('size_sg'))

    def __init__(self):
        self.required_stat_attrs = ('size_sg',)
        self.has_constant_statistics = True
        self.size_dataset = None

    def calculate_constant_statistics(self, data, target):# pylint: disable=unused-argument
        self.size_dataset = len(data)

    def calculate_statistics(self, subgroup_description, target, data, statistics=None): # pylint: disable=unused-argument
        _, size = ps.get_cover_array_and_size(subgroup_description, self.size_dataset, data)
        return SimpleCountQF.tpl(size)

    def gp_get_stats(self, _):
        return {"size_sg" : 1}

    def gp_get_null_vector(self):
        return {"size_sg":0}

    def gp_merge(self, l, r):
        l["size_sg"] += r["size_sg"]

    def gp_get_params(self, _cover_arr, v):
        return SimpleCountQF.tpl(v['size_sg'])

    def gp_to_str(self, stats):
        return str(stats['size_sg'])

    def gp_size_sg(self, stats):
        return stats['size_sg']

    @property
    def gp_requires_cover_arr(self):
        return False


class CountQF(SimpleCountQF, ps.BoundedInterestingnessMeasure):
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.size_sg

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.size_sg



class AreaQF(SimpleCountQF):
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.size_sg * subgroup.depth
