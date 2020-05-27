'''
Created on 29.09.2017

@author: lemmerfn
'''
from collections import namedtuple
from functools import total_ordering
import pysubgroup as ps



@total_ordering
class FITarget:
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
        if cached_statistics is None or not isinstance(cached_statistics, dict):
            statistics = dict()
        elif all(k in cached_statistics for k in FITarget.statistic_types):
            return cached_statistics
        else:
            statistics = cached_statistics

        _, size = ps.get_cover_array_and_size(subgroup_description, len(data), data)

        statistics['size_sg'] = size
        statistics['size_dataset'] = len(data)
        return statistics


class SimpleCountQF(ps.AbstractInterestingnessMeasure):
    tpl = namedtuple('CountQF_parameters', ('subgroup_size'))

    def __init__(self):
        self.required_stat_attrs = ('subgroup_size',)
        self.has_constant_statistics = True
        self.size_dataset = None

    def calculate_constant_statistics(self, data, target):
        self.size_dataset = len(data)

    def calculate_statistics(self, subgroup_description, target, data, statistics=None):
        _, size = ps.get_cover_array_and_size(subgroup_description, self.size_dataset, data)
        return SimpleCountQF.tpl(size)

    def gp_get_stats(self, _):
        return {"subgroup_size" : 1}

    def gp_get_null_vector(self):
        return {"subgroup_size":0}

    def gp_merge(self, l, r):
        l["subgroup_size"] += r["subgroup_size"]

    def gp_get_params(self, _cover_arr, v):
        return SimpleCountQF.tpl(v['subgroup_size'])

    def gp_to_str(self, stats):
        return str(stats['subgroup_size'])

    @property
    def gp_requires_cover_arr(self):
        return False


class CountQF(SimpleCountQF, ps.BoundedInterestingnessMeasure):
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.subgroup_size

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.subgroup_size



class AreaQF(SimpleCountQF):
    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.subgroup_size * subgroup.depth
