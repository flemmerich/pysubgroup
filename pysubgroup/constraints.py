import pysubgroup as ps

class MinSupportConstraint:
    def __init__(self, min_support):
        self.min_support = min_support

    @property
    def is_monotone(self):
        return True

    def is_satisfied(self, subgroup, statistics=None, data=None):
        if hasattr(statistics, 'size'):
            return statistics.size >= self.min_support
        elif hasattr(statistics, 'size_sg'):
            return statistics.size_sg >= self.min_support
        else:
            return ps.get_size(subgroup, len(data), data) >= self.min_support

    def gp_prepare(self, qf):
        self.get_subgroup_size = qf.gp_subgroup_size # pylint: disable=attribute-defined-outside-init

    def gp_is_satisfied(self, node):
        return self.get_subgroup_size(node) >= self.min_support