import pysubgroup as ps

class MinSupportConstraint:
    def __init__(self, min_support):
        self.min_support = min_support

    @property
    def is_monotone(self):
        return True

    def is_satisfied(self, subgroup, statistics=None, data=None):
        if hasattr(statistics, 'size_sg'):
            return statistics.size_sg >= self.min_support
        if isinstance(statistics, dict) and 'size_sg' in statistics:
            return statistics['size_sg'] >= self.min_support
        try:
            return ps.get_size(subgroup, len(data), data) >= self.min_support
        except AttributeError: # special case for gp_growth
            return self.get_size_sg(statistics)

    def gp_prepare(self, qf):
        self.get_size_sg = qf.gp_size_sg # pylint: disable=attribute-defined-outside-init

    def gp_is_satisfied(self, node):
        return self.get_size_sg(node) >= self.min_support
