import unittest
import numpy as np
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data

class TestQFInputs(unittest.TestCase):
    def setUp(self):
        self.data = get_credit_data()

    def test_get_cover_array_and_size(self):
        sel = ps.EqualitySelector('checking_status', b'no checking')
        _, size = ps.get_cover_array_and_size(sel, None, self.data)
        self.assertEqual(size, 394)
        _, size = ps.get_cover_array_and_size(slice(None), len(self.data), None)
        self.assertEqual(size, len(self.data))
        _, size = ps.get_cover_array_and_size(slice(0, 10), len(self.data))
        self.assertEqual(size, 10)
        _, size = ps.get_cover_array_and_size(np.array([1, 3, 5, 7, 11], dtype=int))
        self.assertEqual(size, 5)

# TODO Need to test other qf as well
#    def test_AreaQf(self):
#        pass

    def test_CountQf(self):
        target = ps.FITarget()
        #task = ps.SubgroupDiscoveryTask(self.data, ps.FITarget(), None, None)

        qf = ps.CountQF()
        qf.calculate_constant_statistics(self.data, target)
        sel = ps.EqualitySelector('checking_status', b'no checking')
        print(self.data.columns)
        print(self.data.checking_status.value_counts())
        size = qf.evaluate(sel, target, self.data)
        self.assertEqual(size, 394)
        size = qf.evaluate(slice(None), target, self.data)
        self.assertEqual(size, len(self.data))
        size = qf.evaluate(slice(0, 10), target, self.data)
        self.assertEqual(size, 10)
        size = qf.evaluate(np.array([1, 3, 5, 7, 11], dtype=int), target, self.data)
        self.assertEqual(size, 5)

if __name__ == '__main__':
    unittest.main()
