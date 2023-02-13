import unittest
from pysubgroup.tests.DataSets import get_credit_data

import pandas as pd
import pysubgroup as ps



class TestAlgorithms(unittest.TestCase):
    def setUp(self) -> None:
        data = get_credit_data()
        target = ps.BinaryTarget('class', b'bad')
        searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['class'])
        searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['class'])
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=2, depth=1, qf=ps.StandardQF(0.5))

    def test_BeamSearch_raises(self):
        with self.assertRaises(RuntimeError):
            ps.BeamSearch(1, False).execute(self.task)

    def test_DFSNUmeric_raises(self):
        with self.assertRaises(RuntimeError):
            ps.DFSNumeric().execute(self.task)

    def test_BeamSearch_adaptive(self):
        ps.BeamSearch(1, beam_width_adaptive=True).execute(self.task)




if __name__ == '__main__':
    unittest.main()