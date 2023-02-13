import unittest

import numpy as np
import pandas as pd
import pysubgroup as ps



class TestGpGrowth(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame.from_records([(1,1,0), (1,1,1,), (1,1,0), (1,0,1)], columns=("A", "B", "C"))
    

    def test_export_fi(self):
        target = ps.FITarget()
        searchspace = ps.create_selectors(self.df)
        task = ps.SubgroupDiscoveryTask(self.df, target, searchspace, result_set_size=5, depth=2, qf=ps.CountQF())
        ps.GpGrowth().to_file(task, "./test_gp_fi.txt")

    def test_export_binary(self):
        target = ps.BinaryTarget("A", 1)
        searchspace = ps.create_selectors(self.df, ignore=["A"])
        task = ps.SubgroupDiscoveryTask(self.df, target, searchspace, result_set_size=5, depth=2, qf=ps.StandardQF(0.5))
        ps.GpGrowth().to_file(task, "./test_gp_binary.txt")

    def test_export_model(self):
        model = ps.PolyRegression_ModelClass("A", "B")
        QF = ps.EMM_Likelihood(model)
        searchspace = ps.create_selectors(self.df)
        task = ps.SubgroupDiscoveryTask(self.df, None, searchspace, result_set_size=5, depth=2, qf=QF)
        ps.GpGrowth().to_file(task, "./test_gp_model.txt")



if __name__ == "__main__":
    unittest.main()

