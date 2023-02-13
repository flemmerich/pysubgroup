import unittest
from collections import namedtuple
import numpy as np
import pandas as pd
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase

class TestGeneralisationAwareQf(unittest.TestCase):
    def setUp(self):
        self.qf = ps.CountQF()
        self.ga_qf = ps.GeneralizationAwareQF(self.qf)
        self.prepare_df()
    

    def prepare_df(self):
        A = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        self.A1 = ps.EqualitySelector("columnA", True)
        self.A0 = ps.EqualitySelector("columnA", False)

        B = np.array(["A", "B", "C", "C", "B", "A", "D", "A", "A", "A"])
        self.BA = ps.EqualitySelector("columnB", "A")
        self.BC = ps.EqualitySelector("columnB", "C")
        self.df = pd.DataFrame.from_dict({'columnA': A, 'columnB':B, 'columnC': np.array([[0, 1] for _ in range(5)]).flatten()})


    def test_CountTarget1(self):
        df = self.df
        target = ps.FITarget()
        self.ga_qf.calculate_constant_statistics(df, target)

        ga_score = self.ga_qf.evaluate(ps.Conjunction([self.A1]), target, df)

        A1_score = self.qf.evaluate(ps.Conjunction([self.A1]), target, df)
        zero_score = self.qf.evaluate(ps.Conjunction([]), target, df)

        self.assertEqual(ga_score, A1_score-zero_score)

        ga2_score = self.ga_qf.evaluate(ps.Conjunction([self.A1]), target, df)

        self.assertEqual(ga2_score, ga_score)


    def test_CountTarget2(self):
        df = self.df
        target = ps.FITarget()
        self.ga_qf.calculate_constant_statistics(df, None)

        ga_score = self.ga_qf.evaluate(ps.Conjunction([self.A1, self.BA]), target,  df)

        A_B_score = self.qf.evaluate(ps.Conjunction([self.A1, self.BA]), target, df)
        zero_score = self.qf.evaluate(ps.Conjunction([]), target, df)

        self.assertEqual(ga_score, A_B_score-zero_score)



class TestGeneralisationAware_StandardQf(unittest.TestCase):
    def setUp(self):
        self.df = None
        self.A1 = None
        self.BA = None
        TestGeneralisationAwareQf.prepare_df(self)
        self.ga_qf = ps.GeneralizationAware_StandardQF(0)

    def test_simple(self):
        target = ps.BinaryTarget('columnC', 1)
        qf = ps.StandardQF(0)
        qf.calculate_constant_statistics(self.df, target)

        self.ga_qf.calculate_constant_statistics(self.df, target)

        #print(qf.calculate_statistics(self.A1, self.df))
        #print(qf.calculate_statistics(self.BA, self.df))
        #print(qf.calculate_statistics(ps.Conjunction([self.A1, self.BA]), self.df))
        #print(qf.calculate_statistics(slice(None), self.df))
        ga_stat = self.ga_qf.calculate_statistics(ps.Conjunction([self.A1, self.BA]), target, self.df)

        self.assertEqual(ga_stat.subgroup_stats, ps.SimplePositivesQF.tpl(3, 2))
        self.assertEqual(ga_stat.generalisation_stats, ps.SimplePositivesQF.tpl(5, 3))
        # Ensure cache works properly
        self.assertEqual(ga_stat, self.ga_qf.calculate_statistics(ps.Conjunction([self.A1, self.BA]), target, self.df))

        ga_score = self.ga_qf.evaluate(ps.Conjunction([self.A1, self.BA]), target, self.df)
        ga_score2 = self.ga_qf.evaluate(ps.Conjunction([self.A1, self.BA]), target, self.df)

        self.assertEqual(ga_score, ga_score2)
        self.assertAlmostEqual(ga_score, 0.06666666666666)


class TestAlgorithms(TestAlgorithmsBase, unittest.TestCase):
    # TODO properly specify desired result
    def setUp(self):
        NS_checking = ps.EqualitySelector("checking_status", b"<0")
        NS_foreign_worker = ps.EqualitySelector("foreign_worker", b"yes")
        NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        NS_savings_status = ps.EqualitySelector("savings_status", b"<100")
        NS_job = ps.EqualitySelector("job", b"skilled")
        self.result = [ps.Conjunction([NS_checking, NS_foreign_worker]),
                       ps.Conjunction([NS_checking]),
                       ps.Conjunction([NS_checking, NS_other_parties, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_other_parties]),
                       ps.Conjunction([NS_checking, NS_savings_status, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_savings_status, NS_other_parties, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_job, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_savings_status, NS_other_parties]),
                       ps.Conjunction([NS_checking, NS_job]),
                       ]
        self.qualities = [0.055299999999999995,
                          0.05280000000000001,
                          0.052300000000000006,
                          0.05059999999999999,
                          0.04959999999999999,
                          0.048299999999999996,
                          0.04660000000000001,
                          0.04550000000000001,
                          0.0452,
                          0.044399999999999995]
        data = get_credit_data()
        target = ps.BinaryTarget('class', b'bad')
        searchSpace = ps.create_nominal_selectors(data, ignore=['class'])
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=3, qf=ps.GeneralizationAwareQF(ps.StandardQF(1.0)))

    @unittest.skip
    def test_GA_SimpleDFS(self):
        self.runAlgorithm(ps.SimpleDFS(), "SimpleDFS", self.result, self.qualities, self.task)

    @unittest.skip
    def test_StandardQF_GA_SimpleDFS(self):
        self.task.qf = ps.GeneralizationAware_StandardQF(0.5)
        self.runAlgorithm(ps.SimpleDFS(), "Standard_SimpleDFS", self.result, self.qualities, self.task)
        print(self.task.qf.cache)


if __name__ == '__main__':
    unittest.main()
