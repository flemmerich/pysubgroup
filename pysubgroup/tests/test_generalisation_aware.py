import unittest
from collections import namedtuple
import numpy as np
import pandas as pd
import pysubgroup as ps

task_dummy = namedtuple('task_dummy', ['data'])

class TestGeneralisationAwareQf(unittest.TestCase):
    def setUp(self):
        self.qf = ps.CountQF()
        self.ga_qf = ps.GeneralizationAwareQF(self.qf)
        A = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        self.A1 = ps.EqualitySelector("columnA", True)
        self.A0 = ps.EqualitySelector("columnA", False)

        B = np.array(["A", "B", "C", "C", "B", "A", "D", "A", "A", "A"])
        self.BA = ps.EqualitySelector("columnB", "A")
        self.BC = ps.EqualitySelector("columnB", "C")
        self.df = pd.DataFrame.from_dict({'columnA': A, 'columnB':B, 'columnC': np.array([[0, 1] for _ in range(5)]).flatten()})

    def test_CountTarget1(self):
        df = self.df
        self.ga_qf.calculate_constant_statistics(task_dummy(df))

        ga_score = self.ga_qf.evaluate(ps.Conjunction([self.A1]), df)

        A1_score = self.qf.evaluate(ps.Conjunction([self.A1]), df)
        zero_score = self.qf.evaluate(ps.Conjunction([]), df)

        self.assertEqual(ga_score, A1_score-zero_score)

    def test_CountTarget2(self):
        df = self.df
        self.ga_qf.calculate_constant_statistics(task_dummy(df))

        ga_score = self.ga_qf.evaluate(ps.Conjunction([self.A1, self.BA]), df)

        A_B_score = self.qf.evaluate(ps.Conjunction([self.A1, self.BA]), df)
        zero_score = self.qf.evaluate(ps.Conjunction([]), df)

        self.assertEqual(ga_score, A_B_score-zero_score)



if __name__ == '__main__':
    unittest.main()
