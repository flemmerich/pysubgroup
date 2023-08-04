import unittest

import pandas as pd
import pysubgroup as ps



class TestBinaryTarget(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame.from_records([(1,1,0), (1,1,1,), (1,1,0), (1,0,1)], columns=("A", "B", "C"))
        self.selector = ps.EqualitySelector("A", 1)

    def test_init_errors(self):
        with self.assertRaises(ValueError):
            ps.BinaryTarget()
        with self.assertRaises(ValueError):
            ps.BinaryTarget("A", 1, self.selector)

    def test_calculate_statistics_for_cached(self):
        target = ps.BinaryTarget("C",1)
        statistics = target.calculate_statistics(self.selector, self.df)
        statistics2 = target.calculate_statistics(self.selector, self.df, statistics)
        self.assertIs(statistics, statistics2)

    def test_LiftQf(self):
        qf = ps.LiftQF()
        self.assertIsInstance(qf, ps.StandardQF)

    def test_SimpleBinomialQF(self):
        qf = ps.SimpleBinomialQF()
        self.assertIsInstance(qf, ps.StandardQF)

    def test_LWRAccQF(self):
        qf = ps.WRAccQF()
        self.assertIsInstance(qf, ps.StandardQF)


if __name__ == "__main__":
    unittest.main()

