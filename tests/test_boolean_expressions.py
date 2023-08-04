import unittest

import numpy as np
import pandas as pd
import pysubgroup as ps


class TestRelationsMethods(unittest.TestCase):

    def check_dataframe_query(self, selector, result):
        result1 = np.array(result, dtype=bool)
        np.testing.assert_array_equal(selector.covers(self.df), result1)
        np.testing.assert_array_equal(self.df.query(repr(selector)), self.df[result1])

    def test_DNF(self):
        A1 = ps.EqualitySelector("A1", 1)
        A2 = ps.EqualitySelector("A2", 1, "AA")
        B1 = ps.EqualitySelector("B1", 1)
        B2 = ps.EqualitySelector("B2", "1")

        dnf1 = ps.DNF()
        dnf1.append_or([A1, A2])
        dnf2 = ps.DNF([A1, A2])
        self.assertTrue(dnf1 == dnf2)

        dnf3 = ps.DNF(ps.Conjunction([A1, A2]))
        dnf4 = ps.DNF()
        dnf4.append_and([A1, A2])
        dnf5 = ps.DNF()
        dnf5.append_and(A1)
        dnf5.append_and(A2)
        self.assertTrue(dnf3 == dnf4)
        self.assertTrue(dnf4 == dnf5)

        dnf6 = ps.DNF([])
        dnf6.append_and([B1, B2])
        dnf7 = ps.DNF([])
        dnf7.append_and([A1, A2])
        dnf7.append_or(ps.Conjunction([B1, B2]))
        self.df = pd.DataFrame.from_dict({"A1": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0], #pylint: disable=attribute-defined-outside-init
                                          "A2": [0, 1, 1, 1, 2, 2, 2, 0, 0, 0],
                                          "B1": [0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                                          "B2": ["0", "0", "0", "0", "1", "1", "2", "0", "0", "1"]})
        self.check_dataframe_query(dnf1, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.check_dataframe_query(dnf3, [0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        self.check_dataframe_query(dnf6, [0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
        self.check_dataframe_query(dnf7, [0, 1, 1, 0, 1, 1, 0, 0, 0, 1])

    def test_equality_expressions(self):
        A1 = ps.EqualitySelector("A", 1)
        A2 = ps.EqualitySelector("A", 2, "AA")
        B1 = ps.EqualitySelector("B", 1)

        D1 = ps.Disjunction([A1, A2])
        D1_clone = ps.Disjunction([A1, A2])
        self.assertTrue(D1 == D1_clone)
        self.assertTrue(hash(D1) == hash(D1_clone))

        D_all = ps.Disjunction([A1, A2, B1])
        D1_clone.append_or(B1)
        self.assertTrue(D_all == D1_clone)
        self.assertTrue(hash(D_all) == hash(D1_clone))

        C1 = ps.Conjunction([A1, A2])
        C1_clone = ps.Conjunction([A1, A2])
        self.assertTrue(C1 == C1_clone)
        self.assertTrue(hash(C1) == hash(C1_clone))

        C_all = ps.Conjunction([A1, A2, B1])
        C1_clone.append_and(B1)
        self.assertTrue(C_all == C1_clone)
        self.assertTrue(hash(C_all) == hash(C1_clone))

        self.assertFalse(C1 == D1)
        self.assertFalse(hash(C1) == hash(D1))


if __name__ == '__main__':
    unittest.main()
