import pysubgroup as ps
import numpy as np
import unittest
import pandas as pd


class TestRelationsMethods(unittest.TestCase):
    def test_NominalSelector_ordering(self):
        A1 = ps.NominalSelector("A", 1)
        A2 = ps.NominalSelector("A", 2, "AA")
        B1 = ps.NominalSelector("B", 1)

        B1_clone = ps.NominalSelector("B", 1)
        self.assertTrue(A1 < B1)
        self.assertTrue(A1 < A2)
        self.assertTrue(A2 < B1)
        self.assertTrue(B1 == B1_clone)
        self.assertTrue(hash(B1) == hash(B1_clone))
        A1.attribute_value = 3
        self.assertTrue(A2 < A1)
        A2.attribute_name = "B"
        self.assertTrue(A1 < A2)

        C1 = ps.NominalSelector("checking_status", b"<0")
        C2 = ps.NominalSelector("checking_status", b"<0")

        self.assertTrue(C1 == C2)
        self.assertTrue(hash(C1) == hash(C2))

        l = [A1, A2, B1]
        self.assertEqual(l.index(A1), 0)
        self.assertEqual(l.index(A2), 1)
        self.assertEqual(l.index(B1), 2)

    def test_NumericSelector_ordering(self):
        S1 = ps.NumericSelector("A", 1.2345, 2.0)
        S1_clone = ps.NumericSelector("A", 1.2345, 2.0)
        S2 = ps.NumericSelector("A", 1.2345, 3.0)
        S3 = ps.NumericSelector("A", 1.2346, 3.0)
        S4 = ps.NumericSelector("B", 1.0, 2.0)
        self.assertTrue(S1 < S2)
        self.assertTrue(S1 < S3)
        self.assertTrue(S2 < S3)
        self.assertTrue(S3 < S4)
        self.assertTrue(S1 == S1_clone)
        self.assertTrue(hash(S1) == hash(S1_clone))

    def test_SubgroupDescription_ordering(self):
        A1 = ps.NominalSelector("A", 1)
        A2 = ps.NominalSelector("A", 2, "AA")
        B1 = ps.NominalSelector("B", 1)

        SGD1 = ps.SubgroupDescription([A1, A2])
        SGD1_clone = ps.SubgroupDescription([A1, A2])
        SGD1_order = ps.SubgroupDescription([A2, A1])

        self.assertTrue(SGD1 == SGD1_clone)
        self.assertTrue(hash(SGD1) == hash(SGD1_clone))
        self.assertTrue(SGD1 == SGD1_order)
        self.assertTrue(hash(SGD1) == hash(SGD1_order))

        SGD2 = ps.SubgroupDescription([A1, A2, B1])
        SGD3 = ps.SubgroupDescription([B1])
        self.assertTrue(SGD1 > SGD2)
        self.assertTrue(SGD2 < SGD3)

    def test_nominal_selector_covers(self):
        A = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        A1 = ps.NominalSelector("A", True)
        A0 = ps.NominalSelector("A", False)

        B = np.array(["A", "B", "C", "C", "B", "A", "D", "A", "A", "A"])
        BA = ps.NominalSelector("B", "A")
        BC = ps.NominalSelector("B", "C")

        C = np.array([np.nan, np.nan, 1.1, 1.1, 2, 2, 2, 2, 2, 2])
        CA = ps.NominalSelector("C", 1.1)
        CNan = ps.NominalSelector("C", np.nan)

        df = pd.DataFrame.from_dict({"A": A, "B": B, "C": C})

        np.testing.assert_array_equal(A1.covers(df), A)
        np.testing.assert_array_equal(A0.covers(df), np.logical_not(A))

        np.testing.assert_array_equal(BA.covers(df), [1, 0, 0, 0, 0, 1, 0, 1, 1, 1])
        np.testing.assert_array_equal(BC.covers(df), [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])

        np.testing.assert_array_equal(CA.covers(df), [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(CNan.covers(df), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
