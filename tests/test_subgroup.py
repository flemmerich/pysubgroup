import unittest
import numpy as np
import pandas as pd
import pysubgroup as ps


class TestRelationsMethods(unittest.TestCase):
    def test_EqualitySelector_ordering(self):
        A1 = ps.EqualitySelector("A", 1)
        A1_clone = ps.EqualitySelector("A", 1)
        A2 = ps.EqualitySelector("A", 2, "AA")
        B1 = ps.EqualitySelector("B", 1)

        self.assertTrue(A1_clone is A1)

        B1_clone = ps.EqualitySelector("B", 1)
        self.assertTrue(A1 < B1)
        self.assertTrue(A1 < A2)
        self.assertTrue(A2 < B1)
        self.assertTrue(B1 == B1_clone)
        self.assertTrue(hash(B1) == hash(B1_clone))

        C1 = ps.EqualitySelector("checking_status", b"<0")
        C2 = ps.EqualitySelector("checking_status", b"<0")

        self.assertTrue(C1 == C2)
        self.assertTrue(hash(C1) == hash(C2))

        l = [A1, A2, B1]
        self.assertEqual(l.index(A1), 0)
        self.assertEqual(l.index(A2), 1)
        self.assertEqual(l.index(B1), 2)

    def test_IntervalSelector_ordering(self):
        S1 = ps.IntervalSelector("A", 1.2345, 2.0)
        S1_clone = ps.IntervalSelector("A", 1.2345, 2.0)
        S1_clone = ps.IntervalSelector("A", 1.2345, 2.0)
        S2 = ps.IntervalSelector("A", 1.2345, 3.0)
        S3 = ps.IntervalSelector("A", 1.2346, 3.0)
        S4 = ps.IntervalSelector("B", 1.0, 2.0)
        self.assertTrue(S1 < S2)
        self.assertTrue(S1 < S3)
        self.assertTrue(S2 < S3)
        self.assertTrue(S3 < S4)
        self.assertTrue(S1 == S1_clone)
        self.assertTrue(hash(S1) == hash(S1_clone))
        self.assertTrue(S1 is S1_clone)

    def test_Conjunction_ordering(self):
        self.assert_class_ordering(ps.Conjunction)

    def assert_class_ordering(self, cls):
        A1 = ps.EqualitySelector("A", 1)
        A2 = ps.EqualitySelector("A", 2, "AA")
        B1 = ps.EqualitySelector("B", 1)

        SGD1 = cls([A1, A2])
        SGD1_clone = cls([A1, A2])
        SGD1_order = cls([A2, A1])

        self.assertTrue(SGD1 == SGD1_clone)
        self.assertTrue(hash(SGD1) == hash(SGD1_clone))
        self.assertTrue(SGD1 == SGD1_order)
        self.assertTrue(hash(SGD1) == hash(SGD1_order))

        SGD2 = cls([A1, A2, B1])
        SGD3 = cls([B1])
        self.assertTrue(SGD1 > SGD2)
        self.assertTrue(SGD2 < SGD3)

    def test_Disjunction_ordering(self):
        self.assert_class_ordering(ps.Disjunction)

    def test_nominal_selector_covers(self):
        A = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        A1 = ps.EqualitySelector("columnA", True)
        A0 = ps.EqualitySelector("columnA", False)

        B = np.array(["A", "B", "C", "C", "B", "A", "D", "A", "A", "A"])
        BA = ps.EqualitySelector("columnB", "A")
        BC = ps.EqualitySelector("columnB", "C")

        C = np.array([np.nan, np.nan, 1.1, 1.1, 2, 2, 2, 2, 2, 2])
        CA = ps.EqualitySelector("columnC", 1.1)
        CNan = ps.EqualitySelector("columnC", np.nan)

        df = pd.DataFrame.from_dict({"columnA": A, "columnB": B, "columnC": C})

        np.testing.assert_array_equal(A1.covers(df), A)
        np.testing.assert_array_equal(A0.covers(df), np.logical_not(A))

        np.testing.assert_array_equal(BA.covers(df), [1, 0, 0, 0, 0, 1, 0, 1, 1, 1])
        np.testing.assert_array_equal(BC.covers(df), [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])

        np.testing.assert_array_equal(CA.covers(df), [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(CNan.covers(df), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_str_representations(self):
        A = ps.EqualitySelector("A", 1)
        self.assertEqual(str(A), "A==1")
        self.assertEqual(repr(A), "A==1")

        B = ps.EqualitySelector("BB", 2)
        self.assertEqual(str(B), "BB==2")
        self.assertEqual(repr(B), "BB==2")

        C = ps.EqualitySelector("CCC", True)
        self.assertEqual(str(C), "CCC==True")
        self.assertEqual(repr(C), "CCC==True")

        NegC = ps.NegatedSelector(ps.EqualitySelector("CCC", True))
        self.assertEqual(str(NegC), "NOT CCC==True")
        self.assertEqual(repr(NegC), "(not CCC==True)")

        I1 = ps.IntervalSelector("test", 10, 15)
        self.assertEqual(str(I1), "test: [10:15[")
        self.assertEqual(repr(I1), "test: [10:15[")

        I2 = ps.IntervalSelector("test2", np.sqrt(2), np.sqrt(3))
        self.assertEqual(str(I2), "test2: [1.41:1.73[")
        self.assertEqual(repr(I2), "test2: [1.4142135623730951:1.7320508075688772[")
        #self.assertEqual(repr(NegC), "(not CCC==True)")

    def test_create_selectors_with_nan(self):
        df = pd.DataFrame.from_dict({'A' : np.array([np.nan, np.nan, np.nan]), 'B' : np.array([10, np.nan, np.nan])})
        result = ps.create_selectors(df)
        A_null = ps.EqualitySelector('A', np.nan)
        B_null = ps.EqualitySelector('B', np.nan)
        B_10 = ps.EqualitySelector('B', 10.)
        assert A_null in result
        assert B_null in result
        assert B_10 in result


if __name__ == '__main__':
    unittest.main()
