import unittest
import numpy as np
import pandas as pd
import pysubgroup as ps


class TestRepresentation(unittest.TestCase):
    def setUp(self):
        self.A = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        self.A1 = ps.EqualitySelector("columnA", True)
        self.A0 = ps.EqualitySelector("columnA", False)

        self.B = np.array(["A", "B", "C", "C", "B", "A", "D", "A", "A", "A"])
        self.BA = ps.EqualitySelector("columnB", "A")
        self.BC = ps.EqualitySelector("columnB", "C")

        self.C = np.array([np.nan, np.nan, 1.1, 1.1, 2, 2, 2, 2, 2, 2])
        self.CA = ps.EqualitySelector("columnC", 1.1)
        self.CNan = ps.EqualitySelector("columnC", np.nan)

        self.df = pd.DataFrame.from_dict({"columnA": self.A, "columnB": self.B, "columnC": self.C})


    def test_BitSet(self):
        with ps.BitSetRepresentation(self.df, [self.A1, self.A0, self.BA, self.BC, self.CA, self.CNan]) as representation:
            np.testing.assert_array_equal(self.A1.representation, self.A)  # pylint: disable=no-member
            np.testing.assert_array_equal(self.A0.representation, np.logical_not(self.A))   # pylint: disable=no-member

            np.testing.assert_array_equal(self.BA.representation, [1, 0, 0, 0, 0, 1, 0, 1, 1, 1])   # pylint: disable=no-member
            np.testing.assert_array_equal(self.BC.representation, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])   # pylint: disable=no-member

            np.testing.assert_array_equal(self.CA.representation, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])   # pylint: disable=no-member
            np.testing.assert_array_equal(self.CNan.representation, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])   # pylint: disable=no-member

            np.testing.assert_array_equal(representation.Conjunction([self.BA, self.CNan]).representation, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # pylint: disable=no-member
            np.testing.assert_array_equal(representation.Disjunction([self.BA, self.BC]).representation, [1, 0, 1, 1, 0, 1, 0, 1, 1, 1])  # pylint: disable=no-member


    def test_Set(self):
        with ps.SetRepresentation(self.df, [self.A1, self.A0, self.BA, self.BC, self.CA, self.CNan]) as representation:
            self.assertEqual(self.A1.representation, {2, 3, 6, 7, 8, 9}) # pylint: disable=no-member
            self.assertEqual(self.A0.representation, {0, 1, 4, 5}) # pylint: disable=no-member

            self.assertEqual(self.BA.representation, {0, 5, 7, 8, 9}) # pylint: disable=no-member
            self.assertEqual(self.BC.representation, {2, 3}) # pylint: disable=no-member

            self.assertEqual(self.CA.representation, {2, 3}) # pylint: disable=no-member
            self.assertEqual(self.CNan.representation, {0, 1}) # pylint: disable=no-member

            self.assertEqual(representation.Conjunction([self.BA, self.CNan]).representation, {0})  # pylint: disable=no-member
            self.assertEqual(representation.Conjunction([self.A0, self.CNan]).representation, {0, 1})  # pylint: disable=no-member


    def test_NumpySet(self):
        with ps.NumpySetRepresentation(self.df, [self.A1, self.A0, self.BA, self.BC, self.CA, self.CNan]) as representation:
            np.testing.assert_array_equal(self.A1.representation, [2, 3, 6, 7, 8, 9]) # pylint: disable=no-member
            np.testing.assert_array_equal(self.A0.representation, [0, 1, 4, 5]) # pylint: disable=no-member

            np.testing.assert_array_equal(self.BA.representation, [0, 5, 7, 8, 9]) # pylint: disable=no-member
            np.testing.assert_array_equal(self.BC.representation, [2, 3]) # pylint: disable=no-member

            np.testing.assert_array_equal(self.CA.representation, [2, 3]) # pylint: disable=no-member
            np.testing.assert_array_equal(self.CNan.representation, [0, 1]) # pylint: disable=no-member

            np.testing.assert_array_equal(representation.Conjunction([self.BA, self.CNan]).representation, [0])  # pylint: disable=no-member
            np.testing.assert_array_equal(representation.Conjunction([self.A0, self.CNan]).representation, [0, 1])  # pylint: disable=no-member



if __name__ == '__main__':

    unittest.main()
