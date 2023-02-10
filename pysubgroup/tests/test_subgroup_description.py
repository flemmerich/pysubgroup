import tempfile
import os
import unittest

import pickle
import pysubgroup as ps


class TestRelationsMethods(unittest.TestCase):

    def test_pickle(self):

        A1 = ps.EqualitySelector("A", 1)
        with tempfile.TemporaryDirectory() as td:
            f_name = os.path.join(td, 'test.pickle')
            with open(f_name, 'wb') as f:
                pickle.dump(A1, f)

            with open(f_name, 'rb') as f:
                A2 = pickle.load(f)

            assert A1 == A2

from pysubgroup.tests.DataSets import get_credit_data
import numpy as np
import pandas as pd

class TestBasics(unittest.TestCase):
    def test_get_cover_array_and_size(self):
        df = get_credit_data()#
        # len(df)==1000
        self.assertEqual(ps.get_cover_array_and_size(slice(None), data=df)[1], len(df))
        self.assertEqual(ps.get_cover_array_and_size(slice(3,11), data=df)[1], 8)
        self.assertEqual(ps.get_cover_array_and_size(slice(900,1100), data=df)[1], 100)
        with self.assertRaises(ValueError):
            ps.get_cover_array_and_size(slice(900,1100))
        self.assertEqual(ps.get_cover_array_and_size(np.array([1,0,1], dtype=bool))[1],2)
        with self.assertRaises(NotImplementedError):
            ps.get_cover_array_and_size(np.array(["s", "b"]))

    def test_get_size(self):
        df = get_credit_data()#
        # len(df)==1000
        self.assertEqual(ps.get_size(slice(None), data=df), len(df))
        self.assertEqual(ps.get_size(slice(3,11), data=df), 8)
        self.assertEqual(ps.get_size(slice(900,1100), data=df), 100)
        with self.assertRaises(ValueError):
            ps.get_size(slice(900,1100))
        self.assertEqual(ps.get_size(np.array([1,0,1], dtype=bool)),2)
        self.assertEqual(ps.get_size(ps.EqualitySelector("checking_status", b"<0"), data=df),274)
        self.assertEqual(ps.get_size(np.array([1,2,3], dtype=np.int32)),3)
        with self.assertRaises(NotImplementedError):
            ps.get_size(np.array(["s", "b"]))

    def test_EqualitySelector(self):
        with self.assertRaises(TypeError):
            ps.EqualitySelector(None, 1)
        with self.assertRaises(TypeError):
            ps.EqualitySelector("a", None)
        with self.assertRaises(TypeError):
            ps.EqualitySelector(None, None)
        ps.EqualitySelector("A", np.nan)

    def test_NegatedSelector(self):


        df = pd.DataFrame.from_records([(1,1,0), (1,1,1,), (1,1,0), (1,0,1)], columns=("A", "B", "C"))
        A_sel = ps.EqualitySelector("A", 1)
        A_neg = ps.NegatedSelector(A_sel)
        self.assertEqual(A_neg.attribute_name, "A")

        np.testing.assert_array_equal(A_neg.covers(df), [0,0,0,0])

        B_sel = ps.EqualitySelector("B", 1)
        B_neg = ps.NegatedSelector(B_sel)

        np.testing.assert_array_equal(B_neg.covers(df), [0,0,0,1])

    def test_IntervalSelector(self):
        sel = ps.IntervalSelector("A", float("-inf"), float("+inf"))
        self.assertEqual(str(sel), "A = anything")
        self.assertEqual(sel.selectors, (sel,))

        sel = ps.IntervalSelector("A", float("-inf"), 0)
        sel = ps.IntervalSelector("A", 0, float("+inf"))
        sel = ps.IntervalSelector("A", 0, 1)

        




if __name__ == '__main__':
    unittest.main()
