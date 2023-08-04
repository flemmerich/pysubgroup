import tempfile
import os
import unittest

import pickle
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data
import numpy as np
import pandas as pd


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
        sel = ps.EqualitySelector("A", np.nan)

        self.assertEqual(sel.selectors, (sel,))

    def test_NegatedSelector(self):


        df = pd.DataFrame.from_records([(1,1,0), (1,1,1,), (1,1,0), (1,0,1)], columns=("A", "B", "C"))
        A_sel = ps.EqualitySelector("A", 1)

        A_neg = ps.NegatedSelector(A_sel)
        self.assertEqual(A_neg.attribute_name, "A")
        self.assertEqual(A_neg.selectors, (A_neg,))
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

        sel = ps.IntervalSelector("age", 60, float("+inf"), "the_elderly")
        self.assertEqual(str(sel), "the_elderly")

        with self.assertRaises(AssertionError):
            ps.IntervalSelector("age", 70, 60)

    def test_remove_target_attributes(self):
        selectors = [ps.IntervalSelector("A", 1, 2), ps.EqualitySelector("B", 1), ps.EqualitySelector("C", 1)]

        fi_target=ps.FITarget()
        fi_sel = ps.remove_target_attributes(selectors, fi_target)
        self.assertEqual(selectors[:], fi_sel)

        num_target=ps.NumericTarget("C")
        num_sel = ps.remove_target_attributes(selectors, num_target)
        self.assertEqual(selectors[:-1], num_sel)

        bin_target=ps.BinaryTarget("A", -10)
        bin_sel = ps.remove_target_attributes(selectors, bin_target)
        self.assertEqual(selectors[1:], bin_sel)

    def test_EqualitySelector_from_str(self):
        selector1 = ps.EqualitySelector.from_str("A=='hello'")
        selector2 = ps.EqualitySelector("A", "hello")
        self.assertEqual(selector1, selector2)
        self.assertIs(selector1, selector2)


    def test_IntervalSelector_from_str(self):
        def test_str(s, attribute_name, lb, ub):
            selector = ps.IntervalSelector.from_str(s)
            _selector = ps.IntervalSelector(attribute_name, lb, ub)
            self.assertEqual(selector, _selector)
            self.assertIs(selector, _selector)

        test_str("A = anything", "A", float("-inf"), float("+inf"))
        test_str("A >= 10", "A", 10, float("+inf"))
        test_str("A >= 10.0", "A", 10.0, float("+inf"))
        test_str("A < 12.3", "A", float("-inf"), 12.3)
        test_str("A < 12", "A", float("-inf"), 12)
        test_str("A : [12:34[", "A", 12, 34)
        test_str("A : [12.3:45.6[", "A", 12.3, 45.6)

        with self.assertRaises(ValueError):
            ps.IntervalSelector.from_str("A == B")

    def test_create_nominal_selectors_for_attribut(self):
        df = pd.DataFrame({"A" : np.array([1,1,0], dtype=bool)})
        selectors = ps.create_nominal_selectors_for_attribute(df, "A")
        self.assertTrue(all(sel.is_bool for sel in selectors))

    def test_Conjunction(self):
        conj = ps.Conjunction([ps.EqualitySelector.from_str("A==0"), ps.EqualitySelector.from_str("B==1")])
        with self.assertRaises(RuntimeError):
            conj.append_or(ps.EqualitySelector.from_str("C==12"))
        with self.assertRaises(RuntimeError):
            conj.pop_or()
        sel = conj.pop_and()
        self.assertIs(sel, ps.EqualitySelector("B", 1))
        self.assertEqual(conj, ps.Conjunction([ps.EqualitySelector("A", 0)]))

    def test_Disjunction(self):
        from copy import copy # pylint: disable=import-outside-toplevel
        dis = ps.Disjunction([ps.EqualitySelector.from_str("A==0"), ps.EqualitySelector.from_str("B==1")])
        self.assertEqual(len(dis), 2)
        dis2 = copy(dis)
        self.assertEqual(dis, dis2)
        self.assertFalse(dis is dis2)
        with self.assertRaises(RuntimeError):
            dis.append_and(ps.EqualitySelector.from_str("C==12"))

        df = pd.DataFrame({"A" : np.array([1,1,0], dtype=bool)})
        dis = ps.Disjunction([])
        np.testing.assert_array_equal(dis.covers(df), [False, False, False])

    def test_create_numeric_selectors_for_attribute(self):
        from pysubgroup.tests.t_utils import interval_selectors_from_str # pylint:disable=import-outside-toplevel
        df = pd.DataFrame({"A" : np.array([1,1,0,3,3,4,5], dtype=int)})
        selectors = ps.create_numeric_selectors_for_attribute(df, "A", nbins=4, intervals_only=False)
        result_selectors = interval_selectors_from_str("""A>=1
            A<1
            A>=3
            A<3
            A>=4
            A<4""")
        self.assertEqual(selectors, result_selectors)



if __name__ == '__main__':
    unittest.main()
