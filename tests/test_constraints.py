import unittest

import pandas as pd

import pysubgroup as ps


class TestGeneralisationAwareQf(unittest.TestCase):
    def test_is_satisfied_dict(self):
        constr = ps.MinSupportConstraint(10)
        self.assertTrue(constr.is_satisfied(None, {"size_sg": 12}, None))
        self.assertFalse(constr.is_satisfied(None, {"size_sg": 9}, None))


class TestContainsValueConstraint(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame.from_records(
            [
                (1, 1),
                (0.5, 1),
                (0.5, 0),
                (0, 0),
            ],
            columns=("A", "B"),
        )
        self.subgroup_pos_only = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 1)]
        )
        self.subgroup_both = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0.5)]
        )
        self.subgroup_neg_only = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0)]
        )

    def test_is_monotone(self):
        constr = ps.ContainsValueConstraint(None, None)
        self.assertTrue(constr.is_monotone)

    def test_is_satisfied(self):
        constr = ps.ContainsValueConstraint("B", 1)
        self.assertTrue(constr.is_satisfied(self.subgroup_pos_only, None, self.df))
        self.assertTrue(constr.is_satisfied(self.subgroup_both, None, self.df))
        self.assertFalse(constr.is_satisfied(self.subgroup_neg_only, None, self.df))


class TestMinUniqueValuesConstraint(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame.from_records(
            [
                (1, 1),
                (0.5, 1),
                (0.5, 0),
                (0, 0),
            ],
            columns=("A", "B"),
        )
        self.subgroup_pos_only = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 1)]
        )
        self.subgroup_both = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0.5)]
        )
        self.subgroup_neg_only = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0)]
        )

    def test_is_monotone(self):
        constr = ps.MinUniqueValuesConstraint(None, None)
        self.assertTrue(constr.is_monotone)

    def test_is_satisfied(self):
        constr = ps.MinUniqueValuesConstraint("B", 2)
        self.assertFalse(constr.is_satisfied(self.subgroup_pos_only, None, self.df))
        self.assertTrue(constr.is_satisfied(self.subgroup_both, None, self.df))
        self.assertFalse(constr.is_satisfied(self.subgroup_neg_only, None, self.df))


if __name__ == "__main__":
    unittest.main()
