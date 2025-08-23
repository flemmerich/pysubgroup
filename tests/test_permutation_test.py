import unittest

import pandas as pd

import pysubgroup as ps


class TestPermutationTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame.from_records(
            [
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                (0, 1, 1),
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 0),
                (1, 0, 0),
            ],
            columns=("A", "class", "prediction"),
        )
        self.subgroup = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0)]
        )
        self.target = ps.SoftClassifierTarget("class", "prediction")

    def test_permutation_test(self):
        qf = ps.ROCAUCQF("class")
        qf.calculate_constant_statistics(self.df, self.target)
        task = ps.SubgroupDiscoveryTask(
            data=self.df,
            target=self.target,
            search_space=ps.create_selectors(
                self.df,
                ignore=[self.target.label_column, self.target.prediction_column],
            ),
            qf=qf,
        )
        p_values, _, _ = ps.permutation_test(
            qf=qf,
            result=ps.SubgroupDiscoveryResult([(0, self.subgroup, None)], task),
            target=self.target,
            data=self.df,
            num_random_samples=10,
        )

        self.assertEqual(p_values[0], 1)


if __name__ == "__main__":
    unittest.main()
