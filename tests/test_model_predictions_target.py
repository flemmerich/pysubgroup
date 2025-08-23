import unittest

import numpy as np
import pandas as pd

import pysubgroup as ps


class TestModelPredictionsTarget(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame.from_records(
            [
                (1, 1, 1),
                (1, 1, 0.6),
                (1, 0, 0.4),
                (1, 0, 0),
                (0.5, 1, 1),
                (0.5, 1, 0.5),
                (0.5, 0, 0.5),
                (0.5, 0, 0),
                (0.2, 1, 1),
                (0.2, 1, 1),
                (0.1, 0, 0),
                (0.1, 0, 0),
                (0, 1, 0),
                (0, 1, 0.4),
                (0, 0, 0.6),
                (0, 0, 1),
            ],
            columns=("A", "class", "prediction"),
        )
        self.subgroup_good = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 1)]
        )
        self.subgroup_tied = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0.5)]
        )
        self.subgroup_bad = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0)]
        )
        self.subgroup_only_pos = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0.2)]
        )
        self.subgroup_only_neg = ps.create_subgroup_with_representation(
            self.df, [ps.EqualitySelector("A", 0.1)]
        )
        self.target = ps.SoftClassifierTarget("class", "prediction")

    def test_get_target_columns(self):
        columns = self.target.get_target_columns(self.df)

        self.assertTrue(columns.equals(self.df.loc[:, ["class", "prediction"]]))

    def test_ARLQF_constraint_only_neg(self):
        qf = ps.ARLQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_only_neg, self.target, self.df)

        self.assertEqual(-np.inf, q)

    def test_ARLQF_constraint_only_pos(self):
        qf = ps.ARLQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_only_pos, self.target, self.df)

        self.assertEqual(0 - qf.dataset_quality, q)

    def test_ARLQF_good(self):
        qf = ps.ARLQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_good, self.target, self.df)

        self.assertGreater(0, q)

    def test_ARLQF_optimistic_estimate_good(self):
        qf = ps.ARLQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_good, self.target, self.df)

        self.assertEqual(
            ps.average_ranking_loss([0, 0, 1], [0, 0.4, 1]) - qf.dataset_quality, oe
        )

    def test_ARLQF_good_weights(self):
        qf = ps.ARLQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_good, self.target, self.df)

        qfw = ps.ARLQF(self.target.label_column, 1, 1, 1)
        qfw.calculate_constant_statistics(self.df, self.target)
        qw = qfw.evaluate(self.subgroup_good, self.target, self.df)

        self.assertEqual(q * 1 * 4, qw)

    def test_ARLQF_bad(self):
        qf = ps.ARLQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_bad, self.target, self.df)

        self.assertGreater(q, 0)

    def test_ARLQF_optimistic_estimate_bad(self):
        qf = ps.ARLQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_bad, self.target, self.df)

        self.assertEqual(
            ps.average_ranking_loss([1, 0, 0], [0, 0.6, 1]) - qf.dataset_quality, oe
        )

    def test_ARLQF_bad_weights(self):
        qf = ps.ARLQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_bad, self.target, self.df)

        qfw = ps.ARLQF(self.target.label_column, 1, 1, 1)
        qfw.calculate_constant_statistics(self.df, self.target)
        qw = qfw.evaluate(self.subgroup_bad, self.target, self.df)

        self.assertEqual(q * 1 * 4, qw)

    def test_ROCAUCQF_constraint_only_neg(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_only_neg, self.target, self.df)

        self.assertEqual(-np.inf, q)

    def test_ROCAUCQF_constraint_only_pos(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_only_pos, self.target, self.df)

        self.assertEqual(-np.inf, q)

    def test_ROCAUCQF_good(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_good, self.target, self.df)

        self.assertGreater(0, q)

    def test_ROCAUCQF_optimistic_estimate_good(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_good, self.target, self.df)

        self.assertEqual(qf.dataset_quality - 1, oe)

    def test_ROCAUCQF_optimistic_estimate_good_weights(self):
        qf = ps.ROCAUCQF(self.target.label_column, 0.6, 0.5)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_good, self.target, self.df)

        self.assertEqual((qf.dataset_quality - 1) * (2 * 2) ** 0.5, oe)

    def test_ROCAUCQF_good_weights(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_good, self.target, self.df)

        qfw = ps.ROCAUCQF(self.target.label_column, 1, 1)
        qfw.calculate_constant_statistics(self.df, self.target)
        qw = qfw.evaluate(self.subgroup_good, self.target, self.df)

        self.assertEqual(q * 1 * 4, qw)

    def test_ROCAUCQF_optimistic_estimate_tied(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_tied, self.target, self.df)

        self.assertEqual(qf.dataset_quality - 0.5, oe)

    def test_ROCAUCQF_bad(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_bad, self.target, self.df)

        self.assertGreater(q, 0)

    def test_ROCAUCQF_optimistic_estimate_bad(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_bad, self.target, self.df)

        self.assertEqual(qf.dataset_quality - 0, oe)

    def test_ROCAUCQF_optimistic_estimate_bad_weights(self):
        qf = ps.ROCAUCQF(self.target.label_column, 0.6, 0.5)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_bad, self.target, self.df)

        self.assertEqual((qf.dataset_quality - 0) * (2 * 2) ** 0.5, oe)

    def test_ROCAUCQF_optimistic_estimate_bad_weights_wrong(self):
        # size weight cannot be greater than class balance weight for optimistic estimate
        qf = ps.ROCAUCQF(self.target.label_column, 0.4, 0.5)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_bad, self.target, self.df)

        self.assertEqual(np.inf, oe)

    def test_ROCAUCQF_bad_weights(self):
        qf = ps.ROCAUCQF(self.target.label_column)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_bad, self.target, self.df)

        qfw = ps.ROCAUCQF(self.target.label_column, 1, 1)
        qfw.calculate_constant_statistics(self.df, self.target)
        qw = qfw.evaluate(self.subgroup_bad, self.target, self.df)

        self.assertEqual(q * 1 * 4, qw)

    def test_PRAUCQF_constraint_only_neg(self):
        qf = ps.PRAUCQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_only_neg, self.target, self.df)

        self.assertEqual(-np.inf, q)

    def test_PRAUCQF_constraint_only_pos(self):
        qf = ps.PRAUCQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_only_pos, self.target, self.df)

        self.assertEqual(qf.dataset_quality - 1, q)

    def test_PRAUCQF_good(self):
        qf = ps.PRAUCQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_good, self.target, self.df)

        self.assertGreater(0, q)

    def test_PRAUCQF_optimistic_estimate_good(self):
        qf = ps.PRAUCQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_good, self.target, self.df)

        self.assertEqual(
            qf.dataset_quality - ps.pr_auc_score([0, 0, 1], [0, 0.4, 0.6]), oe
        )

    def test_PRAUCQF_good_weights(self):
        qf = ps.PRAUCQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_good, self.target, self.df)

        qfw = ps.PRAUCQF(self.target.label_column, 1, 1, 1)
        qfw.calculate_constant_statistics(self.df, self.target)
        qw = qfw.evaluate(self.subgroup_good, self.target, self.df)

        self.assertEqual(q * 1 * 4, qw)

    def test_PRAUCQF_bad(self):
        qf = ps.PRAUCQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_bad, self.target, self.df)

        self.assertGreater(q, 0)

    def test_PRAUCQF_optimistic_estimate_bad(self):
        qf = ps.PRAUCQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        oe = qf.optimistic_estimate(self.subgroup_bad, self.target, self.df)

        self.assertEqual(
            qf.dataset_quality - ps.pr_auc_score([1, 0, 0], [0, 0.6, 1]), oe
        )

    def test_PRAUCQF_bad_weights(self):
        qf = ps.PRAUCQF(self.target.label_column, 1)
        qf.calculate_constant_statistics(self.df, self.target)
        q = qf.evaluate(self.subgroup_bad, self.target, self.df)

        qfw = ps.PRAUCQF(self.target.label_column, 1, 1, 1)
        qfw.calculate_constant_statistics(self.df, self.target)
        qw = qfw.evaluate(self.subgroup_bad, self.target, self.df)

        self.assertEqual(q * 1 * 4, qw)


if __name__ == "__main__":
    unittest.main()
