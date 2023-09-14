# flake8: noqa: E501

import unittest
from copy import copy

from algorithms_testing import TestAlgorithmsBase
from t_utils import conjunctions_from_str

import pysubgroup as ps
from pysubgroup.datasets import get_credit_data


class TestStandardQFNumeric(unittest.TestCase):
    def test_constructor(self):
        ps.StandardQFNumeric(0)
        ps.StandardQFNumeric(1.0)
        ps.StandardQFNumeric(0, invert=True)
        ps.StandardQFNumeric(0, invert=False)

        with self.assertRaises(ValueError):
            ps.StandardQFNumeric("test")

        ps.StandardQFNumeric(0, estimator="sum")
        ps.StandardQFNumeric(0, estimator="average")
        ps.StandardQFNumeric(0, estimator="order")

        with self.assertRaises(ValueError):
            ps.StandardQFNumeric(0, estimator="bla")


class TestStandardQFNumericMedian(TestAlgorithmsBase, unittest.TestCase):
    def test_constructor(self):
        ps.StandardQFNumeric(0, centroid="median")
        ps.StandardQFNumeric(1.0, centroid="median")
        ps.StandardQFNumeric(0, invert=True, centroid="median")
        ps.StandardQFNumeric(0, invert=False, centroid="median")

        ps.StandardQFNumeric(0, centroid="sorted_median")

        with self.assertRaises(ValueError):
            ps.StandardQFNumeric("test", centroid="median")

        with self.assertRaises(ValueError):
            ps.StandardQFNumeric(0, centroid="bla")

        ps.StandardQFNumeric(0, estimator="max", centroid="median")
        # ps.StandardQFNumeric(0, estimator="order", centroid='median')

        with self.assertRaises(AssertionError):
            ps.StandardQFNumeric(0, estimator="sum", centroid="median")
        with self.assertRaises(AssertionError):
            ps.StandardQFNumeric(0, estimator="average", centroid="median")

        with self.assertRaises(AssertionError):
            ps.StandardQFNumeric(0, estimator="bla", centroid="median")

    def setUp(self):
        l = conjunctions_from_str(
            """316646.0    job=='b'high qualif/self emp/mgmt''
   310615.0    foreign_worker=='b'yes'' AND job=='b'high qualif/self emp/mgmt''
   297844.5    foreign_worker=='b'yes'' AND own_telephone=='b'yes'' AND property_magnitude=='b'no known property''
   297844.5    own_telephone=='b'yes'' AND property_magnitude=='b'no known property''
   288480.5    job=='b'high qualif/self emp/mgmt'' AND own_telephone=='b'yes''
   283002.0    own_telephone=='b'yes''
   282217.5    class=='b'bad'' AND own_telephone=='b'yes''
   282000.0    other_parties=='b'none'' AND own_telephone=='b'yes''
   276312.5    foreign_worker=='b'yes'' AND job=='b'high qualif/self emp/mgmt'' AND own_telephone=='b'yes''
   275467.5    foreign_worker=='b'yes'' AND other_parties=='b'none'' AND own_telephone=='b'yes'' """
        )
        self.result = [conjunction for quality, conjunction in l]
        self.qualities = [quality for quality, conjunction in l]

        data = get_credit_data()
        target = ps.NumericTarget("credit_amount")
        searchSpace_Nominal = ps.create_nominal_selectors(
            data, ignore=["credit_amount"]
        )
        searchSpace_Numeric = (
            []
        )  # ps.create_numeric_selectors(data, ignore=['credit_amount'], nbins=10)
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(
            data,
            target,
            searchSpace,
            result_set_size=10,
            depth=5,
            qf=ps.StandardQFNumeric(1, False, estimator="max", centroid="median"),
        )

    def Atest_Apriori_no_numba(self):
        algorithm = ps.Apriori(use_numba=False)
        algorithm.use_vectorization = False
        self.runAlgorithm(
            algorithm,
            "Median Apriori, use_numba=False",
            self.result,
            self.qualities,
            self.task,
        )

    def test_Apriori_no_numba_sorted_median(self):
        algorithm = ps.Apriori(use_numba=False)
        algorithm.use_vectorization = False

        qf = ps.StandardQFNumeric(1, False, estimator="max", centroid="sorted_median")
        df = self.task.data
        df = df.sort_values(by=self.task.target.target_variable)
        task = ps.SubgroupDiscoveryTask(
            df,
            self.task.target,
            self.task.search_space,
            qf,
            result_set_size=10,
            depth=5,
        )
        self.runAlgorithm(
            algorithm,
            "Quick Median Apriori, use_numba=False",
            self.result,
            self.qualities,
            task,
        )


#    def test_SimpleDFS(self): # terribly slow
#        self.runAlgorithm(
#            ps.SimpleDFS(), "Median SimpleDFS", self.result, self.qualities, self.task
#        )


class TestNumericTarget(unittest.TestCase):
    def setUp(self):
        records = [(1, 20), (1, 10), (1, 10), (1, 0), (0, -20), (0, -10), (0, -10)]
        self.df = pd.DataFrame.from_records(records, columns=("A", "target"))

    def test_get_base_statistics(self):
        sg = ps.EqualitySelector("A", 1)
        target = ps.NumericTarget("target")
        self.assertEqual(target.get_base_statistics(sg, self.df), (7, 0.0, 4, 10.0))

    def test_calculate_statistics(self):
        target = ps.NumericTarget("target")
        sg = ps.EqualitySelector("A", 1)
        statistics = target.calculate_statistics(sg, self.df, None)
        statistics2 = target.calculate_statistics(
            sg, self.df, cached_statistics=statistics
        )
        self.assertIs(statistics, statistics2)
        del statistics["size_sg"]
        statistics3 = target.calculate_statistics(
            sg, self.df, cached_statistics=statistics
        )


class TestStandardQFNumericTscore(unittest.TestCase):
    def test_basics(self):
        epsilon = 0.001
        records = [
            (1, 20),
            (1, 10),
            (1, 10 + epsilon),
            (1, 0),
            (0, -20),
            (0, -10),
            (0, -10 - epsilon),
        ]
        df = pd.DataFrame.from_records(records, columns=("A", "target"))
        target = ps.NumericTarget("target")
        sg = ps.EqualitySelector("A", 1)
        qf = ps.StandardQFNumericTscore()
        qf.evaluate(sg, target, df, None)
        qf.optimistic_estimate(sg, target, df, None)

        # test empty subgroup
        sg0 = ps.EqualitySelector("A", 2)
        qf.evaluate(sg0, target, df, None)
        qf.optimistic_estimate(sg0, target, df, None)


class TestAlgorithmsWithNumericTarget(TestAlgorithmsBase, unittest.TestCase):
    def setUp(self):
        NS_telephone = ps.EqualitySelector("own_telephone", b"yes")
        NS_foreign_worker = ps.EqualitySelector("foreign_worker", b"yes")
        NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        NS_personal = ps.EqualitySelector("personal_status", b"male single")
        NS_job = ps.EqualitySelector("job", b"high qualif/self emp/mgmt")
        NS_class = ps.EqualitySelector("class", b"bad")

        o = [
            [NS_telephone],
            [NS_foreign_worker, NS_telephone],
            [NS_other_parties, NS_telephone],
            [NS_foreign_worker, NS_telephone, NS_personal],
            [NS_telephone, NS_personal],
            [NS_foreign_worker, NS_other_parties, NS_telephone],
            [NS_job],
            [NS_class, NS_telephone],
            [NS_foreign_worker, NS_job],
            [NS_foreign_worker, NS_other_parties, NS_telephone, NS_personal],
        ]
        self.result = list(map(ps.Conjunction, o))
        self.qualities = [
            383476.7679999999,
            361710.05800000014,
            345352.9920000001,
            338205.08,
            336857.8220000001,
            323586.28200000006,
            320306.81600000005,
            300963.84599999996,
            299447.332,
            297422.98200000013,
        ]

        data = get_credit_data()
        target = ps.NumericTarget("credit_amount")
        searchSpace_Nominal = ps.create_nominal_selectors(
            data, ignore=["credit_amount"]
        )
        searchSpace_Numeric = (
            []
        )  # ps.create_numeric_selectors(data, ignore=['credit_amount'], nbins=10)
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(
            data,
            target,
            searchSpace,
            result_set_size=10,
            depth=5,
            qf=ps.StandardQFNumeric(1, False, "sum"),
        )

    def test_SimpleDFS(self):
        self.runAlgorithm(
            ps.SimpleDFS(), "SimpleDFS", self.result, self.qualities, self.task
        )

    def test_DFS_average(self):
        self.task.qf = ps.StandardQFNumeric(self.task.qf.a, False, "average")
        self.runAlgorithm(
            ps.DFS(ps.BitSetRepresentation),
            "DFS average",
            self.result,
            self.qualities,
            self.task,
        )

    def test_DFS_order_with_numba(self):
        try:
            import numba
        except ImportError:
            self.skipTest("No numba installed")
        self.task.qf = ps.StandardQFNumeric(self.task.qf.a, False, "order")
        self.runAlgorithm(
            ps.DFS(ps.BitSetRepresentation),
            "DFS order with numba",
            self.result,
            self.qualities,
            self.task,
        )

    def test_DFS_order_no_numba(self):
        self.task.qf = ps.StandardQFNumeric(self.task.qf.a, False, "order")
        self.task.qf.estimator.use_numba = False
        self.runAlgorithm(
            ps.DFS(ps.BitSetRepresentation),
            "DFS order no numba",
            self.result,
            self.qualities,
            self.task,
        )

    def test_DFS_sum(self):
        self.task.qf = ps.StandardQFNumeric(self.task.qf.a, False, "sum")
        self.runAlgorithm(
            ps.DFS(ps.BitSetRepresentation),
            "DFS sum",
            self.result,
            self.qualities,
            self.task,
        )

    def test_BeamSearch_sum(self):
        self.task.qf = ps.StandardQFNumeric(self.task.qf.a, False, "sum")
        self.runAlgorithm(
            ps.BeamSearch(beam_width=self.task.result_set_size),
            "BeamSearch sum",
            self.result,
            self.qualities,
            self.task,
        )

    def test_BeamSearch_average(self):
        self.task.qf = ps.StandardQFNumeric(self.task.qf.a, False, "average")
        self.runAlgorithm(
            ps.BeamSearch(beam_width=self.task.result_set_size),
            "BeamSearch average",
            self.result,
            self.qualities,
            self.task,
        )

    def test_BeamSearch_order(self):
        self.task.qf = ps.StandardQFNumeric(self.task.qf.a, False, "order")
        self.runAlgorithm(
            ps.BeamSearch(beam_width=self.task.result_set_size),
            "BeamSearch order",
            self.result,
            self.qualities,
            self.task,
        )

    def test_Apriori_no_numba(self):
        self.runAlgorithm(
            ps.Apriori(use_numba=False),
            "Apriori use_numba=False",
            self.result,
            self.qualities,
            self.task,
        )

    def test_Apriori_with_numba(self):
        self.runAlgorithm(
            ps.Apriori(use_numba=True),
            "Apriori use_numba=True",
            self.result,
            self.qualities,
            self.task,
        )

    def test_DFSNumeric(self):
        self.runAlgorithm(
            ps.DFSNumeric(), "DFS_numeric", self.result, self.qualities, self.task
        )
        # print('   Number of call to qf:', algo.num_calls)

    # def test_SimpleSearch(self):
    #   self.runAlgorithm(ps.SimpleSearch(), "SimpleSearch", self.result, self.qualities, self.task)


import pandas as pd


class TestNumericEstimators(unittest.TestCase):
    def test_estimator1(self):
        records = [(1, 100), (1, 75), (1, 53), (1, 12), (0, 11), (0, 49)]
        df = pd.DataFrame.from_records(records, columns=["A", "Target"])
        T = ps.NumericTarget("Target")
        sel = ps.EqualitySelector("A", 1)

        qf = ps.StandardQFNumeric(a=1)
        self.assertEqual(qf.optimistic_estimate(sel, T, df), 78)

        for a in [1, 0.5, 0]:
            qf = ps.StandardQFNumeric(a=a, estimator="max")
            self.assertEqual(qf.optimistic_estimate(sel, T, df), 3**a * 50)


if __name__ == "__main__":
    unittest.main()


# 383476.7679999999:      own_telephone=='b'yes''
# 361710.05800000014:     foreign_worker=='b'yes'' and own_telephone=='b'yes''
# 345352.9920000001:      other_parties=='b'none'' and own_telephone=='b'yes''
# 338205.08:      foreign_worker=='b'yes'' and own_telephone=='b'yes'' and personal_status=='b'male single''
# 336857.8220000001:      own_telephone=='b'yes'' and personal_status=='b'male single''
# 323586.28200000006:     foreign_worker=='b'yes'' and other_parties=='b'none'' and own_telephone=='b'yes''
# 320306.81600000005:     job=='b'high qualif/self emp/mgmt''
# 300963.84599999996:     class=='b'bad'' and own_telephone=='b'yes''
# 299447.332:     foreign_worker=='b'yes'' and job=='b'high qualif/self emp/mgmt''
# 297422.98200000013:     foreign_worker=='b'yes'' and other_parties=='b'none'' and own_telephone=='b'yes'' and personal_status=='b'male single''

# 639577.0460000001:   duration>=30.0
# 624424.3040000001:   duration>=30.0 AND foreign_worker=='b'yes''
# 579219.206:  duration>=30.0 AND other_parties=='b'none''
# 564066.4640000002:   duration>=30.0 AND foreign_worker=='b'yes'' AND other_parties=='b'none''
# 547252.302:  duration>=30.0 AND num_dependents==1.0
# 532099.56:   duration>=30.0 AND foreign_worker=='b'yes'' AND num_dependents==1.0
# 491104.688:  duration>=30.0 AND num_dependents==1.0 AND other_parties=='b'none''
# 490633.1400000001:   duration>=30.0 AND foreign_worker=='b'yes'' AND other_payment_plans=='b'none''
# 490633.1400000001:   duration>=30.0 AND other_payment_plans=='b'none''
# 475951.94600000005:  duration>=30.0 AND foreign_worker=='b'yes'' AND num_dependents==1.0 AND other_parties=='b'none''
