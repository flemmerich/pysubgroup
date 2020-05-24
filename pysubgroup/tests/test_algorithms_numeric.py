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

import unittest
import pysubgroup as ps


from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase

class TestStandardQFNumeric(unittest.TestCase):
    def test_constructor(self):
        ps.StandardQFNumeric(0)
        ps.StandardQFNumeric(1.0)
        ps.StandardQFNumeric(0, invert=True)
        ps.StandardQFNumeric(0, invert=False)

        with self.assertRaises(ValueError):
            ps.StandardQFNumeric('test')

        ps.StandardQFNumeric(0, estimator='sum')
        ps.StandardQFNumeric(0, estimator='average')
        ps.StandardQFNumeric(0, estimator='order')

        with self.assertRaises(ValueError):
            ps.StandardQFNumeric(0, estimator='bla')


class TestAlgorithmsWithNumericTarget(TestAlgorithmsBase, unittest.TestCase):
    def setUp(self):
        NS_telephone = ps.EqualitySelector("own_telephone", b"yes")
        NS_foreign_worker = ps.EqualitySelector("foreign_worker", b"yes")
        NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        NS_personal = ps.EqualitySelector("personal_status", b'male single')
        NS_job = ps.EqualitySelector("job", b'high qualif/self emp/mgmt')
        NS_class = ps.EqualitySelector("class", b"bad")

        o = [[NS_telephone],
             [NS_foreign_worker, NS_telephone],
             [NS_other_parties, NS_telephone],
             [NS_foreign_worker, NS_telephone, NS_personal],
             [NS_telephone, NS_personal],
             [NS_foreign_worker, NS_other_parties, NS_telephone],
             [NS_job],
             [NS_class, NS_telephone],
             [NS_foreign_worker, NS_job],
             [NS_foreign_worker, NS_other_parties, NS_telephone, NS_personal]
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
            297422.98200000013]

        data = get_credit_data()
        target = ps.NumericTarget('credit_amount')
        searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['credit_amount'])
        searchSpace_Numeric = [] #ps.create_numeric_selectors(data, ignore=['credit_amount'], nbins=10)
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=5, qf=ps.CountCallsInterestingMeasure(ps.StandardQFNumeric(1, False, 'sum')))

    def test_SimpleDFS(self):
        self.runAlgorithm(ps.SimpleDFS(), "SimpleDFS", self.result, self.qualities, self.task)

    def test_DFS_average(self):
        self.task.qf = ps.CountCallsInterestingMeasure(ps.StandardQFNumeric(self.task.qf.a, False, 'average'))
        self.runAlgorithm(ps.DFS(ps.BitSetRepresentation), "DFS average", self.result, self.qualities, self.task)

    def test_DFS_order(self):
        self.task.qf = ps.CountCallsInterestingMeasure(ps.StandardQFNumeric(self.task.qf.a, False, 'order'))
        self.runAlgorithm(ps.DFS(ps.BitSetRepresentation), "DFS order", self.result, self.qualities, self.task)

    def test_DFS_sum(self):
        self.task.qf = ps.CountCallsInterestingMeasure(ps.StandardQFNumeric(self.task.qf.a, False, 'sum'))
        self.runAlgorithm(ps.DFS(ps.BitSetRepresentation), "DFS sum", self.result, self.qualities, self.task)

    def test_BeamSearch_sum(self):
        self.task.qf = ps.CountCallsInterestingMeasure(ps.StandardQFNumeric(self.task.qf.a, False, 'sum'))
        self.runAlgorithm(ps.BeamSearch(), "BeamSearch sum", self.result, self.qualities, self.task)

    def test_BeamSearch_average(self):
        self.task.qf = ps.CountCallsInterestingMeasure(ps.StandardQFNumeric(self.task.qf.a, False, 'average'))
        self.runAlgorithm(ps.BeamSearch(), "BeamSearch average", self.result, self.qualities, self.task)

    def test_BeamSearch_order(self):
        self.task.qf = ps.CountCallsInterestingMeasure(ps.StandardQFNumeric(self.task.qf.a, False, 'order'))
        self.runAlgorithm(ps.BeamSearch(), "BeamSearch order", self.result, self.qualities, self.task)

    def test_Apriori_no_numba(self):
        self.runAlgorithm(ps.Apriori(use_numba=False), "Apriori use_numba=False", self.result, self.qualities, self.task)

    def test_Apriori_with_numba(self):
        self.runAlgorithm(ps.Apriori(use_numba=True), "Apriori use_numba=True", self.result, self.qualities, self.task)

    def test_DFSNumeric(self):
        algo = ps.DFSNumeric()
        self.runAlgorithm(algo, "DFS_numeric", self.result, self.qualities, self.task)
        print('   Number of call to qf:', algo.num_calls)

    #def test_SimpleSearch(self):
    #   self.runAlgorithm(ps.SimpleSearch(), "SimpleSearch", self.result, self.qualities, self.task)


if __name__ == '__main__':
    unittest.main()


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
