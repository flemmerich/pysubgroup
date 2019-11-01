
import unittest
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data

from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase


class BooleanTargetBase(TestAlgorithmsBase):
    # pylint: disable=no-member
    def test_GeneralisingBFS(self):
        self.runAlgorithm(ps.GeneralisingBFS(), "GeneralisingBFS", self.result, self.qualities, self.task)
    def test_GeneralisingApriori(self):
        algorithm = ps.Apriori(combination_name='Disjunction')
        algorithm.optimistic_estimate_name = 'optimistic_generalisation'
        self.runAlgorithm(algorithm, "GeneralisingApriori", self.result, self.qualities, self.task)
    # pylint: enable=no-member

class TestAlgorithms(BooleanTargetBase, unittest.TestCase):
    def setUp(self):
        NS_checking = ps.NominalSelector("checking_status", b"<0")
        NS_foreign_worker = ps.NominalSelector("foreign_worker", b"yes")
        NS_other_parties = ps.NominalSelector("other_parties", b"none")
        NS_savings_status = ps.NominalSelector("savings_status", b"<100")
        NS_job = ps.NominalSelector("job", b"skilled")
        self.result = [ps.Conjunction([NS_checking, NS_foreign_worker]),
                       ps.Conjunction([NS_checking]),
                       ps.Conjunction([NS_checking, NS_other_parties, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_other_parties]),
                       ps.Conjunction([NS_checking, NS_savings_status, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_savings_status, NS_other_parties, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_job, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_savings_status, NS_other_parties]),
                       ps.Conjunction([NS_checking, NS_job]),
                       ]
        self.qualities = [0.055299999999999995,
                          0.05280000000000001,
                          0.052300000000000006,
                          0.05059999999999999,
                          0.04959999999999999,
                          0.048299999999999996,
                          0.04660000000000001,
                          0.04550000000000001,
                          0.0452,
                          0.044399999999999995]
        data = get_credit_data()
        target = ps.NominalTarget('class', b'bad')
        searchSpace = ps.create_nominal_selectors(data, ignore=['class'])
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=4, qf=ps.StandardQF(1.0))


if __name__ == '__main__':
    unittest.main()
#depth =5
#   0.07800000000000001: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR other_parties=='b'co applicant'' OR other_payment_plans=='b'bank''
#   0.07800000000000001: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR other_parties=='b'co applicant'' OR other_payment_plans=='b'bank'' OR purpose=='b'other''
#   0.07790000000000001: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR credit_history=='b'no credits/all paid'' OR other_parties=='b'co applicant'' OR other_payment_plans=='b'bank''
#   0.0779:              checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR other_payment_plans=='b'bank''
#   0.0779:              checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR other_payment_plans=='b'bank'' OR purpose=='b'other''
#   0.07780000000000002: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR credit_history=='b'no credits/all paid'' OR other_payment_plans=='b'bank''
#   0.07780000000000002: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR credit_history=='b'no credits/all paid'' OR other_payment_plans=='b'bank'' OR purpose=='b'other''
#   0.07770000000000002: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR credit_history=='b'no credits/all paid'' OR other_payment_plans=='b'bank'' OR purpose=='b'repairs''
#   0.0776:              checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR other_parties=='b'co applicant'' OR other_payment_plans=='b'bank'' OR purpose=='b'repairs''
#   0.07750000000000001: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR other_payment_plans=='b'bank'' OR purpose=='b'repairs''
#   0.07750000000000001: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR credit_history=='b'all paid'' OR other_parties=='b'co applicant'' OR other_payment_plans=='b'bank''