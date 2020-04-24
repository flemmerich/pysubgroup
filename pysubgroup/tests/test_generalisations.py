import unittest
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase

skip_long_running = True

class BooleanTargetBase(TestAlgorithmsBase):
    # pylint: disable=no-member
    @unittest.skipIf(skip_long_running, "as skip_long_running flag is True")
    def test_GeneralisingBFS(self):
        self.runAlgorithm(ps.GeneralisingBFS(), "GeneralisingBFS", self.result, self.qualities, self.task)

    def test_GeneralisingApriori(self):
        algorithm = ps.Apriori(combination_name='Disjunction')
        algorithm.optimistic_estimate_name = 'optimistic_generalisation'
        self.runAlgorithm(algorithm, "GeneralisingApriori", self.result, self.qualities, self.task)
    # pylint: enable=no-member

class TestAlgorithms(BooleanTargetBase, unittest.TestCase):
    def setUp(self):
        NS_checking = ps.EqualitySelector("checking_status", b"<0")
        NS_checking2 = ps.EqualitySelector("checking_status", b"0<=X<200")
        NS_other_parties = ps.EqualitySelector("other_parties", b"co applicant")
        NS_other = ps.EqualitySelector("purpose", b'other')
        NS_repairs = ps.EqualitySelector("purpose", b'repairs')
        NS_purpose = ps.EqualitySelector("purpose", b'business')

        NS_history = ps.EqualitySelector("credit_history", b"no credits/all paid")
        NS_history2 = ps.EqualitySelector("credit_history", b"all paid")
        NS_empl = ps.EqualitySelector("employment", b"unemployed")
        NS_job = ps.EqualitySelector("job", b"unemp/unskilled non res")
        NS_bank = ps.EqualitySelector("other_payment_plans", b"bank")
        self.result = [ps.Disjunction([NS_checking, NS_checking2, NS_bank]),
                       ps.Disjunction([NS_checking, NS_checking2, NS_history]),
                       ps.Disjunction([NS_checking, NS_checking2]),
                       ps.Disjunction([NS_checking, NS_checking2, NS_other]),
                       ps.Disjunction([NS_checking, NS_checking2, NS_repairs]),
                       ps.Disjunction([NS_checking, NS_checking2, NS_empl]),
                       ps.Disjunction([NS_checking, NS_checking2, NS_other_parties]),
                       ps.Disjunction([NS_checking, NS_checking2, NS_history2]),
                       ps.Disjunction([NS_checking, NS_checking2, NS_purpose]),
                       ps.Disjunction([NS_checking, NS_checking2, NS_job]),
                       ]
        self.qualities = [0.0779,
                          0.07740000000000002,
                          0.0771,
                          0.07680000000000001,
                          0.07670000000000002,
                          0.0767,
                          0.07660000000000003,
                          0.07650000000000003,
                          0.07650000000000001,
                          0.07600000000000001]
        data = get_credit_data()
        target = ps.BinaryTarget('class', b'bad')
        searchSpace = ps.create_nominal_selectors(data, ignore=['class'])
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=3, qf=ps.StandardQF(1.0))


if __name__ == '__main__':
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms)
    complete_suite = unittest.TestSuite([suite1])
    unittest.TextTestRunner(verbosity=2).run(complete_suite)
#depth = 5, a=1.0
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

#depth = 3, a=1.0
#  0.0779:              checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR other_payment_plans=='b'bank''
#  0.07740000000000002: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR credit_history=='b'no credits/all paid''
#  0.0771:              checking_status=='b'0<=X<200'' OR checking_status=='b'<0''
#  0.07680000000000001: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR purpose=='b'other''
#  0.07670000000000002: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR purpose=='b'repairs''
#  0.0767:              checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR employment=='b'unemployed''
#  0.07660000000000003: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR other_parties=='b'co applicant''
#  0.07650000000000003: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR credit_history=='b'all paid''
#  0.07650000000000001: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR purpose=='b'business''
#  0.07600000000000001: checking_status=='b'0<=X<200'' OR checking_status=='b'<0'' OR job=='b'unemp/unskilled non res''