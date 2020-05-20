import unittest
import pytest
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data

from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase

class TestSettings:
    All = True
    Apriori = False
    SimpleDFS = False
    BestFirstSearch = False
    BeamSearch = False
    DFS_bitset = False
    DFS_set = False
    DFS_numpyset = False
    SimpleSearch = False

skip_long_running = True
class BooleanTargetBase():

    # pylint: disable=no-member
    @unittest.skipUnless(TestSettings.All or TestSettings.Apriori, 'flag not set')
    def test_Apriori(self):
        self.runAlgorithm(ps.Apriori(), "Apriori", self.result, self.qualities, self.task)

    @unittest.skipUnless(TestSettings.All or TestSettings.SimpleDFS, 'flag not set')
    def test_SimpleDFS(self):
        self.runAlgorithm(ps.SimpleDFS(), "SimpleDFS", self.result, self.qualities, self.task)

    @unittest.skipUnless(TestSettings.All or TestSettings.BestFirstSearch, 'flag not set')
    def test_BestFirstSearch(self):
        self.runAlgorithm(ps.BestFirstSearch(), "BestFirstSearch", self.result, self.qualities, self.task)

    @unittest.skipUnless(TestSettings.All or TestSettings.BeamSearch, 'flag not set')
    def test_BeamSearch(self):
        self.runAlgorithm(ps.BeamSearch(beam_width=12), "BeamSearch", self.result, self.qualities, self.task)

    @unittest.skipUnless(TestSettings.All or TestSettings.DFS_bitset, 'flag not set')
    def test_DFS_bitset(self):
        self.runAlgorithm(ps.DFS(ps.BitSetRepresentation), "DFS bitset", self.result, self.qualities, self.task)




    @pytest.mark.slow
    @unittest.skipUnless(TestSettings.All or TestSettings.SimpleSearch, 'flag not set')
    def test_SimpleSearch(self):
        self.runAlgorithm(ps.SimpleSearch(), "SimpleSearch", self.result, self.qualities, self.task)

    @unittest.skipUnless(TestSettings.All or TestSettings.DFS_set, 'flag not set')
    def test_DFS_set(self):
        self.runAlgorithm(ps.DFS(ps.SetRepresentation), "DFS set", self.result, self.qualities, self.task)

    @unittest.skipUnless(TestSettings.All or TestSettings.DFS_numpyset, 'flag not set')
    def test_DFS_numpy_sets(self):
        self.runAlgorithm(ps.DFS(ps.NumpySetRepresentation), "DFS numpyset", self.result, self.qualities, self.task)
    # pylint: enable=no-member



class TestAlgorithms(TestAlgorithmsBase, BooleanTargetBase, unittest.TestCase):
    def setUp(self):
        NS_checking = ps.EqualitySelector("checking_status", b"<0")
        NS_foreign_worker = ps.EqualitySelector("foreign_worker", b"yes")
        NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        NS_savings_status = ps.EqualitySelector("savings_status", b"<100")
        NS_job = ps.EqualitySelector("job", b"skilled")
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
        target = ps.BinaryTarget('class', b'bad')
        searchSpace = ps.create_nominal_selectors(data, ignore=['class'])
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=5, qf=ps.StandardQF(1.0))

# also includes numeric attributes and has 12 targets


class TestAlgorithms2(TestAlgorithmsBase, BooleanTargetBase, unittest.TestCase):
    def setUp(self):
        NS_checking = ps.EqualitySelector("checking_status", b"<0")
        NS_foreign_worker = ps.EqualitySelector("foreign_worker", b"yes")
        NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        NS_savings_status = ps.EqualitySelector("savings_status", b"<100")
        NS_job = ps.EqualitySelector("job", b"skilled")
        NS_dependents = ps.EqualitySelector("num_dependents", 1.0)
        self.result = [ps.Conjunction([NS_checking, NS_foreign_worker]),
                       ps.Conjunction([NS_checking]),
                       ps.Conjunction([NS_checking, NS_other_parties, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_other_parties]),
                       ps.Conjunction([NS_checking, NS_savings_status, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_foreign_worker, NS_dependents]),
                       ps.Conjunction([NS_checking, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_dependents]),
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
                          0.04870000000000001,
                          0.048299999999999996,
                          0.0474,
                          0.04660000000000001,
                          0.04550000000000001,
                          0.0452,
                          0.044399999999999995
                          ]
        data = get_credit_data()
        target = ps.BinaryTarget('class', b'bad')
        searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['class'])
        searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['class'])
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=12, depth=5, qf=ps.StandardQF(1.0))

    @pytest.mark.slow
    def test_DFS_numpy_sets(self):
        super.test_DFS_numpy_sets()

# uses an a=0.5 and result_set_size = 12, is much faster because of that


class TestAlgorithms3(TestAlgorithmsBase, BooleanTargetBase, unittest.TestCase):
    def setUp(self):
        NS_checking = ps.EqualitySelector("checking_status", b"<0")
        NS_foreign_worker = ps.EqualitySelector("foreign_worker", b"yes")
        NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        NS_savings_status = ps.EqualitySelector("savings_status", b"<100")
        NS_job = ps.EqualitySelector("job", b"skilled")
        NS_dependents = ps.EqualitySelector("num_dependents", 1.0)
        self.result = [ps.Conjunction([NS_checking, NS_foreign_worker, NS_job, NS_other_parties, NS_savings_status]),  # AND job=='b'skilled'' AND other_parties=='b'none'' AND savings_status=='b'<100'
                       # 0.113713540226172:    checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND job=='b'skilled'' AND savings_status=='b'<100''
                       ps.Conjunction([NS_checking, NS_foreign_worker, NS_job, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_foreign_worker, NS_job]),  # checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND job=='b'skilled''
                       # checking_status=='b'<0'' AND job=='b'skilled'' AND other_parties=='b'none'' AND savings_status=='b'<100''
                       ps.Conjunction([NS_checking, NS_job, NS_other_parties, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_foreign_worker, NS_job, NS_other_parties]),
                       ps.Conjunction([NS_checking, NS_job, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_foreign_worker, NS_other_parties, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_foreign_worker, NS_other_parties]),
                       ps.Conjunction([NS_checking, NS_foreign_worker, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_foreign_worker]),
                       ps.Conjunction([NS_checking, NS_foreign_worker, NS_job, NS_dependents, NS_savings_status]),
                       ps.Conjunction([NS_checking, NS_job, NS_other_parties])]

        self.qualities = [0.11457431093955019,
                          0.113713540226172,
                          0.11201325679119281,
                          0.1117538749727658,
                          0.11161046793076415,
                          0.11145710640046322,
                          0.11045259291161472,
                          0.10929088624672183,
                          0.10875519439407161,
                          0.10866138825404954,
                          0.10832735026213287,
                          0.10813405094128754]
        data = get_credit_data()
        target = ps.BinaryTarget('class', b'bad')
        searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['class'])
        searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['class'])
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=12, depth=5, qf=ps.StandardQF(0.5))

    @pytest.mark.slow
    def test_BestFirstSearch(self):
        super.test_BestFirstSearch()

    @pytest.mark.slow
    def test_SimpleDFS(self):
        super.test_SimpleDFS()

    @pytest.mark.slow
    def test_DFS_numpy_sets(self):
        super.test_DFS_numpy_sets()


if __name__ == '__main__':

    #unittest.main()
    suites = []
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms2))
    suites.append(unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms3))
    complete_suite = unittest.TestSuite(suites)
    unittest.TextTestRunner(verbosity=2).run(complete_suite)

    #import cProfile
    # p=cProfile.Profile()
    # p.enable()
    # t=TestAlgorithms()
    # t.setUp()
    # t.test_BSD()

    # p.disable()
    # p.dump_stats(r"E:\SGD\profile2.prof")

# 0.055299999999999995:   checking_status=b'<0' AND foreign_worker=b'yes'
# 0.05280000000000001:    checking_status=b'<0'
# 0.052300000000000006:   checking_status=b'<0' AND other_parties=b'none' AND foreign_worker=b'yes'
# 0.05059999999999999:    checking_status=b'<0' AND other_parties=b'none'
# 0.04959999999999999:    checking_status=b'<0' AND savings_status=b'<100' AND foreign_worker=b'yes'
# 0.048299999999999996:   checking_status=b'<0' AND savings_status=b'<100'
# 0.04660000000000001:    checking_status=b'<0' AND savings_status=b'<100' AND other_parties=b'none' AND foreign_worker=b'
# 0.04550000000000001:    checking_status=b'<0' AND job=b'skilled' AND foreign_worker=b'yes'
# 0.0452:                 checking_status=b'<0' AND savings_status=b'<100' AND other_parties=b'none'
# 0.044399999999999995:   checking_status=b'<0' AND job=b'skilled'

# <<< best 12 for a=1
# 0.055299999999999995: checking_status=='b'<0'' AND foreign_worker=='b'yes''
# 0.05280000000000001:  checking_status=='b'<0''
# 0.052300000000000006: checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND other_parties=='b'none''
# 0.05059999999999999:  checking_status=='b'<0'' AND other_parties=='b'none''
# 0.04959999999999999:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND savings_status=='b'<100''
# 0.04870000000000001:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND num_dependents==1.0
# 0.048299999999999996: checking_status=='b'<0'' AND savings_status=='b'<100''
# 0.0474:               checking_status=='b'<0'' AND num_dependents==1.0
# 0.04660000000000001:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND other_parties=='b'none'' AND savings_status=='b'<100''
# 0.04550000000000001:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND job=='b'skilled''


# <<< best 12 a=0.5
# 0.11457431093955019:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND job=='b'skilled'' AND other_parties=='b'none'' AND savings_status=='b'<100''
# 0.113713540226172:    checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND job=='b'skilled'' AND savings_status=='b'<100''
# 0.11201325679119281:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND job=='b'skilled''
# 0.1117538749727658:   checking_status=='b'<0'' AND job=='b'skilled'' AND other_parties=='b'none'' AND savings_status=='b'<100''
# 0.11161046793076415:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND job=='b'skilled'' AND other_parties=='b'none''
# 0.11145710640046322:  checking_status=='b'<0'' AND job=='b'skilled'' AND savings_status=='b'<100''
# 0.11045259291161472:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND other_parties=='b'none'' AND savings_status=='b'<100''
# 0.10929088624672183:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND other_parties=='b'none''
# 0.10875519439407161:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND savings_status=='b'<100''
# 0.10866138825404954:  checking_status=='b'<0'' AND foreign_worker=='b'yes''
# 0.10832735026213287:  checking_status=='b'<0'' AND foreign_worker=='b'yes'' AND job=='b'skilled'' AND num_dependents==1.0 AND savings_status=='b'<100''
# 0.10813405094128754:  checking_status=='b'<0'' AND job=='b'skilled'' AND other_parties=='b'none''
