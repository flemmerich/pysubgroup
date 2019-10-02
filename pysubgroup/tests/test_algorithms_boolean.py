import pysubgroup as ps
import pandas as pd

import unittest

from test_DataSets import *

from test_algorithms import *

class TestAlgorithms(TestAlgorithmsBase,unittest.TestCase):
    def setUp(self):
        NS_checking=ps.NominalSelector("checking_status",b"<0")
        NS_foreign_worker=ps.NominalSelector("foreign_worker",b"yes")
        NS_other_parties=ps.NominalSelector("other_parties",b"none")
        NS_savings_status=ps.NominalSelector("savings_status",b"<100")
        NS_job=ps.NominalSelector("job",b"skilled")
        self.result=[ps.SubgroupDescription([NS_checking,NS_foreign_worker]),
                ps.SubgroupDescription([NS_checking]),
                ps.SubgroupDescription([NS_checking, NS_other_parties,NS_foreign_worker]),
                ps.SubgroupDescription([NS_checking, NS_other_parties]),
                ps.SubgroupDescription([NS_checking,NS_savings_status,NS_foreign_worker]),
                ps.SubgroupDescription([NS_checking,NS_savings_status]),
                ps.SubgroupDescription([NS_checking,NS_savings_status,NS_other_parties,NS_foreign_worker]),
                ps.SubgroupDescription([NS_checking,NS_job,NS_foreign_worker]),
                ps.SubgroupDescription([NS_checking, NS_savings_status,NS_other_parties]),
                ps.SubgroupDescription([NS_checking, NS_job]),
            ]
        self.qualities=[0.055299999999999995,
            0.05280000000000001,
            0.052300000000000006,
            0.05059999999999999,
            0.04959999999999999,
            0.048299999999999996,
            0.04660000000000001,
            0.04550000000000001,
            0.0452,
            0.044399999999999995]
        data=getCreditData2()
        target = ps.NominalTarget ('class', b'bad')
        searchSpace = ps.create_nominal_selectors(data, ignore=['class'])
        self.task = ps.SubgroupDiscoveryTask (data, target, searchSpace, result_set_size=10, depth=5, qf=ps.StandardQF(1.0))
    
    def test_Apriori(self):
        self.runAlgorithm(ps.Apriori(),"Apriori")


    def test_SimpleDFS(self):
        self.runAlgorithm(ps.SimpleDFS(),"SimpleDFS")


    def test_BestFirstSearch(self):
        self.runAlgorithm(ps.BestFirstSearch(),"BestFirstSearch")
    
    def test_BeamSearch(self):
        self.runAlgorithm(ps.BeamSearch(beam_width=10),"BeamSearch")

    def test_BSD(self):
        self.runAlgorithm(ps.BSD(),"BSD")

    def test_TID_SD_True(self):
        self.runAlgorithm(ps.TID_SD(True),"TID_SD True")
    
    def test_TID_SD_False(self):
        self.runAlgorithm(ps.TID_SD(False),"TID_SD False")


    

if __name__ == '__main__':
    if True:
        unittest.main()
    else:
        import cProfile
        p=cProfile.Profile()
        p.enable()
        t=TestAlgorithms()
        t.setUp()
        t.test_BSD()
    
        p.disable()
        p.dump_stats(r"E:\SGD\profile2.prof")

#0.055299999999999995:   checking_status=b'<0' AND foreign_worker=b'yes'
#0.05280000000000001:    checking_status=b'<0'
#0.052300000000000006:   checking_status=b'<0' AND other_parties=b'none' AND foreign_worker=b'yes'
#0.05059999999999999:    checking_status=b'<0' AND other_parties=b'none'
#0.04959999999999999:    checking_status=b'<0' AND savings_status=b'<100' AND foreign_worker=b'yes'
#0.048299999999999996:   checking_status=b'<0' AND savings_status=b'<100'
#0.04660000000000001:    checking_status=b'<0' AND savings_status=b'<100' AND other_parties=b'none' AND foreign_worker=b'
#0.04550000000000001:    checking_status=b'<0' AND job=b'skilled' AND foreign_worker=b'yes'
#0.0452: checking_status=b'<0' AND savings_status=b'<100' AND other_parties=b'none'
#0.044399999999999995:   checking_status=b'<0' AND job=b'skilled'

#checking_status=ps.NominalSelector(