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


import pysubgroup as ps

import unittest

from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase


class TestAlgorithmsWithNumericTarget(TestAlgorithmsBase, unittest.TestCase):
    def setUp(self):
        NS_telephone = ps.NominalSelector("own_telephone", b"yes")
        NS_foreign_worker = ps.NominalSelector("foreign_worker", b"yes")
        NS_other_parties = ps.NominalSelector("other_parties", b"none")
        NS_personal = ps.NominalSelector("personal_status", b'male single')
        NS_job = ps.NominalSelector("job", b'high qualif/self emp/mgmt')
        NS_class = ps.NominalSelector("class", b"bad")

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
        self.result = list(map(ps.SubgroupDescription, o))
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
        searchSpace = ps.create_nominal_selectors(data, ignore=['credit_amount'])

        self.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=5, qf=ps.StandardQFNumeric(1, False))

    def test_SimpleDFS(self):
        self.runAlgorithm(ps.SimpleDFS(), "SimpleDFS", self.result, self.qualities, self.task)

    def test_DFSNumeric(self):
        self.runAlgorithm(ps.DFSNumeric(), "DFS_numeric", self.result, self.qualities, self.task)


if __name__ == '__main__':
    unittest.main()
