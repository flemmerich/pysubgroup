import unittest
import numpy as np
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_titanic_data
from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase



#data=get_titanic_data()
#search_space = ps.create_selectors(data, ignore="survived")
#dt = data.dtypes

#task = ps.SubgroupDiscoveryTask(data, ps.FITarget, search_space, result_set_size=10, depth=5, qf=ps.CountQF())
#result = ps.SimpleDFS().execute(task)

#for (q, sg) in result:
#    print(str(q) + ":\t" + str(sg.subgroup_description))

#task = ps.SubgroupDiscoveryTask(data, ps.FITarget, search_space, result_set_size=10, depth=2, qf=ps.AreaQF())
#result = ps.SimpleDFS().execute(task)

#for (q, sg) in result:
#    print(str(q) + ":\t" + str(sg.subgroup_description))

class TestCountQF(TestAlgorithmsBase, unittest.TestCase):
    def test_Apriori(self):
        self.runAlgorithm(ps.Apriori(), "Apriori", self.result[1:], self.qualities[1:], self.task)

    def test_DFS(self):
        self.runAlgorithm(ps.SimpleDFS(), "DFS", self.result[:-1], self.qualities[:-1], self.task)

    def setUp(self):
        NS_cabin = ps.EqualitySelector("Cabin", np.nan)
        NS_embarked = ps.EqualitySelector("Embarked", 'S')
        NS_male = ps.EqualitySelector("Sex", 'male')
        NS_female = ps.EqualitySelector("Sex", 'female')
        #NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        #NS_savings_status = ps.EqualitySelector("savings_status", b"<100")
        #NS_job = ps.EqualitySelector("job", b"skilled")
        self.result = [ps.Conjunction([]),
                       ps.Conjunction([NS_cabin]),
                       ps.Conjunction([NS_embarked]),
                       ps.Conjunction([NS_male]),
                       ps.Conjunction([NS_cabin, NS_embarked]),
                       ps.Conjunction([NS_cabin, NS_male]),
                       ps.Conjunction([NS_embarked, NS_male]),
                       ps.Conjunction([NS_cabin, NS_embarked, NS_male]),
                       ps.Conjunction([NS_female]),
                       ps.Conjunction([NS_cabin, NS_female]),
                       ps.Conjunction([NS_embarked, NS_female])]

        self.qualities = [156, 125, 110, 100, 89, 82, 73, 60, 56, 43, 37]

        data = get_titanic_data()
        self.qualities2 = [np.count_nonzero(conj.covers(data)) for conj in self.result]
        self.assertEqual(self.qualities, self.qualities2)
        searchSpace = ps.create_nominal_selectors(data)
        self.task = ps.SubgroupDiscoveryTask(data, ps.FITarget(), searchSpace, result_set_size=10, depth=5, qf=ps.CountQF())




class TestAreaQF(TestAlgorithmsBase, unittest.TestCase):
    def test_SimpleSearch(self):
        self.runAlgorithm(ps.SimpleSearch(), "SimpleSearch", self.result, self.qualities, self.task)
   # 178: Cabin.isnull() AND Embarked=='S'
   #164: Cabin.isnull() AND Sex=='male'
   #146: Embarked=='S' AND Sex=='male'
   #125: Cabin.isnull()
   #110: Embarked=='S'
   #100: Sex=='male'
   #86:  Cabin.isnull() AND Sex=='female'
   #74:  Embarked=='S' AND Sex=='female'
   #56:  Sex=='female'
   #46:  Cabin.isnull() AND Embarked=='C'
    def setUp(self):
        NS_cabin = ps.EqualitySelector("Cabin", np.nan)
        NS_embarked = ps.EqualitySelector("Embarked", 'S')
        NS_embarked2 = ps.EqualitySelector("Embarked", 'C')
        NS_male = ps.EqualitySelector("Sex", 'male')
        NS_female = ps.EqualitySelector("Sex", 'female')
        #NS_other_parties = ps.EqualitySelector("other_parties", b"none")
        #NS_savings_status = ps.EqualitySelector("savings_status", b"<100")
        #NS_job = ps.EqualitySelector("job", b"skilled")
        self.result = [ps.Conjunction([NS_cabin, NS_embarked]),
                       ps.Conjunction([NS_cabin, NS_male]),
                       ps.Conjunction([NS_embarked, NS_male]),
                       ps.Conjunction([NS_cabin]),
                       ps.Conjunction([NS_embarked]),
                       ps.Conjunction([NS_male]),
                       ps.Conjunction([NS_cabin, NS_female]),
                       ps.Conjunction([NS_embarked, NS_female]),
                       ps.Conjunction([NS_female]),
                       ps.Conjunction([NS_cabin, NS_embarked2]),
                       ]

        self.qualities = [178, 164, 146, 125, 110, 100, 86, 74, 56, 46]

        data = get_titanic_data()
        self.qualities2 = [np.count_nonzero(conj.covers(data))*conj.depth for conj in self.result]
        self.assertEqual(self.qualities, self.qualities2)
        searchSpace = ps.create_nominal_selectors(data)
        self.task = ps.SubgroupDiscoveryTask(data, ps.FITarget(), searchSpace, result_set_size=10, depth=2, qf=ps.AreaQF())

 #  156: True
 #  125: Cabin.isnull()
 #  110: Embarked=='S'
 #  100: Sex=='male'
 #  89:  Cabin.isnull() AND Embarked=='S'
 #  82:  Cabin.isnull() AND Sex=='male'
 #  73:  Embarked=='S' AND Sex=='male'
 #  60:  Cabin.isnull() AND Embarked=='S' AND Sex=='male'
 #  56:  Sex=='female'
 #  43:  Cabin.isnull() AND Sex=='female'
 #  37:  Embarked=='S' AND Sex=='female'

if __name__ == '__main__':
    #unittest.main()
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestCountQF)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestAreaQF)
    #suite3 = unittest.TestLoader().loadTestsFromTestCase(TestAlgorithms3)
    complete_suite = unittest.TestSuite([suite1, suite2])
    unittest.TextTestRunner(verbosity=2).run(complete_suite)
