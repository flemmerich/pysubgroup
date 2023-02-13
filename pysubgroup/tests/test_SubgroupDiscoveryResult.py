import unittest
import matplotlib.pyplot as plt

import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase

show_plots = False
class TestNominalTarget_to_result(unittest.TestCase, TestAlgorithmsBase):
    @classmethod
    def setUpClass(cls):
        data = get_credit_data()
        target = ps.BinaryTarget('class', b'bad')
        searchSpace = ps.create_nominal_selectors(data, ignore=['class'])
        cls.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=5, qf=ps.StandardQF(1.0))
        cls.result = ps.SimpleDFS().execute(cls.task)

    def test_to_dataframe(self):
        df = self.__class__.result.to_dataframe()
        self.assertEqual(len(df), 10)
        self.assertEqual(len(df.columns), 15)

    def test_to_descriptions(self):
        l = self.__class__.result.to_descriptions()
        self.assertEqual(len(l), 10)
        for tpl in l:
            self.assertEqual(len(tpl), 2)
            self.assertTrue(isinstance(tpl[0], float))
            self.assertTrue(isinstance(tpl[1], ps.Conjunction))

    #def test_supportSetVisualization(self):
    #    img = self.__class__.result.supportSetVisualization()
    #    self.assertEqual(img.shape, (10, 274))
    #    if show_plots:
    #        plt.matshow(img)
    #        plt.show()

class TestNumericTarget_to_result(unittest.TestCase, TestAlgorithmsBase):
    @classmethod
    def setUpClass(cls):
        data = get_credit_data()
        target = ps.NumericTarget('credit_amount')
        searchSpace = ps.create_nominal_selectors(data, ignore=['credit_amount'])
        cls.task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=10, depth=3, qf=ps.StandardQFNumeric(1.0))
        cls.result = ps.SimpleDFS().execute(cls.task)

    def test_to_dataframe(self):
        df = self.__class__.result.to_dataframe()
        self.assertEqual(len(df), 10)
        self.assertEqual(len(df.columns), 16)

    def test_to_descriptions(self):
        l = self.__class__.result.to_descriptions()
        self.assertEqual(len(l), 10)
        for tpl in l:
            self.assertEqual(len(tpl), 2)
            self.assertTrue(isinstance(tpl[0], float))
            self.assertTrue(isinstance(tpl[1], ps.Conjunction))


    #def test_supportSetVisualization(self):
    #    img = self.__class__.result.supportSetVisualization()
    #    self.assertEqual(img.shape, (10, 425))
    #    if show_plots:
    #        plt.matshow(img)
    #        plt.show()


if __name__ == '__main__':
    show_plots = False
    unittest.main()
