from timeit import default_timer as timer
import unittest
import pysubgroup as ps
class TestAlgorithmsBase():
    def evaluate_result(self,algorithm_result):
        #compare length such that zip works correctly
        self.assertEqual(len(algorithm_result),len(self.result))
        self.assertEqual(len(algorithm_result),len(self.qualities))
        for (algorithm_q,algorithm_SG),expected_q, expected_SGD in zip(algorithm_result,self.qualities,self.result):
            self.assertEqual(algorithm_SG.subgroupDescription.to_query(),expected_SGD.to_query())
            self.assertEqual(algorithm_q,expected_q)


    def runAlgorithm(self,algorithm,name):
        print("Running " + name)
        start = timer()
        algorithm_result = algorithm.execute(self.task)
        end = timer()
        print("   Runtime for {}: {}".format(name, end - start)) 
        self.evaluate_result( algorithm_result)

    