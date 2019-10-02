from timeit import default_timer as timer
import abc
class TestAlgorithmsBase(abc.ABC):
    def evaluate_result(self,algorithm_result):
        #compare length such that zip works correctly
        self.assertEqual(len(algorithm_result),len(self.result)) # pylint: disable=maybe-no-member
        self.assertEqual(len(algorithm_result),len(self.qualities)) # pylint: disable=maybe-no-member
        for (algorithm_q,algorithm_SG),expected_q, expected_SGD in zip(algorithm_result,self.qualities,self.result): # pylint: disable=maybe-no-member
            self.assertEqual(repr(algorithm_SG.subgroup_description),repr(expected_SGD)) # pylint: disable=maybe-no-member
            self.assertEqual(algorithm_q,expected_q) # pylint: disable=maybe-no-member


    def runAlgorithm(self,algorithm,name):
        print("Running " + name)
        start = timer()
        algorithm_result = algorithm.execute(self.task) # pylint: disable=maybe-no-member
        end = timer()
        print("   Runtime for {}: {}".format(name, end - start)) 
        self.evaluate_result( algorithm_result)

    