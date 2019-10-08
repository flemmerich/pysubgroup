from timeit import default_timer as timer
import abc


class TestAlgorithmsBase(abc.ABC):
    def evaluate_result(self, algorithm_result,result, qualities):
        # compare length such that zip works correctly
        self.assertEqual(len(algorithm_result), len(result)) 
        self.assertEqual(len(algorithm_result), len(qualities))
        for (q, sg) in algorithm_result:
            print ("   "+   str(q) + ":\t" + str(sg.subgroup_description))
        for (algorithm_q, algorithm_SG), expected_q, expected_SGD in zip(algorithm_result, qualities, result): 
            self.assertEqual(repr(algorithm_SG.subgroup_description), repr(expected_SGD))
            self.assertEqual(algorithm_q, expected_q)

    def runAlgorithm(self, algorithm, name, result, qualities, task):
        print()
        print("Running " + name)
        start = timer()
        algorithm_result = algorithm.execute(task)
        end = timer()
        print("   Runtime for {}: {}".format(name, end - start))
        print()
        self.evaluate_result(algorithm_result,result, qualities)
