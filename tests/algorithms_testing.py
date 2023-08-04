from timeit import default_timer as timer
import abc
import pysubgroup as ps

class TestAlgorithmsBase(abc.ABC):

    # pylint: disable=no-member
    def evaluate_result(self, algorithm_result, result, qualities):
        self.assertTrue(isinstance(algorithm_result, ps.SubgroupDiscoveryResult))
        algorithm_result.to_dataframe()
        algorithm_result = algorithm_result.to_descriptions()
        for (q, sg) in algorithm_result:
            print("   " + str(q) + ":\t" + str(sg))
        # compare length such that zip works correctly
        self.assertEqual(len(algorithm_result), len(result))
        self.assertEqual(len(algorithm_result), len(qualities))

        for (algorithm_q, algorithm_SG), expected_q, expected_SGD in zip(algorithm_result, qualities, result):
            self.assertEqual(repr(algorithm_SG), repr(expected_SGD))
            self.assertEqual(algorithm_q, expected_q)


    def runAlgorithm(self, algorithm, name, result, qualities, task):
        print()
        print("Running " + name)
        start = timer()
        algorithm_result = algorithm.execute(task)
        end = timer()
        print("   Runtime for {}: {}".format(name, end - start))

        if hasattr(self.task.qf, 'calls'):
            print('   Number of call to qf:', self.task.qf.calls)
        print()
        self.evaluate_result(algorithm_result, result, qualities)
        return algorithm_result
    # pylint: enable=no-member
