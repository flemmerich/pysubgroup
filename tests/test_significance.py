import unittest
import numpy as np
import pysubgroup as ps
from pysubgroup.datasets import get_credit_data

class TestSignificance(unittest.TestCase):
    def setUp(self):
        self.data = get_credit_data().map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        self.target = ps.BinaryTarget('class', 'good')
        self.task = ps.SubgroupDiscoveryTask(
            self.data,
            self.target,
            search_space=ps.create_selectors(self.data, ignore=['class']),
            qf=ps.WRAccQF(),
            result_set_size=1,
            depth=2
        )
        self.result = ps.BeamSearch().execute(self.task)


    def test_full_workflow(self):     
        tester = ps.Significance(self.task)
        null_dist = tester.generate_null_distribution(num_permutations=50, num_qualities=2)
        result = tester.add_statistical_metrics(self.result)
        
        # Verify output
        self.assertIsInstance(null_dist, np.ndarray)
        self.assertEqual(len(null_dist), 100) # 50 (num_permutation) * 2 (num_qualities)

        # Verify content (only meaningful for WRAcc)
        mask = null_dist != float('-inf')
        if np.any(mask):
            self.assertTrue(np.all(null_dist[mask] >= -0.25), "WRAcc too low")
            self.assertTrue(np.all(null_dist[mask] <= 0.25), "WRAcc too high")

        self.assertIsInstance(result, ps.SubgroupDiscoveryResult)
        

    def test_stats_wrapper(self):
        """Possible to test other search strategies."""
        wrapper = ps.StatsWrapper(ps.BeamSearch(), num_permutations=50)
        wrapped_result = wrapper.execute(self.task)
        
        self.assertIsInstance(wrapped_result, ps.SubgroupDiscoveryResult)
 
        df = wrapped_result.to_dataframe()
        self.assertIn('z_score', df.columns)
        self.assertIn('p_value', df.columns)
        self.assertTrue(0 <= df.iloc[0].p_value <= 1)

    
    def test_visualization(self):
        tester = ps.Significance(self.task)
        null_dist = tester.generate_null_distribution(num_permutations=50, num_qualities=1)
        self.assertEqual(len(null_dist), 50) 

        original_quality = self.result.to_dataframe()['quality'].iloc[0]

        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        from matplotlib import pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_subplot()
        ps.plot_null_distribution(null_dist, original_quality, bw_adjust=1, ax=ax)
        
        self.assertGreater(len(ax.patches), 0)  # At least one histogram bar
        self.assertEqual(len(ax.lines), 2)      # KDE line + vline
        
        plt.close(fig)
        #plt.show()     # just deactivate non-interactive backend and look at the plot...


if __name__ == '__main__':
    unittest.main()