import unittest
import numpy as np
import pysubgroup as ps
from pysubgroup.datasets import get_credit_data

class TestSignificance(unittest.TestCase):
    def setUp(self):
        # Initialize test data and task once for all tests
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
        self.original_result_size = self.task.result_set_size  # Capture original; same reason as function

    def test_basic_functionality(self):
        # Use the pre-configured task from setUp
        tester = ps.Significance(self.task)

        null_dist = tester.generate_null_distribution(
            num_permutations=50, 
            num_qualities=2
        )

        # Verify output
        self.assertIsInstance(null_dist, np.ndarray)
        self.assertEqual(len(null_dist), 100) # num_permutation * num_qualities

        # Verify content - test likely not meaningful when using different quality functions
        self.assertTrue(np.all(null_dist >= -0.25))  # WRAcc minimum
        self.assertTrue(np.all(null_dist <= 0.25))   # WRAcc maximum

    def test_visualization(self):
        # Use data from setUp
        tester = ps.Significance(self.task)
        null_dist = tester.generate_null_distribution(num_permutations=50, num_qualities=1)
        
        # Get original subgroup quality
        result = ps.BeamSearch().execute(self.task)
        original_quality = result.to_dataframe()['quality'].iloc[0]

        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        from matplotlib import pyplot as plt
        
        # Create figure and test plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ps.plot_null_distribution(null_dist, original_quality, bw_adjust=1, ax=ax)
        
        self.assertGreater(len(ax.patches), 0)  # At least one histogram bar
        self.assertEqual(len(ax.lines), 2)      # KDE line + vline
                
        # Check vertical line position
        vline = [line for line in ax.lines if line.get_linestyle() == '--'][0]
        self.assertAlmostEqual(vline.get_xdata()[0], original_quality, places=2)
        
        plt.close(fig)

    def tearDown(self):
        # Reset task to original state
        self.task.result_set_size = self.original_result_size

if __name__ == '__main__':
    unittest.main()