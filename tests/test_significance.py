import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import pysubgroup as ps
from pysubgroup.datasets import get_credit_data

class TestStatisticalSignificance(unittest.TestCase):
    def setUp(self):
        # Load and prepare data
        self.data = get_credit_data().map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        self.target = ps.BinaryTarget('class', 'good')
        self.search_space = ps.create_selectors(self.data, ignore=['class'])
        self.task = ps.SubgroupDiscoveryTask(
            self.data,
            self.target,
            search_space=self.search_space,
            qf=ps.WRAccQF(),
            result_set_size=5,
            depth=2,
            min_quality=0
        )

    def test_null_distribution_size(self):
        """Test if null distribution has correct size (num_permutations * num_qualities)."""
        sig = ps.StatisticalSignificance(self.task)
        null_dist = sig.generate_null_distribution(num_permutations=50, num_qualities=2)
        self.assertEqual(len(null_dist), 50 * 2)  # 50 permutations * 2 qualities each

    def test_result_metrics_inclusion(self):
        """Verify z-scores, p-values, and adjusted p-values are included."""
        decorator = ps.SignificanceDecorator(ps.BeamSearch(), num_permutations=10, adjust_method='holm')
        result = decorator.execute(self.task)
        df = result.to_dataframe()
        
        # Check for expected columns
        self.assertIn('z_score', df.columns)
        self.assertIn('p_value', df.columns)
        self.assertIn('p_value_adj', df.columns)
        
        # Ensure values are within valid ranges
        self.assertTrue((df['p_value'] >= 0).all() and (df['p_value'] <= 1).all())
        self.assertTrue((df['p_value_adj'] >= df['p_value']).all())  # P-value adjustment

    def test_empirical_p_values_non_normal(self):
        """Test empirical p-value calculation when normality failed."""
        with patch.object(ps.StatisticalSignificance, '_check_normality', return_value=False):
            decorator = ps.SignificanceDecorator(ps.BeamSearch(), num_permutations=100)
            result = decorator.execute(self.task)
            df = result.to_dataframe()
            
            # Should have p_values but no z-scores column
            self.assertIn('p_value', df.columns)
            self.assertNotIn('z_score', df.columns) 

    def test_multiple_testing_correction(self):
        """Validate Holm-Bonferroni adjustment correctness."""
        decorator = ps.SignificanceDecorator(ps.BeamSearch(), adjust_method='holm')
        result = decorator.execute(self.task)
        df = result.to_dataframe()
        
        # Check ordering of adjusted p-values (Holm: sorted ascendingly)
        sorted_p = np.sort(df['p_value'])
        sorted_adj_p = np.sort(df['p_value_adj'])
        self.assertTrue((sorted_adj_p >= sorted_p).all())  # Each adj p >= corresponding raw p

    def test_parallel_consistency(self):
        """Ensure parallel and serial null distributions are statistically similar."""
        np.random.seed(42)  # Control randomness
        task = self.task
        
        # Generate serial null distribution
        sig_serial = ps.StatisticalSignificance(task)
        null_serial = sig_serial.generate_null_distribution(num_permutations=100, n_jobs=1)
        
        # Generate parallel null distribution with same seed
        np.random.seed(42)
        sig_parallel = ps.StatisticalSignificance(task)
        null_parallel = sig_parallel.generate_null_distribution(num_permutations=100, n_jobs=-1)
        
        # Compare basic statistics
        self.assertAlmostEqual(np.nanmean(null_serial), np.nanmean(null_parallel), delta=0.1)
        self.assertAlmostEqual(np.nanstd(null_serial), np.nanstd(null_parallel), delta=0.1)

    def test_selector_serialization(self):
        """Verify patched SelectorBase can be serialized."""
        import pickle
        selector = ps.EqualitySelector('checking_status', 'no checking')
        pickled = pickle.dumps(selector)
        unpickled = pickle.loads(pickled)
        self.assertEqual(selector, unpickled)

    def test_non_finite_quality_handling(self):
        """Test scenarios with invalid (-inf) qualities are handled gracefully."""
        # Create task with impossible quality threshold
        task = ps.SubgroupDiscoveryTask(
            self.data,
            self.target,
            search_space=self.search_space,
            qf=ps.WRAccQF(),
            result_set_size=5,
            depth=2,
            min_quality=1.0  # No subgroup can achieve this
        )
        
        decorator = ps.SignificanceDecorator(ps.BeamSearch(), num_permutations=10)
        result = decorator.execute(task)
        df = result.to_dataframe()
        
        # Verify empty result handling
        self.assertTrue(df.empty, "Should return empty dataframe when no subgroups found")
        self.assertEqual(len(result.results), 0, "Result list should be empty")

    def test_non_finite_quality_filtering_in_worker(self):
        """Test that non-finite qualities are filtered out in the worker."""
        from unittest.mock import MagicMock

        # Create a mock task
        task = MagicMock()
        task.target = ps.BinaryTarget('class', 'good')
        task.data = self.data

        # Create a mock result with non-finite qualities
        mock_result = MagicMock()
        mock_result.results = [
            (np.inf, 'selector1', 'description1'),   # Infinite
            (np.nan, 'selector2', 'description2'),   # NaN
            (0.5, 'selector3', 'description3'),      # Valid
            (-np.inf, 'selector4', 'description4')   # Negative infinite
        ]

        # Mock the search strategy to return the mock result
        mock_search = MagicMock()
        mock_search.execute.return_value = mock_result

        # Initialize StatisticalSignificance with mock objects
        sig = ps.StatisticalSignificance(task, mock_search)

        # Simulate a single permutation run via the worker
        worker_output = sig._worker(num_qualities=4)

        # Check only the valid quality (0.5) is retained (everything else is filtered out)
        self.assertEqual(worker_output, [0.5], "Worker should filter out non-finite qualities")

    def test_visualization(self):
        # Set up non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        # Generate results and null distribution
        decorator = ps.SignificanceDecorator(ps.BeamSearch(), num_permutations=50)
        result = decorator.execute(self.task)

        # Get test data
        null_dist = decorator.null_distribution
        original_quality = result.to_dataframe()['quality'].iloc[0]

        # Create plot
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ps.plot_null_distribution(null_dist, original_quality, ax=ax)
        
        # Basic element checks
        self.assertGreater(len(ax.patches), 3, "Should have histogram bars") 
        self.assertEqual(len(ax.lines), 2, "Expected KDE line + quality line")
        
        plt.close(fig)
        # plt.show()     # just deactivate non-interactive backend and look at the plot...

if __name__ == '__main__':
    unittest.main()