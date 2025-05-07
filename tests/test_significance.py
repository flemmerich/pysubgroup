import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from scipy.stats import gumbel_r
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
        with patch.object(ps.StatisticalSignificance, '_worker', return_value=[0.5, 0.4]): # Force 2 qualities per permutation
            sig = ps.StatisticalSignificance(self.task)
            null_dist = sig.generate_null_distribution(num_permutations=50, num_qualities=2)
            self.assertEqual(len(null_dist), 50 * 2)  # 50 permutations * 2 qualities each

    def test_empty_null_distribution(self):
        """Test for scenarios where all permutations yield non-finite qualities."""
        with patch.object(ps.StatisticalSignificance, '_worker', return_value=[]):
            sig = ps.StatisticalSignificance(self.task)
            null_dist = sig.generate_null_distribution(num_permutations=10)
            self.assertEqual(len(null_dist), 0)
            
            base_result = ps.BeamSearch().execute(self.task)
            sig.null_distribution = null_dist
            with self.assertRaises(ValueError):
                sig.add_metrics_to_result(base_result)  # Should raise due to empty null
        
    def test_result_metrics_inclusion(self):
        """Test p-value calculation when normality is passed and test multiple testing correction."""
        with patch.object(ps.StatisticalSignificance, '_check_normality', return_value=True):
            decorator = ps.Stats(ps.BeamSearch(), num_permutations=50, adjust_method='holm')
            result = decorator.execute(self.task)
            df = result.to_dataframe()
            
            self.assertIn('z_score', df.columns)
            self.assertIn('p_value', df.columns)
            self.assertIn('p_value_adj', df.columns)
            
            self.assertTrue((df['p_value'] >= 0).all() and (df['p_value'] <= 1).all())
            if not df.empty:
                self.assertTrue((df['p_value_adj'] >= df['p_value']).all())

    def test_empirical_p_values_non_normal(self):
        """Test empirical p-value calculation when normality failed."""
        with patch.object(ps.StatisticalSignificance, '_check_normality', return_value=False):
            decorator = ps.Stats(ps.BeamSearch(), num_permutations=100)
            result = decorator.execute(self.task)
            df = result.to_dataframe()
            
            # Should have p_values but no z-scores column
            self.assertIn('p_value', df.columns)
            self.assertNotIn('z_score', df.columns) 

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
        
        decorator = ps.Stats(ps.BeamSearch(), num_permutations=50)
        result = decorator.execute(task)       
        self.assertEqual(len(result.results), 0, "Result list should be empty")
        self.assertEqual(len(result.to_dataframe()), 0, "No rows in dataframe")

    def test_non_finite_quality_filtering_in_worker(self):
        """Test that non-finite qualities are filtered out in the worker."""
        # Mock result with non-finite qualities
        mock_search = MagicMock()
        mock_result = MagicMock()
        mock_result.results = [
            (np.inf, 'selector1', 'desc1'), # Infinite
            (np.nan, 'selector2', 'desc2'), # NaN
            (0.5, 'selector3', 'desc3'),    # Valid
            (-np.inf, 'selector4', 'desc4') # Negative infinite
        ]
        mock_search.execute.return_value = mock_result
        
        sig = ps.StatisticalSignificance(self.task, mock_search)
        worker_output = sig._worker(num_qualities=4)
        self.assertEqual(worker_output, [0.5])

    def test_visualization(self):
        # Set up non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        # Generate results and null distribution
        decorator = ps.Stats(ps.BeamSearch(), num_permutations=50)
        result = decorator.execute(self.task)

        # Get test data
        null_dist = decorator.null_distribution
        original_quality = result.to_dataframe()['quality'].iloc[0]

        if len(null_dist) == 0:
            self.skipTest("Null distribution is empty") 
            
        # Create plot
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ps.plot_null_distribution(null_dist, original_quality, ax=ax)
        
        self.assertIsInstance(fig, plt.Figure)
        
        plt.close(fig)

    def test_anderson_gumbel_r_rejection(self):
        """Test Anderson-Darling correctly rejects non-Gumbel data."""
        # Generate normal data (which should fail Gumbel R test)
        sig = ps.StatisticalSignificance(self.task)
        sig.null_distribution = np.random.normal(0, 1, 1000)
        
        # Check Gumbel R test fails
        self.assertFalse(sig._check_gumbel_r(alpha=0.05))

    def test_invalid_alpha_gumbel(self):
        """Test error raising for unsupported alpha in Gumbel R check."""
        sig = ps.StatisticalSignificance(self.task)
        sig.null_distribution = gumbel_r.rvs(size=100)
        
        with self.assertRaises(ValueError):
            sig._check_gumbel_r(alpha=0.2)  # 0.2 not in [0.15, 0.10, ...]

    def test_gumbel_r_branch_computes_correct_pvalues(self):
        """When null_distribution is Gumbel, _check_gumbel_r must pass and p‐values match gumbel_r.sf."""
        # Generate a pure Gumbel-R null distribution and fit μ, β
        null_dist = gumbel_r.rvs(loc=5.0, scale=2.0, size=500, random_state=0)
        mu_hat, beta_hat = gumbel_r.fit(null_dist)

        # Pick a test quality q and compute its expected standardized and p‐value
        q = mu_hat + 1.23 * beta_hat
        expected_std = (q - mu_hat) / beta_hat
        expected_p = gumbel_r.sf(expected_std)

        fake_sdr = ps.SubgroupDiscoveryResult([(q, None, None)], self.task)

        # Run add_metrics_to_result on a StatisticalSignificance carrying our null
        sig = ps.StatisticalSignificance(self.task)
        sig.null_distribution = null_dist
        out = sig.add_metrics_to_result(fake_sdr, alpha=0.05, adjust_method=None)

        self.assertTrue(sig._check_gumbel_r(alpha=0.05),
                        "Gumbel R fit + Anderson-Darling should pass on pure Gumbel data")
        self.assertIsNotNone(out.standardized_values,
                             "standardized_values should be set when Gumbel branch is used")
        # Only one subgroup
        self.assertAlmostEqual(out.standardized_values[0], expected_std, places=6)
        self.assertAlmostEqual(out.p_values[0], expected_p, places=6)

    def test_fallback_to_empirical_when_not_gumbel(self):
        """If Gumbel‐R test fails, p‐values should be empirical (rank‐based)."""
        # Create a null that's not Gumbel, e.g. uniform
        rng = np.random.RandomState(1)
        null_dist = rng.uniform(-1, 1, size=200)

        # Choose an observed quality q
        q = 0.3
        emp_p = (np.sum(null_dist >= q) + 1) / (len(null_dist) + 1)

        fake_sdr = ps.SubgroupDiscoveryResult([(q, None, None)], self.task)

        sig = ps.StatisticalSignificance(self.task)
        sig.null_distribution = null_dist

        # Ensure Gumbel-R check fails
        self.assertFalse(sig._check_gumbel_r(alpha=0.05))
        out = sig.add_metrics_to_result(fake_sdr, alpha=0.05, adjust_method=None)

        self.assertIsNone(out.standardized_values,
                          "standardized_values must be None when using empirical branch")
        self.assertAlmostEqual(out.p_values[0], emp_p, places=6)

    def test_result_metrics_inclusion(self):
        """Test p-value calculation and column names."""
        with patch.object(ps.StatisticalSignificance, '_check_normality', return_value=True):
            decorator = ps.Stats(ps.BeamSearch(), num_permutations=50, adjust_method='holm')
            result = decorator.execute(self.task)
            df = result.to_dataframe()
            
            # Check for correct column name
            self.assertIn('standardized_value', df.columns)
            self.assertIn('p_value', df.columns)
            self.assertIn('p_value_adj', df.columns)
            
            # Validate p-value ranges
            self.assertTrue((df['p_value'] >= 0).all() and (df['p_value'] <= 1).all())
            if not df.empty:
                self.assertTrue((df['p_value_adj'] >= df['p_value']).all())

if __name__ == '__main__':
    unittest.main()
