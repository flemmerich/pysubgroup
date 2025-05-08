import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from scipy.stats import gumbel_r
import pysubgroup as ps
from pysubgroup.datasets import get_credit_data

class TestStatisticalSignificance(unittest.TestCase):
    def setUp(self):
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
        with patch.object(ps.StatisticalSignificance, '_worker', return_value=[0.5, 0.4]):
            sig = ps.StatisticalSignificance(self.task)
            null_dist = sig.generate_null_distribution(num_permutations=50, num_qualities=2)
            self.assertEqual(len(null_dist), 50 * 2)

    def test_empty_null_distribution(self):
        with patch.object(ps.StatisticalSignificance, '_worker', return_value=[]):
            sig = ps.StatisticalSignificance(self.task)
            null_dist = sig.generate_null_distribution(num_permutations=10)
            self.assertEqual(len(null_dist), 0)
            
            base_result = ps.BeamSearch().execute(self.task)
            sig.null_distribution = null_dist
            with self.assertRaises(ValueError):
                sig.add_metrics_to_result(base_result)

    def test_result_metrics_inclusion(self):
        with patch.object(ps.StatisticalSignificance, '_check_normality', return_value=True):
            decorator = ps.Stats(ps.BeamSearch(), num_permutations=50, adjust_method='holm')
            result = decorator.execute(self.task)
            df = result.to_dataframe()
            
            self.assertIn('p_value_normal', df.columns)
            self.assertIn('p_value_adj', df.columns)
            self.assertTrue((df['p_value_normal'] >= 0).all() & (df['p_value_normal'] <= 1).all())
            if not df.empty:
                self.assertTrue((df['p_value_adj'] >= df['p_value_normal']).all())

    def test_empirical_p_values_non_normal(self):
        with patch.object(ps.StatisticalSignificance, '_check_normality', return_value=False):
            decorator = ps.Stats(ps.BeamSearch(), num_permutations=100)
            result = decorator.execute(self.task)
            df = result.to_dataframe()
            
            # Expect p_value_gumbel column instead of normal p-values when non-normal
            self.assertIn('p_value_gumbel', df.columns)
            self.assertNotIn('p_value_normal', df.columns)

    def test_non_finite_quality_handling(self):
        task = ps.SubgroupDiscoveryTask(
            self.data,
            self.target,
            search_space=self.search_space,
            qf=ps.WRAccQF(),
            result_set_size=5,
            depth=2,
            min_quality=1.0
        )
        decorator = ps.Stats(ps.BeamSearch(), num_permutations=50)
        result = decorator.execute(task)       
        self.assertEqual(len(result.results), 0)
        self.assertEqual(len(result.to_dataframe()), 0)

    def test_non_finite_quality_filtering_in_worker(self):
        mock_search = MagicMock()
        mock_result = MagicMock()
        mock_result.results = [
            (np.inf, 'selector1', 'desc1'),
            (np.nan, 'selector2', 'desc2'),
            (0.5, 'selector3', 'desc3'),
            (-np.inf, 'selector4', 'desc4')
        ]
        mock_search.execute.return_value = mock_result
        
        sig = ps.StatisticalSignificance(self.task, mock_search)
        worker_output = sig._worker(num_qualities=4)
        self.assertEqual(worker_output, [0.5])

    def test_visualization(self):
        import matplotlib
        matplotlib.use('Agg')
        
        decorator = ps.Stats(ps.BeamSearch(), num_permutations=50)
        result = decorator.execute(self.task)
        null_dist = decorator.null_distribution
        original_quality = result.to_dataframe()['quality'].iloc[0] if not result.to_dataframe().empty else 0

        if len(null_dist) == 0:
            self.skipTest("Null distribution is empty")
            
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        ps.plot_null_distribution(null_dist, original_quality, ax=ax)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_anderson_gumbel_r_rejection(self):
        sig = ps.StatisticalSignificance(self.task)
        sig.null_distribution = np.random.normal(0, 1, 1000)
        self.assertFalse(sig._check_gumbel_r(alpha=0.05))

    def test_invalid_alpha_gumbel(self):
        sig = ps.StatisticalSignificance(self.task)
        sig.null_distribution = gumbel_r.rvs(size=100)
        with self.assertRaises(ValueError):
            sig._check_gumbel_r(alpha=0.20)

    def test_gumbel_r_branch_computes_correct_pvalues(self):
        null_dist = gumbel_r.rvs(loc=5.0, scale=2.0, size=500, random_state=0)
        mu_hat, beta_hat = gumbel_r.fit(null_dist)
        q = mu_hat + 1.23 * beta_hat
        expected_p = gumbel_r.sf(q, loc=mu_hat, scale=beta_hat)

        fake_sdr = ps.SubgroupDiscoveryResult([(q, None, None)], self.task)
        sig = ps.StatisticalSignificance(self.task)
        sig.null_distribution = null_dist
        out = sig.add_metrics_to_result(fake_sdr, alpha=0.05, adjust_method=None)

        self.assertTrue(sig._check_gumbel_r(alpha=0.05))
        self.assertAlmostEqual(out.p_values[0], expected_p, places=6)

    def test_fallback_to_empirical_when_not_gumbel(self):
        rng = np.random.RandomState(1)
        null_dist = rng.uniform(-1, 1, size=200)
        q = 0.3
        emp_p = (np.sum(null_dist >= q) + 1) / (len(null_dist) + 1)

        fake_sdr = ps.SubgroupDiscoveryResult([(q, None, None)], self.task)
        sig = ps.StatisticalSignificance(self.task)
        sig.null_distribution = null_dist

        self.assertFalse(sig._check_gumbel_r(alpha=0.05))
        out = sig.add_metrics_to_result(fake_sdr, alpha=0.05, adjust_method=None)
        self.assertAlmostEqual(out.p_values[0], emp_p, places=6)

if __name__ == '__main__':
    unittest.main()
