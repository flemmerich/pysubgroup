import pysubgroup as ps
import numpy as np
from scipy.stats import norm
from .utils import permute

class Significance:
    def __init__(self, task, search_strategy=ps.BeamSearch()):
        self.task = task
        self.search_strategy = search_strategy
        self.null_distribution = None


    def generate_null_distribution(self, num_permutations=1000, num_qualities=1):
        """Generate null distribution.

        Parameters:
            num_permutations (int, optional): Number of subgroup discoveries on permuted dataset. Defaults to 1000.
            num_qualities (int, optional): Number of permutations for null distribution. Defaults to 1 (only the best).

        Returns:
            np.array: Array of quality scores for null distribution.
        """
        baseline = []

        for _ in range(num_permutations):
            null_data = permute(self.task.data, self.task.target.target_selector.attribute_name)
    
            null_task = ps.SubgroupDiscoveryTask(
                null_data,
                self.task.target,
                self.task.search_space,
                qf=self.task.qf,
                result_set_size=num_qualities,
                depth=self.task.depth
            )
            result = self.search_strategy.execute(null_task)

            df = result.to_dataframe()
            # List (qualities) to be filled with qualities or -inf (or 0's?) until full (num_qualities)
            if df.empty:
                qualities = [float('-inf')] * num_qualities
            else:
                qualities = df['quality'].values[:num_qualities].tolist()
                qualities += [float('-inf')] * (num_qualities - len(qualities)) 
                
            baseline.extend(qualities)

        self.null_distribution = np.array(baseline)
        return self.null_distribution
    
    @staticmethod
    def calculate_z_scores(observed_qualities, null_distribution):
        """Calculate Z-scores for observed subgroup qualities"""
        mean_null = np.mean(null_distribution)
        std_null = np.std(null_distribution, ddof=1) # ddof=1 for samples

        # Handle edge case if all null qualities are identical
        # If observed quality > mean_null: z = +inf (p = 0)
        # If observed quality < mean_null: z = -inf (p = 1)
        # If observed quality == mean_null: z = 0 (p = 0.5)
        if std_null == 0:
            return np.where(observed_qualities > mean_null, np.inf, 
                np.where(observed_qualities < mean_null, -np.inf, 0))
        return (observed_qualities - mean_null) / std_null

    @staticmethod
    def calculate_p_values(z_scores):
        """Calculate one-tailed (as extreme or more extreme) p-values."""
        return norm.sf(z_scores)


    def add_statistical_metrics(self, result):
        if self.null_distribution is None:
            raise ValueError("Generate null distribution first.")
        
        observed_qualities = [qual for qual, _, _ in result.results]
        z_scores = self.calculate_z_scores(observed_qualities, self.null_distribution)
        p_values = self.calculate_p_values(z_scores)

        # Store metrics directly on result object
        new_result = ps.SubgroupDiscoveryResult(result.results.copy(), result.task)
        new_result.z_scores = z_scores
        new_result.p_values = p_values

        # Generation of dataframe (monkey-patch)
        original_to_df = new_result.to_dataframe
        def enhanced_to_dataframe():
            df = original_to_df()
            df['z_score'] = new_result.z_scores
            df['p_value'] = new_result.p_values
            return df
        
        new_result.to_dataframe = enhanced_to_dataframe
        return new_result
    
    
class StatsWrapper:
    def __init__(self, search_method, num_permutations=1000):
        """Wrapper that adds statistical significance metrics (z-score, p-value) to subgroup discovery results

        Parameters:
            search_method: Subgroup search strategy (e.g. BeamSearch)
            num_permutations (int, optional): Number of permutations for null distribution. Defaults to 1000.
        """
        self.search_method = search_method
        self.num_permutations = num_permutations
        
    def execute(self, task):
        result = self.search_method.execute(task)
        sig = Significance(task, self.search_method)
        sig.generate_null_distribution(self.num_permutations)
        return sig.add_statistical_metrics(result)


