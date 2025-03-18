import pysubgroup as ps
import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed

class Significance:
    def __init__(self, task, search_strategy=ps.BeamSearch()):
        self.task = task
        self.search_strategy = search_strategy
        self.null_distribution = None


    @staticmethod
    def permute(data, target_attribute):
        """Permute the target column to break associations.

        Parameters:
            data (pd.DataFrame): The dataset to be analyzed.
            target_attribute (pd.Series): The target attribute to permute.

        Returns:
            pd.DataFrame: The dataset with permuted target attribute.
        """
        null_data = data.copy()
        null_data[target_attribute] = np.random.permutation(null_data[target_attribute].values)
        return null_data
    

    def generate_null_distribution(self, num_permutations=1000, num_qualities=1, n_jobs=-1):
        """Generates null distribution. 

        Parallel execution of jobs by passing simple paramters (class references, types, DataFrame)
        to worker. It is necessary to extract all components rather than passing the whole objects
        to avoid problems with pickle.
        
        Parameters:
            num_permutations (int, optional): Number of null hypothesis iterations
            num_qualities (int, optional): Max number of qualities to collect per permutation
            n_jobs (int, optional): Parallel jobs. Defaults to -1 (all cores)
            
        Returns:
            np.ndarray: Flattened array of null distribution qualities
        
        IMPORTANT Pickle Note: Must explicitly pass all parameters needed for task recreation
        to create new instances to ensure thread safety due to pickle limitations.
        """
       
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._worker)(
                # Pass only essential primitives/classes
                self.task.data,
                self.task.target.target_selector.attribute_name,
                self.task.target.target_selector.attribute_value,
                self.task.target.__class__,         # e.g. <class 'pysubgroup.binary_target.BinaryTarget'>
                self.task.qf.__class__,             # e.g. <class 'pysubgroup.binary_target.WRAccQF'>
                self.search_strategy.__class__,     # e.g. <class 'pysubgroup.algorithms.BeamSearch'>
                self.task.depth,
                num_qualities,
                getattr(self.task, 'constraints', None)
                # ignore                            # TODO: how to get the ignore value?
            ) for _ in range(num_permutations)
        )
        self.null_distribution = np.concatenate(results) # results are multiple list -> flatten into single list of qualities
        return self.null_distribution


    @staticmethod
    def _worker(data, target_attr, target_value, target_cls, qf_cls, 
                strategy_cls, depth, num_qualities, constraints=None):
        """Worker function that recreates independent instances in isolation.
        
        Constructs new SubgroupDiscoveryTask objects using only class references and simple
        parameters to ensure pickle safety. Performs permutation and quality calculation.
        
        Parameters:
            data: Dataset
            target_attr: Name of target column
            target_value: Name of target value
            target_cls: Target class reference (not instance)
            qf_cls: Quality function class reference
            strategy_cls: Search strategy class reference
            depth: Search depth parameter
            num_qualities: Number of top qualities to return
            
        Returns:
            list: Quality score (or -inf if no subgroups found)
        """
        null_data = Significance.permute(data, target_attr)
        
        # Recreate objects from class references
        target = target_cls(target_attr, target_value)
        strategy = strategy_cls()
        qf = qf_cls()
        
        task = ps.SubgroupDiscoveryTask(
            null_data,
            target,
            ps.create_selectors(null_data, ignore=[target_attr]),
            qf=qf,
            result_set_size=num_qualities,
            depth=depth,
            constraints=constraints
        )
        
        result = strategy.execute(task)
        df = result.to_dataframe()

        if df.empty:
            return [float('-inf')] * num_qualities
        qualities = df['quality'].values[:num_qualities].tolist()
        return qualities + [float('-inf')] * (num_qualities - len(qualities))


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


