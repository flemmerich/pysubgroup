import pysubgroup as ps
import numpy as np
from scipy.stats import shapiro, norm, anderson
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
    

    def generate_null_distribution(self, num_permutations=1000, num_qualities=1, n_jobs=-1, store=True):
        """Generates null distribution. 

        Parallel execution of jobs by passing simple paramters (class references, types, DataFrame)
        to worker. It is necessary to extract all components rather than passing the whole objects
        to avoid problems with pickle.
        
        Parameters:
            num_permutations (int, optional): Number of null hypothesis iterations
            num_qualities (int, optional): Max number of qualities to collect per permutation
            n_jobs (int, optional): Parallel jobs. Defaults to -1 (all cores)
            store (boolean, optional): Parameter to assure null distribution is saved but extended one is not.
            
        Returns:
            np.ndarray: Flattened array of null distribution qualities
        
        IMPORTANT Pickle Note: Must explicitly pass all parameters needed for task recreation
        to create new instances to ensure thread safety due to pickle limitations.
        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._worker)(
                self.task.data,
                self.task.target.target_selector.attribute_name,
                self.task.target.target_selector.attribute_value,
                self.task.target.__class__,
                self.task.qf.__class__,
                self.search_strategy.__class__,
                self.task.depth,
                num_qualities,
                # getattr(self.task, 'constraints', None) 
                # TODO: 'ignore' and other parameter are still missing. Don't know how to access them...
            ) for _ in range(num_permutations)
        ) 
        null_dist = np.concatenate(results) # Flattens lists
        if store: 
            self.null_distribution = null_dist 
        return null_dist

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
            list: Quality score
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
        qualities = df['quality'].values.tolist() if not df.empty else []
        return [q for q in qualities if np.isfinite(q)] # Filter non-finite. Required for normality test and p-value


    @staticmethod
    def calculate_z_scores(observed_qualities, null_distribution):
        """Calculate Z-scores for observed subgroup qualities"""
        if len(null_distribution) == 0:
            return np.full_like(observed_qualities, np.nan)
        
        mean_null = np.mean(null_distribution)
        std_null = np.std(null_distribution, ddof=1) # ddof=1 for samples

        return (observed_qualities - mean_null) / std_null

    @staticmethod
    def calculate_p_values(z_scores):
        """Calculate one-tailed (as extreme or more extreme) p-values."""
        return norm.sf(z_scores)


    @staticmethod
    def empirical_p_values(observed_qualities, null_distribution):
        """Calculate empirical p-values using extended null distribution and Laplace smoothing."""
        return [
            (np.sum(null_distribution >= q) + 1) / (len(null_distribution) + 1)
            for q in observed_qualities
        ]


    def add_statistical_metrics(self, result, alpha=0.05):
        if self.null_distribution is None:
            raise ValueError("Generate null distribution first")

        observed = [q for q, _, _ in result.results]
        
        if self._check_normality(alpha):
            z_scores = self.calculate_z_scores(observed, self.null_distribution)
            p_values = self.calculate_p_values(z_scores)
        else:
            extended_null = self.generate_null_distribution(
                num_permutations=5*len(self.null_distribution),
                num_qualities=1,
                n_jobs=-1,
                store=False # prevents self.null_distribution from overwriting
            )
            p_values = self.empirical_p_values(observed, extended_null)
            z_scores = None

        return self._extend_result(result, z_scores, p_values)


    def _check_normality(self, alpha=0.05):
        """Checks for normality using Shapiro or Anderson."""
        # Filter non-finite. Required for normality test and p-value.
        finite_null = self.null_distribution[np.isfinite(self.null_distribution)]
        if len(finite_null) < 3:  # Shapiro-Wilk requires min 3 samples
            return False

        # Shapiro-Wilk for small samples (inaccurate p-value for N > 5000) 
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
        if len(finite_null) <= 5000:
            _, p = shapiro(finite_null)
            return p >= alpha
        else:
            # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html#scipy.stats.anderson)
            # Anderson only supports fixed alphas: [0]:15%, [1]:10%, [2]:5%, [3]:2.5%, [4]:1%
            result = anderson(finite_null)
            return result.statistic < result.critical_values[2] # [2]:5%


    def _extend_result(self, result, z_scores, p_values):
        new_result = ps.SubgroupDiscoveryResult(result.results.copy(), result.task)
        new_result.z_scores = z_scores
        new_result.p_values = p_values
        
        # Monkey-patch dataframe
        original_to_df = new_result.to_dataframe
        def patched_to_dataframe():
            df = original_to_df()
            if z_scores is not None:
                df['z_score'] = new_result.z_scores
                df['p_value'] = new_result.p_values
            else:
                df['p_value'] = new_result.p_values
            return df
        new_result.to_dataframe = patched_to_dataframe
        return new_result

    
class StatsWrapper:
    def __init__(self, search_method, num_permutations=1000, n_jobs=-1):
        """Wrapper that adds statistical significance metrics (z-score, p-value) to subgroup discovery results

        Parameters:
            search_method: Subgroup search strategy (e.g. BeamSearch)
            num_permutations (int, optional): Number of permutations for null distribution. Defaults to 1000.
            n_jobs (int, optional): Parallel(-1, all cores) or Serial(1). Defaults to -1.
        """
        self.search_method = search_method
        self.num_permutations = num_permutations
        self.n_jobs = n_jobs
        self.sig = None # store Significance instance fr acces to null distribution (in test)

    def execute(self, task):
        result = self.search_method.execute(task)
        self.sig = Significance(task, self.search_method) # store instance
        # Generate initial null distribution with 1 quality per permutation
        self.sig.generate_null_distribution(
            num_permutations=self.num_permutations, 
            num_qualities=1,
            n_jobs=self.n_jobs)
        return self.sig.add_statistical_metrics(result)


