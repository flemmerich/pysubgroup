import pysubgroup as ps
from pysubgroup.utils import SubgroupDiscoveryResult     # import separately to prevent loop
from pysubgroup.subgroup_description import SelectorBase # import separately to prevent loop
import numpy as np
from scipy.stats import shapiro, norm, anderson
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

# TODO: adaptive permutation strategies (e.g. stop early if p-value is clearly not significant).

# -------------------------------------------------------------------------------------------
# Monkey-patch for __new__ and __getnewargs_ex__ of SelectorBase - needed for parallelization
# -------------------------------------------------------------------------------------------

# Store original implementations
_original_new = SelectorBase.__new__
_original_getnewargs_ex = SelectorBase.__getnewargs_ex__

def _patched_new(cls, *args, **kwargs):
    tmp = _original_new(cls, *args, **kwargs)
    tmp.set_descriptions(*args, **kwargs)
    tmp.__new_args__ = args, kwargs
    
    if tmp not in SelectorBase.__refs__:
        return tmp
        
    for ref in SelectorBase.__refs__:
        if ref == tmp:
            if not hasattr(ref, '__new_args__'):
                ref.__new_args__ = tmp.__new_args__
            return ref
    return tmp

def _patched_getnewargs_ex(self):
    return getattr(self, '__new_args__', ((), {}))

# Apply the patches
SelectorBase.__new__ = _patched_new
SelectorBase.__getnewargs_ex__ = _patched_getnewargs_ex

# -------------------------------------------------------------------------------------------
# End of patch. Changed only changed few loc. Wouldn't presume to change module code directly.
# -------------------------------------------------------------------------------------------


class StatisticalSignificance:
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

        Parameters:
            num_permutations (int, optional): Number of null hypothesis iterations
            num_qualities (int, optional): Max number of qualities to collect per permutation
            n_jobs (int, optional): Parallel jobs. Defaults to -1 (all cores)
            store (boolean, optional): Parameter to assure null distribution is saved but extended one is not.
            
        Returns:
            np.ndarray: Flattened array of null distribution qualities
        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._worker)(num_qualities) for _ in range(num_permutations)
        ) 
        null_dist = np.concatenate(results)
        if store: 
            self.null_distribution = null_dist
        return null_dist


    def _worker(self, num_qualities):
        """Create permuted data based on target type and perform subgroup discovery."""
        target = self.task.target
        if isinstance(target, ps.BinaryTarget):
            target_attr = target.target_selector.attribute_name
            permuted_data = self.permute(self.task.data, target_attr)
        elif isinstance(target, ps.NumericTarget):
            target_attr = target.target_variable
            permuted_data = self.permute(self.task.data, target_attr)
        elif isinstance(target, ps.FITarget):
            # For frequent itemsets, shuffle entire dataset rows
            permuted_data = self.task.data.sample(frac=1).reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")

        new_task = ps.SubgroupDiscoveryTask(
            data=permuted_data,
            target=self.task.target,
            search_space=self.task.search_space,
            qf=self.task.qf,
            result_set_size=num_qualities,
            constraints=self.task.constraints
        )
        
        result = self.search_strategy.execute(new_task)
        return [q for q, _, _ in result.results]
        #return [q for q, _, _ in result.results if np.isfinite(q)] # Filter non-finite. Required for normality test and p-value
    
    # shapiro, anderson, and multipletests require finite inputs.
    def add_metrics_to_result(self, result, alpha=0.05, adjust_method=None):
        """Add metrics to result object

        Args:
            result (SubgroupDiscoveryResult): The SubgroupDiscoveryResult object
            alpha (float, optional): Significance level for normality test and p-value threshold. Defaults to 0.05.
            adjust_method (optional): If provided, performs multiple testing correction at this alpha level.

        Raises:
            ValueError: If self.null_distribution not exists.

        Returns:
            SignificantSubgroupResult: 
        """
        if self.null_distribution is None:
            raise ValueError("Generate null distribution first")

        #observed = [q for q, _, _ in result.results]
        observed = [q if np.isfinite(q) else np.nan for q, _, _ in result.results]
        valid_observed = [q for q in observed if not np.isnan(q)]
        
        if self._check_normality(alpha):
            z_scores = self.calculate_z_scores(valid_observed, self.null_distribution)
            p_values = self.calculate_p_values(z_scores)
        else:
            extended_null = self.generate_null_distribution(
                num_permutations=3*len(self.null_distribution), # take 3 times as much for empirical calculation
                num_qualities=1,
                n_jobs=-1,
                store=False # prevents self.null_distribution from overwriting
            )
            p_values = self.empirical_p_values(valid_observed, extended_null)
            z_scores = None

        # Apply multiple testing correction if specified
        adjusted_p_values = None
        if adjust_method is not None:
            adjusted_p_values = self.adjust_p_values(
                p_values, 
                method=adjust_method,
                alpha=alpha
            )
        return SignificantSubgroupResult(
                result.results.copy(), 
                result.task,
                z_scores,
                p_values,
                adjusted_p_values
            )
    

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
        """Calculate empirical p-values using larger null distribution and Laplace smoothing to avoid p=0."""
        return [
            (np.sum(null_distribution >= q) + 1) / (len(null_distribution) + 1)
            for q in observed_qualities
        ]
  

    @staticmethod
    def adjust_p_values(p_values, method='holm', alpha=0.05):
        """
        Apply multiple testing correction to p-values.
        Possible methods: bonferroni, holm, fdr_bh
        
        Parameters:
            p_values (list): List of p-values to adjust
            method (str): Correction method (see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)
            alpha (float): Significance level for normality test and p-value threshold. Defaults to 0.05.
            
        Returns:
            list: Adjusted p-values
        """
        # Type cast for efficiency and operations
        pvals = np.array(p_values)
        
        valid_mask = np.isfinite(pvals)
        if not np.all(valid_mask):
            pvals[~valid_mask] = 1.0 # set non-finite values to 1.0 (non-significant)

        _, pvals_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
        
        return pvals_adj.tolist()


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


class SignificantSubgroupResult(SubgroupDiscoveryResult):
    """Enhanced result class with statistical metrics"""
    def __init__(self, results, task, z_scores, p_values, adj_p_values=None):
        super().__init__(results, task)
        self.z_scores = z_scores
        self.p_values = p_values
        self.adj_p_values = adj_p_values

    def to_dataframe(self):
        df = super().to_dataframe()
        if self.z_scores is not None:
            df['z_score'] = self.z_scores
        df['p_value'] = self.p_values
        if self.adj_p_values is not None:
            df['p_value_adj'] = self.adj_p_values
        return df
        

class SignificanceDecorator:
    def __init__(self, search_method, num_permutations=1000, adjust_method='holm', alpha=0.05, n_jobs=-1):
        """Wrapper that adds statistical significance metrics (z-score, p-value) to subgroup discovery results

        Parameters:
            search_method: Subgroup search strategy (e.g. BeamSearch)
            num_permutations (int, optional): Number of permutations for null distribution. Defaults to 1000.
            adjust_method (str, optional): Normality test performed. Default to 'holm'.
            alpha (float, optional): Significance level for normality test and p-value threshold.
            n_jobs (int, optional): Parallel(-1, all cores) or Serial(1). Defaults to -1.
        """
        self.search_method = search_method
        self.num_permutations = num_permutations
        self.adjust_method = adjust_method
        self.alpha = alpha
        self.n_jobs = n_jobs

    def execute(self, task):
        # Run original search
        base_result = self.search_method.execute(task)
        
        # Calculate significance metrics
        significance = StatisticalSignificance(task, self.search_method)
        significance.generate_null_distribution(
            num_permutations=self.num_permutations,
            num_qualities=1,
            n_jobs=self.n_jobs
        )
        
        # Return enhanced result
        return significance.add_metrics_to_result(
            base_result,
            alpha=self.alpha,
            adjust_method=self.adjust_method
        )
