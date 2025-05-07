import pysubgroup as ps
from pysubgroup.utils import SubgroupDiscoveryResult     # import separately to prevent loop
from pysubgroup.subgroup_description import SelectorBase # import separately to prevent loop
import numpy as np
from scipy.stats import shapiro, norm, gumbel_r, anderson
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

# TODO: adaptive permutation strategies (e.g. stop early if p-value is clearly not significant).

# -------------------------------------------------------------------------------------------
# Monkey-patch for __new__ and __getnewargs_ex__ of SelectorBase for serialization
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
# End of patch. Changed only few loc. 
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
            target_attribute (str): The name of the target column to permute.

        Returns:
            pd.DataFrame: The dataset with permuted target attribute.
        """
        null_data = data.copy()
        null_data[target_attribute] = np.random.permutation(null_data[target_attribute].values)
        return null_data


    @staticmethod
    def column_permutation(data):
        """Permute each column independently. Used for frequent itemsets to test item independence."""
        return data.apply(np.random.permutation, axis=0)


    def generate_null_distribution(self, num_permutations=1000, num_qualities=1, n_jobs=-1):
        """Generates null distribution. 

        Parameters:
            num_permutations (int, optional): Number of permutations. Defaults to 1000.
            num_qualities (int, optional): Max number of top qualities to collect per permutation. Defaults to 1.
            n_jobs (int, optional): Parallel jobs. Defaults to -1 (all cores)
            
        Returns:
            np.ndarray: Flattened array of null distribution qualities
        """
        with tqdm_joblib(desc="Generating null distribution", total=num_permutations):
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._worker)(num_qualities) for _ in range(num_permutations)
            )

        if any(results):
            null_dist = np.concatenate(results)
        else:
            null_dist = np.array([])

        self.null_distribution = null_dist
        return null_dist

    def _worker(self, num_qualities):
        """Create permuted data based on target type and perform subgroup discovery."""
        target = self.task.target
        original_data = self.task.data

        if isinstance(target, ps.BinaryTarget):
            target_attr = target.target_selector.attribute_name
            permuted_data = self.permute(original_data, target_attr)
        elif isinstance(target, ps.NumericTarget):
            target_attr = target.target_variable
            permuted_data = self.permute(original_data, target_attr)
        elif isinstance(target, ps.FITarget):
            permuted_data = self.column_permutation(original_data) 
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")

        new_task = ps.SubgroupDiscoveryTask(
            data=permuted_data,
            target=self.task.target,
            search_space=self.task.search_space,
            qf=self.task.qf,
            result_set_size=num_qualities,
            depth=self.task.depth,
            min_quality=self.task.min_quality,
            constraints=self.task.constraints
        )
        
        result = self.search_strategy.execute(new_task)
        return [q for q, _, _ in result.results if np.isfinite(q)] # Filter non-finite. Required for normality test and p-value
    

    def add_metrics_to_result(self, result, alpha=0.05, adjust_method=None):
        """Adds statistical significance metrics to subgroup discovery results.

        Args:
            result (SubgroupDiscoveryResult): Result object from subgroup discovery.
            alpha (float): Significance level for distribution tests. Defaults to 0.05.
            adjust_method (str, optional): Multiple testing correction method (e.g., 'holm', 'fdr_bh').

        Returns:
            SignificantSubgroupResult: Enhanced result with:
                - `standardized_values`: Z-scores (normal) or (x-μ)/β (Gumbel R)
                - `p_values`: One-tailed p-values based on detected distribution
                - `adj_p_values`: Adjusted p-values (if `adjust_method` specified)

        Raises:
            ValueError: If `alpha` is invalid for Anderson-Darling tests.
        """
        # Early return if there are no subgroups to analyze (base_result is empty)
        if not result.results:
            return SignificantSubgroupResult([], result.task, None, [], None)
        
        if self.null_distribution is None or len(self.null_distribution) == 0:
            raise ValueError("Null distribution is empty or not generated")

        # Extract observed qualities
        observed_qualities = np.asarray([q for q, _, _ in result.results])

        is_normal = self._check_normality(alpha)
        is_gumbel_r = self._check_gumbel_r(alpha)

        if is_normal:
            z_scores = self.calculate_z_scores(observed_qualities, self.null_distribution)
            p_values = self.calculate_p_values(z_scores)
            standardized_values = z_scores
        elif is_gumbel_r:
            mu, beta = gumbel_r.fit(self.null_distribution)
            standardized_values = (observed_qualities - mu) / beta
            p_values = gumbel_r.sf(standardized_values)
            z_scores = None  # Avoid confusion with normal z-scores
        else:
            p_values = self.empirical_p_values(observed_qualities, self.null_distribution)
            standardized_values = None

        # Apply multiple testing correction
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
                standardized_values,
                p_values,
                adjusted_p_values
            )
    

    @staticmethod
    def calculate_z_scores(observed_qualities, null_distribution):
        """Calculate Z-scores for observed subgroup qualities"""
        mean_null = np.mean(null_distribution)
        std_null = np.std(null_distribution, ddof=1) # ddof=1 for samples
        return (observed_qualities - mean_null) / std_null


    @staticmethod
    def calculate_p_values(z_scores):
        """Calculate one-tailed (as extreme or more extreme) p-values."""
        return norm.sf(z_scores)


    @staticmethod
    def empirical_p_values(observed_qualities, null_distribution):
        """Calculate empirical p-values using Laplace smoothing to avoid p=0."""
        return [
            (np.sum(null_distribution >= q) + 1) / (len(null_distribution) + 1)
            if np.isfinite(q) else 1.0 # set non-finite values to 1.0 (non-significant)
            for q in observed_qualities
        ]
  

    @staticmethod
    def adjust_p_values(p_values, method='holm', alpha=0.05):
        """
        Apply multiple testing correction to p-values.
        Possible methods: 'bonferroni', 'holm', 'fdr_bh' (see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)
        
        Parameters:
            p_values (list): List of p-values to adjust
            method (str): Correction method
            alpha (float): Significance level for normality test and p-value threshold. Defaults to 0.05.
            
        Returns:
            list: Adjusted p-values
        """
        pvals = np.array(p_values)
        if pvals.size == 0:
            return []
            
        valid_mask = np.isfinite(pvals)
        if not np.all(valid_mask):
            pvals[~valid_mask] = 1.0 # set non-finite values to 1.0 (non-significant)

        _, pvals_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
        
        return pvals_adj


    def _check_normality(self, alpha=0.05):
        """Checks if the null distribution follows a normal distribution using Shapiro-Wilk (n ≤ 5000) 
        or Anderson-Darling (n > 5000).

        Parameters:
            alpha (float): Significance level. For Anderson-Darling, must be one of [0.15, 0.10, 0.05, 0.025, 0.01].

        Returns:
            bool: True if the null distribution appears normal at the given significance level.

        Raises:
            ValueError: If `alpha` is not supported for the Anderson-Darling test.
        """
        if len(self.null_distribution) < 3:  # Shapiro-Wilk requires min 3 samples
            return False

        # Shapiro-Wilk for smaller samples (accurate p-value for N <= 5000) 
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
        if len(self.null_distribution) <= 5000:
            _, p = shapiro(self.null_distribution)
            return p >= alpha
        
        # Anderson-Darling for larger datasets (N > 5000)
        # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html#scipy.stats.anderson)
        supported_alphas = [0.15, 0.10, 0.05, 0.025, 0.01]
        if alpha not in supported_alphas:
            raise ValueError(
                f"Alpha must be one of {supported_alphas} for Anderson-Darling test. Got {alpha}."
            )
        
        result = anderson(self.null_distribution)
        idx = supported_alphas.index(alpha)
        return result.statistic < result.critical_values[idx]


    def _check_gumbel_r(self, alpha=0.05):
        """Checks if the null distribution follows a Gumbel R distribution using the Anderson-Darling test.

        Parameters:
            alpha (float): Significance level. Must be one of [0.15, 0.10, 0.05, 0.025, 0.01].

        Returns:
            bool: True if the standardized null distribution passes the Anderson-Darling test for Gumbel R.

        Raises:
            ValueError: If `alpha` is not in the list of supported significance levels.
        """
        data = self.null_distribution
        if len(data) < 2:
            return False

        # Fit parameters and standardize
        try:
            mu, beta = gumbel_r.fit(data)
            standardized_data = (data - mu) / beta
        except:
            return False

        # Validate alpha
        supported_alphas = [0.15, 0.10, 0.05, 0.025, 0.01]
        if alpha not in supported_alphas:
            raise ValueError(
                f"Alpha must be one of {supported_alphas} for Gumbel R test. Got {alpha}."
            )

        # Anderson-Darling test for Gumbel R
        try:
            result = anderson(standardized_data, dist='gumbel_r')
        except ValueError:
            return False

        idx = supported_alphas.index(alpha)
        return result.statistic < result.critical_values[idx]


class SignificantSubgroupResult(SubgroupDiscoveryResult):
    """Subgroup discovery result enhanced with statistical significance metrics.

    Attributes:
        standardized_values (list[float]): 
            - For normal distribution: Z-scores relative to null distribution.
            - For Gumbel R: Standardized values (x-μ)/β using fitted parameters.
            - None if empirical p-values are used.
        p_values (list[float]): One-tailed p-values (normal/Gumbel survival function or empirical).
        adj_p_values (list[float]): Adjusted p-values after multiple testing correction.
    """
    def __init__(self, results, task, standardized_values, p_values, adj_p_values=None):
        super().__init__(results, task)
        self.standardized_values = standardized_values 
        self.p_values = p_values
        self.adj_p_values = adj_p_values

    def to_dataframe(self):
        """Converts results to a DataFrame with added statistical columns:
            - `standardized_value`: See class attribute documentation.
            - `p_value`: Raw significance level.
            - `p_value_adj`: Adjusted p-value (if applicable).
        """
        df = super().to_dataframe()
        if self.standardized_values is not None:
            df['standardized_value'] = self.standardized_values
        df['p_value'] = self.p_values
        if self.adj_p_values is not None:
            df['p_value_adj'] = self.adj_p_values
        return df


class Stats:
    def __init__(self, search_method, num_permutations=1000, num_qualities=1, adjust_method='holm', alpha=0.05, n_jobs=-1):
        """Wrapper that adds statistical significance metrics (z-score, p-value) to subgroup discovery results

        Parameters:
            search_method: Subgroup search strategy (e.g. BeamSearch)
            num_permutations (int, optional): Number of permutations for null distribution. Defaults to 1000.
            num_qualities (int, optional): Max number of top qualities to collect per permutation. Defaults to 1.
            adjust_method (str, optional): Method for multiple testing correction (e.g., 'holm', 'bonferroni', 'fdr_bh'). Defaults to 'holm'.
            alpha (float, optional): Significance level for normality test and p-value threshold. Defaults to 0.05.
            n_jobs (int, optional): Parallel(-1, all cores) or Serial(1, num of cores). Defaults to -1.
        """
        self.search_method = search_method
        self.num_permutations = num_permutations
        self.num_qualities = num_qualities
        self.adjust_method = adjust_method
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.null_distribution = None

    def execute(self, task):
        # Run original search
        base_result = self.search_method.execute(task)
        
        # Calculate significance metrics
        significance = StatisticalSignificance(task, self.search_method)
        self.null_distribution = significance.generate_null_distribution(
            num_permutations=self.num_permutations,
            num_qualities=self.num_qualities,
            n_jobs=self.n_jobs,
        )
        
        # Return enhanced result
        return significance.add_metrics_to_result(
            base_result,
            alpha=self.alpha,
            adjust_method=self.adjust_method
        )
    