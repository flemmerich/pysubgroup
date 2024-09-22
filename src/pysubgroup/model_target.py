from collections import namedtuple

import numpy as np

import pysubgroup as ps

# Define a named tuple to store regression parameters and subgroup size
beta_tuple = namedtuple("beta_tuple", ["beta", "size_sg"])


class EMM_Likelihood(ps.AbstractInterestingnessMeasure):
    """Exceptional Model Mining likelihood-based interestingness measure.

    This class computes the difference in likelihoods between a subgroup model
    and the inverse (complement) model, providing a measure of how exceptional
    the subgroup is with respect to the entire dataset.
    """

    # Define a named tuple to store model parameters and likelihoods
    tpl = namedtuple(
        "EMM_Likelihood",
        ["model_params", "subgroup_likelihood", "inverse_likelihood", "size"],
    )

    def __init__(self, model):
        """Initialize the EMM_Likelihood measure with a given model.

        Parameters:
            model: An instance of a model class that provides fit and likelihood methods.
        """
        self.model = model
        self.has_constant_statistics = False
        self.required_stat_attrs = EMM_Likelihood.tpl._fields
        self.data_size = None

    def calculate_constant_statistics(self, data, target):
        """Calculate statistics that remain constant over all subgroups.

        Parameters:
            data: The dataset as a pandas DataFrame.
            target: The target variable (unused in this context).
        """
        self.model.calculate_constant_statistics(data, target)
        self.data_size = len(data)
        self.has_constant_statistics = True

    def calculate_statistics(
        self, subgroup, target, data, statistics=None
    ):  # pylint: disable=unused-argument
        """Calculate statistics specific to a subgroup.

        Parameters:
            subgroup: The subgroup description.
            target: The target variable (unused in this context).
            data: The dataset as a pandas DataFrame.
            statistics: Previously calculated statistics (optional).

        Returns:
            An EMM_Likelihood.tpl namedtuple containing model parameters,
            subgroup likelihood, inverse likelihood, and subgroup size.
        """
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, self.data_size, data)
        params = self.model.fit(cover_arr, data)
        return self.get_tuple(sg_size, params, cover_arr)

    def get_tuple(self, sg_size, params, cover_arr):
        """Compute the likelihoods for the subgroup and its complement.

        Parameters:
            sg_size: Size of the subgroup.
            params: Model parameters obtained from fitting the subgroup.
            cover_arr: Boolean array indicating the instances in the subgroup.

        Returns:
            An EMM_Likelihood.tpl namedtuple with the computed statistics.
        """
        # Compute likelihoods for all data instances
        all_likelihood = self.model.likelihood(
            params, np.ones(self.data_size, dtype=bool)
        )
        # Sum of likelihoods for subgroup instances
        sg_likelihood_sum = np.sum(all_likelihood[cover_arr])
        # Sum of likelihoods for all instances
        total_likelihood_sum = np.sum(all_likelihood)
        # Compute average likelihood for the complement (inverse) subgroup
        dataset_average = np.nan
        if (self.data_size - sg_size) > 0:
            dataset_average = (total_likelihood_sum - sg_likelihood_sum) / (
                self.data_size - sg_size
            )
        # Compute average likelihood for the subgroup
        sg_average = np.nan
        if sg_size > 0:
            sg_average = sg_likelihood_sum / sg_size
        return EMM_Likelihood.tpl(params, sg_average, dataset_average, sg_size)

    def evaluate(self, subgroup, target, data, statistics=None):
        """Evaluate the interestingness of a subgroup.

        Parameters:
            subgroup: The subgroup description.
            target: The target variable (unused in this context).
            data: The dataset as a pandas DataFrame.
            statistics: Previously calculated statistics (optional).

        Returns:
            The difference between subgroup likelihood and inverse likelihood.
        """
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.subgroup_likelihood - statistics.inverse_likelihood

    def gp_get_params(self, cover_arr, v):
        """Get parameters for GP-Growth algorithm.

        Parameters:
            cover_arr: Boolean array indicating the instances in the subgroup.
            v: Statistics vector from GP-Growth.

        Returns:
            An EMM_Likelihood.tpl namedtuple with the computed statistics.
        """
        params = self.model.gp_get_params(v)
        sg_size = params.size_sg
        return self.get_tuple(sg_size, params, cover_arr)

    @property
    def gp_requires_cover_arr(self):
        """Indicate whether the GP-Growth algorithm requires a cover array.

        Returns:
            True, since the cover array is required.
        """
        return True

    def __getattr__(self, name):
        """Delegate attribute access to the underlying model.

        Parameters:
            name: Name of the attribute.

        Returns:
            The attribute from the model if it exists.
        """
        return getattr(self.model, name)


class PolyRegression_ModelClass:
    """Polynomial Regression Model Class for Exceptional Model Mining.

    Provides methods to fit a polynomial regression model to a subgroup and
    compute likelihoods for Exceptional Model Mining.
    """

    def __init__(self, x_name="x", y_name="y", degree=1):
        """Initialize the Polynomial Regression Model.

        Parameters:
            x_name (str): Name of the independent variable in the data.
            y_name (str): Name of the dependent variable in the data.
            degree (int): Degree of the polynomial (currently only degree=1 is supported).

        Raises:
            ValueError: If degree is not equal to 1.
        """
        self.x_name = x_name
        self.y_name = y_name
        if degree != 1:
            raise ValueError("Currently only degree == 1 is supported")
        self.degree = degree
        self.x = None
        self.y = None
        self.has_constant_statistics = True
        super().__init__()

    def calculate_constant_statistics(
        self, data, target
    ):  # pylint: disable=unused-argument
        """Calculate statistics that remain constant over all subgroups.

        Parameters:
            data: The dataset as a pandas DataFrame.
            target: The target variable (unused in this context).
        """
        self.x = data[self.x_name].to_numpy()
        self.y = data[self.y_name].to_numpy()
        self.has_constant_statistics = True

    @staticmethod
    def gp_merge(u, v):
        """Merge two statistics vectors for the GP-Growth algorithm.

        Parameters:
            u (numpy.ndarray): Left statistics vector.
            v (numpy.ndarray): Right statistics vector.
        """
        v0 = v[0]
        u0 = u[0]
        if v0 == 0 or u0 == 0:
            d = 0
        else:
            d = v0 * u0 / (v0 + u0) * (v[1] / v0 - u[1] / u0) * (v[2] / v0 - u[2] / u0)
        u += v
        u[3] += d

    def gp_get_null_vector(self):
        """Get a null vector for initialization in the GP-Growth algorithm.

        Returns:
            numpy.ndarray: Zero-initialized array of size 5.
        """
        return np.zeros(5)

    def gp_get_stats(self, row_index):
        """Get statistics for a single row (used in GP-Growth algorithm).

        Parameters:
            row_index (int): Index of the row in the dataset.

        Returns:
            numpy.ndarray: Statistics vector for the given row.
        """
        x = self.x[row_index]
        return np.array([1, x, self.y[row_index], 0, x * x])

    def gp_get_params(self, v):
        """Extract model parameters from the statistics vector.

        Parameters:
            v (numpy.ndarray): Statistics vector.

        Returns:
            beta_tuple: Contains regression coefficients and subgroup size.
        """
        size = v[0]
        if size < self.degree:
            return beta_tuple(np.full(self.degree + 1, np.nan), size)
        v1 = v[1]
        # Compute slope and intercept for linear regression
        slope = v[0] * v[3] / (v[0] * v[4] - v1 * v1)
        intercept = v[2] / v[0] - slope * v[1] / v[0]
        return beta_tuple(np.array([slope, intercept]), v[0])

    def gp_to_str(self, stats):
        """Convert statistics to a string representation.

        Parameters:
            stats (numpy.ndarray): Statistics vector.

        Returns:
            str: String representation of the statistics.
        """
        return " ".join(map(str, stats))

    def gp_size_sg(self, stats):
        """Get the size of the subgroup from the statistics.

        Parameters:
            stats (numpy.ndarray): Statistics vector.

        Returns:
            float: Size of the subgroup.
        """
        return stats[0]

    @property
    def gp_requires_cover_arr(self):
        """Indicate whether the GP-Growth algorithm requires a cover array.

        Returns:
            False, since the cover array is not required.
        """
        return False

    def fit(self, subgroup, data=None):
        """Fit the polynomial regression model to the subgroup data.

        Parameters:
            subgroup: The subgroup description.
            data: The dataset as a pandas DataFrame (optional).

        Returns:
            beta_tuple: Contains regression coefficients and subgroup size.
        """
        cover_arr, size = ps.get_cover_array_and_size(subgroup, len(self.x), data)
        if size <= self.degree + 1:
            return beta_tuple(np.full(self.degree + 1, np.nan), size)
        # Fit polynomial regression model to subgroup data
        return beta_tuple(
            np.polyfit(self.x[cover_arr], self.y[cover_arr], deg=self.degree), size
        )

    def likelihood(self, stats, sg):
        """Compute the likelihoods for the subgroup instances.

        Parameters:
            stats (beta_tuple): Regression parameters and subgroup size.
            sg (numpy.ndarray): Boolean array indicating subgroup instances.

        Returns:
            numpy.ndarray: Likelihood values for the subgroup instances.
        """
        from scipy.stats import norm  # pylint: disable=import-outside-toplevel

        if any(np.isnan(stats.beta)):
            return np.full(self.x[sg].shape, np.nan)
        # Compute the residuals and evaluate the normal probability density function
        residuals = np.polyval(stats.beta, self.x[sg]) - self.y[sg]
        return norm.pdf(residuals)

    def loglikelihood(self, stats, sg):
        """Compute the log-likelihoods for the subgroup instances.

        Parameters:
            stats (beta_tuple): Regression parameters and subgroup size.
            sg (numpy.ndarray): Boolean array indicating subgroup instances.

        Returns:
            numpy.ndarray: Log-likelihood values for the subgroup instances.
        """
        from scipy.stats import norm  # pylint: disable=import-outside-toplevel

        # Compute the residuals and evaluate the normal log-probability density function
        residuals = np.polyval(stats.beta, self.x[sg]) - self.y[sg]
        return norm.logpdf(residuals)
