from collections import namedtuple
from scipy.stats import norm
import numpy as np
import pysubgroup as ps
beta_tuple = namedtuple('beta_tuple', ['beta', 'size'])


class EMM_Likelihood(ps.AbstractInterestingnessMeasure):
    tpl = namedtuple('EMM_Likelihood', ['model_params', 'subgroup_likelihood', 'inverse_likelihood', 'size'])
    def __init__(self, model):
        self.model = model
        self.has_constant_statistics = False
        self.required_stat_attrs = EMM_Likelihood.tpl._fields
        self.data_size = None

    def calculate_constant_statistics(self, task):
        self.model.calculate_constant_statistics(task)
        self.data_size = len(task.data)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, data=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, self.data_size, data)

        params = self.model.fit(cover_arr, data)
        return self.get_tuple(sg_size, params, cover_arr)

    def get_tuple(self, sg_size, params, cover_arr):
        #numeric stability?
        all_likelihood = self.model.likelihood(params, np.ones(self.data_size, dtype=bool))
        sg_likelihood_sum = np.sum(all_likelihood[cover_arr])
        total_likelihood_sum = np.sum(all_likelihood)
        dataset_average = np.nan
        if (self.data_size - sg_size) > 0:
            dataset_average = (total_likelihood_sum - sg_likelihood_sum)/(self.data_size - sg_size)
        sg_average = np.nan
        if sg_size > 0:
            sg_average = sg_likelihood_sum/sg_size
        return EMM_Likelihood.tpl(params, sg_average, dataset_average, sg_size)

    def evaluate(self, subgroup, statistics=None):
        statistics = self.ensure_statistics(subgroup, statistics)
        #numeric stability?
        return statistics.subgroup_likelihood - statistics.inverse_likelihood

    def gp_get_params(self, cover_arr, v):
        params = self.model.gp_get_params(v)
        sg_size = params.size
        return self.get_tuple(sg_size, params, cover_arr)


    def supports_weights(self):
        return False

    def is_applicable(self, _):
        return True

    def __getattr__(self, name):
        return getattr(self.model, name)

class PolyRegression_ModelClass:
    def __init__(self, x_name='x', y_name='y', degree=1):
        self.x_name = x_name
        self.y_name = y_name
        if degree != 1:
            raise ValueError('Currently only degree == 1 is supported')
        self.degree = degree
        self.x = None
        self.y = None
        self.has_constant_statistics = True
        super().__init__()

    def calculate_constant_statistics(self, task):
        data = task.data
        self.x = data[self.x_name].to_numpy()
        self.y = data[self.y_name].to_numpy()
        self.has_constant_statistics = True

    @staticmethod
    def gp_merge(u, v):
        v0 = v[0]
        u0 = u[0]
        if v0 == 0 or u0 == 0:
            d = 0
        else:
            d = v0 * u0/(v0 + u0)*(v[1]/v0 - u[1]/u0)*(v[2]/v0 - u[2]/u0)
        u += v
        u[3] += d

    def gp_get_null_vector(self):
        return np.zeros(5)

    def gp_get_stats(self, row_index):
        x = self.x[row_index]
        return np.array([1, x, self.y[row_index], 0, x*x])

    def gp_get_params(self, v):
        size = v[0]
        if size < self.degree:
            return beta_tuple(np.full(self.degree + 1, np.nan), size)
        v1 = v[1]
        slope = v[0] * v[3] / (v[0]*v[4] - v1 * v1)
        intersept = v[2]/v[0] - slope * v[1]/v[0]
        return beta_tuple(np.array([slope, intersept]), v[0])

    def fit(self, subgroup, data=None):
        cover_arr, size = ps.get_cover_array_and_size(subgroup, len(self.x), data)
        if size <= self.degree + 1:
            return beta_tuple(np.full(self.degree + 1, np.nan), size)
        return beta_tuple(np.polyfit(self.x[cover_arr], self.y[cover_arr], deg=self.degree), size)

    def likelihood(self, stats, sg):
        if any(np.isnan(stats.beta)):
            return np.full(self.x[sg].shape, np.nan)
        return norm.pdf(np.polyval(stats.beta, self.x[sg]) - self.y[sg])

    def loglikelihood(self, stats, sg):
        return norm.logpdf(np.polyval(stats.beta, self.x[sg]) - self.y[sg])
