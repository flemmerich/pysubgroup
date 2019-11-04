import numpy as np
import pysubgroup as ps

class LinearRegressionModel:
    def __init__(self, x_name, y_name):
        self.x_name = x_name
        self.y_name = y_name
        self.X = None
        self.Y = None
        self.has_constant_statistics = True
        super().__init__()

    def calculate_constant_statistics(self, task):
        data = task.data
        self.X = data[self.x_name].to_numpy()
        self.Y = data[self.y_name].to_numpy()
        self.has_constant_statistics = True

    def gp_merge(self, u, v):
        d = v[1] * u[1]/(v[1] + u[1])*(v[2]/v[1] - u[2]/u[1])*(v[3]/v[1] - u[3]/u[1])
        u += v
        u[4] += d

    def gp_null_vector(self):
        return np.zeros(5)

    def gp_get_stats(self, row_index):
        x = self.X[row_index]
        return np.array([1, x, self.Y[row_index], 0, x*x])


