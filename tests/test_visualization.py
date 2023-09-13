import matplotlib.pyplot as plt

import pysubgroup as ps
from pysubgroup.datasets import get_titanic_data, get_credit_data


plt.interactive(False)


import unittest

import pysubgroup as ps



class TestVizualization(unittest.TestCase):
    def setUp(self):
        data = get_titanic_data()

        target = ps.BinaryTarget ('Survived', True)
        searchspace = ps.create_selectors(data, ignore=['Survived'])
        task = ps.SubgroupDiscoveryTask (
            data,
            target,
            searchspace,
            result_set_size=5,
            depth=2,
            qf=ps.WRAccQF())
        self.result = ps.DFS(ps.BitSetRepresentation).execute(task)
        self.result_df = self.result.to_dataframe()
        self.data=data

    def test_plot_sgbars(self):
        ps.plot_sgbars(self.result_df, )
        ps.plot_sgbars(self.result_df, dynamic_widths=True)

    def test_plot_roc(self):
        ps.plot_roc(self.result_df, self.data)
        ps.plot_roc(self.result_df, self.data, annotate=True)

    def test_plot_npspace(self):
        ps.plot_npspace(self.result_df, self.data)
        ps.plot_npspace(self.result_df, self.data, annotate=False)
        ps.plot_npspace(self.result_df, self.data, fixed_limits=True)

    def test_similarity_dendogram(self):
        ps.similarity_dendrogram(self.result, self.data)
        ps.similarity_dendrogram(self.result.to_descriptions(), self.data)

    def test_similarity_sgs(self):
        ps.similarity_sgs(self.result.to_descriptions(), self.data, color=True)
        ps.similarity_sgs(self.result.to_descriptions(), self.data, color=False)

    def test_supportSetVisualization(self):
        ps.supportSetVisualization(self.result)
        ps.supportSetVisualization(self.result, in_order=False)
        ps.supportSetVisualization(self.result, drop_empty=False)



class TestVizualization2(unittest.TestCase):
    def setUp(self):
        self.data = get_credit_data()
        self.target = ps.NumericTarget("credit_amount")
        searchSpace_Nominal = ps.create_nominal_selectors(
            self.data, ignore=["credit_amount"]
        )
        searchSpace_Numeric = (
            []
        )  # ps.create_numeric_selectors(data, ignore=['credit_amount'], nbins=10)
        searchSpace = searchSpace_Nominal + searchSpace_Numeric
        task = ps.SubgroupDiscoveryTask(
            self.data,
            self.target,
            searchSpace,
            result_set_size=10,
            depth=2,
            qf=ps.StandardQFNumeric(1, False, estimator="max", centroid='median'),
        )
        self.result = ps.DFS(ps.BitSetRepresentation).execute(task)
    def test_plot_distribution_numeric(self):
        ps.plot_distribution_numeric(self.result, self.target, self.data, 20)
        ps.plot_distribution_numeric(self.result, self.target, self.data, 20, show_dataset=False)
        ps.plot_distribution_numeric(self.result.to_descriptions(), self.target, self.data, 20)
        ps.plot_distribution_numeric(self.result.to_descriptions()[0][1], self.target, self.data, 20)

        subgroups = [subgroup for quality, subgroup in self.result.to_descriptions()]
        ps.plot_distribution_numeric(subgroups, self.target, self.data, 20)




if __name__ == "__main__":
    unittest.main()