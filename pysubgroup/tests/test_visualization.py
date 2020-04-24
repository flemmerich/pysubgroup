import matplotlib.pyplot as plt
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_titanic_data

if __name__ == '__main__':
    plt.interactive(False)


    data = get_titanic_data()
    target = ps.BinaryTarget('Survived', 0)
    search_space = ps.create_selectors(data, ignore=['Survived'])
    task = ps.SubgroupDiscoveryTask(data, target, search_space,
                                    result_set_size=5, depth=2,
                                    qf=ps.ChiSquaredQF())

    result = ps.SimpleDFS().execute(task)

    #dfs = ps.results_as_df(data, result)
    #fig = ps.plot_roc(dfs, data, ps.ChiSquaredQF())
    #fig.show()
    #plt.show()
