import pysubgroup as ps
import pandas as pd
import matplotlib.pyplot as plt
from DataSets import *
plt.interactive(False)


data = getTitanicData()
target = ps.NominalTarget('Survived', 0)
search_space = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask(data, target, search_space,
                                result_set_size=5, depth=2,
                                qf=ps.ChiSquaredQF())

result = ps.SimpleDFS().execute(task)

dfs = ps.results_as_df(data, result)
fig = ps.plot_roc( dfs, data, ps.ChiSquaredQF())
fig.show()
plt.show()
