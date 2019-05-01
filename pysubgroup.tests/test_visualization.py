import pysubgroup as ps
import pandas as pd
import matplotlib.pyplot as plt
plt.interactive(False)


data = pd.read_csv("~/datasets/titanic.csv")
target = ps.NominalTarget('survived', 0)
searchSpace = ps.create_selectors(data, ignore=['survived'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace, 
                                 result_set_size=5, depth=2,
                                 qf=ps.ChiSquaredQF())

result = ps.SimpleDFS().execute(task)

dfs = ps.utils.resultsAsDataFrame (data, result)
plt = ps.plot_roc (data, dfs, ps.ChiSquaredQF())
plt.show()