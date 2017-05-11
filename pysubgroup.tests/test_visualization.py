import pysubgroup as ps
import pandas as pd

data = pd.read_csv("C:/data/titanic.csv")
target = ps.NominalSelector ('survived', 0)
searchSpace = ps.createSelectors(data, ignore=['survived'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace, 
                                 resultSetSize=5, depth=2, 
                                 qf=ps.ChiSquaredQF())

result = ps.SimpleDFS().execute(task)

dfs = ps.utils.resultsAsDataFrame (data, result)
ps.plot_roc (data, dfs, ps.ChiSquaredQF())