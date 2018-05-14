import pysubgroup as ps
import pandas as pd

import pprint

pp = pprint.PrettyPrinter(indent=4)

data = pd.read_csv("~/datasets/titanic.csv")
searchSpace = ps.createSelectors(data, ignore="survived")
dt = data.dtypes

task = ps.SubgroupDiscoveryTask (data, ps.FITarget, searchSpace, resultSetSize=10, depth=5, qf=ps.CountQF())
result = ps.SimpleDFS().execute(task)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))


task = ps.SubgroupDiscoveryTask (data, ps.FITarget, searchSpace, resultSetSize=10, depth=3, qf=ps.AreaQF())
result = ps.SimpleDFS().execute(task)

for (q, sg) in result:
    print (f"{q}\t{sg.subgroupDescription}\t{sg.count(data)}")