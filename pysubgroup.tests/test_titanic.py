import pysubgroup as ps
import pandas as pd

data = pd.read_csv("C:/data/titanic.csv")
target = ps.NominalSelector ('survived', True)
searchSpace = ps.createSelectors(data, ignore=['survived'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace,
                                 resultSetSize=5, depth=2,
                                 qf=ps.StandardQF(1))

result = ps.Apriori().execute(task)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))
