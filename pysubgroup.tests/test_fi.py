from scipy.io import arff
import pysubgroup as ps
import pandas as pd

import pprint

pp = pprint.PrettyPrinter(indent=4)

data = pd.read_csv("C:/data/titanic.csv")
searchSpace = ps.createSelectors(data)

task = ps.SubgroupDiscoveryTask (data, ps.FITarget, searchSpace, resultSetSize=10, depth=1, qf=ps.CountQF())
result = ps.SimpleDFS().execute(task)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))   
