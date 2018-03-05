from scipy.io import arff

import pysubgroup as ps
import pandas as pd

data = pd.DataFrame (arff.loadarff("C:\data\Datasets\credit-g.arff") [0])

target = ps.NominalTarget ('class', b'bad')
searchSpace = ps.createSelectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=10, depth=1, qf=ps.ChiSquaredQF(direction="bidirect"))

result = ps.SimpleDFS().execute(task)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))   

# print WRAccQF().evaluateFromDataset(data, Subgroup(target, []))
