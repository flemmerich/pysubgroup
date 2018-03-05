from scipy.io import arff

import pysubgroup as ps
import pandas as pd
from timeit import default_timer as timer

data = pd.DataFrame (arff.loadarff("C:\data\Datasets\credit-g.arff") [0])

target = ps.NominalTarget ('class', b'bad')
searchSpace = ps.createSelectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=10, depth=3, qf=ps.ChiSquaredQF(direction="bidirect"))

start = timer()
result = ps.SimpleDFS().execute(task)
end = timer()

print("Time elapsed: ", (end - start)) 

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))   

# print WRAccQF().evaluateFromDataset(data, Subgroup(target, []))
