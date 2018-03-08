from scipy.io import arff

import pysubgroup as ps
import pandas as pd

data = pd.DataFrame (arff.loadarff("../data/credit-g.arff") [0])

target = ps.NominalTarget ('class', b'bad')
searchSpace = ps.createNominalSelectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=10, depth=3, qf=ps.StandardQF(1.0))

result = ps.BeamSearch(beamWidth=10).execute(task)
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))

print ("******")
result = ps.SimpleDFS().execute(task)
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))

# print WRAccQF().evaluateFromDataset(data, Subgroup(target, []))
