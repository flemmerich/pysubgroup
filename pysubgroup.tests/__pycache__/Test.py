from scipy.io import arff

import pysubgroup.subgroup as sg
import pysubgroup.algorithms as algo
import pysubgroup.measures as measures
import pandas as pd

data = pd.DataFrame (arff.loadarff("C:\data\Datasets\credit-g.arff") [0])

target = sg.NominalSelector ('class', b'bad')
searchSpace = sg.createSelectors(data, ignore_attributes=['class'])
task = algo.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=10, depth=1, qf=measures.ChiSquaredQF(direction="bidirect"))

result = algo.SimpleDFS().execute(task)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))   

# print WRAccQF().evaluateFromDataset(data, Subgroup(target, []))
