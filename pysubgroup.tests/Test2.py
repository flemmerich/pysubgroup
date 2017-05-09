import pysubgroup.subgroup as sg
import pysubgroup.algorithms as algo
import pysubgroup.measures as measures
import pandas as pd

data = pd.read_csv("C:/data/titanic.csv")

target = sg.NominalSelector ('survived', 0)
searchSpace = sg.createSelectors(data, ignore_attributes=['survived'])
task = algo.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=3, depth=1, qf=measures.WRAccQF())

result = algo.SimpleDFS().execute(task)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))