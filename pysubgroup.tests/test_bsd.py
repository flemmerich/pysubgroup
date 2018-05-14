from scipy.io import arff

import pysubgroup as ps
import pandas as pd
from timeit import default_timer as timer


data = pd.DataFrame (arff.loadarff("../data/credit-g.arff") [0])

target = ps.NominalTarget('class', b'bad')
searchSpace = ps.createNominalSelectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=10, depth=5, qf=ps.StandardQF(0.5))

#start = timer()
#result = ps.BSD_Bitarray().execute(task)
#end = timer()
#print("Time elapsed: ", (end - start))
#for (q, sg) in result:
#    print (str(q) + ":\t" + str(sg.subgroupDescription))
# print WRAccQF().evaluateFromDataset(data, Subgroup(target, []))


start = timer()
result = ps.BSD().execute(task)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))


print ("******")
start = timer()
result = ps.SimpleDFS().execute(task)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))