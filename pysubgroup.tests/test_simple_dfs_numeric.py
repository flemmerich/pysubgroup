import pysubgroup as ps
import pandas as pd
from scipy.io import arff
from timeit import default_timer as timer


data = pd.DataFrame (arff.loadarff("../data/credit-g.arff") [0])
target = ps.NumericTarget ('credit_amount')
searchSpace = ps.createNominalSelectors(data, ignore=['credit_amount'])

task = ps.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=10, depth=5, qf=ps.StandardQF_numeric(1, False))
print (searchSpace)

start = timer()
result = ps.BestFirstSearch().execute(task)
end = timer()
print(f"Time elapsed: {end - start}")
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription))
