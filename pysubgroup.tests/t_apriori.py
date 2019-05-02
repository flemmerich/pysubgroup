from timeit import default_timer as timer
import pandas as pd
from scipy.io import arff

import pysubgroup as ps

data = pd.DataFrame (arff.loadarff("../data/credit-g.arff") [0])

target = ps.NominalTarget ('class', b'bad')
search_space = ps.create_nominal_selectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask (data, target, search_space, result_set_size=10, depth=5, qf=ps.StandardQF(1.0))

start = timer()
result = ps.Apriori().execute(task)
end = timer()
print("Time elapsed: ", (end - start))

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroup_description))

print("******")

start = timer()
result = ps.SimpleDFS().execute(task)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroup_description))
