from scipy.io import arff

import pysubgroup as ps
import pandas as pd
from timeit import default_timer as timer


data = pd.DataFrame(arff.loadarff("../data/credit-g.arff") [0])

target = ps.NominalTarget('class', b'bad')
search_space = ps.create_nominal_selectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask (data, target, search_space, result_set_size=10, depth=5, qf=ps.ChiSquaredQF())


start = timer()
result = ps.BSD().execute(task)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))


print ("******")
start = timer()
result = ps.TID_SD().execute(task, use_sets=True)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))


print ("******")
start = timer()
result = ps.TID_SD().execute(task)
end = timer()
print("Time elapsed: ", (end - start))
for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))
