import pysubgroup as ps
import pandas as pd
from scipy.io import arff
from timeit import default_timer as timer


data = pd.DataFrame(arff.loadarff("../data/credit-g.arff") [0])
target = ps.NumericTarget('credit_amount')
search_space = ps.create_nominal_selectors(data, ignore=['credit_amount'])

task = ps.SubgroupDiscoveryTask (data, target, search_space, result_set_size=10, depth=3, qf=ps.StandardQFNumeric(1, False))
print (search_space)

start = timer()
result = ps.SimpleDFS().execute(task)
end = timer()
print(f"Time elapsed: {end - start}")
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroup_description))
