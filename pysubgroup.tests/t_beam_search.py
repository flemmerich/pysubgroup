from scipy.io import arff

import pysubgroup as ps
import pandas as pd

data = pd.DataFrame (arff.loadarff("../data/credit-g.arff") [0])

target = ps.NominalTarget ('class', b'bad')
search_space = ps.create_nominal_selectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask (data, target, search_space, result_set_size=10, depth=3, qf=ps.StandardQF(1.0))

result = ps.BeamSearch(beam_width=10).execute(task)
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroup_description))

print ("******")
result = ps.SimpleDFS().execute(task)
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroup_description))

# print WRAccQF().evaluate_from_dataset(data, Subgroup(target, []))
