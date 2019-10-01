import pysubgroup as ps
import pandas as pd

import pprint

pp = pprint.PrettyPrinter(indent=4)

data = pd.read_csv("../data/titanic.csv")
search_space = ps.create_selectors(data, ignore="survived")
dt = data.dtypes

task = ps.SubgroupDiscoveryTask (data, ps.FITarget, search_space, result_set_size=10, depth=5, qf=ps.CountQF())
result = ps.SimpleDFS().execute(task)

for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))

task = ps.SubgroupDiscoveryTask (data, ps.FITarget, search_space, result_set_size=10, depth=3, qf=ps.AreaQF())
result = ps.SimpleDFS().execute(task)

for (q, sg) in result:
    print(f"{q}\t{sg.subgroup_description}\t{sg.count(data)}")
