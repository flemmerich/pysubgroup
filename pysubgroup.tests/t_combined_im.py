import pysubgroup as ps
import pandas as pd


data = pd.read_table("../data/titanic.csv")
target = ps.NominalTarget('Survived', 0)
search_space = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask (data, target, search_space,
                                 result_set_size=5, depth=2,
                                 qf=ps.CombinedInterestingnessMeasure([ps.StandardQF(1), ps.GAStandardQF(1)]))

result = ps.SimpleDFS().execute(task, use_optimistic_estimates=False)

for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))
