import pysubgroup as ps
import pandas as pd


data = pd.read_csv("~/datasets/titanic.csv")
target = ps.NominalTarget ('survived', 0)
searchSpace = ps.create_selectors(data, ignore=['survived'])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace,
                                 result_set_size=5, depth=2,
                                 qf=ps.CombinedInterestingnessMeasure([ps.StandardQF(1), ps.GAStandardQF(1)]))

result = ps.SimpleDFS().execute(task, use_optimistic_estimates=False)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroup_description))
