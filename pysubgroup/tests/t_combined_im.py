import pandas as pd
import pysubgroup as ps


data = pd.read_table("../data/titanic.csv")
target = ps.BinaryTarget('Survived', 0)
search_space = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask(data, target, search_space,
                                result_set_size=5, depth=2,
                                qf=ps.CombinedInterestingnessMeasure([ps.StandardQF(1), ps.GeneralizationAware_StandardQF(1)]))

result = ps.SimpleDFS().execute(task, use_optimistic_estimates=False)

print(result.to_dataframe())
