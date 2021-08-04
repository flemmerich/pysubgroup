import pandas as pd
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_titanic_data



data = get_titanic_data()
target = ps.BinaryTarget('Survived', 0)
search_space = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask(data, target, search_space,
                                result_set_size=5, depth=2,
                                qf=ps.CombinedInterestingnessMeasure([ps.StandardQF(1), ps.GeneralizationAware_StandardQF(1)]))

result = ps.SimpleDFS().execute(task, use_optimistic_estimates=False)

print(result.to_dataframe())
