import pysubgroup as ps
import pandas as pd

adult = pd.read_csv('../data/adult_age_test.csv', sep='\t', index_col=0)
new_adult = adult[:3000]
effect = 'age_prediction_change'

target = ps.NumericTarget(effect)
search_space = ps.create_selectors(new_adult, ignore=[effect, 'income'])
task = ps.SubgroupDiscoveryTask(new_adult, target, search_space, qf=ps.IncrementalQFNumeric(1), min_quality=0)
result = ps.BeamSearch().execute(task)
for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))
