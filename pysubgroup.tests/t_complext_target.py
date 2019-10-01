import pysubgroup as ps
import pandas as pd

adult = pd.read_csv('../data/adult_age_test.csv', sep='\t', index_col=0)
new_adult = adult[:2000]
attr = 'age_change'
effect = 'age_prediction_change'

target = ps.ComplexTarget((attr, effect))
search_space = ps.create_selectors(new_adult, ignore=[attr, effect, 'income'])
task = ps.SubgroupDiscoveryTask(new_adult, target, search_space, qf=ps.CorrelationQF('entropy'))
result = ps.BeamSearch().execute(task)
for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))
