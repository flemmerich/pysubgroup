import pysubgroup as ps
import pandas as pd
data = pd.read_table("../data/titanic.csv")
target = ps.NominalTarget('Survived', True)

searchspace = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask(
    data,
    target,
    searchspace,
    result_set_size=5,
    depth=2,
    qf=ps.ChiSquaredQF())
result = ps.BeamSearch().execute(task)

for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))

