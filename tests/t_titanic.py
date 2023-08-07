import pysubgroup as ps
from pysubgroup.datasets import get_titanic_data

data = get_titanic_data()
target = ps.BinaryTarget("Survived", True)

searchspace = ps.create_selectors(data, ignore=["Survived"])
task = ps.SubgroupDiscoveryTask(
    data, target, searchspace, result_set_size=5, depth=2, qf=ps.ChiSquaredQF()
)
result = ps.BeamSearch().execute(task)

print(result.to_dataframe())
