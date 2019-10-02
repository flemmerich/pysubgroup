import pysubgroup as ps
from timeit import default_timer as timer
from pysubgroup.tests.DataSets import getCreditData
data = getCreditData()

print("running")
target = ps.NominalTarget('class', b'bad')
search_space = ps.create_selectors(data, ignore=['class'])
task = ps.SubgroupDiscoveryTask(data, target, search_space, result_set_size=10, depth=3, qf=ps.ChiSquaredQF(direction="bidirect"))

start = timer()
result = ps.SimpleDFS().execute(task)
end = timer()

print("Time elapsed: ", (end - start)) 

for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))

# print WRAccQF().evaluate_from_dataset(data, Subgroup(target, []))
