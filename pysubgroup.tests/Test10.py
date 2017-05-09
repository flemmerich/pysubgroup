import cProfile
import heapq
import time

from scipy.io import arff

from Selectors import NominalSelector
import SubgroupDiscoveryTask, Selectors, BSD, SimpleDFS
from pysubgroup import SGFilter
from pysubgroup.InterestingnessMeasures import ChiSquaredQF, StandardQF


data = arff.loadarff("C:\data\Datasets\credit-g.arff") [0]
# print meta['class']
target = NominalSelector ('class', b'bad')

# sel2 = NominalSelector ('checking_status', 'no checking')

# sg = Subgroup(NominalSelector ('class', "bad"), [NominalSelector ('checking_status', "'no checking'")])
# qf = WRAccQF()
# print qf.evaluateFromDataset(data, sg)
# print sel.covers(data[0])
# print sum(1 for i in data if sel.covers(i))
print ("XXX")

searchSpace = Selectors.createSelectors(data, intervals_only=False)
searchSpace = [x for x in searchSpace if x.attributeName != target.attributeName]
print (searchSpace)

task = SubgroupDiscoveryTask.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=20, depth=1, qf=ChiSquaredQF(direction="bidirect"))
algo = SimpleDFS.SimpleDFS()
start_time = time.clock()
result = algo.execute(task)
stop_time = time.clock()
print (stop_time - start_time, "seconds")
# cProfile.run('result = algo.execute(task)')

print (result)
result = heapq.nlargest(len(result), result)
for x, sg in result:
    sg.calculateStatistics(data)
#result = SGFilter.filterByCount(result, 500, False, False)

for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroupDescription) + str(sg.statistics["sg_size"]))   

# print WRAccQF().evaluateFromDataset(data, Subgroup(target, []))
