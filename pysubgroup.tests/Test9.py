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

searchSpace = Selectors.createSelectors(data, intervals_only=False)
searchSpace = [x for x in searchSpace if x.attributeName != target.attributeName]
print (searchSpace)

task = SubgroupDiscoveryTask.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=50, depth=1, qf=ChiSquaredQF(direction="bidirect"))
algo = SimpleDFS.SimpleDFS()
start_time = time.clock()
result = algo.execute(task)
stop_time = time.clock()
print (stop_time - start_time, "seconds")
# cProfile.run('result = algo.execute(task)')

print (result)
result.sort(key=lambda x: x[0], reverse=True)

result = SGFilter.uniqueAttributes(result)
result = SGFilter.overlapFilter(result, data, similarity_level=0.8)
for i, (q, sg) in enumerate(result):
    print ("(" + str(i) + ") " + str(q) + ":\t" + str(sg.subgroupDescription))   

# print WRAccQF().evaluateFromDataset(data, Subgroup(target, []))
