import cProfile
import heapq
import time

import SGDUtils
from Selectors import NominalSelector
import Subgroup
import SubgroupDiscoveryTask, Selectors, BSD, SimpleDFS
import numpy as np
import pandas as pd
from pysubgroup import GeneralizationAwaresInterestingnessMeasures


# Create some data
d = np.array([[1, 'test', 'A', 'true', 2], \
               [2, 'test2', 'B', 'true', 5], \
               [5, 'test', 'B', 'False', 1], \
               [3, 'test', '', 'False', 100]])
df = pd.DataFrame(d)
df.columns = ['intTest', 'strTest', 'strTest2', 'trg', 'weights']

df['intTest'] = df['intTest'].astype(int)
df['weights'] = df['weights'].astype(int)
data = df.to_records(False, True)

# print meta['class']
target = NominalSelector ('trg', 'False')
sel1 = NominalSelector('strTest', 'test')
sel2 = NominalSelector('strTest2', 'B')
sg = Subgroup.Subgroup (target, Subgroup.SubgroupDescription([sel1, sel2]))

qf = GeneralizationAwaresInterestingnessMeasures.GAStandardQF(0.5)
print (GeneralizationAwaresInterestingnessMeasures.getMaxGeneralizationTargetShare(data, sg))
