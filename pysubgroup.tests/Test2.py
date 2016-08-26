from __future__ import division 
import cProfile
import time

from scipy.io import arff

import SubgroupDiscoveryTask, Selectors, BSD, SimpleDFS
from Selectors import NominalSelector, NumericSelector
from Subgroup import Subgroup
from InterestingnessMeasures import WRAccQF
import heapq


data, meta = arff.loadarff('C:/data/Datasets/credit-g.arff')
# print meta['class']
target = NominalSelector ('class', "bad")

sel = NumericSelector ('credit_amount', 4741, float("inf"))
print sum (1 for x in data if sel.covers(x))

sg = Subgroup(target, [sel])
print WRAccQF().evaluateFromDataset(data, sg)
