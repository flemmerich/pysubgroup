'''
Created on 29.04.2016

@author: lemmerfn
'''
from pysubgroup import InterestingnessMeasures

class SubgroupDiscoveryTask(object):
    def __init__(self, data, target, searchSpace, qf=InterestingnessMeasures.WRAccQF(), resultSetSize=10, depth=3, minQuality=0, weightingAttribute=None):
        self.data = data
        self.target = target
        self.searchSpace = searchSpace
        self.qf = qf
        self.resultSetSize = resultSetSize
        self.depth = depth
        self.minQuality = minQuality
        self.weightingAttribute = weightingAttribute