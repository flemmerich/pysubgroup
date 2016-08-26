'''
Created on 29.04.2016

@author: lemmerfn
'''
from InterestingnessMeasures import WRAccQF

class SubgroupDiscoveryTask(object):
    def __init__(self, data, target, searchSpace, qf=WRAccQF(), resultSetSize=10, depth=3, minQuality=0):
        self.data = data
        self.target = target
        self.searchSpace = searchSpace
        self.qf = qf
        self.resultSetSize = resultSetSize
        self.depth = depth
        self.minQuality = minQuality