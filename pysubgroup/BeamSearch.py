'''
Created on 29.04.2016

@author: lemmerfn
'''
from pysubgroup.Subgroup import Subgroup
from heapq import heappush

class BeamSearch(object):
    '''
    Implements the BeamSearch algorithm
    '''
    def __init__(self, beamWidth, beamWidthAdaptive=False):
        self.beamWidth = beamWidth
        self.beamWidthAdaptive = beamWidthAdaptive
    
    def execute (self, task):
        if self.beamWidthAdaptive:
            self.beamWidth = task.resultSetSize
            
        beam = []
        
        for sel in task.searchSpace:
            sg = Subgroup(task.target, [sel])
            quality = task.qf.evaluateFromDataset (task.data, sg)
            heappush (beam, (quality, sg))
        
        print beam
