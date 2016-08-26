'''
Created on 02.05.2016

@author: lemmerfn
'''
from heapq import heappush, heappop

def addIfRequired (result, sg, quality, task):
    if (quality > task.minQuality):
        if (len(result) < task.resultSetSize):
            heappush(result, (quality, sg))
        elif (quality > result[0][0]):
            heappop(result)
            heappush(result, (quality, sg))


def minimumRequiredQuality (result, task):
    if (len(result) < task.resultSetSize):
        return task.minQuality
    else:
        return result [0][0]
    
# Returns the cutpoints for discretization
def equalFrequencyDiscretization (data, attributeName, nbins=5):
    numberInstances = len(data[attributeName])
    sortedData = sorted(data[attributeName])
    cutpoints = []
    for i in range(1, nbins):
        position = i * numberInstances / nbins
        while True:
            if (position >= numberInstances):
                break
            val = sortedData [position]
            if (not val in cutpoints):
                break
            position += 1
        if (not val in cutpoints):
            cutpoints.append(val)
    return cutpoints

