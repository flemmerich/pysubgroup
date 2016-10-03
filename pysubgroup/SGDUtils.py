'''
Created on 02.05.2016

@author: lemmerfn
'''
from heapq import heappush, heappop
import numpy as np

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
        position = i * numberInstances // nbins
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

def printResultSet(data, result, statisticsToShow, weightingAttribute=None, printHeader=True, includeTarget=False):
    if printHeader:
        s = "Quality\tSubgroup"
        for stat in statisticsToShow:
            s += "\t" + stat
        print (s) 
    for (q, sg) in result:
        sg.calculateStatistics(data, weightingAttribute)
        s = str(q) + ":\t" + str(sg.subgroupDescription) 
        if includeTarget:
            s += str(sg.target)
        for stat in statisticsToShow:
            s += "\t" + str(sg.statistics[stat])
        print (s) 

def resultAsTable(data, result, statisticsToShow, weightingAttribute=None, printHeader=True, includeTarget=False):
    
    table = []
    if printHeader:
        row = ["Quality", "Subgroup"]
        for stat in statisticsToShow:
            row.append(stat)
        table.append(row)
    for (q, sg) in result:
        sg.calculateStatistics(data, weightingAttribute)
        row = [str(q), str(sg.subgroupDescription)]
        if includeTarget:
            row.append(str(sg.target))
        for stat in statisticsToShow:
            row.append(str(sg.statistics[stat]))
        table.append(row) 
    return table


def extractStatisticsFromDataset (data, subgroup, weightingAttribute=None): 
    if (weightingAttribute == None):
        sgInstances = subgroup.subgroupDescription.covers(data)
        positives = subgroup.target.covers(data)
        instancesSubgroup = np.sum(sgInstances)
        positivesDataset = np.sum(positives)
        instancesDataset = len(data)
        positivesSubgroup = np.sum(np.logical_and(sgInstances, positives))
        return (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)  
    else:
        weights = data[weightingAttribute]
        sgInstances = subgroup.subgroupDescription.covers(data)
        positives = subgroup.target.covers(data)                         

        instancesDataset = np.sum(weights)
        instancesSubgroup = np.sum(np.dot(sgInstances, weights))
        positivesDataset = np.sum(np.dot(positives, weights))
        positivesSubgroup = np.sum(np.dot(np.logical_and(sgInstances, positives), weights))
        return (instancesDataset, positivesDataset, instancesSubgroup, positivesSubgroup)
