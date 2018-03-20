'''
Created on 02.05.2016

@author: lemmerfn
'''
from functools import partial
from heapq import heappush, heappop
import itertools
import numpy as np
import pandas as pd

all_statistics = ('size_sg', 'size_dataset', 'positives_sg', 'positives_dataset', 'size_complement', 'relative_size_sg', 'relative_size_complement', 'coverage_sg', 'coverage_complement', 'target_share_sg', 'target_share_complement', 'target_share_dataset', 'lift')
all_statistics_weighted = all_statistics + ('size_sg_weighted', 'size_dataset_weighted', 'positives_sg_weighted', 'positives_dataset_weighted', 'size_complement_weighted', 'relative_size_sg_weighted', 'relative_size_complement_weighted', 'coverage_sg_weighted', 'coverage_complement_weighted', 'target_share_sg_weighted', 'target_share_complement_weighted', 'target_share_dataset_weighted', 'lift_weighted')
all_statistics_numeric = ('size_sg', 'size_dataset', 'mean_sg', 'mean_dataset', 'std_sg', 'std_dataset', 'median_sg', 'median_dataset', 'max_sg', 'max_dataset', 'min_sg', 'min_dataset', 'mean_lift', 'median_lift')


def addIfRequired (result, sg, quality, task, check_for_duplicates=False):
    if (quality > task.minQuality):
        if (check_for_duplicates and (quality, sg) in result):
            return
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
def equalFrequencyDiscretization (data, attributeName, nbins=5, weightingAttribute=None):
    cutpoints = []
    if weightingAttribute == None:
        cleanedData = data[attributeName]
        cleanedData = cleanedData[~np.isnan(cleanedData)]
        sortedData = sorted(cleanedData)
        numberInstances = len(sortedData)
        for i in range(1, nbins):
            position = i * numberInstances // nbins
            while True:
                if (position >= numberInstances):
                    break
                val = sortedData [position]
                if (not val in cutpoints):
                    break
                position += 1
            # print (sortedData [position])
            if (not val in cutpoints):
                cutpoints.append(val)
    else:
        cleanedData = data[[attributeName, weightingAttribute]]
        cleanedData = cleanedData[~np.isnan(cleanedData[attributeName])]
        cleanedData.sort(order=attributeName)
    
        overall_weights = cleanedData[weightingAttribute].sum()
        remaining_weights = overall_weights
        bin_size = overall_weights / nbins
        sum_of_weights = 0
        for row in cleanedData:
            sum_of_weights += row[weightingAttribute]
            if (sum_of_weights > bin_size):
                if (not row[attributeName] in cutpoints):
                    cutpoints.append(row[attributeName])
                    remaining_weights = remaining_weights - sum_of_weights
                    if remaining_weights < 1.5 * (bin_size):
                        break
                    sum_of_weights = 0
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
        row = ["quality", "subgroup"]
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

def resultsAsDataFrame (data, result, statisticsToShow=all_statistics, autoround=False, weightingAttribute=None, includeTarget=False):
    res = resultAsTable(data, result, statisticsToShow, weightingAttribute, True, includeTarget)
    headers = res.pop(0)
    df = pd.DataFrame(res, columns=headers, dtype=np.float64)
    if (autoround):
        df = results_df_autoround(df)
    return df

def conditional_invert (val, invert):
    return - 2* (invert -0.5) * val


def results_df_autoround (df):
    return df.round({
                'quality' : 3,
                'size_sg' : 0,
                'size_dataset' : 0,
                'positives_sg': 0,
                'positives_dataset':0,
                'size_complement' : 0,
                'relative_size_sg':3,
                'relative_size_complement':3,
                'coverage_sg':3,
                'coverage_complement':3,
                'target_share_sg':3,
                'target_share_complement':3,
                'target_share_dataset':3,
                'lift':3,
                  
                'size_sg_weighted': 1,
                'size_dataset_weighted':1,
                'positives_sg_weighted':1,
                'positives_dataset_weighted':1,
                'size_complement_weighted':1,
                'relative_size_sg_weighted':3,
                'relative_size_complement_weighted':3,
                'coverage_sg_weighted':3,
                'coverage_complement_weighted':3,
                'target_share_sg_weighted':3,
                'target_share_complement_weighted':3,
                'target_share_dataset_weighted':3,
                'lift_weighted':3})


def perc_formatter (x):
    return "{0:.1f}%".format(x * 100)

def float_formatter (x, digits=2):
    return ("{0:." + str(digits) + "f}").format(x)

def to_latex (data, result, statistics_to_show):
    df = resultsAsDataFrame (data, result) [statistics_to_show]
    latex = df.to_latex(index=False, col_space=10, formatters={
                'quality' : partial (float_formatter, digits=3),
                'size_sg' : partial (float_formatter, digits=0),
                'size_dataset' : partial (float_formatter, digits=0),
                'positives_sg': partial (float_formatter, digits=0),
                'positives_dataset':partial (float_formatter, digits=0),
                'size_complement' : partial (float_formatter, digits=0),
                'relative_size_sg':perc_formatter,
                'relative_size_complement':perc_formatter,
                'coverage_sg':perc_formatter,
                'coverage_complement':perc_formatter,
                'target_share_sg':perc_formatter,
                'target_share_complement':perc_formatter,
                'target_share_dataset':perc_formatter,
                'lift':partial (float_formatter, digits=1),
                  
                'size_sg_weighted': partial (float_formatter, digits=1),
                'size_dataset_weighted':partial (float_formatter, digits=1),
                'positives_sg_weighted':partial (float_formatter, digits=1),
                'positives_dataset_weighted':partial (float_formatter, digits=1),
                'size_complement_weighted':partial (float_formatter, digits=1),
                'relative_size_sg_weighted':perc_formatter,
                'relative_size_complement_weighted':perc_formatter,
                'coverage_sg_weighted': perc_formatter,
                'coverage_complement_weighted' : perc_formatter,
                'target_share_sg_weighted': perc_formatter,
                'target_share_complement_weighted': perc_formatter,
                'target_share_dataset_weighted': perc_formatter,
                'lift_weighted': perc_formatter}
    )
    latex = latex.replace (' AND ', ' $\wedge$ ')
    return latex


def isCategoricalAttribute (data, attribute_name):    
    return attribute_name in data.select_dtypes(exclude=['number']).columns.values

def isNumericalAttribute (data, attribute_name):
    return attribute_name in data.select_dtypes(include=['number']).columns.values

def remove_selectors_with_attributes(selector_list, attribute_list):
    return [x for x in selector_list if not x.attributeName in attribute_list]

def effective_sample_size(weights):
    return sum(weights) ** 2 / sum(weights ** 2)

# from https://docs.python.org/3/library/itertools.html#recipes
def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)))

