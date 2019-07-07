'''
Created on 02.05.2016

@author: lemmerfn
'''
import itertools
from functools import partial
from heapq import heappush, heappop

import numpy as np
import pandas as pd

all_statistics = ('size_sg', 'size_dataset', 'positives_sg', 'positives_dataset', 'size_complement', 'relative_size_sg',
                  'relative_size_complement', 'coverage_sg', 'coverage_complement', 'target_share_sg',
                  'target_share_complement', 'target_share_dataset', 'lift')
all_statistics_weighted = all_statistics + (
    'size_sg_weighted', 'size_dataset_weighted', 'positives_sg_weighted', 'positives_dataset_weighted',
    'size_complement_weighted', 'relative_size_sg_weighted', 'relative_size_complement_weighted',
    'coverage_sg_weighted',
    'coverage_complement_weighted', 'target_share_sg_weighted', 'target_share_complement_weighted',
    'target_share_dataset_weighted', 'lift_weighted')
all_statistics_numeric = (
    'size_sg', 'size_dataset', 'mean_sg', 'mean_dataset', 'std_sg', 'std_dataset', 'median_sg', 'median_dataset',
    'max_sg',
    'max_dataset', 'min_sg', 'min_dataset', 'mean_lift', 'median_lift')

complex_statistics = ('sg_size', 'dataset_size', 'complement_sg_size', 'sg_corr',
                      'dataset_corr', 'complement_sg_corr', 'corr_lift',)


def add_if_required(result, sg, quality, task, check_for_duplicates=False):
    if quality > task.min_quality:
        if check_for_duplicates and (quality, sg) in result:
            return
        if len(result) < task.result_set_size:
            heappush(result, (quality, sg))
        elif quality > result[0][0]:
            heappop(result)
            heappush(result, (quality, sg))


def minimum_required_quality(result, task):
    if len(result) < task.result_set_size:
        return task.min_quality
    else:
        return result[0][0]


# Returns the cutpoints for discretization
def equal_frequency_discretization(data, attribute_name, nbins=5, weighting_attribute=None):
    cutpoints = []
    if weighting_attribute is None:
        cleaned_data = data[attribute_name]
        cleaned_data = cleaned_data[~np.isnan(cleaned_data)]
        sorted_data = sorted(cleaned_data)
        number_instances = len(sorted_data)
        for i in range(1, nbins):
            position = i * number_instances // nbins
            while True:
                if position >= number_instances:
                    break
                val = sorted_data[position]
                if val not in cutpoints:
                    break
                position += 1
            # print (sorted_data [position])
            if val not in cutpoints:
                cutpoints.append(val)
    else:
        cleaned_data = data[[attribute_name, weighting_attribute]]
        cleaned_data = cleaned_data[~np.isnan(cleaned_data[attribute_name])]
        cleaned_data.sort(order=attribute_name)

        overall_weights = cleaned_data[weighting_attribute].sum()
        remaining_weights = overall_weights
        bin_size = overall_weights / nbins
        sum_of_weights = 0
        for row in cleaned_data:
            sum_of_weights += row[weighting_attribute]
            if sum_of_weights > bin_size:
                if not row[attribute_name] in cutpoints:
                    cutpoints.append(row[attribute_name])
                    remaining_weights = remaining_weights - sum_of_weights
                    if remaining_weights < 1.5 * (bin_size):
                        break
                    sum_of_weights = 0
    return cutpoints


def print_result_set(data, result, statistics_to_show, weighting_attribute=None, print_header=True,
                     include_target=False):
    if print_header:
        s = "Quality\tSubgroup"
        for stat in statistics_to_show:
            s += "\t" + stat
        print(s)
    for (q, sg) in result:
        sg.calculate_statistics(data, weighting_attribute)
        s = str(q) + ":\t" + str(sg.subgroup_description)
        if include_target:
            s += str(sg.target)
        for stat in statistics_to_show:
            s += "\t" + str(sg.statistics[stat])
        print(s)


def result_as_table(data, result, statistics_to_show, weighting_attribute=None, print_header=True,
                    include_target=False, complex_target=False):
    table = []
    if print_header:
        row = ["quality", "subgroup"]
        for stat in statistics_to_show:
            row.append(stat)
        table.append(row)
    for (q, sg) in result:
        if complex_target:
            sg.calculate_corr_statistics(data)
        else:
            sg.calculate_statistics(data, weighting_attribute)
        row = [str(q), str(sg.subgroup_description)]
        if include_target:
            row.append(str(sg.target))
        for stat in statistics_to_show:
            row.append(str(sg.statistics[stat]))
        table.append(row)
    return table


def results_as_df(data, result, statistics_to_show=all_statistics, autoround=False, weighting_attribute=None,
                  include_target=False, complex_target=False):
    res = result_as_table(data, result, statistics_to_show, weighting_attribute, True, include_target, complex_target)
    headers = res.pop(0)
    df = pd.DataFrame(res, columns=headers, dtype=np.float64)
    if autoround:
        df = results_df_autoround(df)
    return df


as_df = results_as_df


def conditional_invert(val, invert):
    return - 2 * (invert - 0.5) * val


def results_df_autoround(df):
    return df.round({
        'quality': 3,
        'size_sg': 0,
        'size_dataset': 0,
        'positives_sg': 0,
        'positives_dataset': 0,
        'size_complement': 0,
        'relative_size_sg': 3,
        'relative_size_complement': 3,
        'coverage_sg': 3,
        'coverage_complement': 3,
        'target_share_sg': 3,
        'target_share_complement': 3,
        'target_share_dataset': 3,
        'lift': 3,

        'size_sg_weighted': 1,
        'size_dataset_weighted': 1,
        'positives_sg_weighted': 1,
        'positives_dataset_weighted': 1,
        'size_complement_weighted': 1,
        'relative_size_sg_weighted': 3,
        'relative_size_complement_weighted': 3,
        'coverage_sg_weighted': 3,
        'coverage_complement_weighted': 3,
        'target_share_sg_weighted': 3,
        'target_share_complement_weighted': 3,
        'target_share_dataset_weighted': 3,
        'lift_weighted': 3})


def perc_formatter(x):
    return "{0:.1f}%".format(x * 100)


def float_formatter(x, digits=2):
    return ("{0:." + str(digits) + "f}").format(x)


def to_latex(data, result, statistics_to_show):
    df = results_as_df(data, result)[statistics_to_show]
    latex = df.to_latex(index=False, col_space=10, formatters={
        'quality': partial(float_formatter, digits=3),
        'size_sg': partial(float_formatter, digits=0),
        'size_dataset': partial(float_formatter, digits=0),
        'positives_sg': partial(float_formatter, digits=0),
        'positives_dataset': partial(float_formatter, digits=0),
        'size_complement': partial(float_formatter, digits=0),
        'relative_size_sg': perc_formatter,
        'relative_size_complement': perc_formatter,
        'coverage_sg': perc_formatter,
        'coverage_complement': perc_formatter,
        'target_share_sg': perc_formatter,
        'target_share_complement': perc_formatter,
        'target_share_dataset': perc_formatter,
        'lift': partial(float_formatter, digits=1),

        'size_sg_weighted': partial(float_formatter, digits=1),
        'size_dataset_weighted': partial(float_formatter, digits=1),
        'positives_sg_weighted': partial(float_formatter, digits=1),
        'positives_dataset_weighted': partial(float_formatter, digits=1),
        'size_complement_weighted': partial(float_formatter, digits=1),
        'relative_size_sg_weighted': perc_formatter,
        'relative_size_complement_weighted': perc_formatter,
        'coverage_sg_weighted': perc_formatter,
        'coverage_complement_weighted': perc_formatter,
        'target_share_sg_weighted': perc_formatter,
        'target_share_complement_weighted': perc_formatter,
        'target_share_dataset_weighted': perc_formatter,
        'lift_weighted': perc_formatter}
                        )
    latex = latex.replace(' AND ', ' $\wedge$ ')
    return latex


def is_categorical_attribute(data, attribute_name):
    return attribute_name in data.select_dtypes(exclude=['number']).columns.values


def is_numerical_attribute(data, attribute_name):
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


def overlap(sg, another_sg, data):
    cover_sg = sg.covers(data)
    cover_another_sg = another_sg.covers(data)
    union = np.logical_or(cover_sg, cover_another_sg)
    intercept = np.logical_and(cover_sg, cover_another_sg)
    sim = np.sum(intercept) / np.sum(union)
    return sim


#####
# bitset operations
#####
def to_bits(list_of_ints):
    v = 0
    for x in list_of_ints:
        v += 1 << x
    return v


def count_bits(bitset_as_int):
    c = 0
    while bitset_as_int > 0:
        c += 1
        bitset_as_int &= bitset_as_int - 1
    return c


def find_set_bits(bitset_as_int):
    while bitset_as_int > 0:
        x = bitset_as_int.bit_length() - 1
        yield x
        bitset_as_int = bitset_as_int - (1 << x)


#####
# TID-list operations
#####
def intersect_of_ordered_list(list_1, list_2):
    result = []
    i = 0
    j = 0
    while i < len(list_1) and j < len(list_2):
        if list_1[i] < list_2[j]:
            i += 1
        elif list_2[j] < list_1[i]:
            j += 1
        else:
            result.append(list_1[i])
            j += 1
            i += 1
    return result
