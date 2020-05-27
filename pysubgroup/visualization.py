from functools import partial

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt

import pysubgroup as ps


def plot_sgbars(result_df, _, ylabel="target share", title="Discovered Subgroups", dynamic_widths=False, _suffix=""):
    shares_sg = result_df["target_share_sg"]
    shares_compl = result_df["target_share_complement"]
    sg_relative_sizes = result_df["relative_size_sg"]
    x = np.arange(len(result_df))

    base_width = 0.8
    if dynamic_widths:
        width_sg = 0.02 + base_width * sg_relative_sizes
        width_compl = base_width - width_sg
    else:
        width_sg = base_width / 2
        width_compl = base_width / 2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, shares_sg, width_sg, align='edge')
    rects2 = ax.bar(x + width_sg, shares_compl, width_compl, align='edge', color='#61b76f')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + base_width / 2)
    ax.set_xticklabels(result_df.index, rotation=90)

    ax.legend((rects1[0], rects2[0]), ('subgroup', 'complement'))
    fig.set_size_inches(12, len(result_df))

    return fig


def plot_roc(result_df, data, qf=ps.StandardQF(0.5), levels=40, annotate=False):
    instances_dataset = len(data)
    positives_dataset = np.max(result_df['positives_dataset'])
    negatives_dataset = instances_dataset - positives_dataset

    xlist = np.linspace(0.01, 0.99, 100)
    ylist = np.linspace(0.01, 0.99, 100)
    X, Y = np.meshgrid(xlist, ylist)
    f = np.vectorize(partial(qf.evaluate, instances_dataset, positives_dataset), otypes=[np.float])
    Z = f(X * negatives_dataset + Y * positives_dataset, Y * positives_dataset)
    max_val = np.max([np.max(Z), -np.min(Z)])

    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap("bwr")

    plt.contourf(X, Y, Z, levels, cmap=cm, vmin=-max_val, vmax=max_val)

    for i, sg in result_df.iterrows():
        rel_positives_sg = sg['positives_sg'] / positives_dataset
        rel_negatives_sg = (sg['size_sg'] - sg['positives_sg']) / negatives_dataset
        ax.plot(rel_negatives_sg, rel_positives_sg, 'ro', color='black')
        if annotate:
            label_margin = 0.01
            ax.annotate(str(i), (rel_negatives_sg + label_margin, rel_positives_sg + label_margin))

    # plt.colorbar(cp)
    plt.title('Discovered subgroups')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    return fig


def plot_npspace(result_df, data, annotate=True, fixed_limits=False):

    fig, ax = plt.subplots()

    for i, sg in result_df.iterrows():
        target_share_sg = sg['target_share_sg']
        size_sg = sg['size_sg']
        ax.plot(size_sg, target_share_sg, 'ro', color='black')
        if annotate:
            ax.annotate(str(i), (size_sg + 5, target_share_sg + 0.001))

    if fixed_limits:
        plt.xlim((0, len(data)))
        plt.ylim((0, 1))

    plt.title('Discovered subgroups')
    plt.xlabel('Size of Subgroup')
    plt.ylabel('Target Share Subgroup')

    return fig


def plot_distribution_numeric(sg, data, bins):
    fig, _ = plt.subplots()
    target_values_sg = data[sg.covers(data)][sg.target.get_attributes()].values
    target_values_data = data[sg.target.get_attributes()].values
    plt.hist(target_values_sg, bins, alpha=0.5, label=str(sg.subgroup_description), density=True)
    plt.hist(target_values_data, bins, alpha=0.5, label="Overall Data", density=True)
    plt.legend(loc='upper right')
    return fig


def compare_distributions_numeric(sgs, data, bins):
    fig, _ = plt.subplots()
    for sg in sgs:
        target_values_sg = data[sg.covers(data)][sg.target.get_attributes()].values
        plt.hist(target_values_sg, bins, alpha=0.3, label=str(sg.subgroup_description), density=True)
    plt.legend(loc='upper right')
    return fig


def similarity_sgs(sgd_results, data, color=True):
    sgs = [x[1] for x in sgd_results]
    #sgNames = [str(sg.subgroup_description) for sg in sgs]
    dists = [[ps.overlap(sg, sg2, data) for sg2 in sgs] for sg in sgs]
    dist_df = pd.DataFrame(dists)
    if color:
        dist_df = dist_df.style.background_gradient()
    return dist_df


def similarity_dendrogram(result, data):
    fig, _ = plt.subplots()
    dist_df = similarity_sgs(result, data, color=False)
    mat = 1 - dist_df.values
    dists = squareform(mat)
    linkage_matrix = linkage(dists, "single")
    dendrogram(linkage_matrix, labels=dist_df.index)
    return fig

def supportSetVisualization(result, in_order=True, drop_empty=True):
    df = result.task.data
    n_items = len(result.task.data)
    n_SGDs = len(result.results)
    covs = np.zeros((n_items, n_SGDs), dtype=bool)
    for i, (_, r, _) in enumerate(result.to_subgroups):
        covs[:, i] = r.covers(df)

    img_arr = covs.copy()

    sort_inds_x = np.argsort(np.sum(covs, axis=1))[::-1]
    img_arr = img_arr[sort_inds_x, :]
    if not in_order:
        sort_inds_y = np.argsort(np.sum(covs, axis=0))
        img_arr = img_arr[:, sort_inds_y]
    if drop_empty:
        keep_entities = np.sum(img_arr, axis=1) > 0
        print("Discarding {} entities that are not covered".format(n_items - np.count_nonzero(keep_entities)))
        img_arr = img_arr[keep_entities, :]
    return img_arr.T
