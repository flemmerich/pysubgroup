from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import pysubgroup.boolean_target as bt

def plot_sgbars (sgs, shares_sg, shares_compl, sg_relative_sizes, ylabel="target share", title="Discovered Subgroups", dynamic_widths=False, suffix=""):
    x = np.arange (len(sgs))

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
    ax.set_xticklabels(sgs, rotation=90)
    

    ax.legend((rects1[0], rects2[0]), ('subgroup', 'complement'))
    fig.set_size_inches(12, len(sgs))
    
    return fig

def plot_roc (data, result_df, qf=bt.StandardQF(0.5), levels=40):
    instances_dataset = len(data)
    positives_dataset = np.max(result_df['positives_dataset'])
    negatives_dataset = instances_dataset - positives_dataset
    
    xlist = np.linspace(0.01, 0.99, 100)
    ylist = np.linspace(0.01, 0.99, 100)
    X, Y = np.meshgrid(xlist, ylist)
    f = np.vectorize(partial(qf.evaluateFromStatistics, instances_dataset, positives_dataset), otypes=[np.float])
    Z = f (X * negatives_dataset + Y * positives_dataset, Y * positives_dataset)
    max_val = np.max ([np.max(Z), -np.min(Z)])
        
            
    plt.figure()
    cm = plt.cm.bwr
    
    plt.contourf(X, Y, Z, levels, cmap=cm, vmin=-max_val, vmax=max_val)
    
    for i, sg in result_df.iterrows():
        rel_positives_sg = sg['positives_sg'] / positives_dataset
        rel_negatives_sg = (sg['size_sg'] - sg['positives_sg']) / negatives_dataset
        plt.plot(rel_negatives_sg, rel_positives_sg, 'ro', color='black')
    
    # plt.colorbar(cp)
    plt.title('Discovered subgroups')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    return plt.figure()

