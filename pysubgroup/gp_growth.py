import copy
from itertools import combinations, chain
from heapq import heappush, heappop
from collections import Counter, namedtuple
from collections import defaultdict
import numpy as np
import pysubgroup as ps

from pysubgroup.tests.DataSets import get_credit_data

data = get_credit_data()

searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['class'])
searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['class'])
searchSpace = searchSpace_Nominal + searchSpace_Numeric
target = ps.FITarget()
task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=12, depth=5, qf=ps.CountQF())

def get_path(node):
    ref=node
    path=[]
    while True:
        path.append(ref.cls)
        ref = ref.parent
        if ref is None:
            break
    return path
class GP_Growth:
    def execute(self, task):

        l=[]
        for selector in task.search_space:
            cov_arr=selector.covers(data)
            l.append( (np.count_nonzero(cov_arr), selector, cov_arr))
        s=sorted(l, reverse=True)
        selectors_sorted = [selector for size, selector, arr in s]
        arrs = np.vstack(list(arr for size, selector, arr in s)).T
        node_classes = [[] for _ in range(len(selectors_sorted))]
        GP_node = namedtuple('GP_node', ['cls', 'node_id', 'parent', 'children', 'stats', 'curr_children', 'curr_stats'])
        root = GP_node(-1,-1, None, {}, {"size":0}, [], 0)
        nodes = []
        for row_index, row in enumerate(arrs):
            print(row_index)
            nn = np.nonzero(row)[0]
            node = root
            for pos, child_index in enumerate(nn):
                node.stats['size'] += 1
                if pos < len(nn):
                    if child_index not in node.children:
                        new_node = GP_node(child_index, len(nodes), node, {}, {"size":0}, [], 0)
                        nodes.append(new_node)
                        node.children[child_index] = new_node
                        node_classes[child_index].append(new_node)

                    node = node.children[child_index]
        node = nodes[10]
        print("{} {}".format(get_path(node), node.stats['size']))
        print("done") 
    def recurse(self, root, cls, prefix, node_classes):
        for node in node_classes[cls]:
            self.pass_upwards(node, node.curr_stats)
    def pass_upwards(self, ref, values):
        path = []
        while ref is not None:
            path.append(ref)
            for stat, value in values.items:
                ref[stat] += value
            ref = ref.parent
        return path
GP_Growth().execute(task)
        
