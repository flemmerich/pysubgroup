from collections import  namedtuple
from collections import defaultdict
import numpy as np
import pysubgroup as ps


from pysubgroup.tests.DataSets import get_credit_data

data = get_credit_data()

searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['class'])
searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['class'])
searchSpace = searchSpace_Nominal + searchSpace_Numeric
target = ps.FITarget()
task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=30, depth=5, qf=ps.CountQF())
def get_path(node):
    ref = node
    path = []
    while True:
        path.append(ref.cls)
        ref = ref.parent
        if ref is None:
            break
    return path



class GP_Growth:
    def __init__(self):
        self.GP_node = namedtuple('GP_node', ['cls', 'id', 'parent', 'children', 'stats'])
        self.minSupp = 500

    def prepare_selectors(self, search_space):
        self.get_stats = task.qf.gp_get_stats
        self.get_null_vector = task.qf.gp_get_null_vector
        self.merge = task.qf.gp_merge
        l = []
        for selector in search_space:
            cov_arr = selector.covers(data)
            l.append((np.count_nonzero(cov_arr), selector, cov_arr))
        l = [(size, selector, arr) for size, selector, arr in l if size > self.minSupp]

        s = sorted(l, reverse=True)
        selectors_sorted = [selector for size, selector, arr in s]
        arrs = np.vstack(list(arr for size, selector, arr in s)).T
        return selectors_sorted, arrs

    def execute(self, task): 
        selectors_sorted, arrs = self.prepare_selectors(task.search_space)
        GP_node = self.GP_node
        
        root = GP_node(-1, -1, None, {}, self.get_null_vector())
        nodes = []
        for row_index, row in enumerate(arrs):
            d = self.get_stats(row_index)
            nn = np.nonzero(row)[0]
            node = root
            for cls in nn:
                if cls not in node.children:
                    new_child = GP_node(cls, len(nodes), node, {}, self.get_null_vector())
                    nodes.append(new_child)
                    node.children[cls] = new_child
                self.merge(node.stats, d)
                node = node.children[cls]
            self.merge(node.stats, d)
        nodes.append(root)

        cls_nodes = defaultdict(list)
        for node in nodes:
            cls_nodes[node.cls].append(node)
        print(self.get_stats_for_class(cls_nodes))
        
        patterns = self.recurse(cls_nodes, [])
        out = []
        for indices, stats in sorted(patterns, key=lambda x: x[1]["size"]):
            selectors = [selectors_sorted[i] for i in indices]
            print(selectors, stats)
            out.append((stats["size"], ps.Conjunction(selectors)))
        return out


    def check_constraints(self, node):
        return node['size'] > self.minSupp

    def recurse(self, cls_nodes, prefix):
        if len(cls_nodes) == 0:
            raise RuntimeError
        results = []
        stats_dict = self.get_stats_for_class(cls_nodes)
        
        results.append((prefix, cls_nodes[-1][0].stats))
        for cls, nodes in cls_nodes.items():
            if cls >= 0:
                if self.check_constraints(stats_dict[cls]):
                    new_tree = self.create_new_tree_from_nodes(nodes)
                    if len(new_tree) > 0:
                        results.extend(self.recurse(new_tree, (*prefix, cls)))
        return results

    def get_stats_for_class(self, cls_nodes):
        out = {}
        for key, nodes in cls_nodes.items():
            s = self.get_null_vector()
            for node in nodes:
                self.merge(s, node.stats)
            out[key] = s
        return out


    def create_new_tree_from_nodes(self, nodes):
        new_nodes = {}
        for node in nodes:
            nodes_upwards = self.get_nodes_upwards(node)
            self.create_copy_of_path(nodes_upwards[1:], new_nodes, node.stats)
 
        #self.remove_infrequent_nodes(new_nodes)
        cls_nodes = defaultdict(list)
        for new_node in new_nodes.values():
            cls_nodes[new_node.cls].append(new_node)
        return cls_nodes

    def remove_infrequent_nodes(self, new_nodes):
        keys = list(new_nodes.keys())
        for key in keys:
            node = new_nodes[key]
            if node.stats["size"] < self.minSupp:
                del new_nodes[key]

    def create_copy_of_path(self, nodes, new_nodes, stats):
        parent = None
        for node in reversed(nodes):
            if node.id not in new_nodes:
                new_node = self.GP_node(node.cls, node.id, parent, {}, self.get_null_vector())
                new_nodes[node.id] = new_node
            else:
                new_node = new_nodes[node.id]
            for key, val in stats.items():
                new_node.stats[key] += val
            if parent is not None:
                parent.children[new_node.cls] = new_node
            parent = new_node
        
    def get_nodes_upwards(self, node):
        ref = node
        path = []
        while True:
            path.append(ref)
            ref = ref.parent
            if ref is None:
                break
        return path



gp = GP_Growth().execute(task)
for i in range(5):
    print()
dfs1 = ps.SimpleDFS().execute(task)
dfs = []
for size, sg in dfs1:
    print('{}    {}'.format(sg.subgroup_description, size))
    dfs.append((size, sg.subgroup_description))
print(len(dfs))
print(len(gp))
for l, r in zip(reversed(gp), dfs):
    print(l)
    print(r)
    assert(l==r)
