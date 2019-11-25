from collections import  namedtuple, defaultdict
from itertools import combinations
import numpy as np
import pysubgroup as ps
from tqdm import tqdm


class GP_Growth:

    def __init__(self):
        self.GP_node = namedtuple('GP_node', ['cls', 'id', 'parent', 'children', 'stats'])
        self.minSupp = 200
        self.tqdm = tqdm
        self.depth = 0

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
        task.qf.calculate_constant_statistics(task)
        self.depth = task.depth
        selectors_sorted, arrs = self.prepare_selectors(task.search_space)
        GP_node = self.GP_node

        root = GP_node(-1, -1, None, {}, self.get_null_vector())
        nodes = []
        for row_index, row in self.tqdm(enumerate(arrs), 'creating tree', total=len(arrs)):
            new_stats = self.get_stats(row_index)
            classes = np.nonzero(row)[0]
            self.normal_insert(root, nodes, new_stats, classes)
        nodes.append(root)
        cls_nodes = defaultdict(list)
        for node in nodes:
            cls_nodes[node.cls].append(node)

        patterns = self.recurse(cls_nodes, [])
        out = []
        for indices, gp_params in self.tqdm(patterns, 'computing quality function',):
            if len(indices) > 0:
                selectors = [selectors_sorted[i] for i in indices]
                #print(selectors, stats)
                sg = ps.Conjunction(selectors)
                statistics = task.qf.gp_get_params(sg.covers(data), gp_params)
                #qual1 = task.qf.evaluate(sg, task.qf.calculate_statistics(sg, task.data))
                qual2 = task.qf.evaluate(sg, statistics)
                out.append((qual2, sg))
        return out

    def normal_insert(self, root, nodes, new_stats, classes):
        node = root
        for cls in classes:
            if cls not in node.children:
                new_child = self.GP_node(cls, len(nodes), node, {}, self.get_null_vector())
                nodes.append(new_child)
                node.children[cls] = new_child
            self.merge(node.stats, new_stats)
            node = node.children[cls]
        self.merge(node.stats, new_stats)
        return node

    def insert_into_tree(self, root, nodes, new_stats, classes, max_depth):
        ''' Creates a tree of a maximum depth = depth
        '''
        if len(classes) <= max_depth:
            self.normal_insert(root, nodes, new_stats, classes)
            return
        for prefix in combinations(classes, max_depth -1):
            node = self.normal_insert(root, nodes, new_stats, classes)
            # do normal insert for prefix
            index_for_remaining = classes.index(prefix) + 1
            for cls in classes[index_for_remaining:]:
                if cls not in node.children:
                    new_child = self.GP_node(cls, len(nodes), node, {}, self.get_null_vector())
                    nodes.append(new_child)
                    node.children[cls] = new_child
                    self.merge(node.stats, new_stats)


    def check_constraints(self, node):
        return node[0] > self.minSupp

    def recurse(self, cls_nodes, prefix):
        if len(cls_nodes) == 0:
            raise RuntimeError
        results = []
        stats_dict = self.get_stats_for_class(cls_nodes)
        
        results.append((prefix, cls_nodes[-1][0].stats))
        if len(prefix) >= self.depth:
            return results
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
            self.merge(new_node.stats, stats)
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

if __name__ == '__main__':
    from pysubgroup.tests.DataSets import get_credit_data
    from pysubgroup import model_target

    data = get_credit_data()
    #warnings.filterwarnings("error")
    print(data.columns)
    searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['duration', 'credit_amount'])
    searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['duration', 'credit_amount'])
    searchSpace = searchSpace_Nominal + searchSpace_Numeric
    target = ps.FITarget()
    QF=model_target.EMM_Likelihood(model_target.PolyRegression_ModelClass(x_name='duration', y_name='credit_amount'))
    task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=50, depth=3, qf=QF)


    import time
    start_time = time.time()
    gp = GP_Growth().execute(task)
    print("--- %s seconds ---" % (time.time() - start_time))
    #gp = [(qual, sg) for qual, sg in gp if sg.depth <= task.depth]
    gp = sorted(gp)


    start_time = time.time()
    dfs1 = ps.SimpleDFS().execute(task)
    print("--- %s seconds ---" % (time.time() - start_time))
    dfs = [(qual, sg.subgroup_description) for qual, sg in dfs1]
    dfs = sorted(dfs, reverse=True)
    gp = sorted(gp, reverse=True)


    gp = gp[:task.result_set_size]
    for l, r in zip(gp, dfs):
        print('gp:', l)
        print('df:', r)
        assert(abs(l[0]-r[0]) < 0.00000001)
        assert(l[1] == r[1])
