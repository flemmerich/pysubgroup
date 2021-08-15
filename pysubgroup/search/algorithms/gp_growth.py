from collections import  namedtuple, defaultdict
from itertools import combinations
import numpy as np
import pysubgroup as ps
from tqdm import tqdm
from copy import copy
import itertools
class GpGrowth:

    def __init__(self, mode='b_u' ):
        self.GP_node = namedtuple('GP_node', ['cls', 'id', 'parent', 'children', 'stats'])
        self.minSupp = 10
        self.tqdm = tqdm
        self.depth = 0
        self.mode = mode  #specify eihther b_u (bottom up) or t_d (top down)
        # Future: There also is the option of a stable mode which never creates the prefix trees

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
        arrs = np.vstack([arr for size, selector, arr in s]).T
        return selectors_sorted, arrs

    def nodes_to_cls_nodes(self, nodes):
        cls_nodes = defaultdict(list)
        for node in nodes:
            cls_nodes[node.cls].append(node)
        return cls_nodes


    def execute(self, task):
        assert(self.mode in ('b_u', 't_d'))
        task.qf.calculate_constant_statistics(task)
        self.depth = task.depth
        selectors_sorted, arrs = self.prepare_selectors(task.search_space)
        self.requires_cover_arr = task.qf.gp_requires_cover_arr

        # Create tree
        root = self.GP_node(-1, -1, None, {}, self.get_null_vector())
        nodes = []
        for row_index, row in self.tqdm(enumerate(arrs), 'creating tree', total=len(arrs)):
            self.normal_insert(root, nodes, self.get_stats(row_index), np.nonzero(row)[0])
        nodes.append(root)

        # mine tree
        cls_nodes = self.nodes_to_cls_nodes(nodes)
        if self.mode == 'b_u':
            patterns = self.recurse(cls_nodes, [])
        elif self.mode == 't_d':
            patterns = self.recurse_top_down(cls_nodes, root)
        else:
            raise RuntimeError('mode needs to be either b_u or t_d')

        # compute quality functions
        return self.calculate_quality_function_for_patterns(patterns, selectors_sorted, arrs)

    def calculate_quality_function_for_patterns(self, patterns, selectors_sorted, arrs):
        out = []
        for indices, gp_params in self.tqdm(patterns, 'computing quality function',):
            if len(indices) > 0:
                selectors = [selectors_sorted[i] for i in indices]
                #print(selectors, stats)
                sg = ps.Conjunction(selectors)
                if self.requires_cover_arr:
                    statistics = task.qf.gp_get_params(np.all([arrs[i] for i in indices]), gp_params)
                else:
                    statistics = task.qf.gp_get_params(None, gp_params)
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
        #return node[0] >= self.minSupp
        return node['size'] >= self.minSupp

    def recurse(self, cls_nodes, prefix, is_single_path=False):
        if len(cls_nodes) == 0:
            raise RuntimeError
        results = []

        results.append((prefix, cls_nodes[-1][0].stats))
        if len(prefix) >= self.depth:
            return results
        
        stats_dict = self.get_stats_for_class(cls_nodes)
        if is_single_path:
            root_stats = cls_nodes[-1][0].stats
            del stats_dict[-1]
            all_combinations = ps.powerset(stats_dict.keys(), max_length=self.depth - len(prefix))
            
            for comb in all_combinations:
                results.append((prefix+comb, root_stats))
        else:
            for cls, nodes in cls_nodes.items():
                if cls >= 0:
                    if self.check_constraints(stats_dict[cls]):
                        if len(prefix) == (self.depth - 1):
                            results.append(((*prefix, cls), stats_dict[cls]))
                        else:
                            is_single_path_now = len(nodes) == 1
                            new_tree = self.create_new_tree_from_nodes(nodes)
                            if len(new_tree) > 0:
                                results.extend(self.recurse(new_tree, (*prefix, cls), is_single_path_now))
        return results

    def get_prefixes_top_down(self, alpha, max_length):
        if len(alpha) == 0:
            return [()]
        if len(alpha) == 1 or max_length == 1:
            return [(alpha[0],)]
        prefixes = [(alpha[0],)]
        prefixes.extend([(alpha[0], *suffix) for suffix in self.get_prefixes_top_down(alpha[1:], max_length-1)])
        return prefixes


    def recurse_top_down(self, cls_nodes, root, depth_in=0):

        alpha = []
        curr_depth = depth_in
        while True:
            if root.cls == -1:
               pass
            else:
                alpha.append(root.cls)
            if len(root.children) == 1 and curr_depth <= self.depth:
                curr_depth += 1
                root = next(iter(root.children.values()))
            else:
                break
        prefixes = self.get_prefixes_top_down(alpha, max_length=self.depth - depth_in + 1)

        # Bug: If we have a longer path that branches. eg. consider the tree from items A - B - C and A - B - D
        # and depth - depth_in == 2 then prefixes = [(A), (A, B)] but the sets
        # (A, C) and (A, D) are also valid
        # basically if we have prefixes of diffrent length this does not work properly
        if len(root.children) == 0 or curr_depth >= self.depth:
            results = []
            stats_dict = self.get_stats_for_class(cls_nodes)
            for prefix in prefixes:
                cls = max(prefix)
                if self.check_constraints(stats_dict[cls]):
                    results.append((prefix, stats_dict[cls]))
            return results
        else:
            suffixes = [((), root.stats)]
            stats_dict = self.get_stats_for_class(cls_nodes)
            for cls in cls_nodes:
                if cls >= 0 and cls not in alpha:
                    if self.check_constraints(stats_dict[cls]):
                        # Future: There is also the possibility to compute the stats_dict of the prefix tree
                        # without creating the prefix tree first
                        # This might be useful if curr_depth == self.depth - 2
                        # as we need not recreate the tree

                        if curr_depth == (self.depth - 1):
                            suffixes.append(((cls,), stats_dict[cls]))
                        else:
                            new_root, nodes = self.get_top_down_tree_for_class(cls_nodes, cls)
                            if len(nodes) > 0:
                                new_cls_nodes = self.nodes_to_cls_nodes(nodes)
                                print("  " * curr_depth, cls, curr_depth, len(new_cls_nodes))
                                suffixes.extend(self.recurse_top_down(new_cls_nodes, new_root, curr_depth+1))

        return [((*pre, *(suf[0])), suf[1]) for pre, suf in itertools.product(prefixes, suffixes)]

    def remove_infrequent_class(self, nodes, cls_nodes, stats_dict):
        # returns cleaned tree

        infrequent_classes = []
        for cls in cls_nodes:
            if not self.check_constraints(stats_dict[cls]):
                infrequent_classes.append(cls)
        infrequent_classes = sorted(infrequent_classes, reverse=True)
        for cls in infrequent_classes:
            for node_to_remove in cls_nodes[cls]:
                self.merge_trees_top_down(nodes, node_to_remove.parent, node_to_remove)



    def get_top_down_tree_for_class(self, cls_nodes, cls):
        # Future: Can eventually also remove infrequent nodes already during tree creation
        base_root = None
        nodes = []
        if len(cls_nodes[cls]) > 0:
            base_root = self.create_copy_of_tree_top_down(cls_nodes[cls][0], nodes)
            for other_root in cls_nodes[cls][1:]:
                self.merge_trees_top_down(nodes, base_root, other_root)
        return base_root, nodes

    def create_copy_of_tree_top_down(self, root, nodes=None, parent=None):
        if nodes is None:
            nodes = []
        #if len(nodes) == 0:
        #    root_cls = -1
        children = {}
        new_root = self.GP_node(root.cls, len(nodes), parent, children, root.stats.copy())
        nodes.append(new_root)
        for child_cls, child in root.children.items():
            new_child = self.create_copy_of_tree_top_down(child, nodes, new_root)
            children[child_cls] = new_child
        return new_root

    def merge_trees_top_down(self, nodes, mutable_root, other_root):
        self.merge(mutable_root.stats, other_root.stats)
        for cls in other_root.children:
            if cls not in mutable_root.children:
                self.create_copy_of_tree_top_down(other_root.children[cls], nodes, mutable_root)
            else:
                self.merge_trees_top_down(nodes, mutable_root.children[cls], other_root.children[cls])
        

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
                new_node = self.GP_node(node.cls, node.id, parent, {}, stats.copy())
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

    def to_file(self, task, path):
        task.qf.calculate_constant_statistics(task)
        self.depth = task.depth
        selectors_sorted, arrs = self.prepare_selectors(task.search_space)

        # Create tree
        root = self.GP_node(-1, -1, None, {}, self.get_null_vector())
        nodes = []
        with open(path, 'w') as f:
            for row_index, row in self.tqdm(enumerate(arrs), 'creating tree', total=len(arrs)):
                #print(np.nonzero(row)[0])
                f.write(" ".join(map(str, np.nonzero(row)[0])) + " "+ task.qf.gp_to_str(self.get_stats(row_index))+"\r\n")
        

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
    #QF=model_target.EMM_Likelihood(model_target.PolyRegression_ModelClass(x_name='duration', y_name='credit_amount'))
    QF=ps.CountQF()
    task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=200, depth=4, qf=QF)
    GpGrowth(mode='b_u').to_file(task,'E:/tmp/gp_credit.txt')

    import time
    start_time = time.time()
    gp = GpGrowth(mode='b_u').execute(task)
    print("--- %s seconds ---" % (time.time() - start_time))
    #gp = [(qual, sg) for qual, sg in gp if sg.depth <= task.depth]
    gp = sorted(gp)
    quit()

    start_time = time.time()
    dfs1 = ps.SimpleDFS().execute(task)
    print("--- %s seconds ---" % (time.time() - start_time))
    dfs = [(qual, sg.subgroup_description) for qual, sg in dfs1]
    dfs = sorted(dfs, reverse=True)
    gp = sorted(gp, reverse=True)

    def better_sorted(l):
        the_dict=defaultdict(list)
        prev_key=l[0][0]
        for key, val in l:
            
            if abs(prev_key-key)<10**-11:
                the_dict[prev_key].append(val)
            else:
                the_dict[key].append(val)
                prev_key = key
        print(len(the_dict))
        result = []
        for key, vals in the_dict.items():
            for val in sorted(vals):
                result.append((key, val))
        return result
    dfs = better_sorted(dfs)
    gp = better_sorted(gp)
    gp = gp[:task.result_set_size]

    for i, (l, r) in enumerate(zip(gp, dfs)):
        print(i)
        print('gp:', l)
        print('df:', r)
        assert(abs(l[0]-r[0]) < 10 ** -7)
        assert(l[1] == r[1])
