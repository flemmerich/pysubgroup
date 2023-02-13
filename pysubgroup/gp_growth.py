from collections import namedtuple, defaultdict
from itertools import combinations
import numpy as np
import pysubgroup as ps
import itertools

def identity(x, *args, **kwargs):#pylint: disable=unused-argument
    return x


class GpGrowth:

    def __init__(self, mode='b_u' ):
        self.GP_node = namedtuple('GP_node', ['cls', 'id', 'parent', 'children', 'stats'])
        self.minSupp = 10
        self.tqdm = identity
        self.depth = 0
        self.mode = mode  #specify eihther b_u (bottom up) or t_d (top down)
        self.constraints_monotone = []
        # Future: There also is the option of a stable mode which never creates the prefix trees


    def prepare_selectors(self, search_space, data):
        l = []
        for selector in search_space:
            cov_arr = selector.covers(data)
            l.append((np.count_nonzero(cov_arr), selector, cov_arr))

        l = [(size, selector, arr) for size, selector, arr in l if all(constraint.is_satisfied(arr, None, data) for constraint in self.constraints_monotone)]
        s = sorted(l, reverse=True)
        selectors_sorted = [selector for size, selector, arr in s]
        if len(selectors_sorted)==0:
            arrs = np.empty((0,0),dtype=np.bool_)
        else:
            arrs = np.vstack([arr for size, selector, arr in s]).T
        return selectors_sorted, arrs

    def nodes_to_cls_nodes(self, nodes):
        cls_nodes = defaultdict(list)
        for node in nodes:
            cls_nodes[node.cls].append(node)
        return cls_nodes


    def setup_from_quality_function(self, qf):
        # pylint: disable=attribute-defined-outside-init
        self.get_stats = qf.gp_get_stats
        self.get_null_vector = qf.gp_get_null_vector
        self.merge = qf.gp_merge
        self.requires_cover_arr = qf.gp_requires_cover_arr
        # pylint: enable=attribute-defined-outside-init

    def setup_constraints(self, constraints, qf):

        self.constraints_monotone = constraints
        for constraint in self.constraints_monotone:
            constraint.gp_prepare(qf)

        if len(constraints)==1:
            self.check_constraints = constraints[0].gp_is_satisfied

    
    def check_constraints(self, node): #pylint: disable=method-hidden
        return all(constraint.gp_is_satisfied(node) for constraint in self.constraints_monotone)


    def setup(self, task):
        task.qf.calculate_constant_statistics(task.data, task.target)
        self.depth = task.depth
        self.setup_constraints(task.constraints_monotone, task.qf)

        self.setup_from_quality_function(task.qf)
        
    def create_initial_tree(self, arrs):
        # Create tree
        root = self.GP_node(-1, -1, None, {}, self.get_null_vector())
        nodes = []
        for row_index, row in self.tqdm(enumerate(arrs), 'creating tree', total=len(arrs)):
            self.normal_insert(root, nodes, self.get_stats(row_index), np.nonzero(row)[0])
        nodes.append(root)
        return root, nodes


    def execute(self, task):
        assert self.mode in ('b_u', 't_d')
        self.setup(task)

        selectors_sorted, arrs = self.prepare_selectors(task.search_space, task.data)
        root, nodes = self.create_initial_tree(arrs)



        # mine tree
        cls_nodes = self.nodes_to_cls_nodes(nodes)
        if self.mode == 'b_u':
            patterns = self.recurse(cls_nodes, [])
        elif self.mode == 't_d':
            patterns = self.recurse_top_down(cls_nodes, root)
        else:
            raise RuntimeError('mode needs to be either b_u or t_d')

        # compute quality functions
        result = self.calculate_quality_function_for_patterns(task, patterns, selectors_sorted, arrs)
        
        result = ps.prepare_subgroup_discovery_result(result, task)
        return ps.SubgroupDiscoveryResult(result, task)

    def calculate_quality_function_for_patterns(self, task, patterns, selectors_sorted, arrs):
        out = []
        for indices, gp_params in self.tqdm(patterns, 'computing quality function',):
            if len(indices) > 0:
                selectors = [selectors_sorted[i] for i in indices]
                #print(selectors, stats)
                sg = ps.Conjunction(selectors)
                if self.requires_cover_arr:
                    if len(indices)==1:
                        cover_arr = arrs[:,indices[0]]
                    else:
                        cover_arr = np.all([arrs[:,i] for i in indices])
                    statistics = task.qf.gp_get_params(cover_arr, gp_params)
                else:
                    statistics = task.qf.gp_get_params(None, gp_params)
                #qual1 = task.qf.evaluate(sg, task.qf.calculate_statistics(sg, task.data))
                qual2 = task.qf.evaluate(sg, task.target, task.data, statistics)
                out.append((qual2, sg, statistics))
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

    def recurse(self, cls_nodes, prefix, is_single_path=False):
        if len(cls_nodes) == 0:
            raise RuntimeError
        results = []

        results.append((prefix, cls_nodes[-1][0].stats))
        if len(prefix) >= self.depth:
            return results

        stats_dict = self.get_stats_for_class(cls_nodes)
        if is_single_path:
            if len(cls_nodes)==1 and -1 in cls_nodes:
                return results
            del stats_dict[-1] # remove root node
            all_combinations = ps.powerset(stats_dict.keys(), max_length=self.depth - len(prefix)+1)
            for comb in all_combinations:
                # it might still be, that stats_dict[comb[-1]] is wrong if that is the case then
                # stats_dict[comb[0]] is correct
                if len(comb)>0:
                    results.append((prefix+comb, stats_dict[comb[-1]]))
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
        self.setup(task)
        _, arrs = self.prepare_selectors(task.search_space, task.data)

        # Create tree
        to_str = task.qf.gp_to_str
        with open(path, 'w', encoding="utf-8") as f:
            for row_index, row in self.tqdm(enumerate(arrs), 'creating tree', total=len(arrs)):
                #print(np.nonzero(row)[0])
                f.write(" ".join(map(str, np.nonzero(row)[0])) + " "+ to_str(self.get_stats(row_index))+"\n")
