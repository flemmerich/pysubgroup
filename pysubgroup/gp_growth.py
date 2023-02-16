from collections import namedtuple, defaultdict
from pathlib import Path
import warnings
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
        self.results = []
        self.task = []
        # Future: There also is the option of a stable mode which never creates the prefix trees


    def prepare_selectors(self, search_space, data):
        l = []
        assert len(search_space) > 0, "Provided searchspace was empty"
        for selector in search_space:
            cov_arr = selector.covers(data)
            l.append((np.count_nonzero(cov_arr), selector, cov_arr))

        l = [(size, selector, arr) for size, selector, arr in l if all(constraint.is_satisfied(arr, slice(None), data) for constraint in self.constraints_monotone)]
        s = sorted(l, reverse=True)
        self.remove_selectors_with_low_optimistic_estimate(s, len(search_space))

        selectors_sorted = [selector for size, selector, arr in s]
        if len(selectors_sorted)==0:
            arrs = np.empty((0,0),dtype=np.bool_)
        else:
            arrs = np.vstack([arr for size, selector, arr in s]).T
        #print(selectors_sorted)
        return selectors_sorted, arrs


    def remove_selectors_with_low_optimistic_estimate(self, s, search_space_size):
        if not hasattr(self.task.qf, "optimistic_estimate"):
            return
        if search_space_size > self.task.result_set_size:
            # remove selectors which have to lo of an optimistic estimate
            #selectors_map = {selector : i for i,(_, selector, _) in enumerate(s)}
            stats=[]
            for _, _, cov_arr in s:
                statistics = self.task.qf.calculate_statistics(cov_arr, self.task.target, self.task.data)
                stats.append(statistics)
                quality = self.task.qf.evaluate(cov_arr, self.task.target, self.task.data, statistics)
                ps.add_if_required(self.results, None, quality, self.task, statistics=statistics)
            del statistics
            to_pop=[]
            min_quality = ps.minimum_required_quality(self.results, self.task)
            for i,((_, _, cov_arr), statistics) in enumerate(zip(s, stats)):
                if not self.task.qf.optimistic_estimate(cov_arr, self.task.target, self.task.data, statistics) > min_quality:
                    to_pop.append(i)
            self.task.min_quality = np.nextafter(float(min_quality), self.task.min_quality)
            for i in reversed(to_pop):
                s.pop(i)
            self.results.clear()



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

        if self.mode=="t_d" and len(self.constraints_monotone) == 0:
            warnings.warn("Poor runtime expected: Top down method does not use optimistic estimates and no constraints were provided", UserWarning)

        if len(constraints)==1:
            self.check_constraints = constraints[0].gp_is_satisfied


    def check_constraints(self, node): #pylint: disable=method-hidden
        return all(constraint.gp_is_satisfied(node) for constraint in self.constraints_monotone)

    def setup(self, task):
        self.task = task
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
        assert self.mode in ('b_u', 't_d'), 'mode needs to be either b_u or t_d'
        self.setup(task)

        selectors_sorted, arrs = self.prepare_selectors(task.search_space, task.data)
        root, nodes = self.create_initial_tree(arrs)



        # mine tree
        cls_nodes = self.nodes_to_cls_nodes(nodes)
        if self.mode == 'b_u':
            self.recurse(cls_nodes, tuple())
        elif self.mode == 't_d':
            results = self.recurse_top_down(cls_nodes, root)
            results = self.calculate_quality_function_for_patterns(task, results, arrs)
            for quality, sg, stats in results:
                ps.add_if_required(self.results, sg, quality, self.task, statistics=stats)
        self.results = self.convert_results_to_subgroups(self.results, selectors_sorted)

        self.results = ps.prepare_subgroup_discovery_result(self.results, task)
        return ps.SubgroupDiscoveryResult(self.results, task)


    def convert_results_to_subgroups(self, results, selectors_sorted):
        new_result = []
        for quality, indices, stats in results:
            selectors = [selectors_sorted[i] for i in indices]
            sg = ps.Conjunction(selectors)
            new_result.append((quality, sg, stats))
        return new_result



    def calculate_quality_function_for_patterns(self, task, results, arrs):
        out = []
        for indices, gp_params in self.tqdm(results, 'computing quality function',):
            if self.requires_cover_arr:
                if len(indices)==1:
                    cover_arr = arrs[:,indices[0]]
                else:
                    cover_arr = np.all([arrs[:,i] for i in indices])
                statistics = task.qf.gp_get_params(cover_arr, gp_params)
                sg = cover_arr
            else:
                statistics = task.qf.gp_get_params(None, gp_params)
                sg = None
            #qual1 = task.qf.evaluate(sg, task.qf.calculate_statistics(sg, task.data))
            qual2 = task.qf.evaluate(sg, task.target, task.data, statistics)
            out.append((qual2, indices, statistics))
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


    def add_if_required(self, prefix, gp_stats):
        statistics = self.task.qf.gp_get_params(None, gp_stats)
        quality = self.task.qf.evaluate(None, None, None, statistics)
        ps.add_if_required(self.results, prefix, quality, self.task, statistics=statistics)


    def recurse(self, cls_nodes, prefix, is_single_path=False):
        if len(cls_nodes) == 0:
            raise RuntimeError # pragma: no cover
        self.add_if_required(prefix, cls_nodes[-1][0].stats)
        if len(prefix) >= self.depth:
            return # pragma: no cover



        stats_dict = self.get_stats_for_class(cls_nodes)
        if not self.requires_cover_arr:
            statistics = self.task.qf.gp_get_params(None, cls_nodes[-1][0].stats)
            optimistic_estimate = self.task.qf.optimistic_estimate(None, self.task.target, self.task.data, statistics)
            if not optimistic_estimate >= ps.minimum_required_quality(self.results, self.task):
                return
        if is_single_path:
            if len(cls_nodes)==1 and -1 in cls_nodes:
                return
            del stats_dict[-1] # remove root node
            all_combinations = ps.powerset(stats_dict.keys(), max_length=self.depth - len(prefix)+1)

            for comb in all_combinations:
                # it might still be, that stats_dict[comb[-1]] is wrong if that is the case then
                # stats_dict[comb[0]] is correct
                if len(comb)>0:
                    self.add_if_required(prefix+comb, stats_dict[comb[-1]])
        else:
            for cls, nodes in cls_nodes.items():
                if cls >= 0:
                    if self.check_constraints(stats_dict[cls]):
                        if len(prefix) == (self.depth - 1):
                            self.add_if_required((*prefix, cls), stats_dict[cls])
                        else:
                            is_single_path_now = len(nodes) == 1
                            new_tree = self.create_new_tree_from_nodes(nodes)
                            if len(new_tree) > 0:
                                self.recurse(new_tree, (*prefix, cls), is_single_path_now)


    def recurse_top_down(self, cls_nodes, root, depth_in=0):
        #print(f"{depth_in}"+"\t"*depth_in+str(root.cls))
        #print("init root", root.cls)
        #print(depth_in)
        #self.check_tree_is_ordered(root)
        results = []

        curr_depth = depth_in
        stats_dict = self.get_stats_for_class(cls_nodes)
        is_valid_class = {key : self.check_constraints(gp_stats) for key, gp_stats in stats_dict.items()}
        #init_class = root.cls
        #direct_child = None
        init_root = root
        alpha = []
        while True:
            if root.cls == -1:
                pass
            else:
                alpha.append(root.cls)
            if len(root.children) == 1 and curr_depth <= self.depth:
                #print(f"Path optmization {len(root.children)}")
            #    curr_depth += 1

                potential_root = next(iter(root.children.values()))
                if is_valid_class[potential_root.cls]:
                    root=potential_root
                else:
                    break
            else:
                break
        #self.get_prefixes_top_down(alpha,  max_length=self.depth - depth_in + 1) #
        #assert len(alpha) > 0
        prefixes =  list(ps.powerset(alpha, max_length=self.depth - depth_in + 1))[1:]

        #prefixes = list(map(lambda x: sum(x, tuple()), prefixes))
        #print(root.cls, list(root.children), prefixes)
        #print("AAA", list(cls_nodes.keys()))



        if init_root.cls == -1:
            prefixes.append(tuple())
        for prefix in prefixes:
            if len(prefix)==0:
                if is_valid_class[init_root.cls]:
                    results.append((prefix, stats_dict[init_root.cls]))
                continue
            cls = prefix[-1]
            if is_valid_class[cls]:
                results.append((prefix, stats_dict[cls]))

        #suffixes = [((), root.stats)]

        suffixes = []
        if curr_depth == (self.depth - 1):
            #print(f"{depth_in}"+"\t"*depth_in+"B")
            for cls, stats in stats_dict.items():
                if cls < 0 or cls in alpha:
                    continue
                assert cls > max(alpha), f"{cls} {max(alpha)}, {alpha}, {list(stats_dict.keys())}"
                suffixes.append(((cls,), stats))
        else:
            #print(f"{depth_in}"+"\t"*depth_in+"A")
            for cls in stats_dict:
                if cls < 0 or cls in alpha:
                    continue
                if is_valid_class[cls]:
                    # Future: There is also the possibility to compute the stats_dict of the prefix tree
                    # without creating the prefix tree first
                    # This might be useful if curr_depth == self.depth - 2
                    # as we need not recreate the tree
                    new_root, nodes = self.get_top_down_tree_for_class(cls_nodes, cls, is_valid_class)
                    #self.check_tree_is_ordered(new_root)
                    #self.check_tree_is_ordered(init_root)
                    if len(nodes) > 0:
                        new_cls_nodes = self.nodes_to_cls_nodes(nodes)
                        #new_dict = self.get_stats_for_class(new_cls_nodes)
                        #for key, value in new_dict.items():
                        #    if isinstance(stats_dict[key], dict):
                        #        continue
                        #    assert stats_dict[key][0]>=value[0], f"{stats_dict[key][0]} {value[0]}"
                        #    assert stats_dict[key][1]>=value[1], f"{stats_dict[key][1]} {value[1]}"
                        #print("  " * curr_depth, cls, curr_depth, len(new_cls_nodes))
                        suffixes.extend(self.recurse_top_down(new_cls_nodes, new_root, curr_depth+1))
        #if prefixes == [(12,), (13,)]:
        #    print(f"{depth_in}"+"\t"*depth_in+ "pre, suf", prefixes)

        # the combination below can be optimized to avoid the if by first grouping them by length
        results.extend([((*pre, *suf), gp_stats) for pre, (suf, gp_stats) in itertools.product(prefixes, suffixes) if len(pre)+len(suf)<=self.depth and (len(pre)==0 or pre[-1]<suf[0])])
        #if prefixes == [(12,), (13,)]:
        #    print(f"{depth_in}"+"\t"*depth_in+ "results", results)
        #print()
        return results

    def check_tree_is_ordered(self, root, prefix=None): # pragma: no cover
        """Verify that the nodes of a tree are sorted in ascending order"""
        if prefix is None:
            prefix= []
        s = {root.cls}
        for cls, child in root.children.items():
            assert child.cls > root.cls, f"{prefix} , {cls}"
            s2=self.check_tree_is_ordered(child, prefix+[cls])
            s = s.union(s2)
        return s




    def get_top_down_tree_for_class(self, cls_nodes, cls, is_valid_class):
        # Future: Can eventually also remove infrequent nodes already during tree creation
        base_root = None
        nodes = []
        if len(cls_nodes[cls]) > 0 and is_valid_class[cls]:
            base_root = self.create_copy_of_tree_top_down(cls_nodes[cls][0], nodes, is_valid_class=is_valid_class)
            for from_root in cls_nodes[cls][1:]:
                self.merge_trees_top_down(nodes, base_root, from_root, is_valid_class)
        return base_root, nodes

    def create_copy_of_tree_top_down(self, from_root, nodes=None, parent=None, is_valid_class=None):
        if nodes is None:
            nodes = [] # pragma: no cover
        #if len(nodes) == 0:
        #    root_cls = -1
        children = {}
        new_root = self.GP_node(from_root.cls, len(nodes), parent, children, from_root.stats.copy())
        nodes.append(new_root)
        for child_cls, child in from_root.children.items():
            if is_valid_class is None or child_cls in is_valid_class:
                new_child = self.create_copy_of_tree_top_down(child, nodes, new_root, is_valid_class=is_valid_class)
                children[child_cls] = new_child
        return new_root

    def merge_trees_top_down(self, nodes, mutable_root, from_root, is_valid_class):
        self.merge(mutable_root.stats, from_root.stats)
        for cls in from_root.children:
            if cls not in mutable_root.children:
                new_child = self.create_copy_of_tree_top_down(from_root.children[cls], nodes, mutable_root, is_valid_class=is_valid_class)
                mutable_root.children[cls] = new_child
            else:
                self.merge_trees_top_down(nodes, mutable_root.children[cls], from_root.children[cls], is_valid_class=is_valid_class)


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
        path=Path(path).absolute()
        print(path)
        with open(path, 'w', encoding="utf-8") as f:
            for row_index, row in self.tqdm(enumerate(arrs), 'creating tree', total=len(arrs)):
                f.write(" ".join(map(str, np.nonzero(row)[0])) + " "+ to_str(self.get_stats(row_index))+"\n")
