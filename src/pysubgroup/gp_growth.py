import itertools
import warnings
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np

import pysubgroup as ps


def identity(x, *args, **kwargs):  # pylint: disable=unused-argument
    """
    Identity function used as a placeholder for tqdm when progress bars are not needed.

    Parameters:
        x: The input value to return.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The input value x.
    """
    return x


class GpGrowth:
    """
    Implementation of the GP-Growth algorithm.

    GP-Growth is a generalization of FP-Growth and SD-Map capable of working with different
    Exceptional Model Mining targets on top of Frequent Itemset Mining and Subgroup Discovery.

    This class provides methods to perform pattern mining using GP-Growth, supporting both
    bottom-up ('b_u') and top-down ('t_d') modes.

    Attributes:
        GP_node (namedtuple): Structure representing a node in the GP-tree.
        minSupp (int): Minimum support threshold (currently unused).
        tqdm (function): Function for progress bars (default is identity function).
        depth (int): Maximum depth of the search.
        mode (str): Mode of the algorithm ('b_u' for bottom-up, 't_d' for top-down).
        constraints_monotone (list): List of monotonic constraints.
        results (list): List to store the resulting subgroups.
        task (SubgroupDiscoveryTask): The subgroup discovery task to execute.
    """

    def __init__(self, mode="b_u"):
        """
        Initializes the GpGrowth algorithm with the specified mode.

        Parameters:
            mode (str): The mode of the algorithm ('b_u' for bottom-up, 't_d' for top-down).
        """
        self.GP_node = namedtuple(
            "GP_node", ["cls", "id", "parent", "children", "stats"]
        )
        self.minSupp = 10
        self.tqdm = identity  # Placeholder for progress bar function
        self.depth = 0
        self.mode = mode  # Specify either 'b_u' (bottom-up) or 't_d' (top-down)
        self.constraints_monotone = []
        self.results = []
        self.task = []
        # Future: There is also the option of a stable mode
        # which never creates the prefix trees

    def prepare_selectors(self, search_space, data):
        """
        Prepares the selectors by computing their coverage arrays and filtering based on constraints.

        Parameters:
            search_space (list): The list of selectors to consider.
            data (DataFrame): The dataset to be analyzed.

        Returns:
            tuple: A tuple containing:
                - selectors_sorted (list): The sorted list of selectors after filtering.
                - arrs (ndarray): A 2D NumPy array where each column corresponds to the coverage array of a selector.
        """
        selectors = []
        assert len(search_space) > 0, "Provided search space was empty"
        for selector in search_space:
            cov_arr = selector.covers(data)
            selectors.append((np.count_nonzero(cov_arr), selector, cov_arr))

        # Filter selectors based on monotonic constraints
        selectors = [
            (size, selector, arr)
            for size, selector, arr in selectors
            if all(
                constraint.is_satisfied(arr, slice(None), data)
                for constraint in self.constraints_monotone
            )
        ]
        # Sort selectors in decreasing order of support (size)
        sorted_selectors = sorted(selectors, reverse=True)

        # Remove selectors with low optimistic estimate if necessary
        self.remove_selectors_with_low_optimistic_estimate(
            sorted_selectors, len(search_space)
        )

        # Extract the sorted selectors and their coverage arrays
        selectors_sorted = [selector for size, selector, arr in sorted_selectors]
        if len(selectors_sorted) == 0:
            arrs = np.empty((0, 0), dtype=np.bool_)
        else:
            arrs = np.vstack([arr for size, selector, arr in sorted_selectors]).T
        return selectors_sorted, arrs

    def remove_selectors_with_low_optimistic_estimate(self, s, search_space_size):
        """
        Removes selectors from the list that have an optimistic estimate below the minimum required quality.

        Parameters:
            s (list): List of selectors with their size and coverage arrays.
            search_space_size (int): The size of the initial search space.
        """
        if not hasattr(self.task.qf, "optimistic_estimate"):
            return
        if search_space_size > self.task.result_set_size:
            # Remove selectors which have too low of an optimistic estimate
            stats = []
            # Evaluate each selector and update the result set
            for _, _, cov_arr in s:
                statistics = self.task.qf.calculate_statistics(
                    cov_arr, self.task.target, self.task.data
                )
                stats.append(statistics)
                quality = self.task.qf.evaluate(
                    cov_arr, self.task.target, self.task.data, statistics
                )
                ps.add_if_required(
                    self.results, None, quality, self.task, statistics=statistics
                )
            del statistics
            to_pop = []
            # Determine the minimum required quality based on current results
            min_quality = ps.minimum_required_quality(self.results, self.task)
            # Identify selectors to remove
            for i, ((_, _, cov_arr), statistics) in enumerate(zip(s, stats)):
                if (
                    not self.task.qf.optimistic_estimate(
                        cov_arr, self.task.target, self.task.data, statistics
                    )
                    > min_quality
                ):
                    to_pop.append(i)
            # Update the minimum quality for the task
            self.task.min_quality = np.nextafter(
                float(min_quality), self.task.min_quality
            )
            # Remove the selectors with low optimistic estimate
            for i in reversed(to_pop):
                s.pop(i)
            self.results.clear()

    def nodes_to_cls_nodes(self, nodes):
        """
        Groups nodes by their class labels.

        Parameters:
            nodes (list): List of nodes to group.

        Returns:
            defaultdict: A dictionary mapping class labels to lists of nodes.
        """
        cls_nodes = defaultdict(list)
        for node in nodes:
            cls_nodes[node.cls].append(node)
        return cls_nodes

    def setup_from_quality_function(self, qf):
        """
        Sets up function pointers from the quality function.

        Parameters:
            qf: The quality function used in the task.
        """
        # pylint: disable=attribute-defined-outside-init
        self.get_stats = qf.gp_get_stats
        self.get_null_vector = qf.gp_get_null_vector
        self.merge = qf.gp_merge
        self.requires_cover_arr = qf.gp_requires_cover_arr
        # pylint: enable=attribute-defined-outside-init

    def setup_constraints(self, constraints, qf):
        """
        Prepares constraints for use in the algorithm.

        Parameters:
            constraints (list): List of constraints to apply.
            qf: The quality function used in the task.
        """
        self.constraints_monotone = constraints
        for constraint in self.constraints_monotone:
            constraint.gp_prepare(qf)

        if self.mode == "t_d" and len(self.constraints_monotone) == 0:
            warnings.warn(
                """Poor runtime expected: Top-down method does not use
                optimistic estimates and no constraints were provided""",
                UserWarning,
            )

        if len(constraints) == 1:
            # Optimize constraint checking if only one constraint is present
            self.check_constraints = constraints[0].gp_is_satisfied

    def check_constraints(self, node):  # pylint: disable=method-hidden
        """
        Checks if a node satisfies all monotonic constraints.

        Parameters:
            node: The node to check.

        Returns:
            bool: True if the node satisfies all constraints, False otherwise.
        """
        return all(
            constraint.gp_is_satisfied(node) for constraint in self.constraints_monotone
        )

    def setup(self, task):
        """
        Prepares the algorithm by setting up the task, depth, constraints, and quality function.

        Parameters:
            task (SubgroupDiscoveryTask): The task to execute.
        """
        self.task = task
        task.qf.calculate_constant_statistics(task.data, task.target)
        self.depth = task.depth
        self.setup_constraints(task.constraints_monotone, task.qf)
        self.setup_from_quality_function(task.qf)

    def create_initial_tree(self, arrs):
        """
        Creates the initial FP-tree from the coverage arrays.

        Parameters:
            arrs (ndarray): A 2D NumPy array where each column corresponds to the coverage array of a selector.

        Returns:
            tuple: A tuple containing:
                - root (GP_node): The root node of the tree.
                - nodes (list): A list of all nodes in the tree.
        """
        # Create root node
        root = self.GP_node(-1, -1, None, {}, self.get_null_vector())
        nodes = []
        # Build the tree by inserting transactions
        for row_index, row in self.tqdm(
            enumerate(arrs), "creating tree", total=len(arrs)
        ):
            self.normal_insert(
                root, nodes, self.get_stats(row_index), np.nonzero(row)[0]
            )
        nodes.append(root)
        return root, nodes

    def execute(self, task):
        """
        Executes the GP-Growth algorithm on the given task.

        Parameters:
            task (SubgroupDiscoveryTask): The subgroup discovery task to execute.

        Returns:
            SubgroupDiscoveryResult: The result of the subgroup discovery.
        """
        assert self.mode in ("b_u", "t_d"), "mode needs to be either b_u or t_d"
        self.setup(task)

        selectors_sorted, arrs = self.prepare_selectors(task.search_space, task.data)
        root, nodes = self.create_initial_tree(arrs)

        # Mine the tree
        cls_nodes = self.nodes_to_cls_nodes(nodes)
        if self.mode == "b_u":
            self.recurse(cls_nodes, tuple())
        else:  # self.mode == "t_d"
            results = self.recurse_top_down(cls_nodes, root)
            results = self.calculate_quality_function_for_patterns(task, results, arrs)
            for quality, sg, stats in results:
                ps.add_if_required(
                    self.results, sg, quality, self.task, statistics=stats
                )
        # Convert the results to subgroups
        self.results = self.convert_results_to_subgroups(self.results, selectors_sorted)

        self.results = ps.prepare_subgroup_discovery_result(self.results, task)
        return ps.SubgroupDiscoveryResult(self.results, task)

    def convert_results_to_subgroups(self, results, selectors_sorted):
        """
        Converts patterns (indices) to actual subgroups.

        Parameters:
            results (list): List of results containing qualities, indices, and statistics.
            selectors_sorted (list): The list of sorted selectors.

        Returns:
            list: A list of tuples containing quality, subgroup, and statistics.
        """
        new_result = []
        for quality, indices, stats in results:
            selectors = [selectors_sorted[i] for i in indices]
            sg = ps.Conjunction(selectors)
            new_result.append((quality, sg, stats))
        return new_result

    def calculate_quality_function_for_patterns(self, task, results, arrs):
        """
        Calculates the quality function for the given patterns.

        Parameters:
            task (SubgroupDiscoveryTask): The task containing the quality function.
            results (list): List of patterns with their aggregated parameters.
            arrs (ndarray): The coverage arrays of the selectors.

        Returns:
            list: A list of tuples containing quality, indices, and statistics.
        """
        out = []
        for indices, gp_params in self.tqdm(
            results,
            "computing quality function",
        ):
            if self.requires_cover_arr:
                # Reconstruct the cover array for the pattern
                if len(indices) == 1:
                    cover_arr = arrs[:, indices[0]]
                else:
                    cover_arr = np.all([arrs[:, i] for i in indices])
                statistics = task.qf.gp_get_params(cover_arr, gp_params)
                sg = cover_arr
            else:
                statistics = task.qf.gp_get_params(None, gp_params)
                sg = None
            # Evaluate the quality of the subgroup
            qual2 = task.qf.evaluate(sg, task.target, task.data, statistics)
            out.append((qual2, indices, statistics))
        return out

    def normal_insert(self, root, nodes, new_stats, classes):
        """
        Inserts a transaction into the FP-tree.

        Parameters:
            root (GP_node): The root node of the tree.
            nodes (list): List of all nodes in the tree.
            new_stats: The statistics associated with the transaction.
            classes (array-like): The class labels (selectors) present in the transaction.

        Returns:
            GP_node: The leaf node where the transaction ends.
        """
        node = root
        for cls in classes:
            if cls not in node.children:
                # Create a new child node if necessary
                new_child = self.GP_node(
                    cls, len(nodes), node, {}, self.get_null_vector()
                )
                nodes.append(new_child)
                node.children[cls] = new_child
            # Merge the statistics
            self.merge(node.stats, new_stats)
            node = node.children[cls]
        # Merge statistics at the leaf node
        self.merge(node.stats, new_stats)
        return node

    def add_if_required(self, prefix, gp_stats):
        """
        Adds a pattern to the result set if it meets the quality threshold.

        Parameters:
            prefix (tuple): The current pattern (tuple of class indices).
            gp_stats: The aggregated statistics for the pattern.
        """
        statistics = self.task.qf.gp_get_params(None, gp_stats)
        quality = self.task.qf.evaluate(None, None, None, statistics)
        ps.add_if_required(
            self.results, prefix, quality, self.task, statistics=statistics
        )

    def recurse(self, cls_nodes, prefix, is_single_path=False):
        """
        Recursively mines patterns in bottom-up mode.

        Parameters:
            cls_nodes (defaultdict): Dictionary mapping class labels to nodes.
            prefix (tuple): The current pattern prefix.
            is_single_path (bool): Flag indicating if the current path is a single path.
        """
        if len(cls_nodes) == 0:
            raise RuntimeError  # pragma: no cover
        # Add current pattern to results
        self.add_if_required(prefix, cls_nodes[-1][0].stats)
        if len(prefix) >= self.depth:
            return  # pragma: no cover

        stats_dict = self.get_stats_for_class(cls_nodes)
        if not self.requires_cover_arr:
            # Prune using optimistic estimate if possible
            statistics = self.task.qf.gp_get_params(None, cls_nodes[-1][0].stats)
            optimistic_estimate = self.task.qf.optimistic_estimate(
                None, self.task.target, self.task.data, statistics
            )
            if not optimistic_estimate >= ps.minimum_required_quality(
                self.results, self.task
            ):
                return
        if is_single_path:
            # Handle single-path optimization
            if len(cls_nodes) == 1 and -1 in cls_nodes:
                return
            del stats_dict[-1]  # Remove root node
            all_combinations = ps.powerset(
                stats_dict.keys(), max_length=self.depth - len(prefix) + 1
            )

            for comb in all_combinations:
                if len(comb) > 0:
                    self.add_if_required(prefix + comb, stats_dict[comb[-1]])
        else:
            # Recursively mine each child node
            for cls, nodes in cls_nodes.items():
                if cls >= 0:
                    if self.check_constraints(stats_dict[cls]):
                        if len(prefix) == (self.depth - 1):
                            self.add_if_required((*prefix, cls), stats_dict[cls])
                        else:
                            is_single_path_now = len(nodes) == 1
                            new_tree = self.create_new_tree_from_nodes(nodes)
                            assert len(new_tree) > 0
                            self.recurse(new_tree, (*prefix, cls), is_single_path_now)

    def recurse_top_down(self, cls_nodes, root, depth_in=0):
        """
        Recursively mines patterns in top-down mode.

        Parameters:
            cls_nodes (defaultdict): Dictionary mapping class labels to nodes.
            root (GP_node): The current root node.
            depth_in (int): The current depth in the recursion.

        Returns:
            list: A list of patterns with their aggregated statistics.
        """
        results = []

        curr_depth = depth_in
        stats_dict = self.get_stats_for_class(cls_nodes)
        is_valid_class = {
            key: self.check_constraints(gp_stats)
            for key, gp_stats in stats_dict.items()
        }
        init_root = root
        alpha = []
        # Traverse down single paths
        while True:
            if root.cls == -1:
                pass
            else:
                alpha.append(root.cls)
            if len(root.children) == 1 and curr_depth <= self.depth:
                potential_root = next(iter(root.children.values()))
                if is_valid_class[potential_root.cls]:
                    root = potential_root
                else:
                    break
            else:
                break
        # Generate prefixes from alpha
        prefixes = list(ps.powerset(alpha, max_length=self.depth - depth_in + 1))[1:]

        if init_root.cls == -1:
            prefixes.append(tuple())
        for prefix in prefixes:
            if len(prefix) == 0:
                if is_valid_class[init_root.cls]:
                    results.append((prefix, stats_dict[init_root.cls]))
                continue
            cls = prefix[-1]
            assert is_valid_class[cls]
            results.append((prefix, stats_dict[cls]))

        suffixes = []
        if curr_depth == (self.depth - 1):
            # Handle leaf nodes
            for cls, stats in stats_dict.items():
                if cls < 0 or cls in alpha:
                    continue
                assert cls > max(
                    alpha
                ), f"{cls} {max(alpha)}, {alpha}, {list(stats_dict.keys())}"
                suffixes.append(((cls,), stats))
        else:
            # Recursively mine child nodes
            for cls in stats_dict:
                if cls < 0 or cls in alpha:
                    continue
                if is_valid_class[cls]:
                    new_root, nodes = self.get_top_down_tree_for_class(
                        cls_nodes, cls, is_valid_class
                    )
                    assert len(nodes) > 0
                    new_cls_nodes = self.nodes_to_cls_nodes(nodes)
                    suffixes.extend(
                        self.recurse_top_down(new_cls_nodes, new_root, curr_depth + 1)
                    )
        # Combine prefixes and suffixes to form new patterns
        results.extend(
            [
                ((*pre, *suf), gp_stats)
                for pre, (suf, gp_stats) in itertools.product(prefixes, suffixes)
                if len(pre) + len(suf) <= self.depth
                and (len(pre) == 0 or pre[-1] < suf[0])
            ]
        )
        return results

    def check_tree_is_ordered(self, root, prefix=None):  # pragma: no cover
        """
        Verifies that the nodes of a tree are sorted in ascending order.

        Parameters:
            root (GP_node): The root node of the tree.
            prefix (list): The current path prefix.

        Returns:
            set: A set of class labels in the tree.
        """
        if prefix is None:
            prefix = []
        s = {root.cls}
        for cls, child in root.children.items():
            assert child.cls > root.cls, f"{prefix} , {cls}"
            s2 = self.check_tree_is_ordered(child, prefix + [cls])
            s = s.union(s2)
        return s

    def get_top_down_tree_for_class(self, cls_nodes, cls, is_valid_class):
        """
        Creates a subtree for a specific class in top-down mode.

        Parameters:
            cls_nodes (defaultdict): Dictionary mapping class labels to nodes.
            cls (int): The class label to create the subtree for.
            is_valid_class (dict): Dictionary indicating valid classes.

        Returns:
            tuple: A tuple containing:
                - base_root (GP_node): The root of the new subtree.
                - nodes (list): A list of nodes in the new subtree.
        """
        base_root = None
        nodes = []
        if len(cls_nodes[cls]) > 0 and is_valid_class[cls]:  # pragma: no branch
            base_root = self.create_copy_of_tree_top_down(
                cls_nodes[cls][0], nodes, is_valid_class=is_valid_class
            )
            for from_root in cls_nodes[cls][1:]:
                self.merge_trees_top_down(nodes, base_root, from_root, is_valid_class)
        return base_root, nodes

    def create_copy_of_tree_top_down(
        self, from_root, nodes=None, parent=None, is_valid_class=None
    ):
        """
        Creates a copy of the tree starting from a specific root in top-down mode.

        Parameters:
            from_root (GP_node): The root node to copy from.
            nodes (list): List to store the new nodes.
            parent (GP_node): The parent of the new root node.
            is_valid_class (dict): Dictionary indicating valid classes.

        Returns:
            GP_node: The new root node of the copied subtree.
        """
        if nodes is None:
            nodes = []  # pragma: no cover
        children = {}
        new_root = self.GP_node(
            from_root.cls, len(nodes), parent, children, from_root.stats.copy()
        )
        nodes.append(new_root)
        for child_cls, child in from_root.children.items():
            if (
                is_valid_class is None or child_cls in is_valid_class
            ):  # pragma: no branch
                new_child = self.create_copy_of_tree_top_down(
                    child, nodes, new_root, is_valid_class=is_valid_class
                )
                children[child_cls] = new_child
        return new_root

    def merge_trees_top_down(self, nodes, mutable_root, from_root, is_valid_class):
        """
        Merges two trees in top-down mode.

        Parameters:
            nodes (list): List of nodes in the mutable tree.
            mutable_root (GP_node): The root of the mutable tree to merge into.
            from_root (GP_node): The root of the tree to merge from.
            is_valid_class (dict): Dictionary indicating valid classes.
        """
        self.merge(mutable_root.stats, from_root.stats)
        for cls in from_root.children:
            if cls not in mutable_root.children:
                # Add new child to mutable root
                new_child = self.create_copy_of_tree_top_down(
                    from_root.children[cls],
                    nodes,
                    mutable_root,
                    is_valid_class=is_valid_class,
                )
                mutable_root.children[cls] = new_child
            else:
                # Merge existing child nodes
                self.merge_trees_top_down(
                    nodes,
                    mutable_root.children[cls],
                    from_root.children[cls],
                    is_valid_class=is_valid_class,
                )

    def get_stats_for_class(self, cls_nodes):
        """
        Aggregates statistics for each class label.

        Parameters:
            cls_nodes (defaultdict): Dictionary mapping class labels to nodes.

        Returns:
            dict: A dictionary mapping class labels to aggregated statistics.
        """
        out = {}
        for key, nodes in cls_nodes.items():
            s = self.get_null_vector()
            for node in nodes:
                self.merge(s, node.stats)
            out[key] = s
        return out

    def create_new_tree_from_nodes(self, nodes):
        """
        Creates a new tree from a list of nodes for recursive mining.

        Parameters:
            nodes (list): List of nodes to build the new tree from.

        Returns:
            defaultdict: A dictionary mapping class labels to nodes in the new tree.
        """
        new_nodes = {}
        for node in nodes:
            nodes_upwards = self.get_nodes_upwards(node)
            self.create_copy_of_path(nodes_upwards[1:], new_nodes, node.stats)

        cls_nodes = defaultdict(list)
        for new_node in new_nodes.values():
            cls_nodes[new_node.cls].append(new_node)
        return cls_nodes

    def create_copy_of_path(self, nodes, new_nodes, stats):
        """
        Creates a copy of a path in the tree, updating statistics.

        Parameters:
            nodes (list): The list of nodes in the path.
            new_nodes (dict): Dictionary to store new nodes.
            stats: The statistics to merge into the nodes.
        """
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
        """
        Retrieves all nodes from a given node up to the root.

        Parameters:
            node (GP_node): The starting node.

        Returns:
            list: A list of nodes from the given node up to the root.
        """
        ref = node
        path = []
        while True:
            path.append(ref)
            ref = ref.parent
            if ref is None:
                break
        return path

    def to_file(self, task, path):
        """
        Writes the tree to a file in a specific format.

        Parameters:
            task (SubgroupDiscoveryTask): The task containing the quality function.
            path (str or Path): The file path to write to.
        """
        self.setup(task)
        _, arrs = self.prepare_selectors(task.search_space, task.data)

        # Create tree
        to_str = task.qf.gp_to_str
        path = Path(path).absolute()
        print(path)
        with open(path, "w", encoding="utf-8") as f:
            for row_index, row in self.tqdm(
                enumerate(arrs), "creating tree", total=len(arrs)
            ):
                f.write(
                    " ".join(map(str, np.nonzero(row)[0]))
                    + " "
                    + to_str(self.get_stats(row_index))
                    + "\n"
                )
