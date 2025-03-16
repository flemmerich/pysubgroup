import pysubgroup as ps
import numpy as np
from .utils import permute

class Significance:
    def __init__(self, task):
        self.task = task # SubgroupDiscoveryTask object

    def generate_null_distribution(self, num_permutations=1000, num_qualities=1):
        """Generate null distribution without any random seeding"""
        baseline = []
        original_result_size = self.task.result_set_size
        self.task.result_set_size = max(num_qualities, self.task.result_set_size or 1) # TODO: just improve...

        for _ in range(num_permutations):
            null_data = permute(self.task.data, self.task.target.target_selector.attribute_name)
            
            # Discover subgroups
            null_task = ps.SubgroupDiscoveryTask(
                null_data,
                self.task.target,
                self.task.search_space,
                qf=self.task.qf,
                result_set_size=num_qualities, # self.task.result_set_size ??
                depth=self.task.depth
            )
            result = ps.BeamSearch().execute(null_task)

            # Collect qualities
            if result.to_dataframe().empty:
                qualities = [0] * num_qualities
            else:
                df = result.to_dataframe()
                qualities = df['quality'].values[:num_qualities].tolist()
                qualities += [0] * (num_qualities - len(qualities))
                
            baseline.extend(qualities)

        self.task.result_set_size = original_result_size
        return np.array(baseline)
    
"""       
Example (if num_qualities=3):
    If 2 subgroups found: [0.5, 0.4] → becomes [0.5, 0.4, 0.0]
    If 0 subgroups found: [0.0, 0.0, 0.0]
    If 5 subgroups found: [0.5, 0.4, 0.3] (only top 3 kept)

Why this matters:
    Ensures consistent array dimensions for statistical analysis
    Handles variable subgroup discovery results gracefully
    Maintains fixed-length outputs for permutation iterations
"""