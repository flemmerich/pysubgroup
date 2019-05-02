import pysubgroup as ps
import pandas as pd
import numpy as np

import pprint
pp = pprint.PrettyPrinter(indent=4)

data = np.array ([[1, 2, 3, 4, 5], ["F", "F", "F", "Tr", "Tr"]]).T
data = pd.DataFrame(data, columns=["Target", "A"])
data["Target"] = pd.to_numeric (data["Target"])


target = ps.NumericTarget ('Target')
print(data[target.target_variable])
sg = ps.Subgroup(target, ps.NominalSelector("A", "Tr"))
print(target.get_base_statistics(data, sg))
sg.calculate_statistics(data)
# pp.pprint (sg.statistics)
qf = ps.StandardQFNumeric(1.0)
print(qf.optimistic_estimate_from_dataset(data, sg))
