import pprint
import numpy as np
import pandas as pd
import pysubgroup as ps


pp = pprint.PrettyPrinter(indent=4)

data = np.array([[1, 2, 3, 4, 5], ["F", "F", "F", "Tr", "Tr"]]).T
data = pd.DataFrame(data, columns=["Target", "A"])
data["Target"] = pd.to_numeric(data["Target"])


target = ps.NumericTarget('Target')
print(data[target.target_variable])
sgd = ps.EqualitySelector("A", "Tr")
target.calculate_statistics(sgd, data)

qf = ps.StandardQFNumeric(1.0)
print(qf.evaluate(sgd, target, data))
print(qf.optimistic_estimate(sgd, target, data))
