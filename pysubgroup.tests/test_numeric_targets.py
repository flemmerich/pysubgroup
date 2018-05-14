from scipy.io import arff
import pysubgroup as ps
import pandas as pd

import pprint
pp = pprint.PrettyPrinter(indent=4)

data = pd.DataFrame (arff.loadarff("../data/credit-g.arff") [0])

target = ps.NumericTarget ('credit_amount')
sg = ps.Subgroup (target, ps.NominalSelector("purpose", b"other"))
print (target.get_base_statistics(data, sg))
sg.calculateStatistics(data)
# pp.pprint (sg.statistics)

qf = ps.StandardQF_numeric (1.0)
print (qf.evaluateFromDataset(data, sg))
