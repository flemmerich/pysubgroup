from scipy.io import arff
import pysubgroup as ps
import pandas as pd


data = pd.DataFrame (arff.loadarff("../data/credit-g.arff") [0])

target = ps.NumericTarget ('credit_amount')
sg = ps.Subgroup (target, ps.NominalSelector("purpose", b"other"))
print(target.get_base_statistics(data, sg))
sg.calculate_statistics(data)
# pp.pprint (sg.statistics)

qf = ps.StandardQFNumeric (1.0)
print (qf.evaluate_from_dataset(data, sg))
