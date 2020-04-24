import unittest
import pysubgroup as ps
from pysubgroup.tests.DataSets import get_credit_data

from pysubgroup.tests.algorithms_testing import TestAlgorithmsBase
data = get_credit_data()

target = ps.NumericTarget('credit_amount')
sg = ps.Subgroup(target, ps.NominalSelector("purpose", b"other"))
print(target.get_base_statistics(data, sg))
sg.calculate_statistics(data)
# pp.pprint (sg.statistics)

qf = ps.StandardQFNumeric(1.0)
print(qf.evaluate_from_dataset(data, sg))
