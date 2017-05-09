from scipy.io import arff

from Selectors import NominalSelector, NumericSelector
from pysubgroup import GeneralizationAwaresInterestingnessMeasures
from pysubgroup.Subgroup import Subgroup, SubgroupDescription


data = arff.loadarff("C:\data\Datasets\credit-g.arff") [0]

target = NominalSelector ('class', b'bad')
sel1 = NominalSelector('purpose', b"'new car'")
sel2 = NominalSelector('other_payment_plans', b'bank')
sel3 = NumericSelector ('age', 12.52342545, 56)
sg = Subgroup (target, SubgroupDescription([sel1, sel2]))

print (SubgroupDescription([sel1, sel2, sel3]))
qf = GeneralizationAwaresInterestingnessMeasures.GAStandardQF(0.5)
print (qf.evaluateFromDataset(data, sg))
