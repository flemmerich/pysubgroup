from scipy.io import arff

from Selectors import NominalSelector, NumericSelector
from pysubgroup import GeneralizationAwaresInterestingnessMeasures
from pysubgroup.Subgroup import Subgroup, SubgroupDescription


data = arff.loadarff("C:\data\Datasets\credit-g.arff") [0]

target = NominalSelector ('class', b'bad')
sel1 = NominalSelector('purpose', b"'new car'")
sel2 = NominalSelector('other_payment_plans', b'bank', "BANKER")

sel3 = NumericSelector('age', 10, 20, "youngguy")
print (SubgroupDescription([sel1, sel2, sel3]))

