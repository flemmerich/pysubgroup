import pysubgroup as ps
from pysubgroup.datasets import get_credit_data

data = get_credit_data()

target = ps.NumericTarget("credit_amount")
sgd = ps.EqualitySelector("purpose", b"other")

stats = target.calculate_statistics(sgd, data)
print(stats)


qf = ps.StandardQFNumeric(1.0)
score = qf.evaluate(sgd, target, data)
print(score)

score = qf.evaluate(sgd, target, data, stats)
print(score)
