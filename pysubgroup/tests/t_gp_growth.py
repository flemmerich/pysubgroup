import time
from collections import defaultdict
from pysubgroup.tests.DataSets import get_credit_data
from pysubgroup import model_target
import pysubgroup as ps



data = get_credit_data()
#warnings.filterwarnings("error")
searchSpace_Nominal = ps.create_nominal_selectors(data, ignore=['duration', 'credit_amount', 'class'])
searchSpace_Numeric = ps.create_numeric_selectors(data, ignore=['duration', 'credit_amount', 'class'])
searchSpace = searchSpace_Nominal + searchSpace_Numeric
#searchSpace = [ps.EqualitySelector("checking_status", b'<0'), ps.EqualitySelector("own_telephone", b'yes'), ps.IntervalSelector("age", 30, 36)]
#QF=model_target.EMM_Likelihood(model_target.PolyRegression_ModelClass(x_name='duration', y_name='credit_amount'))
#target = ps.FITarget()
#QF=ps.CountQF()

QF=ps.StandardQF(0.5)
target = ps.BinaryTarget('class', b'bad')
constr = [ps.MinSupportConstraint(30)]
task = ps.SubgroupDiscoveryTask(data, target, searchSpace, result_set_size=30, depth=2, qf=QF, constraints=constr)
ps.GpGrowth(mode='t_d').to_file(task,'E:/tmp/gp_credit.txt')


start_time = time.time()
gp = ps.GpGrowth(mode='t_d').execute(task)
print("--- %s seconds ---" % (time.time() - start_time))
#gp = [(qual, sg) for qual, sg in gp if sg.depth <= task.depth]
gp = sorted(gp.to_descriptions(include_stats=True), reverse=True)
def print_result(result):
    for i, (q, pattern, stats) in enumerate(gp):
        print(pattern, stats)
        if i > 10:
            break

start_time = time.time()
dfs1 = ps.SimpleDFS().execute(task)
print("--- %s seconds ---" % (time.time() - start_time))
dfs = dfs1.to_descriptions(include_stats=True)
dfs = sorted(dfs, reverse=True)
gp = sorted(gp, reverse=True)
#if len(gp[0][1].selectors)==0:
#    gp = gp[1:]

def better_sorted(l):
    the_dict=defaultdict(list)
    prev_key=l[0][0]
    for key, val in l:

        if abs(prev_key-key)<10**-11:
            the_dict[prev_key].append(val)
        else:
            the_dict[key].append(val)
            prev_key = key
    print(len(the_dict))
    result = []
    for key, vals in the_dict.items():
        for val in sorted(vals):
            result.append((key, val))
    return result
#dfs = better_sorted(dfs)
#gp = better_sorted(gp)
#gp = gp[:task.result_set_size]
assert len(gp) == len(dfs), f"{len(gp)} != {len(dfs)}"
for i, (l, r) in enumerate(zip(gp, dfs)):
    print(i)
    print('gp:', l)
    print('df:', r)
    if not abs(l[0]-r[0]) < 10 ** -7:
        print("<<< ERROR <<<")
    #assert(abs(l[0]-r[0]) < 10 ** -7)
    #assert(l[1] == r[1])
    if i > 20:
        break
for q, sg, qual in dfs:
    print(str(q)+" "+str(sg))