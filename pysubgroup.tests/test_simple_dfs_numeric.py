import pysubgroup as ps
import pandas as pd

data = pd.read_table("C:/test/search_vs_nav.tsv")


target = ps.NumericTarget ("resistance")
searchSpace = ps.createSelectors(data, ignore=['se_curr_id_counts',
       'in_curr_id_counts', 'in_prev_id_counts', 'pageviews',
       'se_curr_id_counts_norm', 'in_curr_id_counts_norm',
       'in_prev_id_counts_norm', 'in_by_curr_id_entropy',
       'in_by_prev_id_entropy', 'spreading_entropy', 'in_by_prev_id_gini',
       'in_by_curr_id_gini', 'spreading_gini', 'resistance', 'search_nav',
       'sourcing', 'pageviews_log', 'bowtie_trans', 'bowtie', 'bowtie_trans_search', "topic" ])
task = ps.SubgroupDiscoveryTask (data, target, searchSpace, resultSetSize=10000, depth=1, qf=ps.StandardQF_numeric(1, False))
print (searchSpace)

result = ps.SimpleDFS().execute(task)
for q, sg in result:
    sg.calculateStatistics(data)
result = ps.minimumStatisticFilter(result, "size_sg", 10)
ps.printResultSet(data, result, ["size_sg", "mean_sg", "mean_dataset"])
