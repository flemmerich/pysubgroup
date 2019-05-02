'''
Created on 10.05.2017

@author: lemmerfn
'''
import pandas as pd
import pysubgroup as ps


data = pd.read_csv("../data/titanic.csv")
target = ps.NominalSelector ('survived', 0)

s1 = ps.Subgroup (target, [])
s2 = ps.Subgroup (target, [])

print(s1 == s2)