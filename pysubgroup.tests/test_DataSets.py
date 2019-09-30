import pandas as pd
from pathlib import Path
from scipy.io import arff

def getCreditData():
    root = Path.cwd()
    if root.name=="pysubgroup.tests":
        root=root.parent
    creditPath=root.joinpath("data/credit-g.arff")
    print("Reading      Path:", creditPath)
    data = pd.DataFrame(arff.loadarff(creditPath) [0])
    return data

def getTitanicData():
    root = Path.cwd()
    if root.name=="pysubgroup.tests":
        root=root.parent
    titanicPath=root.joinpath("data/titanic.csv")
    print("Reading      Path:", titanicPath)
    return pd.read_csv(titanicPath,sep=",",header=[0])


