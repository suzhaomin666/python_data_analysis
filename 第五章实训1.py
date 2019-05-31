import numpy as np
import pandas as pd
from scipy.interpolate import lagrange
datamis=pd.read_csv('missing_data.csv')

print(datamis.notnull().sum())

#拉格朗日插值
def ploy(s,n,k=2):
    y=s[list(range(n-k,n))+list(range(n+1,n+1+k))]
    y=y[y.notnull()]
    return lagrange(y.index,list(y))(n)
for i in datamis.columns:
    for j in range(len(datamis)):
        if(datamis[i].isnull())[j]:
            datamis[i][j]=ploy(datamis[i],j)

print(datamis)
print(datamis.notnull().sum())