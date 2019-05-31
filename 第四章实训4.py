import numpy as np
import pandas as pd
datamast=pd.read_csv("Training_Master.csv",encoding='GBK')
datainfo=pd.read_csv("Training_LogInfo.csv")
dataupda=pd.read_csv("Training_Userupdate.csv")
#宽转长
detailPivot=datamast.pivot_table(index=['Idx','UserInfo_2'])
dd1=pd.crosstab(index=datamast['Idx'],columns=datamast['UserInfo_2'])
print(dd1)
# detailPivot=pd.pivot_table()