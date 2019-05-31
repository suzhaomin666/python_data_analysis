import numpy as np
import pandas as pd
data_model=pd.read_csv('model.csv',encoding='GBK')

print(data_model)

#定义标准差标准化函数
def StandardScaler(data):
    data=(data-data.mean())/data.std()
    return data
#对其中的三列进行处理
data1=StandardScaler(data_model['电量趋势下降指标'])
data2=StandardScaler(data_model['线损指标'])
data3=StandardScaler(data_model['告警类指标'])
#合并输出
data5=pd.concat([data1,data2,data3],axis=1)
print(data5)