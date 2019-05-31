import pandas as pd
data1=pd.read_csv('Training_Master.csv',encoding="GBK")
print(data1.ndim)
print(data1.shape)
print(data1.memory_usage())
print(data1.describe())

