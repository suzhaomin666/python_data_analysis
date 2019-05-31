import pandas as pd

data_loss=pd.read_csv('ele_loss.csv',encoding='GBK')
data_alarm=pd.read_csv('alarm.csv',encoding='GBK')

print(data_loss.shape)
print(data_alarm.shape)
#将以两个主键进行合并
data_new=pd.merge(data_loss,data_alarm,left_on=['ID','date'],right_on=['ID','date'],how='inner')
print(data_new)
