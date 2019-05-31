import pandas as pd
import numpy as np
#读入文件
datainfo=pd.read_csv("Training_LogInfo.csv")
dataupda=pd.read_csv("Training_Userupdate.csv")
# print(datainfo)
#分组
detailGroup1=datainfo[['Idx','Listinginfo1','LogInfo3']].groupby(by='Idx')
detailGroup2=dataupda[['Idx','ListingInfo1','UserupdateInfo2']].groupby(by='Idx')
#分组查看数据
# print(detailGroup1.head())
# print(detailGroup2.head())
#
print('最早、最晚更新及登录时间\n',detailGroup1.agg([np.max,np.min]),detailGroup2.agg([np.max,np.min]))

print('信息更新次数和登录次数\n',np.size(detailGroup1),np.size(detailGroup2))

