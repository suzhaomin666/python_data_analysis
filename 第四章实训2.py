import pandas as pd
datainfo=pd.read_csv("Training_LogInfo.csv")
dataupda=pd.read_csv("Training_Userupdate.csv")
# print(datainfo)
# print(dataupda)
datainfo_Lisdate=pd.to_datetime(datainfo[:]['Listinginfo1'])
datainfo_Logdate=pd.to_datetime(datainfo[:]['LogInfo3'])
dateupda_Lisdate=pd.to_datetime(dataupda[:]['ListingInfo1'])
dataupda_Userdate=pd.to_datetime(dataupda[:]['UserupdateInfo2'])
# print(datainfo[:]['LogInfo3'])
datainfo_timedelta=datainfo_Lisdate-datainfo_Logdate
dataupda_timedelta=dateupda_Lisdate-dataupda_Userdate
print('提取登录表时间信息\n',[i.day for i in datainfo_Logdate],[i.minute for i in datainfo_Logdate],[i.year for i in datainfo_Logdate])
# [i.day for i in datainfo_timedelta]
print('提取更新表时间信息\n',[i.day for i in dataupda_Userdate],[i.minute for i in dataupda_Userdate],[i.year for i in dataupda_Userdate])
# upda_deltaday=[i.day for i in dataupda_timedelta]
print('用户登录表的天数差\n',datainfo_timedelta)
print('用户更新表的天数差\n',dataupda_timedelta)

