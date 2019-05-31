import numpy as np
import pandas as pd
mtcardata=pd.read_csv('mtcars.csv')

#查看前几条
print('\n','头几行数据\n',mtcardata.head(),'\n','数据的维度\n',mtcardata.shape,'\n','数据的大小\n',mtcardata.size)
print('数据的描述性统计\n',mtcardata.describe())
