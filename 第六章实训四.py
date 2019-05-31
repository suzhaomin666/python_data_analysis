from sklearn.datasets import load_wine#wine数据集
from sklearn.cluster import KMeans#K-Means聚类模型
from sklearn.model_selection import train_test_split#数据集划分
from sklearn.preprocessing import StandardScaler#标准差标准化

from sklearn.linear_model import LinearRegression#线性回归模型
from sklearn.metrics import fowlkes_mallows_score,silhouette_score,accuracy_score,\
    precision_score,recall_score,f1_score,cohen_kappa_score,classification_report,roc_curve,\
    explained_variance_score,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score    #聚类、分类、回归评分标准
import matplotlib.pyplot as plt#数据可视化

wine = load_wine()
data = wine['data']
target = wine['target']
#数据集划分为训练集，测试集
data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.2,random_state=125)
#标准差标准化（规则）
stdScaler = StandardScaler().fit(data_train)
data_std_train = stdScaler.transform(data_train)
data_std_test = stdScaler.transform(data_test)
x_train,y_train,x_test,y_test = data_train,target_train,data_test,target_test
clf = LinearRegression().fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(y_pred)
#回归结果可视化
plt.figure(figsize=(10,7))
plt.plot(range(y_test.shape[0]),y_test,linewidth=1.7,linestyle='-')
plt.plot(range(y_test.shape[0]),y_pred,linewidth=1.5,linestyle='-.')
plt.legend(['真实值','预测值'])
plt.show()
#评价回归模型
print('Boston数据线性回归模型的平均绝对误差为：',mean_absolute_error(y_test,y_pred))
print('Boston数据线性回归模型的均方差为：',mean_squared_error(y_test,y_pred))
print('Boston数据线性回归模型的中值绝对误差为：',median_absolute_error(y_test,y_pred))
print('Boston数据线性回归模型的可解释方差值为：',explained_variance_score(y_test,y_pred))
print('Boston数据线性回归模型的R^2为：',r2_score(y_test,y_pred))
