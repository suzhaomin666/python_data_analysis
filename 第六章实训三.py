from sklearn.datasets import load_wine
from sklearn.metrics import fowlkes_mallows_score,silhouette_score,accuracy_score,\
    precision_score,recall_score,f1_score,cohen_kappa_score,classification_report,roc_curve,\
    explained_variance_score,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score    #聚类、分类、回归评分标准
from sklearn.svm import SVC#SVM分类模型
import numpy as np#·numpy科学计算包
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine = load_wine()
data = wine['data']
target = wine['target']
#数据集划分为训练集，测试集
data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.2,random_state=125)
#标准差标准化（规则）
stdScaler = StandardScaler().fit(data_train)
data_std_train = stdScaler.transform(data_train)
data_std_test = stdScaler.transform(data_test)
#分类及预测
svm = SVC().fit(data_std_train,target_train)#建立svc模型
target_pred = svm.predict(data_std_test)#结果预测
true = np.sum(target_pred == target_test)
print('预测正确结果：',true)
print('预测错误结果：',target_test.shape[0]-true)
#评价报告
print('svm分类结果报告为：','\n',classification_report(target_test,target_pred))
