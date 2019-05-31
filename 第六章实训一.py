from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
data = wine['data']
target = wine['target']
#数据集划分为训练集，测试集
data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.2,random_state=125)
#标准差标准化（规则）
stdScaler = StandardScaler().fit(data_train)
data_std_train = stdScaler.transform(data_train)
data_std_test = stdScaler.transform(data_test)
#pca降维
pca_model = PCA(n_components=10).fit(data_std_train)#规则
data_pca_train = pca_model.transform(data_std_train)
data_pca_test = pca_model.transform(data_std_test)
print(data_pca_test)
