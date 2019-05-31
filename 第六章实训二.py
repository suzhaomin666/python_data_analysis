from sklearn.datasets import load_wine
from sklearn.cluster import KMeans#K-Means聚类模型
from sklearn.metrics import fowlkes_mallows_score,silhouette_score,accuracy_score,\
    precision_score,recall_score,f1_score,cohen_kappa_score,classification_report,roc_curve,\
    explained_variance_score,mean_absolute_error,mean_squared_error,median_absolute_error,r2_score    #聚类、分类、回归评分标准
import matplotlib.pyplot as plt#数据可视化
import numpy as np#·numpy科学计算包

wine = load_wine()
data = wine['data']
target = wine['target']
#聚类模型
kmeans = KMeans(n_clusters=3,random_state=42).fit(data)
print('聚类模型为：',kmeans)
#聚类评分
#方法1FMI评分：
s = []
for i in range(2,11):
    kmeans1 = KMeans(n_clusters=i,random_state=42).fit(data)
    score1 = fowlkes_mallows_score(target,kmeans1.labels_)
    print('FMI第%d类,评分为：%f'%(i,score1))
    s.append(score1)
print('FMI最优评分为：%f'%np.max(s))
#方法2轮廓系数评分：
sil_score = []
for j in range(2,15):
    kmeans2 = KMeans(n_clusters=j, random_state=42).fit(data)
    score2=silhouette_score(data,kmeans2.labels_)
    sil_score.append(score2)
plt.rcParams['font.sans-serif'] = 'simhei'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,6))
plt.title('轮廓系数评分折线图')
plt.plot(range(2,15),sil_score,linewidth=1.5,linestyle='-',c='red')
plt.xticks(range(2,15,1))
plt.show()
