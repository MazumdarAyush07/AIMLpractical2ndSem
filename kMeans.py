import warnings
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import silhouette_score

warnings.filterwarnings("ignore")
iris=load_iris()

data=pd.DataFrame(iris.data,columns=iris.feature_names)
print(data.head)

model=KMeans(n_clusters=3,init="random",algorithm="full")
model.fit(data)

#%matplotlib inline
print(model.labels_)
plt.scatter(data.iloc[:,0],data.iloc[:,1],c=model.labels_,cmap='brg')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

print(silhouette_score(data,model.labels_))

score = []
k_range = range(2,50)
for k in k_range:
    model=KMeans(n_clusters=k,init="random",algorithm="full")
    model.fit(data)
    score.append(silhouette_score(data,model.labels_))

plt.figure(figsize=(20,10))
plt.bar(k_range,score)
plt.xticks(k_range)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.show()