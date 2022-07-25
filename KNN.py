from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

iris=load_iris()
data=pd.DataFrame(iris.data,columns=iris.feature_names)
print(data.head())

data['Species']=pd.DataFrame(iris.target)

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

model=KNeighborsClassifier(n_neighbors = 5)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(accuracy_score(y_pred,y_test).round(2)*100)

score = []
k_range = range(1,31)
for k in k_range:
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    score.append(accuracy_score(y_pred,y_test).round(2)*100)

for k in k_range:
    print(k,':',score[k-1])

#%matplotlib inline

plt.plot(k_range,score)
plt.xlabel('Neighbors')
plt.ylabel('Accuracy')
plt.show()