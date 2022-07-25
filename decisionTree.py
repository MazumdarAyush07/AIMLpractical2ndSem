from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

iris=load_iris()
print(iris)

data=pd.DataFrame(iris.data,columns=iris.feature_names)
print(data)

data['Species']=pd.DataFrame(iris.target)
print(data)

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

model=DecisionTreeClassifier(criterion="entropy",splitter="best")
model_fit=model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(accuracy_score(y_pred,y_test).round(2)*100)

#%matplotlib inline

plt.figure(figsize=(10,5))
plot_tree(model_fit,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.show()
