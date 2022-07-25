from sklearn.datasets import load_iris
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score

iris=load_iris()
data=pd.DataFrame(iris.data,columns=iris.feature_names)
print(data.head())

data['Species']=pd.DataFrame(iris.target)
print(data.head())

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

model = MLPClassifier(hidden_layer_sizes=(10,),
                     max_iter=5000,
                     activation='logistic',
                     solver = 'sgd',
                     learning_rate_init=0.001)

scores = cross_val_score(model,x,y,cv=10)
print('Iteration \t Accuracy')
for idx,score in enumerate(scores):
    print('%d\t\t%0.2f'%(idx,score))
print("Average Accuracy: %0.2f Standard deviation of Accuracy: %0.2f" % (scores.mean(),scores.std()))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model = GaussianNB()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print(confusion_matrix(y_pred,y_test))
print("Accuracy:",accuracy_score(y_pred,y_test).round(2)*100)
print("Precision:",precision_score(y_pred,y_test,average='macro').round(2)*100)
print("Recall Score",recall_score(y_pred,y_test,average='macro').round(2)*100)