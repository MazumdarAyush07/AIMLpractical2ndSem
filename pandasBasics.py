import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

data=pd.read_excel('test_data.xlsx')
print(data.head)

print(data.isnull().sum())
print(data.dropna())
print(data.dropna(axis=1))

imr=SimpleImputer(missing_values=np.NaN,strategy="mean")
imr=imr.fit(data)
imputed_data=imr.transform(data)
print(data)
print(imputed_data)

data=pd.read_csv('iris.csv')
print(data.head())

data.columns=['sepal length','sepal width','petal length','petal width','class']
print(data.head())

print(np.unique(data['class']))

mapping={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
data['class']=data['class'].map(mapping)
print(data.head())

le=LabelEncoder()
data['class']=le.fit_transform(data['class'])
print(data.head())