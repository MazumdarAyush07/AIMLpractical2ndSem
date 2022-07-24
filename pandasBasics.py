import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

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

np.unique(data['class']map(mapping))
print(data.head())
