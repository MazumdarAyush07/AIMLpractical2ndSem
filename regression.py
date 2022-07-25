from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt

boston=load_boston()

print(boston.DESCR)
data=pd.DataFrame(boston.data,columns=boston.feature_names)
print(boston.keys())

data['MEDV']=pd.DataFrame(boston.target)
print(data.head())

print(pd.DataFrame(data.corr().round(2)))

x=data['RM']
y=data['MEDV']

print(pd.DataFrame([x,y]).transpose().head())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

print(type(x_train))
print(type(y_train))

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print(sqrt(mean_squared_error(y_pred,y_test)))

#%matplotlib inline

plt.scatter(x_test,y_test,label='Actual')
plt.plot(x_test,y_pred,color='red',label='Fit')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.legend()
plt.show()