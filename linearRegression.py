import pandas as pd
import numpy as np
from random import randint
from matplotlib import pyplot as plt

def reg_fun(x):
    return 7*x-3

x=[]
y=[]
for i in range(5):
    t=randint(0,20)
    print(t,reg_fun(t))
    x.append(t)
    y.append(reg_fun(t)+randint(-10,10))

#%matplotlib inline
plt.plot(x,[reg_fun(i) for i in x],color="green",label="True Distribution")
plt.scatter(x,y,color="blue",label="Target")
plt.legend()
plt.show()

xbar=np.mean(x)
ybar=np.mean(y)
d=0
n=0
for i in range(5):
    d+=(x[i]-xbar)**2
    n+=(x[i]-xbar)*(y[i]-ybar)
    print(np.round((x[i]-xbar)**2,2),np.round((x[i]-xbar)*(y[i]-ybar),2))
w1=n/d
w0=ybar-w1*xbar

print('xbar:',xbar,'ybar',ybar,'w0:',w0,'w1:',w1)
print('n:',n,'d:',d)

def model(w0,w1,x):
    return w0+w1*x
plt.plot(x,[model(w0,w1,i) for i in x],color="red",label="Predicted")
plt.scatter(x,y,color="blue",label="Target")
plt.legend()
plt.show()