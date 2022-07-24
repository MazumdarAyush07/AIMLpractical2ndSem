import numpy as np

array=np.arange(9)
print(type(array))
print(array)

print(array.shape)
print(type(array.shape))

print(array[3])
array[3]=7
print((array))

array=array.reshape(3,3)
print(array)

z=np.zeros((2,4))
print(z)

o=np.ones((2,4))
print(o)

f=np.full((2,2),3)
print(f)

i=np.eye(3,3)
print(i)

my_array=np.arange(8)
print(my_array)

my_array=my_array.reshape(2,4)
print(my_array)

my_array=my_array.T
print(my_array)

print("Max:", my_array.max())
print("Min:", my_array.min())
print("Mean:", my_array.mean())
print("Standard Deviation:", my_array.std(axis=1))

num=[]
for i in range(0,5):
    num.append(np.random.randint(0,2))
num=np.array(num)
print(num)
print(np.unique(num))

x=np.arange(1,4)
y=np.arange(1,7,2)
print(np.add(x,y))

num=np.arange(1,10,dtype=float).reshape(3,3)
print(num)
print(np.max(num))
print(np.max(num,axis=0))
print(np.max(num,axis=1))

num[1,2]=np.NaN
print(num)
print(np.max(num,axis=0))