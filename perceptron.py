import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
import random
def load_data(file_name):#加载数据
    data=[]
    file=open(file_name)
    for line in file.readlines():
        line_handle=line.strip().split(',')#去除空格并且以逗号分离
        num= len(line_handle)
        unit=[]
        for i in range(num):
            unit.append(float(line_handle[i]))
        data.append(unit)
    return np.array(data)
def fun(X,Y,W,a):#感知机算法 a是学习率
    time=0
    key = 1
    while key:
        key = 0
        for i in range(X.shape[0]):
            if (data_Y[i] * X[i].dot(W.T)) <= 0:
                key = 1
                W = W + a * Y[i] * X[i]
                time=time+1
    print(time)
    return W
def sum(wrong_time,xi,y,N,gram):
    sum=0
    for j in range(N):
        sum=sum+wrong_time[j]*y[j]*gram[j][xi]
    return sum
def fun_2(X,Y,W,a):#对偶算法
    gram=X.dot(X.T)
    wrong_time=np.zeros((X.shape[0],1))
    k=1
    while k:
        k=0
        for i in range(X.shape[0]):
            if (Y[i]*sum(wrong_time,i,Y,X.shape[0],gram))<=0:
                k=1
                wrong_time[i]=wrong_time[i]+a

    for i in range(X.shape[0]):
        W=W+wrong_time[i]*Y[i]*X[i]
    return W
data=load_data('ex1data1.txt')
print(data.shape)
data_X=data[:,:-1]
data_Y=data[:,-1:]



ones=np.ones((data_X.shape[0],1))
data_X=np.hstack((ones,data_X))#这样就不用单独设b，w1*1+w2*x1+w3*x2这样w1就相当于b.  hstack是水平方向平铺
W=np.zeros((1,data_X.shape[1]))#初始化W

a=0.1
W_1=fun(data_X,data_Y,W,a)
W_2=fun_2(data_X,data_Y,W,a)
print(W_1)

print(W_2)



#显示下数据
lable_1=np.where(data_Y.ravel()==1)
lable_0=np.where(data_Y.ravel()==-1)
mp.scatter(data_X[lable_1,1],data_X[lable_1,2],color='r',marker='o')
mp.scatter(data_X[lable_0,1],data_X[lable_0,2],color='r',marker='x')
X1=np.arange(0,10,0.001)
X2=-(W_2[0][0]+W_2[0][1]*X1)/W_2[0][2]
mp.plot(X1,X2)
mp.show()
