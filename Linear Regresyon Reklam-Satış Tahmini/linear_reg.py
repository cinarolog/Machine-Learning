# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 00:34:28 2022

@author: cinar
"""

#%% import data and libraries


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("advertising.csv",index_col=0)


#%% Investigate datas information 

df.head(5)
df.tail(5)
df.describe().T
df.info()
df.isnull().sum()



#%% Data visualization

data_1=df[["TV","sales"]]

#input
X=data_1["TV"]

#output
y=data_1["sales"]

type(X)
type(y)


plt.figure(figsize=(8,6))
sns.scatterplot(data=data_1,x="TV",y="sales",color="blue")
plt.title("TV-Sales")
plt.show()




#%% Create Model

from sklearn.linear_model import LinearRegression

lr=LinearRegression()



#%% Preparing  Check the shape of dataframe  shape(n,p)

print(" Shape of X",X.shape)
print(" Shape of y",y.shape)


X=X.values.reshape(-1,1)
X.shape #(200, 1)

y=y.values.reshape(-1,1)
y.shape #(200, 1)



#%% Train test split

from sklearn.model_selection import train_test_split

""" train_test_split returns numpyarray"""
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0,shuffle=True)

#shapes


print(" Shape of X_test",X_test.shape)
print(" Shape of X_train",X_train.shape)
print(" Shape of y_test",y_test.shape)
print(" Shape of y_train",y_train.shape)

type(X_train)


#%% Run model and Fitting

lr.fit(X_train,y_train)



#%% Calculate (intercept_,coef)
    #        y=(B0 + B1.x)

print(" B0=Intercept",lr.intercept_) #B0=Intercept [7.29249377] y axis
print("B1=Coefficent or slope",lr.coef_) #B1=Coefficent [[0.04600779]]  eğim



"""
y=(B0 + B1.x)

y= 7.29249377 + 0.04600779.x

"""


#%% Prediction


y_pred=lr.predict(X_test)
y_pred

y_pred.shape
y_test.shape


#%% Visualization y_test(y_true) and y_pred

fig,ax=plt.subplots(figsize=(10,6))

ax.scatter(X_test,y_test,label="Grand Truth",color="red")

ax.scatter(X_test,y_pred,label="Prediction",color="green")

plt.title("TV Sales Prediction")
plt.xlabel("TV")
plt.ylabel("SAles")
plt.legend(loc="upper left")

#%%

# Her bir tahmin noktasındaki değişimi görelim
indexler = range(1,41)

# Gerçek Data -> Grand Truth
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(indexler, y_test, label='Grand Truth', color='red', linewidth=2)



# Tahmin -> Prediction
ax.plot(indexler, y_pred, label='Prediction', color='green', linewidth=2)

plt.title('GERÇEK - PREDICTION')
plt.xlabel('Data Index')
plt.ylabel('Sales')
plt.legend(loc='upper left')
plt.show()



#%% draw the residuals

#Hata : Residual -> (𝑦−𝑦̂ )
#y_test - y_pred
# Her bir tahmin noktasındaki hatayı görelim

indexler = range(1,41)
# Residuls -> Hatalar
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(indexler, y_test - y_pred, label='Residuals', color='red', linewidth=2)

# sıfır doğrusunu çiz
ax.plot(indexler, np.zeros(40), color='black')

plt.title('HATALAR (RESIDUALS)')
plt.xlabel('Data Index')
plt.ylabel('Sales')
plt.legend(loc='upper left')
plt.show()

#%% Model Confidence

from sklearn.metrics import r2_score, mean_squared_error
# R^2 yi hesaplayalım

r_2=r2_score(y_test, y_pred)
print("R2",r_2) #R2 0.6714477229302764

print("R2 %",r_2*100) #R2 % 67.14477229302764


# MSE  >>>>>>> RMSE
#MSE

mse=mean_squared_error(y_test,y_pred)
print("MSE",mse) #MSE 8.970991242413616

#RMSE

import math


rmse=math.sqrt(mse)
print("RMSE",rmse) #RMSE 2.995161304907236











