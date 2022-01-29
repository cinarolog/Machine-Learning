# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 17:10:49 2022

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



#%% Data visualization    EDA

sns.pairplot(df)

X=df[["TV","radio","newspaper"]]
y=df["sales"]


df.columns

sns.pairplot(df,x_vars=df.columns[:3],y_vars=df.columns[3],height=5)


#%% Create Model

from sklearn.linear_model import LinearRegression

lr=LinearRegression()


#%% Preparing  Check the shape of dataframe  shape(n,p)

print(" Shape of X",X.shape)
print(" Shape of y",y.shape)



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

print(" B0=Intercept",lr.intercept_) #B0=Intercept [2.99489303] y axis
print("Coefficent or slope",lr.coef_) #B1-B2-B3 slope [[ 0.04458402  0.19649703 -0.00278146]]  eğim


katsayilar=pd.DataFrame(lr.coef_,columns=["B1 TV","B2 radio","B3 newspaper"])
katsayilar


#%% Prediction


y_pred=lr.predict(X_test)
y_pred

y_pred.shape
y_test.shape

y_test[:10]

y_pred[:10]

#%% Visualization y_test(y_true) and y_pred

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
print("R2",r_2) #R2 R2 0.8601145185017868

print("R2 %",r_2*100) #R2 % 86.01145185017867


# MSE  >>>>>>> RMSE
#MSE

mse=mean_squared_error(y_test,y_pred)
print("MSE",mse) #MSE 8.970991242413616

#RMSE

import math


rmse=math.sqrt(mse)
print("RMSE",rmse) #RMSE 2.995161304907236


#%% OLS  

import statsmodels.api as sm

X_train_ols=sm.add_constant(X_train)

sm_model=sm.OLS(y_train,X_train_ols)

sonuc=sm_model.fit()

print(sonuc.summary())
"""
p> 0.05   ise önemsiz değişken
p<0.05  ise önemli değişken

                coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.9949      0.330      9.076      0.000       2.343       3.647
TV             0.0446      0.001     30.212      0.000       0.042       0.047
radio          0.1965      0.009     21.994      0.000       0.179       0.214
newspaper     -0.0028      0.006     -0.451 *****0.653*****   -0.015       0.009

newspaper önemsizzz!!!!!!

"""

#%% Korelasyon

sns.heatmap(df.corr(),annot=True)

#%% Sonuclara göre tekrar model oluştur backward elimination

X_train_new=X_train[["TV","radio"]]
X_train_new
X_train_new.shape

X_test_new=X_test[["TV","radio"]]
X_test_new

# modeli tekrar kurgula

lr.fit(X_train_new,y_train)

y_pred_new=lr.predict(X_test_new)


#ols_yeni

X_train_new_ols=sm.add_constant(X_train_new)

sm_model=sm.OLS(y_train,X_train_new_ols)

sonuc=sm_model.fit()

print(sonuc.summary())


#%% Model Confidence 2


r_2=r2_score(y_test, y_pred_new)
print("R2",r_2) #R2 0.8604541663186569

print("R2 %",r_2*100) #R2 % 86.04541663186569


# MSE  >>>>>>> RMSE
#MSE

mse=mean_squared_error(y_test,y_pred_new)
print("MSE",mse) #MSE 4.391429763581881

#RMSE

rmse=math.sqrt(mse)
print("RMSE",rmse) #RMSE 2.0955738506628396














