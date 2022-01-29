# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:17:22 2022

@author: cinar
"""


#%% import data and libraries

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv("wine.data",header=None)
df=data.copy()


#%% EDA Exploratory Data Analysis

df.shape
df.info()
df.describe().T

df.groupby(0).size()

df.isnull().sum()
df.columns



col_names=[
    "class",
    "alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"]


df.columns=col_names
df.columns
df.info()


#categorical variables

categorical = [var for var in df.columns if df[var].dtype == 'O']
print('Kategorik Değişken sayısı: {}'.format(len(categorical)))

#numerical variables

numerical = [var for var in df.columns if df[var].dtype != 'O']
print('Numerik Değişken sayısı: {}\n'.format(len(numerical) - 1))
print('Numerik Sütunlar :', numerical[1:])


df[numerical].head()




#%% Feature and Label

X=df.drop(["class"],axis=1)
X.head()

y=df["class"]

#class counts
y.value_counts(sort=False)
df.groupby("class").size()


#%% Train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


X_train.shape,X_test.shape
y_train.shape,y_test.shape

y_train=y_train.values.reshape(-1,1)
y_test=y_test.values.reshape(-1,1)

y_train.shape,y_test.shape

#%% Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)

"""
standard Scaler returns the numpy array.
"""


cols=[
    "alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"]

type(X_train)


X_train=pd.DataFrame(X_train,columns=[cols])
X_test=pd.DataFrame(y_train,columns=[cols])

type(X_train)


#%% Model Create

from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(X_train,y_train)



#%% Prediction

y_pred=gnb.predict(X_test)
y_pred


#%% Accuracy Score

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,y_pred)
acc

print("Accuracy Score :",acc*100)

"""

print("Accuracy Score :",acc*100)
Accuracy Score : 91.66666666666666

"""


#%% Overfitting Underfitting


"""
# Overfitting: If Train Data Score is TOO HIGH, but Test Data Score is TOO LOW.
# Underfitting: If Train Data Score is VERY LOW, but Test Data Score is VERY HIGH.

"""



y_pred_train=gnb.predict(X_train)
y_pred_train


acc=accuracy_score(y_train,y_pred_train)
acc

print("Accuracy Score :",acc*100)

"""
print("Accuracy Score :",acc*100)
Accuracy Score : 98.59154929577466

"""



"""
this model is good enough..


"""














