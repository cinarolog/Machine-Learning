# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:30:57 2022

@author: cinar
"""

#%% import data and libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns 
import warnings



data=pd.read_csv("diabetes.csv")
df=data.copy()
df2=data.copy()
#%% data info

df.head()
df.info()
"""
0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
"""
df.shape
df.describe().T

healthy=df[df.Outcome==0]
sick=df[df.Outcome==1]

#%%

""" Exploratory Data analysis"""

#%% Age-Glucose
plt.scatter(healthy.Age,healthy.Glucose,color="green",alpha=0.4)
plt.scatter(sick.Age,sick.Glucose,color="red",alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.title("Age-Glucose")
plt.legend()


#%% Age Bloodpressure

plt.scatter(healthy.Age,healthy.BloodPressure,color="orange",alpha=0.5)
plt.scatter(sick.Age,sick.BloodPressure,color="blue",alpha=0.5)
plt.xlabel("Age")
plt.ylabel("BloodPressure")
plt.title("Age BloodPressure")


#%% input output6

X=df.drop("Outcome",axis=1)
y=data["Outcome"]

X.columns
cols=X.columns

#%% Scaling

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X=scaler.fit_transform(X)
type(X)

X=pd.DataFrame(X,columns=[cols])

type(X)


#%% train test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


X_train.shape
y_train.shape


#%%

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

#Train
knn.fit(X_train,y_train)

#Predict
y_pred=knn.predict(X_test)
   
y_pred
y_pred.shape


from sklearn.metrics import accuracy_score ,roc_auc_score

acc=accuracy_score(y_test,y_pred)
print("Acuuracy Score :",acc*100)

roc_auc=roc_auc_score(y_test, y_pred)
print("RocAuc Score :",roc_auc*100)
"""

print("Acuuracy Score :",acc*100)
Acuuracy Score : 70.12987012987013

roc_auc=roc_auc_score(y_test, y_pred)

print("RocAuc Score :",roc_auc*100)
RocAuc Score : 65.85858585858585


"""
#%% LogisticRegression

from sklearn.linear_model import LogisticRegression
# model
logreg = LogisticRegression(solver='liblinear', random_state=0)

logreg.fit(X_train, y_train)
#LogisticRegression(random_state=0, solver='liblinear')


y_pred2 = logreg.predict(X_test)

y_pred2

acc=accuracy_score(y_test,y_pred2)
print("Acuuracy Score :",acc*100)


roc_auc2=roc_auc_score(y_test, y_pred2)
print("RocAuc2 Score :",roc_auc2*100)

logreg.predict_proba(X_test)[:,0]
logreg.predict_proba(X_test)[:,1]

"""

print("Acuuracy Score :",acc*100)
Acuuracy Score : 75.32467532467533

roc_auc2=roc_auc_score(y_test, y_pred2)

print("RocAuc2 Score :",roc_auc2*100)
RocAuc2 Score : 73.53535353535354

"""


