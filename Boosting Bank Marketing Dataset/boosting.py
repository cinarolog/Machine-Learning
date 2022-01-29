# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:16:18 2022

@author: cinar
"""


#%% import data and libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('bank-additional.csv', sep=';')

df=data.copy()



#%% Data Analysis and information

df.head(5)
df.shape
df.info()
df.describe().T
df.isnull().sum()# none missing values =0

df.drop("duration",axis=1,inplace=True)

df["y"].value_counts()
len(df)

# not balanace dataset
df["y"].value_counts() / np.float(len(df))

#numeric features
df.select_dtypes(include=["int64","float64"]).columns
numeric_cols=df.select_dtypes(include=["int64","float64"]).columns

#Categorical features

df.select_dtypes(include=["object"]).columns
cat_cols=df.select_dtypes(include=["object"]).columns
# 19 variables
# 10 categoric
# 9 numeric



#Numerical variables

df.hist(column=numeric_cols,figsize=(10,10))
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()

"""
Ordinal: Order is important in ordinal variables. 
(like 'very satisfied', 'satisfied', 'less satisfied', 'not satisfied'.)
Nominal: For nominal variables, the order is not important. (Like 'blue', 'green', 'red'.)
"""
# ordinal variables

df['poutcome'].value_counts()
df['poutcome'] = df['poutcome'].map({'failure': -1,'nonexistent': 0,'success': 1})
df['poutcome'].value_counts()


df['default'].value_counts()
df['default'] = df['default'].map({'yes': -1,'unknown': 0,'no': 1})
df['default'].value_counts()


df['housing'] = df['housing'].map({'yes': -1,'unknown': 0,'no': 1})
df['loan'] = df['loan'].map({'yes': -1,'unknown': 0,'no': 1})

#Nominal variables:
    
#before OHE
cat_cols
nominal = ['job','marital','education','contact','month','day_of_week']
df.shape
df.columns
df = pd.get_dummies(df,columns=nominal)
# after OHE
df.shape    


#y
df['y']=df['y'].map({'yes': 1,'no': 0})
df.head()    


#%% input output

X=df.drop("y",axis=1)
y=df["y"]


#%%Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train.shape, X_test.shape
y_train.shape, y_test.shape



#%% Feature Scaling

cols=X_train.columns

X_train[numeric_cols]


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
type(X_train)

X_train=pd.DataFrame(X_train,columns=[cols])
X_test=pd.DataFrame(X_test,columns=[cols])

X_train[cols]


#%% Create model AdaBoosting

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

# Create adaboost classifer object
ada_boost = AdaBoostClassifier(n_estimators=400, learning_rate=1, random_state=0)

# Train Adaboost Classifer
model_ada_boost = ada_boost.fit(X_train, y_train)


#Predict the response for test dataset
y_pred_ada_boost = model_ada_boost.predict(X_test)

# roc_auc
roc_auc_scores=roc_auc_score(y_test, y_pred_ada_boost)
print("Roc_Auc Score :",roc_auc_scores*100) #Roc_Auc Score : 62.478891077962295

#accuracy
acc=accuracy_score(y_test,y_pred_ada_boost)
print("Accuracy Scores :",acc*100) #Accuracy Scores : 90.89805825242718




#%% XGBoost

# Import the XGBoost classifier
from xgboost import XGBClassifier

# Create xgboost classifer object
xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=1, random_state=0)

# Train XGBoost Classifer
model_xgb = xgb.fit(X_train, y_train)


#Predict the response for test dataset
y_pred_xgb = model_xgb.predict(X_test)

# roc_auc
roc_auc_scores2=roc_auc_score(y_test, y_pred_xgb)
print("Roc_Auc Score :",roc_auc_scores2*100) #Roc_Auc Score : 63.840416549394874

#accuracy
acc2=accuracy_score(y_test,y_pred_xgb)
print("Accuracy Scores :",acc2*100) #Accuracy Scores : 89.07766990291263












