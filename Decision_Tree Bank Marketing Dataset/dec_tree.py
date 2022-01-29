# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 20:48:02 2022

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


#%% Create model decision tree<<<< gini

from sklearn.tree import DecisionTreeClassifier

dec_tree=DecisionTreeClassifier(criterion="gini",max_depth=3,random_state=0)

dec_tree.fit(X_train,y_train)

y_pred=dec_tree.predict(X_test)


#%% Confidence

from sklearn.metrics import roc_auc_score,accuracy_score

#accuracy
acc=accuracy_score(y_test, y_pred)
print("Accuracy Score :",acc*100) #Accuracy Score : 91.2621359223301

#roc_auc
roc_auc=roc_auc_score(y_test,y_pred)
print("ROC_AUC Score:",roc_auc*100)#ROC_AUC Score: 57.9510273008725


#%% Compare train_set   test_set roc auc

y_pred_train=dec_tree.predict(X_train)

roc_auc_train=roc_auc_score(y_train,y_pred_train)
print("Roc_auc_train Score :",roc_auc_train*100) #Roc_auc_train Score : 60.8538812785388

acc_train=accuracy_score(y_train, y_pred_train)
print("Accuracy_train_Score",acc_train*100) #Accuracy_train_Score 90.34901365705615

"""
There is not overfitting...
"""



#%% Visualisation Decisin Tree

from sklearn import tree

tree.plot_tree(dec_tree.fit(X_train,y_train))
plt.show()


#%% Create model decision tree<<<< gini

from sklearn.tree import DecisionTreeClassifier

dec_tree2=DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=0)

dec_tree2.fit(X_train,y_train)

y_pred2=dec_tree2.predict(X_test)




#%% Confidence 2

from sklearn.metrics import roc_auc_score,accuracy_score

#accuracy
acc2=accuracy_score(y_test, y_pred2)
print("Accuracy Score :",acc2*100) #Accuracy Score : 91.14077669902912

#roc_auc
roc_auc2=roc_auc_score(y_test,y_pred2)
print("ROC_AUC Score:",roc_auc2*100)#ROC_AUC Score: 56.70208274697439


#%% Compare train_set   test_set roc auc2

y_pred_train2=dec_tree2.predict(X_train)

roc_auc_train2=roc_auc_score(y_train,y_pred_train2)
print("Roc_auc_train Score :",roc_auc_train2*100) #Roc_auc_train Score : 59.804337899543384

acc_train2=accuracy_score(y_train, y_pred_train2)
print("Accuracy_train_Score",acc_train2*100) #Accuracy_train_Score 90.13657056145675

"""
There is not overfitting...
"""



#%% Visualisation Decisin Tree2

from sklearn import tree

tree.plot_tree(dec_tree2.fit(X_train,y_train))
plt.show()






