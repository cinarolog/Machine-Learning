# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 00:57:05 2022

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



#%% SVM

# SVC classifier
from sklearn.svm import SVC
# accuracy
from sklearn.metrics import accuracy_score
# classification_report
from sklearn.metrics import classification_report
# ROC-AUC
from sklearn.metrics import roc_auc_score

svc = SVC()

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

#accuracy
acc=accuracy_score(y_test,y_pred)
print("Accuracy Score with Default Hyperparameters :", acc*100)
"""
Accuracy Score : 90.53398058252428
"""


# classification_report
print(classification_report(y_test, y_pred))


#roc_auc
roc_auc_scores=roc_auc_score(y_test,y_pred)
print("Roc_Auc Score with Default Hyperparameters :", roc_auc_scores*100)
"""
Roc_Auc Score with Default Hyperparameters : 58.14100759921194
"""

#%%SVM with RBF Kernel & C=100

# rbf kernel and C=100
svc = SVC(C=100.0) 

# fit classifier
svc.fit(X_train,y_train)

# test set ile tahmin
y_pred_2=svc.predict(X_test)

#accuracy
acc_2=accuracy_score(y_test,y_pred)
print("Accuracy_2 Score with Default Hyperparameters :", acc_2*100)

#roc_auc
roc_auc_scores_2=roc_auc_score(y_test,y_pred)
print("Roc_Auc2 Score with Default Hyperparameters :", roc_auc_scores_2*100)




#%%#  Grid Search CV import SVC classifier 

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# default hyperparameter and model => kernel=rbf, C=1.0 and gamma=scale
svc_grid=SVC() 


# hyperparameter tuning için parametre gridi
parameters = [ # {'C':[1, 10, 100], 'kernel':['linear']},
               {'C':[0.1, 1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':['scale', 'auto', 0.001, 0.01, 0.1, 0.9]},
               # {'C':[0.1, 1, 10, 100], 'kernel':['poly'], 'degree': [2,3], 'gamma':['scale', 'auto', 0.001, 0.01, 0.1, 1]} 
             ]


# Grid Search
# n_jobs=4 -> 4 işlemci core'unu da kullansın
# scoring => 'balanced_accuracy', 'f1', ‘precision’, ‘recall’, ‘roc_auc’ 
grid_search = GridSearchCV(estimator = svc_grid,  
                           param_grid = parameters,
                           scoring = 'balanced_accuracy',
                           cv = 5,
                           verbose=1,
                           n_jobs=4)


# Model Train fitt
grid_search.fit(X_train, y_train)
# Fitting 5 folds for each of 30 candidates, totalling 150 fits
# [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
# [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    3.3s
# [Parallel(n_jobs=4)]: Done 150 out of 150 | elapsed:   16.2s finished
# GridSearchCV(cv=5, estimator=SVC(), n_jobs=4,
#              param_grid=[{'C': [0.1, 1, 10, 100, 1000],
#                           'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.9],
#                           'kernel': ['rbf']}],
#              scoring='balanced_accuracy', verbose=1)
# ​

# GridSearchCV 
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))


# best parameters
print('En iyi sonucu veren Parametreler :','\n\n', (grid_search.best_params_))


# GridSearch'ün verdiği en iyi estimator
print('\n\nSeçilen Estimator:','\n\n', (grid_search.best_estimator_))













