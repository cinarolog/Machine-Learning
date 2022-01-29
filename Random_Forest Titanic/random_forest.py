# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 00:16:38 2022

@author: cinar
"""


#%% import libraries and data 1-2

import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("train.csv")
df=data.copy()


#%% 3 Collect datas info and more...

df.shape #(891, 12)
df.head()
df.info()
df.describe().T
df.isnull().sum()
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177 missing values
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687 missing values
Embarked         2
"""


""" 3.1 missing values for age """


df["Age"].isnull().sum()/df.shape[0]*100 # 19.865319865319865

ax =df['Age'].hist(bins=15, density=True, stacked=True, alpha=0.7)

df['Age'].plot(kind='density')

ax.set(xlabel='Age')
plt.xlim(0, 90)
plt.grid()
plt.show()


# mean -> average
# skipna -> missing values (skip)
df['Age'].mean(skipna=True) 
df['Age'].median(skipna=True)


""" 3.2 missing values for Cabin """

df['Cabin'].isnull().sum() / df.shape[0] * 100 #77.10437710437711

"""
there are lots of missing values therefore we should drop thıs column from dataset
"""


"""3.3 missing values for Embarked """

df['Embarked'].isnull().sum() / df.shape[0] * 100
# 0.22446689113355783


print('(C = Cherbourg, Q = Queenstown, S = Southampton):')
print(df['Embarked'].value_counts() / df.shape[0] * 100)
"""
S    72.278339
C    18.855219
Q     8.641975
"""

sns.countplot(x='Embarked', data=df, palette='Set1')
plt.show()

print('En fazla binilen liman: ', df['Embarked'].value_counts().idxmax())

"""3.4 Final Decision and Implementation"""
# Final decision for columns with missing values:
# We will fill in missing values with Age -> Median method (median = 28)
# Embarked -> we will fill the missing values as 'S'
# Cabin -> we will omit this column because there are too many (77%) missing values

# We will fill in missing values with Age -> Median method (median = 28) fillna()
df["Age"].fillna(df["Age"].median(skipna=True), inplace=True)

# Embarked -> we will fill the missing values as 'S'
df["Embarked"].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)

# Cabin -> we will omit this column because there are too many (77%) missing values
df.drop('Cabin', axis=1, inplace=True)

df.isnull().sum()
"""
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0

"""

df.head()
df.columns

"""
3.5 useless feature analysis

There are two related columns: SibSp, Parch. SibSp: # of siblings / spouses aboard the Titanic.
number of children (couple, husband-wife, for).# Parch: # of parents / children aboard the Titanic.
number of parents for children.# As you can see, both variables are very related to each other and 
there is a high correlation between them.
It will be healthier to collect these two variables under one variable: Is he traveling alone?
Let the name of our variable be YalnizSeyahat. And this will be a categorical variable. 0 or 1.
Looking at SibSp and Parch if the sum of the two is greater than zero then it's not traveling alone -> 0
we will say.
if their sum is zero then it is traveling alone -> 1​.

"""
df['aloneTravel'] = np.where((df["SibSp"] + df["Parch"]) > 0, 0, 1)

#   drop SibSp and Parch 
df.drop('SibSp', axis=1, inplace=True)

df.drop('Parch', axis=1, inplace=True)
df.head()

"""3.6 Categorical variables"""

df.info()

# 4   Sex          891 non-null    object 
# 6   Ticket       891 non-null    object 
# 8   Embarked     891 non-null    object 

df=pd.get_dummies(df,columns=["Pclass","Embarked","Sex"],drop_first=True)
df.head()

#  drop "PassengerId", "Name" and "Ticket"

df.drop('PassengerId', axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.head()



#%% 4 Exploratory Data Analysis  EDA

df.shape #(891, 9)

col_names=df.columns
col_names
df.info()

"""4.1 Age EDA"""

plt.figure(figsize=(10,6))
#  Survived == 1
ax = sns.kdeplot(df["Age"][df.Survived == 1], color="green", shade=True)
#  Survived == 0
sns.kdeplot(df["Age"][df.Survived == 0], color="red", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Survival vs. Death Density Graph for Age')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


"""4.2 Fare (Ücret) için EDA"""

plt.figure(figsize=(10,6))
#  Survived == 1
ax = sns.kdeplot(df["Fare"][df.Survived == 1], color="green", shade=True)
#  Survived == 0
sns.kdeplot(df["Fare"][df.Survived == 0], color="red", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
ax.set(xlabel='Fare')
plt.xlim(-20,200)
plt.show()


"""4.3 Passanger Class (Yolcu Sınıfı) için EDA"""

sns.barplot('Pclass', 'Survived', data=data, color="green")
plt.show()


"""4.4 Aile ile veya Yalnız Seyahat için EDA"""

sns.barplot('aloneTravel', 'Survived', data=df, color="orange")
plt.show()

"""4.5 Sex (Cinsiyet) için EDA"""

sns.barplot('Sex', 'Survived', data=data, color="blue")
plt.show()



#%% input output

y=df["Survived"] #output

X=df

#%%  Feature Scaling

from sklearn.preprocessing import MinMaxScaler


df.describe()
cols = df.columns
cols


scaler = MinMaxScaler()
df = scaler.fit_transform(df)
type(df)

#do it  datayı DataFrame 

df = pd.DataFrame(df, columns=[cols])
df.head()
type(df)


#%% Train test Split

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=1)
X_train.shape
#(712, 8)
X_test.shape
#(179, 8)
y_train

y_test


#%% 8. Model Create Random forest

"""
max_depth: How deep the Decision Trees will go (number of splits).
max_features: The maximum number of variables to use for split in a DT.
criterion: DT criterion (gini, entropy).
n_estimators: Total number of DTs to be created.
"""
from sklearn.ensemble import RandomForestClassifier

fit_rf=RandomForestClassifier(random_state=42)


#%%  Hyperparameter Optimization GridSearchCV

np.random.seed(42)

#  start timer 
start = time.time()

# parameter grid
param_dist = {'max_depth': [2, 3, 4],
              'n_estimators': [100, 200, 400],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

# n_jobs = 4 
cv_rf = GridSearchCV(fit_rf, 
                     cv = 5,
                     param_grid=param_dist, 
                     n_jobs = 4)

# cv_rf fit 
cv_rf.fit(X_train, y_train)

# best parameters
print('GridSearch best parameters: \n', cv_rf.best_params_)

# timer stop
end = time.time()

# write time
print('Grid Search için geçen zaman: {0: .2f}'.format(end - start))


fit_rf.set_params(criterion = 'entropy',
                  max_features = 'log2', 
                  max_depth = 4,
                  n_estimators = 200)

#%% train model

fit_rf.fit(X_train, y_train)

pd.concat((pd.DataFrame(X_train.columns, columns = ['feature']), 
           pd.DataFrame(fit_rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)





#%% Prediction

y_pred = fit_rf.predict(X_test)

# accuracy​
print(" Accuracy Score'u: {0:0.4f}".format(accuracy_score(y_test, y_pred)))
# Accuracy Score'u: 0.7877
# classification report

print(classification_report(y_test, y_pred))










#%%