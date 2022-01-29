# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 20:35:41 2022

@author: cinar
"""

#%% 1-2 import data and libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



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


#%% 8. Model Create

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear', random_state=0)
logreg.fit(X_train, y_train)
#LogisticRegression(random_state=0, solver='liblinear')


#%% 9. Tahmin

y_pred = logreg.predict(X_test)

y_pred

logreg.predict_proba(X_test)[:,0]

logreg.predict_proba(X_test)[:,1]


#%% 10. Accuracy

from sklearn.metrics import accuracy_score

print("Accuracy Score'u: {0:0.4f}".format(accuracy_score(y_test, y_pred)))

"""
 Accuracy Score'u: 0.7556

"""












