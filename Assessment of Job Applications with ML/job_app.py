# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 00:10:19 2022

@author: cinar
"""


#%% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings


#%% import data and informatıon

data=pd.read_csv("DecisionTreesClassificationDataSet.csv",sep=",")

df=data.copy()


df.head(5)

"""
   Deneyim Yili SuanCalisiyor?  ...  StajBizdeYaptimi? IseAlindi
0            11              Y  ...                  N         Y
1             0              N  ...                  Y         Y
2             5              Y  ...                  Y         Y
3             3              N  ...                  Y         Y
4             0              N  ...                  N         N

"""
df.shape

"""
Out[10]: (16, 7)

"""

df.info()

"""
---  ------                   --------------  ----- 
 0   Deneyim Yili             16 non-null     int64 
 1   SuanCalisiyor?           16 non-null     object
 2   Eski Calistigi Firmalar  16 non-null     int64 
 3   Egitim Seviyesi          16 non-null     object
 4   Top10 Universite?        16 non-null     object
 5   StajBizdeYaptimi?        16 non-null     object
 6   IseAlindi                16 non-null     object
dtypes: int64(2), object(5)

"""

df.describe().T
"""
Deneyim Yili              16.0  4.5000  5.680376  0.0  0.75  2.5  5.75  19.0
Eski Calistigi Firmalar   16.0  1.5625  2.220173  0.0  0.00  1.0  2.00   7.0
"""


#%% mapping

fix_mapping_edu = {'BS': 0, 'MS': 1, 'PhD': 2}
fix_mapping = {'Y': 1, 'N': 0}

df['IseAlindi'] = df['IseAlindi'].map(fix_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(fix_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(fix_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(fix_mapping)
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(fix_mapping_edu)

df.head(5)
"""
   Deneyim Yili  SuanCalisiyor?  ...  StajBizdeYaptimi?  IseAlindi
0            11               1  ...                  0          1
1             0               0  ...                  1          1
2             5               1  ...                  1          1
3             3               0  ...                  1          1
4             0               0  ...                  0          0

[5 rows x 7 columns]
"""
df.describe().T

"""
                         count    mean       std  min   25%  50%   75%   max
Deneyim Yili              16.0  4.5000  5.680376  0.0  0.75  2.5  5.75  19.0
SuanCalisiyor?            16.0  0.3125  0.478714  0.0  0.00  0.0  1.00   1.0
Eski Calistigi Firmalar   16.0  1.5625  2.220173  0.0  0.00  1.0  2.00   7.0
Egitim Seviyesi           16.0  0.6250  0.885061  0.0  0.00  0.0  1.25   2.0
Top10 Universite?         16.0  0.4375  0.512348  0.0  0.00  0.0  1.00   1.0
StajBizdeYaptimi?         16.0  0.3750  0.500000  0.0  0.00  0.0  1.00   1.0
IseAlindi                 16.0  0.6250  0.500000  0.0  0.00  1.0  1.00   1.0
"""

df.info()
"""
0   Deneyim Yili             16 non-null     int64
 1   SuanCalisiyor?           16 non-null     int64
 2   Eski Calistigi Firmalar  16 non-null     int64
 3   Egitim Seviyesi          16 non-null     int64
 4   Top10 Universite?        16 non-null     int64
 5   StajBizdeYaptimi?        16 non-null     int64
 6   IseAlindi                16 non-null     int64
dtypes: int64(7)
"""


df.isnull().sum()

"""
Deneyim Yili               0
SuanCalisiyor?             0
Eski Calistigi Firmalar    0
Egitim Seviyesi            0
Top10 Universite?          0
StajBizdeYaptimi?          0
IseAlindi                  0
dtype: int64
"""


#%% Exploratory Data Analysis


#correlation
corr=df.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True,fmt=".2f")
plt.title("Correlation  matrix")
plt.show()

plt.figure(figsize=(10,6))
sns.clustermap(corr,annot=True,fmt=".2f",color="green")
plt.title("Cluster map")
plt.show()


plt.figure(figsize=(10,6))
plt.hist(df)
plt.title("Histogram of DataFrame")
plt.show()

target=df["IseAlindi"]
staj=df["StajBizdeYaptimi?"]

plt.figure(figsize=(10,6))
plt.scatter(staj,target,color="green")
plt.title("Staj and Target")
plt.xlabel("Staj")
plt.ylabel("Target")
plt.legend(loc="upper left")
plt.show()





#%% train test split

y=df["IseAlindi"]

X=df.drop("IseAlindi",axis=1)


#%% Create model decision tree<<<< gini

from sklearn.tree import DecisionTreeClassifier

dec_tree=DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=0)

dec_tree.fit(X,y)

y_pred=dec_tree.predict([[20,1,2,2,0,0]])

print(y_pred)

"""
[1]
array([1], dtype=int64)
"""

print(dec_tree.predict([[5, 1, 3, 0, 0, 0]]))


# In[ ]: Prediction

print("Prediction 1 : ",dec_tree.predict([[2, 0, 7, 0, 1, 0]]))

print("Prediction 2 : ",dec_tree.predict([[2, 1, 7, 0, 0, 0]]))

print("Prediction 3 : ",dec_tree.predict([[20, 0, 5, 1, 1, 1]]))

"""
print("Prediction 1 : ",dec_tree.predict([[2, 0, 7, 0, 1, 0]]))
Prediction 1 :  [0]

print("Prediction 2 : ",dec_tree.predict([[2, 1, 7, 0, 0, 0]]))
Prediction 2 :  [1]

print("Prediction 3 : ",dec_tree.predict([[20, 0, 5, 1, 1, 1]]))
Prediction 3 :  [1]

"""



