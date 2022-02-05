# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 23:05:54 2022

@author: cinar
"""

#%% import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model
import pickle

#%% import data

df = pd.read_csv("multilinearregression.csv",sep = ";")

#%% 3 Collect datas info and more...

df.shape #8, 4)

df.head()
"""
   alan  odasayisi  binayasi   fiyat
0   180          5        10  510000
1   225          4        18  508000
2   260          3         2  548000
3   320          6        10  650000
4   335          4         9  628000

"""
df.info()
"""
0   alan       8 non-null      int64
1   odasayisi  8 non-null      int64
2   binayasi   8 non-null      int64
3   fiyat      8 non-null      int64
"""
df.describe().T
"""
           count        mean           std  ...       50%       75%       max
alan         8.0     302.500     73.920807  ...     327.5     347.5     400.0
odasayisi    8.0       4.250      1.035098  ...       4.0       5.0       6.0
binayasi     8.0       9.125      5.841661  ...      10.0      11.0      18.0
fiyat        8.0  610125.000  79982.922284  ...  630000.0  657500.0  725000.0

[4 rows x 8 columns]
"""
df.isnull().sum()
"""
alan         0
odasayisi    0
binayasi     0
fiyat        0
dtype: int64

"""

df[['alan', 'odasayisi', 'binayasi']]


df['fiyat']

#%%

# linear regression

reg = linear_model.LinearRegression()
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])

# Prediction 

reg.predict([[230,4,10]])

reg.predict([[230,6,0]])

reg.predict([[355,3,20]])

reg.predict([[230,4,10], [230,6,0], [355,3,20]])

#%%

reg.coef_

reg.intercept_

"""
reg.coef_
Out[19]: array([ 1018.99865454, 14893.82374984, -2606.68429997])

reg.intercept_
Out[20]: 262365.1503032055

"""

#%%


# Multiple Linear regression f
# y= a + b1X1 + b2X2 + b3X3 + ...

a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230
x2 = 4
x3 = 10
y = a + b1*x1 + b2*x2 + b3*x3

y


#%%

import pickle

#model save
model_file="price_prediction_mlr.pickle"

pickle.dump(reg, open(model_file,"wb"))


#model load
my_model=pickle.load(open("price_prediction_mlr.pickle","rb"))


my_model.predict([[230,4,10]])

my_model.predict([[230,6,0]])

my_model.predict([[355,3,20]])

my_model.predict([[230,4,10], [230,6,0], [355,3,20]])



"""
my_model.predict([[230,4,10]])
Out[6]: array([530243.29284619])

my_model.predict([[230,6,0]])
Out[7]: array([586097.7833456])

my_model.predict([[355,3,20]])
Out[8]: array([616657.45791365])

my_model.predict([[230,4,10], [230,6,0], [355,3,20]])
Out[9]: array([530243.29284619, 586097.7833456 , 616657.45791365])

"""












