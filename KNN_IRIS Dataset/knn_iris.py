# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 19:33:55 2022

@author: cinar
"""

#%% import data and libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


data=pd.read_csv("iris.csv",index_col=0)
df=data.copy()


#%% Data information

df.shape#  n=150  p=4  +1 y
df.info
df.describe().T
df.columns
type(df["SepalLengthCm"])


df.groupby("Species").size()
"""
Species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64
"""

#%% Features and Labels 

#Features
feature_columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X=df[feature_columns].values
X

#Labels
y=df["Species"]
y
y.shape

#%% Label Encoding   categorical>>>> numerical

from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()

#before
print("setosa",y[0:10])
print("versicolour",y[50:60])
print("virginica",y[100:110])

#after

y=le.fit_transform(y)

print("setosa",y[0:10])
print("versicolour",y[50:60])
print("virginica",y[100:110])


#%% Train-test-split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#train data
print("X_train shape",X_train.shape)
print("y_train shape",y_train.shape)

#test data
print("X_test shape",X_test.shape)
print("y_test shape",y_test.shape)



#%% Scaling 

"""
Our values are in cm. so we don't need to scale.
low values.
"""

#%% Visualization

plt.figure()
sns.pairplot(df,hue="Species",size=3,markers=["o","s","D"])
plt.show()


#Box plot

plt.figure(figsize=(10,6))
df.boxplot(by="Species")
plt.show()

#%% Model Create

from sklearn.neighbors import KNeighborsClassifier

def fit_knn(train_data,label_data,test_data,k):
    
    knn=KNeighborsClassifier(n_neighbors=k)
    #Train
    knn.fit(train_data,label_data)
    #Predict
    prediction_label=knn.predict(test_data)
    #Return
    return prediction_label



#%% Prediction

y_predict=fit_knn(X_train,y_train,X_test,3)
y_predict
y_predict.shape

"""
array([2, 1, 0, 2, 0, 2, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 0, 0, 2, 1,
       0, 0, 2, 0, 0, 1, 1, 0])

y_predict.shape
Out[45]: (30,)

"""


#%% Model Confidence


def accuracy(test_label,pred_label):
    
    #true prediction counts
    correct=np.sum(test_label==pred_label)
    
    n=len(test_label)
    
    accur=correct/n
    
    return accur


confidence=accuracy(y_test,y_predict)
confidence
print("Confidence:",confidence *100)


from sklearn.metrics import accuracy_score
accuracy_sklearn = accuracy_score(y_test, y_predict)*100
print('Model Confidence (Accuracy) ' + str(round(accuracy_sklearn, 2)) + ' %.')


"""
Our Model Accuracy (Accuracy) is 96.67%.
As you can see, the value given to us by Accuracy and Scikit-Learn, 
which we calculated manually, is exactly the same.

"""



#%% Choose k value for knn

"""
According to experience, 
the K value should not be greater than the square root (n) of the available data.
The k value is usually an odd number.
"""
n = len(df)
n#150

import math
k_max = math.sqrt(n)
k_max#12.24744871391589

normal_accuracy = []  

#probably k values

for k in range(1,13):
    y_predict = fit_knn(X_train, y_train, X_test, k)
    accur = accuracy_score(y_test, y_predict)
    normal_accuracy.append(accur)
    
   ​
plt.xlabel("k")
plt.ylabel("accuracy")
plt.plot(range(1,13), normal_accuracy, c='g')
plt.grid(True)  
plt.show()
normal_accuracy = []  
​

k_value = range(1, 31)
​

for k in k_value:
    y_predict = fit_knn(X_train, y_train, X_test, k)
    accur = accuracy_score(y_test, y_predict)
    normal_accuracy.append(accur)

​
plt.xlabel("k")
plt.ylabel("accuracy")
plt.plot(range(1,31), normal_accuracy, c='g')
plt.grid(True)  
plt.show()














