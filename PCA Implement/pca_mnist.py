# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 17:51:50 2022

@author: cinar
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')

from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


#%%

digits = datasets.load_digits()
X = digits.data
y = digits.target
X
y

X.shape
y.shape



#%%


plt.figure(figsize=(16, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i,:].reshape([8,8]), cmap='gray');
    





#%%

# PCA 
pca = decomposition.PCA(n_components=2)

X_reduced = pca.fit_transform(X)

print('%d-boyutlu data 2 boyutlu uzaya yansıtıldı (projecting).' % X.shape[1])

X_reduced.shape


#2D

plt.figure(figsize=(12,10))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()

plt.title('MNIST - PCA Projeksiyonu');




#%%
# PCA  variance

pca = decomposition.PCA().fit(X)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)

plt.xlabel('Principal Component Sayısı')
plt.ylabel('Toplam Açıklanan Variance')

plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.axvline(21, c='b')
plt.axhline(0.9, c='r')

plt.show()





