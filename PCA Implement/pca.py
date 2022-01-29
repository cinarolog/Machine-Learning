# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:42:25 2022

@author: cinar
"""

#%% import data and libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')

from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D


iris = datasets.load_iris()
X = iris.data
y = iris.target


#%% visualization


fig = plt.figure(1, figsize=(6, 5))

plt.clf()

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))


y_clr = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_clr, 
           cmap=plt.cm.nipy_spectral)

plt.show()



#%% train test split


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, 
                                                    stratify=y, 
                                                    random_state=42)

X_train.shape


#%% DecisionTree

from sklearn.tree import DecisionTreeClassifier

# Decision tree, max_depth = 2
dec_tree = DecisionTreeClassifier(max_depth=2, random_state=42)

# modeli fit edelim
dec_tree.fit(X_train, y_train)


preds = dec_tree.predict_proba(X_test)

print(preds)
# accuracy - PCA olmadan

y_pred = dec_tree.predict(X_test)

print('Accuracy without PCA  : {:.2f}'.format(accuracy_score(y_test, y_pred)))



#%% PCA

from sklearn import decomposition

pca=decomposition.PCA(n_components=2)

X
X.mean()

X_centered = X - X.mean(axis=0)

X_centered
X_centered.shape

pca.fit(X_centered)

X_pca = pca.transform(X_centered)
X_pca

X_pca.shape


plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')

plt.legend(loc=0)
plt.show()



#%%  Train, test split -> X_pca


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3, 
                                                    stratify=y, 
                                                    random_state=42)
# Decision tree, max_depth = 2
clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)

# modeli fit edelim
clf_pca.fit(X_train, y_train)

# accuracy - PCA olmadan

y_pred_pca = clf_pca.predict(X_test)

print('Accuracy with PCA: {:.2f}'.format(accuracy_score(y_test, y_pred_pca)))



#%% PCA Analysis

pca.components_

pca.explained_variance_ratio_

for i, component in enumerate(pca.components_):
    print("{}. component: {}% variance".format(i + 1, 
          round(100 * pca.explained_variance_ratio_[i], 2)))










