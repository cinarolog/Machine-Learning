# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:11:46 2022

@author: cinar
"""

#%%

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# Dataset'i import edelim

df = pd.read_csv('Mall_Customers.csv', index_col='CustomerID')
df   

#%%

df.hhead
df.info()

df.isnull().sum()


#%%


df.drop_duplicates(inplace=True)

X = df.iloc[:, [2, 3]].values
X


from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')

plt.title('Elbow Metodu')

plt.xlabel('Cluster Sayısı')
plt.ylabel('WCSS')

plt.show()



#%%# fit ve predict 


y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(15,7))


# K = 
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50)
sns.scatterplot(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], color = 'grey', label = 'Cluster 4',s=50)
sns.scatterplot(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], color = 'orange', label = 'Cluster 5',s=50)

sns.scatterplot(kmeans.cluster_centers_[:, 0], 
                kmeans.cluster_centers_[:, 1], 
                color = 'red', 
                label = 'Centroids',
                s=300,
                marker=',')

plt.grid(False)

plt.title('Müşteri Clusterları')

plt.xlabel('Aylık Gelir (k$)')
plt.ylabel('Spending Score (1-100)')

plt.legend()
plt.show()






#%%














#%%













#%%

















#%%














#%%













#%%

















