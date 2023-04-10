"""clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RYneJRQ3TkuToQ1x5V_DezJhdNzJhOH3
"""

import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

dataset = pd.read_csv('Live.csv')
print(dataset.shape)

dataset.shape

dataset.Column1.unique()

dataset.Column2.unique()

dataset.Column3.unique()

dataset.Column4.unique()

dataset = dataset.iloc[:,:12]
dataset

dataset = dataset.iloc[:,1:]

categorical_features = dataset.select_dtypes(include=['object']).columns
numerical_features = dataset.select_dtypes(include=['int64','float64']).columns
print(categorical_features)
print(numerical_features)

dataset.isnull().sum()

dataset.status_published = pd.to_datetime(dataset.status_published).dt.year
dataset

dataset.status_type.unique()

dataset.status_published.unique()

len(dataset.status_published.unique())

dataset2=dataset.copy()
labelencoder_X = LabelEncoder()
for i in categorical_features:
  dataset.loc[:,i] = labelencoder_X.fit_transform(dataset.loc[:,i])
dataset

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [0])], remainder='passthrough')
dataset = ct.fit_transform(dataset)

dataset

dataset = pd.DataFrame(dataset)
dataset.columns = ct.get_feature_names_out()
dataset

dataset.shape

dataset = pd.DataFrame(dataset)
ns = Normalizer()
dataset = ns.fit_transform(dataset)
dataset

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42,n_init='auto')
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(dataset)

plt.scatter(dataset[y_kmeans == 0, 4], dataset[y_kmeans == 0, 8], s = 100, c = 'orange', label = 'Cluster 1')
plt.scatter(dataset[y_kmeans == 1, 4], dataset[y_kmeans == 1, 8], s = 100, c = 'violet', label = 'Cluster 2')
plt.scatter(dataset[y_kmeans == 2, 4], dataset[y_kmeans == 2, 8], s = 100, c = 'lightgreen', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 4], kmeans.cluster_centers_[:, 8], s = 300, c = 'red', label = 'Centroids')
plt.title('Clusters of posts')
plt.xlabel('Publish Date Value')
plt.ylabel('Like score')
plt.legend()
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(dataset, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(dataset)

plt.scatter(dataset[y_hc == 0, 4], dataset[y_hc == 0, 8], s = 100, c = 'orange', label = 'Cluster 1')
plt.scatter(dataset[y_hc == 1, 4], dataset[y_hc == 1, 8], s = 100, c = 'violet', label = 'Cluster 2')
plt.scatter(dataset[y_hc == 2, 4], dataset[y_hc == 2, 8], s = 100, c = 'lightgreen', label = 'Cluster 3')

plt.title('Clusters of customers')
plt.xlabel('Publish Date Value')
plt.ylabel('Like score')
plt.legend()
plt.show()