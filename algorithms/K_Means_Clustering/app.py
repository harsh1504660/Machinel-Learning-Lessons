from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans_scratch import KMeans_1

centroids = [(-5,-5),(5,5),(-2.5,2.5)]
cluster_std = [1,1,1]

x,y = make_blobs(n_samples=100,cluster_std=cluster_std,centers=centroids,n_features=2,random_state=2)

km = KMeans_1(n_clusters=3,max_iteration=200)
y_means = km.fit_predict(x)
plt.scatter(x[y_means==0,0],x[y_means == 0,1],color='red')
plt.scatter(x[y_means==1,0],x[y_means == 1,1],color='blue')
plt.scatter(x[y_means==2,0],x[y_means == 2,1],color='yellow')
plt.show()