import random
import numpy as np
class KMeans_1 :
    def __init__(self,n_clusters=2,max_iteration=100):
        self.n_clusters = n_clusters
        self.max_iteration = max_iteration
        self.centroid = None

    def fit_predict(self,x):
        random_index = random.sample(range(0,x.shape[0]),self.n_clusters)
        self.centroid = x[random_index]
        
        for i in range(self.max_iteration):
            # assign clusters
            cluster_group = self.assign_clusters(x)

            # move centoid
            old_centroid = self.centroid
            self.centroid = self.move_centroid(x,cluster_group)

            # check finish
            if (old_centroid == self.centroid).all():
                break
        return cluster_group
    def assign_clusters(self,x):
        cluster_group = []
        distances = []
        for row in x:
            for centroid in self.centroid:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distances = min(distances)
            index_pos = distances.index(min_distances)
            print(index_pos)
            distances.clear()
            cluster_group.append(index_pos)
        return np.array(cluster_group)
    
    def move_centroid(self , x , cluster_group):
        new_centroid = []
        cluster_type = np.unique(cluster_group)
        for type in cluster_type:
            new_centroid.append(x[cluster_group ==type].mean(axis=0))
        return np.array(new_centroid)
