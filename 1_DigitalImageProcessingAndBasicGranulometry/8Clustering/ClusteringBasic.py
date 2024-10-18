#! /usr/bin/env python


import matplotlib.pylab as plt

#preparing data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=3, n_features=3, random_state=0)

# plot data
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel("$x1$")
plt.ylabel("$x2$")
plt.savefig("output/data-kMeans.png")
plt.show()

#determine number of clusters looking for 'elbow' on WSSE vs cluster numbers plot
from sklearn.cluster import KMeans
wsse = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, n_init=1)
    kmeans.fit(X)
    wsse.append(kmeans.inertia_)

# make scree plot
plt.plot(range(1, 20), wsse)
plt.xlabel("Clusters")
plt.ylabel("WSSE")
plt.savefig("output/kMenas_Scree.png")
plt.show()

#make k-means clustering
kmeans = KMeans(n_clusters=3, n_init=1)
kmeans.fit(X)
##clusters labels
labels = kmeans.predict(X)
##locations of centroids
centroids  = kmeans.cluster_centers_

#plot
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0],centroids[:,1], c='r', marker='v', label="centers")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("output/kMeans-clustering.png")
plt.show()