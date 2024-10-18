#! /usr/bin/env python

import numpy as np
import matplotlib.pylab as plt

from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
Y = digits.target

#reverse color
X = 255-X

#example digit
plt.imshow(X[100].reshape(8,8), cmap="gray")
plt.savefig("output/digitSingleOne.png")
plt.show()


#find optimal number of clusters
from sklearn.cluster import KMeans
k = range(1,20)
WSSE = []

for i in k:
    model = KMeans(n_clusters = i, n_init='auto')
    model.fit_predict(X)
    WSSE.append(model.inertia_)

plt.plot(k,WSSE)
plt.xlabel('K')
plt.ylabel('Sum of Squared Errors')
plt.savefig("output/digitsScree.png")
plt.show()




#learn classifier
from sklearn.cluster import KMeans

clusters = 10
kmeans = KMeans(n_clusters=clusters, n_init='auto')
kmeans.fit(X)
Z = kmeans.predict(X)


#print selected class
##get images of numbers form given class
clusterIndex = 2
#get images with specific number of cluster
imageIndex = np.where(Z==clusterIndex)[0]
#plt.figure(figsize=(10,10))
for k in range(0, imageIndex.shape[0]):
    image = X[imageIndex[k],:]
    image = image.reshape(8, 8)
    plt.imshow(image, cmap='gray')
    plt.savefig("output/digits/{}_{}.png".format(clusterIndex, k))
    plt.show()