import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
'''
In this example, each sample has 2 features and we would like to cluster them into 3 classes.
'''
plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
print(X.shape)
print(y[:100])
# correct number of clusters:
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("3 Clusters")

y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Two Clusters")


y_pred = KMeans(n_clusters=4, random_state=random_state).fit_predict(X)

plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Four Clusters")


y_pred = KMeans(n_clusters=5,
                random_state=random_state).fit_predict(X)

plt.subplot(224)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Five Clusters")

plt.show()