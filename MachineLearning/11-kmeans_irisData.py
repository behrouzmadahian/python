import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
# np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X[:10])
print('size of data:', X.shape)
print(y[:10])
est = KMeans(n_clusters=3, init='k-means++').fit(X)
print('Labels of each point::\n')
print(est.labels_)
print('Total Sum of squares of distances of points from their center of cluster: ')
print(est.inertia_, '\n')

##########
# 3D plotting of the data:
fig = plt.figure(figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla() # clear current axes!!
labels = est.labels_
ydict = {0: 'setosa', 1: 'versicolor', 2: 'Virginica'}
with plt.style.context('seaborn-whitegrid'):

   for lab, col in zip((0,1, 2), ('blue', 'red','green')):
        ax.scatter(X[est.labels_==lab, 3], X[est.labels_==lab, 0], X[est.labels_==lab, 2], c=col,
                 label=ydict.get(lab))  # c: color

   ax.w_xaxis.set_ticklabels([])
   ax.w_yaxis.set_ticklabels([])
   ax.w_zaxis.set_ticklabels([])
   ax.set_xlabel('Petal width')
   ax.set_ylabel('Sepal length')
   ax.set_zlabel('Petal length')
   plt.legend(loc='upper right')

# Plot the ground truth
plt.show()

print(X[y == 1, 3].mean())
