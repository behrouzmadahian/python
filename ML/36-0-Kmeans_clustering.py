import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from matplotlib import pyplot as plt
import importlib
mpl_toolkits =importlib.import_module('mpl_toolkits')
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)
'''
There are 3 kinds of flowers Setosa labeled:0,'Versicolour' labeled 1, and 'Virginica' labeled 2
for each flower we have 4 attributes. 
The goal is to see if we can put these flowers into 3 clusters b y only looking at these attributes
'''
iris = datasets.load_iris()
X = iris.data
print(X.shape)
y = iris.target
print('target labels: ')
print(y)
estimator=KMeans(n_clusters=3)

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
labels=estimator.fit_predict(X)
#labels = estimator.labels_
print('predicted labels: ',labels)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],c=labels.astype(np.float),s=80,marker='o')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('3 clusters')

# Plot the ground truth
fig = plt.figure(2, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',fontsize=20)


# Reorder the labels to have colors matching the cluster results
colors_dict={0:1,1:0,2:2}
target_colors=[colors_dict[i] for i in y]
print(target_colors)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=target_colors, s=80,marker='p')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
plt.show()
