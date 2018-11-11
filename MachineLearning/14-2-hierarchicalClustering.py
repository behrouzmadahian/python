from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
# generate 3 clusters: a with 100 points, b with 50, c with 50:
np.random.seed(4711)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=100)
print(a.shape)
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=50)
c = np.random.multivariate_normal([0, 0], [[1, 0], [0, 2]], size=50)

X = np.concatenate((a, b),) #X is n*p
X = np.concatenate((X, c),)
print(X.shape)  # 150 samples with 2 dimensions
plt.scatter(X[:,0], X[:,1])
plt.show()
######
# generate the linkage matrix
Z = linkage(X, 'ward')
##########
# Another thing you can and should definitely do is check the Cophenetic Correlation Coefficient of your clustering
# with help of the cophenet() function.
# This (very very briefly) compares (correlates) the actual pairwise distances
# of all your samples to those implied by the hierarchical clustering. The closer the value is to 1, the better the
# clustering preserves the original distances
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
print('*'*20)
c, coph_dists = cophenet(Z, pdist(X))
print(c)
print(coph_dists.shape)
#########################
#plotting the dendogram:
# calculate full dendrogram
labs=list(range(1,201))
print(len(labs))
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    labels=labs,
    orientation='top', #plot the root at the top of the plot
    above_threshold_color='black',
    color_threshold=80 #decrease th threshold to get coloring on more clusters
    #increase it to get one uniform coloring!
)
plt.show()
plt.savefig('HierarchicalClustering.png')
print(0.7*max(Z[:,2])) #coloring threshold is set to this number!!
print(Z.shape)
print(Z[1:3,:])