import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
d = np.random.rand(500, 40)
print(d.shape)
################################################################
# note as a first step data needs to be standardizd.
from sklearn.preprocessing import StandardScaler
d3 = StandardScaler().fit_transform(d)
print('shape of standardized data')
print(d3.shape)
################################################################
# covariance matrix:
# The classic approach to PCA is to perform the eigen decomposition on the covariance matrix which is a d*d matrix
# representing covariance between features
# shape of data: features * samples
print('-'*20)
print('shape of transposed matrix for SVD: ', np.shape(d3.T))
# transpose d3 to be gene*sample; observations must be arranged in columns! and variables in rows!
cov_mat = np.cov(d3.T)
print('portion of cov matrix:')
print(cov_mat[1:5, 1:5])
print(cov_mat.shape)
##############################
# Next, we perform an eigen decomposition on the covariance matrix:
# columns of eig_vec contain the eigen vectors!
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
print(eig_vecs.shape)
###############################
# singular value decomposition:
# While the eigen decomposition of the covariance or correlation matrix may be more intuitive, most PCA implementations
# perform a Singular Vector Decomposition (SVD) to improve the computational efficiency.
print('shape of transposed matrix for SVD: ', np.shape(d3.T))
u, s, v = np.linalg.svd(d3.T)
print('DIMS: u: %s,  s: %s, v: %s' % (u.shape, s.shape, v.shape))
##########################
# Make a list of (eigenvalue, eigenvector) tuples
argsort = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[argsort]
eig_vecs = eig_vecs[:, argsort]

# Sort the (eigenvalue, eigenvector) tuples from high eigen value to to low

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_vals:
    print(i)
######################
# plotting the explained variance;:
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('Commulative sum of explained varianvce: ', cum_var_exp)
sns.set()
plt.figure(figsize=(6, 4))
plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
#####################################
# projection Matrix:
# it's about time to get to the really interesting part: The construction of the projection matrix
# that will be used to transform data onto the new feature subspace. Although, the name
# projection matrix has a nice ring to it, it is basically just a matrix of our concatenated top k eigenvectors.
# lets make the projection matrix from the first 3 components
matrix_w = eig_vecs[:, :3]

print(matrix_w.shape)
#####################################
# Projection Onto the New Feature Space
dprojected = np.matmul(d3, matrix_w)
print(d3.shape, dprojected.shape)
#####################
# plotting the data in the  new space:
ydict = {1: "MSC", 2: 'NTC', 3: 'SMK'}
y = np.array([1]*200+[2]*100+[3]*200)
sns.set()
plt.figure(figsize=(6, 4))
for lab, col in zip((1, 2), ('blue', 'red')):
    plt.scatter(dprojected[y == lab, 0],
                dprojected[y == lab, 1],
                label=ydict.get(lab), c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
#########################################################
# Shortcut - PCA in scikit-learn
from sklearn.decomposition import PCA
sklearn_pca = PCA(n_components=3)
data_projected_sklearn = sklearn_pca.fit_transform(d3)  # d3 is the scaled data!
print(data_projected_sklearn.shape)
# with plt.style.context('seaborn-whitegrid'):
plt.figure(figsize=(6, 4))
for lab, col in zip((1, 2), ('blue', 'red')):
        plt.scatter(data_projected_sklearn[y == lab, 0],
                    data_projected_sklearn[y == lab, 1],
                    label=ydict.get(lab),
                    c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
