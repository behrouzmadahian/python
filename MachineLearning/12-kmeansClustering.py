import numpy as np
from sklearn.cluster import KMeans
'''' Interfacing with R '''
import pandas as pd
d = pd.read_csv('/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/UnionGExpressions.txt',
              sep='\t')
ylabs=list(pd.read_csv('/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/SampleID.txt'
                  ,sep='\t'))
print(d.shape)

print(d.columns)
d1=d[d.columns[2:]].as_matrix()
print(d1.shape)
genes=list((d[d.columns[1]]))
genes=list(set(genes))#getting unique genes!
#############
#here I want to combine probes hitting the same gene and just use their median!
d2=np.zeros((111,120))
for j in range(len(genes)):
    inds=[i for i,x in enumerate(genes) if x==genes[j]] # we use enumerate since we want the position number in the list!!
    tmp=d1[inds,]
    tmp1=np.median(tmp,axis=0) #axis=0: row median!!
    d2[j,]=tmp1

d2 = d2.transpose()

print(d2.shape)

#######
#n_init:
#Number of time the k-means algorithm will be run with different centroid seeds. The final results will be
# the best output of n_init consecutive runs in terms of inertia.
km=KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
             precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
est=km.fit(d2)
print(est.labels_)

kmfit=km.fit_predict(d2) #d2 must be sample*feature
print(kmfit,len(kmfit))


