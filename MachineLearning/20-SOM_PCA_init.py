import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
import time
import sklearn.cluster as cluster

#SGA females:
dSGA = pd.read_csv('/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/SGAFemale.txt',
                sep='\t')
print dSGA.shape
coln=list(dSGA.columns.values)
rownames=list(dSGA[dSGA.columns[0]])
print rownames[0:10]
print coln
dSGA1=dSGA[dSGA.columns[1:]].as_matrix()
print dSGA1.shape
#getting only the weight columns:
dw=dSGA1[:,5:10]
print dw[0:5,:]
######
#lets normalize the SGA data to the population:
popFem=pd.read_csv('/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/popInfoFemale.txt',
                sep='\t',header=None).as_matrix()
print 'Population Parameters: '
print popFem
print
for i in range(5):
    dw[:,i]=(dw[:,i]-popFem[0,i])/popFem[1,i]
print dw[0:5,0:5]
print dw.shape
TIME=np.round_(np.array([1,123,366,1462,2558])/365.25,1)
print TIME
############
#plotting all the data!
for i in range(len(dw[:,1])):
    plt.plot(TIME,dw[i,:],'-',c='lightskyblue')
    plt.ylim([-6,10])

plt.axhline(y=0, xmin=TIME[0], xmax=TIME[4], hold=None,c='red')
plt.xticks(TIME)
plt.xlabel('AGE(Yr)')
plt.ylabel('Pop Scaled Weight')
#adding mean curve:
MeanC=np.mean(dw,axis=0)
plt.plot(TIME,MeanC,'r--',c='black',linewidth=3)
#plt.setp(p,linewidth='3')
plt.ylim([-6,10])
plt.text(4,-1.5,'Mean curve')
print MeanC
plt.show()
############################################
#SOM:
#we can ask to initialize the codebook with vectors from the subspace spanned by the first two eigen values of the
#correlation matrix
som=somoclu.Somoclu(10,10,data=dw,maptype='toroid',initialization='pca')
#scale0: initial learning rate
#scaleN: the learning rate of the final epoch
som.train(epochs=10,scale0=0.1,scaleN=0.005)
som.view_umatrix(bestmatches=True)
#even with PCA initialization, there is a slight difference between different runs of the model.
#this is due to the fact that each core gets different batches of the data and there is no control
#over the order of each batch
############################################
#Post processing the codebook with an arbitrary clustering algorithm that is included in scikit-learn
#the default is k-means with 8 clusters.
#after clustering, the labels for each node are abailable in the SOM object in the cluster class variable
#if we do not pass colors to the matrix viewing functions and clustering is alredy done, the plotting routines
#automatically color the best matchin units according to th clustering structure.
algorithm=cluster.KMeans(n_clusters=10)
som.cluster(algorithm=algorithm)
som.view_umatrix(bestmatches=True)
print 'Clusters of the nodes:'
print som.clusters
print "flattened clusters: "
mapClusts=som.clusters.flatten() #making a flat list
print mapClusts
#print mapClusts.shape
#######
#now we want to see which data point is assigned to which node on the map:
print som.bmus.shape
print som.bmus[0:10,]
themap=[ [i,j]for i in range(10)for j in range(10) ]
#print themap
#themap1=zip(mapClusts,themap)
#print themap1
#print themap1[1][1]
clusts=np.zeros(1110).reshape(1110,1)
print len(themap)
for j in range(len(som.bmus[:,0])):
    p=list(som.bmus[j])
    for i in range(len(themap)):
        if p==list(themap[i]):
            ind=i
            break
    clusts[j]=mapClusts[ind]

print 'CLUSTERS: '
print clusts[0:10]
##########
print rownames
print clusts.shape
print dw.shape
dw1=np.append(dSGA,clusts,1)
print dw1.shape
print dw1[0:2,]
#########
#writing the results to file:
print coln
coln.extend(['Clusters'])
print coln
np.savetxt('SOMClust.txt',dw1,fmt='%.3f',header='\t'.join(coln),delimiter='\t',newline='\n')
#######
#Pairs plotting!!!!
#