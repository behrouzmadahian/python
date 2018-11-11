import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import somoclu
import time
#SGA females:
dSGA = pd.read_csv('/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/SGAFemale.txt',
                sep='\t')
print dSGA.shape
colnames=list(dSGA.columns.values)
rownames=list(dSGA[dSGA.columns[0]])
print rownames[0:10]
print colnames
dSGA=dSGA[dSGA.columns[1:]].as_matrix()
print dSGA.shape
#getting only the weight columns:
dw=dSGA[:,5:10]
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
plt.ylim([-6,10])
plt.text(4,-1.5,'Mean curve')
print MeanC
plt.show()
######################
#SOM:
######################
#PLANAR MAP:
ncl=20

#initialization
som=somoclu.Somoclu(10,10,data=dw,maptype="planar",gridtype="rectangular",neighborhood='gaussian',
                    initialization='random')
som.train()
####plot the component planes of the trained codebook of the ESOM
som.view_component_planes()
#############################################################
#we can plot the U-matrix, together with the best matchin units for each data point. We color code
#the units with the classes of the data points and also add the labels of the data points
som.view_umatrix(bestmatches=True,labels=rownames)
#############################################################
#zooming in to a region of interest:<upper right corner here
som.view_umatrix(bestmatches=True,labels=rownames,zoom=((8,10),(8,10)))
##########################
#repeating with Hexagonal topology, and hexagonal neurons
print 'Hexagonal GRID......'
som = somoclu.Somoclu(10, 10, data=dw, maptype="toroid", gridtype="hexagonal", neighborhood='gaussian',
                      initialization='random')
som.train()
####plot the component planes of the trained codebook of the ESOM
som.view_component_planes()
#############################################################
# we can plot the U-matrix, together with the best matchin units for each data point. We color code
# the units with the classes of the data points and also add the labels of the data points
som.view_umatrix(bestmatches=True, labels=rownames)
#############################################################
# zooming in to a region of interest:<upper right corner here
som.view_umatrix(bestmatches=True, labels=rownames, zoom=((8, 10), (8, 10)))