'''' Interfacing with R '''
import numpy as np
#next two lines are necessary in order to be able to use numpy array and matrix in R functions
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import pandas as pd
from  rpy2 import  rinterface #is needed for logical NA

from rpy2.robjects import r
from rpy2.robjects.packages import importr
d=pd.read_csv('/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/UnionGExpressions.txt',
              sep='\t')
ylabs=list(pd.read_csv('/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/SampleID.txt'
                  ,sep='\t'))
print((d.shape)

d1=d[d.columns[2:]].as_matrix()

genes=list((d[d.columns[1]]))
genes=list(set(genes))
#############
#here I want to combine probes hitting the same gene and just use their median!
d2=np.zeros((111,120))
for j in range(len(genes)):
    inds=[i for i,x in enumerate(genes) if x==genes[j]] # we use enumerate since we want the position number in the list!!
    tmp=d1[inds,]
    tmp1=np.median(tmp,axis=0)
    d2[j,]=tmp1

d2 = d2.transpose()

print((d2.shape)
#normalizing the columns :
def normalize(d):
    for i in range(d.shape[1]):
        d[:,i]=(d[:,i]-np.mean(d[:,i]))/np.std(d[:,i])
    return d
d3=normalize(d2)
print(d3.shape
stats=importr('stats')
gplots=importr('gplots')

grdevices=importr('grDevices')
graphics=importr('graphics')

var1=['red']*40;var2=['green']*40;var3=['blue']*40;
var=var1+var2+var3
print(len(var)
print(var
#########################
########heatmap1

grdevices.png(file='/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/R_int_heatmap1.png',
              width=6500
              , height=6500)
graphics.par(mar=np.array([2,3,2,2]))
graphics.par(oma=np.array([4,0.1,0.1,6]))
stats.heatmap(d3,col=gplots.greenred(100),labRow=ylabs,labCol=genes, Colv=rinterface.NA_Logical,
              RowSideColors=np.array(var),symm=False,scale='none',cexRow=6,cexCol=6)
graphics.par(lend=1)
graphics.legend('topright', legend=np.array(['NTC', 'MSC', 'SMK']), col=np.array(['green', 'red', 'blue']), lty=1,
                lwd=12
                , inset=0, cex=6)
grdevices.dev_off()

###########################################

grdevices.png(file='/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/R_int_heatmap2.png',
              width=1500
              , height=1200)

gplots.heatmap_2(d3, key=False, symkey=False, dendrogram='row', col=gplots.greenred(100),
                 RowSideColors=np.array(var))
graphics.legend('topright',legend=np.array(['NTC','MSC','SMK']),col=np.array(['green','red','blue']),lty=1,lwd=12
                ,inset=0,cex=1.6)
grdevices.dev_off()
#gplots.heatmap_2(d3, key=False, symkey=False, dendrogram='row', col=gplots.greenred(100),
#                 RowSideColors=np.array(var),
 #                distfun=lambda x: stats.dist(x),
  #               hclustfun=lambda x: stats.hclust(x, method='ward.D2'))