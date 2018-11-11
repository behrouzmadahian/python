import numpy as np
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2 import robjects
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
from rpy2.robjects import r
grdevices=importr('grDevices')
#accessing pi in R:
pi=r['pi']
print pi
stats=importr('stats')
graphics=importr('graphics')
base=importr('base')
##
sum=r.sum
print sum(np.array([1,2,3,4]))
#######plot1
d=stats.rnorm(10000)

hist=graphics.hist
grdevices.png(file='/Users/behrouzmadahian/Dropbox/Research_and_Development/Python/DEVELOPMENTS/R_int_hist.png',width=512
              ,height=800)
hist(d,col='blue',ylab='Freg',xlab='rnorm',main='')
grdevices.dev_off()
#####plot2
lattice=importr('lattice')
xyplot=lattice.xyplot
x=range(10)
y=range(10,20)
r.plot(x,y)
raw_input()


