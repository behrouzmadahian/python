#matplotlib: the single most used python package for 2d graphics
#Iphython and pylab mode:
#Iphython:
#is an enhanced interacctive python shell that has lots of interesting
#features including named inputs and outputs, access to shell commands
#improved debugging and many more.
#when we start it with the command argument  -pylab
#it allows for interactive matplotlib sessions that have matlab/mathematica-like
#functionality
###############################################################################
#plotting:
#to get this work, you need to install packages: dateutil, pyparsing, six and matplotlib.
#install pip then from command line in the folder the packages are located do:
#pip install nameOfpackage
#pylab:
#provides a procedural interface to the matplotlib object oriented plotting
#library
#########################################################
#simple plot:

import pylab
import numpy as np
x1=np.random.rand(20)*20
print(x1)
x=np.array(range(20))
y=np.array(range(20,40))
#create a new figure of size 8*6 points with 80 dots per inch!
pylab.figure(figsize=(8,6),dpi=80)
pylab.plot(x,y,color='green',linewidth=2,linestyle='-',label='First Plot')
pylab.plot(x1,y,'bo',label='scatter Plot')
pylab.ylabel('Y')
pylab.xlabel('X')
pylab.xticks([0,10,30],[r'$+0$',r'$\pi$',r'3$\pi$'])
#legend
pylab.legend(loc='upper right')
#saving the plot:
pylab.savefig('firstPlot.png',dpi=72)
pylab.show()
######################################################################
#sub plots:
def f(t):
    return(np.exp(-t)*np.cos(2*np.pi*t))
t1=np.arange(0,5,0.1)
print( t1)
t2=np.arange(0,5,0.2)
#
pylab.figure(1)
#211: two rows and 1 column
pylab.subplot(211)
pylab.plot(t1,f(t1),'bo',t2,f(t2),'k')
pylab.subplot(212)
pylab.plot(t2,t2*np.cos(2*np.pi*t2),'r--')
pylab.show()
###################
#histogramming:
mu, sigma=100,15
print( np.random.randn(100))
x=mu+sigma*np.random.randn(10000)#normal with mean=100 and SD=15
print( x[1:10])
n,bins,patches=pylab.hist(x,50,normed=1,facecolor='g',alpha=0.75)
pylab.xlabel('Smarts')
pylab.ylabel('Probability')
pylab.title('Histogram of IQ')
pylab.text(60,0.025,r'$\mu=100, \ \sigma=15$')
pylab.axis([40,160,0,0.03])
pylab.grid(True)
import scipy
#i want to add the density curve as well:
import scipy.stats as ss

x=np.linspace(40,160)
print( '-----------------------')
print( x)
rv=ss.norm(mu,sigma)
h=pylab.plot(x,rv.pdf(x),color='red')
pylab.show()

#


























































