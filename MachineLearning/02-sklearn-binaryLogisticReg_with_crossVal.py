import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import statsmodels as sm
'''
C: inverse of regularization strength'''
mydata = sm.datasets.spector.load()  # load data from Spector and Mazzeo
x = mydata.exog
print(x[:10])
y = mydata.endog
print(y)
kf_total = model_selection.KFold(n_splits=10, shuffle=False)
lr = LogisticRegression(fit_intercept=True, penalty='l2', C=5)
accu = cross_val_score(lr, x, y, cv=kf_total, scoring='accuracy')
print('Average accuracy:', np.mean(accu), '\n', accu)

