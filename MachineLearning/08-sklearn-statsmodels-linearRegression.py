import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
d = pd.read_csv('linearRegression.txt', sep='\t')
print(d)
print(d.shape)

x = d[d.columns[0:2]].values
y = d[d.columns[2]].values
print(x)
print(y)
reg = linear_model.LinearRegression()
rs = reg.fit(x, y)
trainPred = rs.predict(x)
plt.plot(trainPred, y, 'ro')
plt.plot(y, y, 'r-', color='blue')
plt.xlabel('Predictions')
plt.ylabel('Y')
plt.show()
####
# R square: coefficient of determination:
print('R^2: \n')
print(round(rs.score(x, y), 2))
# parameter estimates:

print('Intercept:  %.4f' % rs.intercept_)
print('beta1 \t  beta2:\n')
print(rs.coef_)
################################
# using statsmodels
import statsmodels.api as sm
x1 = sm.add_constant(x)
model = sm.OLS(y, x1)
rs = model.fit()
print("parameter Estimates:")
print(rs.params)
print("t statistics:")
print(rs.tvalues)
print("P values:")
print(rs.pvalues)
#######
print("F test for model fit:")
print(rs.f_test(np.identity(3)))
print("model degree of Freedom:")
print(rs.df_model)
print("reidual degree of freedom:")
print(rs.df_resid)
print(rs.summary())
###############################
