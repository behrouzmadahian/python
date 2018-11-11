import pandas as pd
from statsmodels.formula.api import ols
from matplotlib import pyplot as plt
# LINEAR REGRESSION USING CATEGORICAL VARIABLES
from patsy.contrasts import Treatment
d = pd.read_csv('linearRegression.csv', sep=',')
print(d.shape)
print(d.head(10))
###########################################
# this part is only for our observation is not needed for modeling:
# treatment contrasts: k categories are coded with k-1 levels!
levels = [1, 2, 3, 4]
contrast = Treatment(reference=1).code_without_intercept(levels)  # reference=0: use the first level as reference.
print(contrast.matrix)
print(contrast.matrix[d.race-1, :][0:20])  # it starts the levels from zero! the reason for subtractions!
############################################
# Fitting the model:
# We make treatment contrast for race!!

model = ols("write~ + C(race, Treatment) + read + math + science", data=d)
rs = model.fit()
print(rs.summary())
print('model predictions: ')
# print rs.predict(d)
plt.plot(d.write, rs.predict(d), 'ro')
plt.plot(d.write, d.write, 'r-', color='blue')
plt.xlabel('Prediction')
plt.ylabel("Write")

plt.show()