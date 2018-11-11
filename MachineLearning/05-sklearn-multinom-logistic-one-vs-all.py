import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
'''
The model fits one-vs-rest model if multi_class='ovr' and uses the cross entropy loss if set to 'multinomial'
if using multinomial you have to change the solver to lbfgs or newton-cg
here the response has 3 levels in nominal scale
the model using sklearn library focuses on prediction!!!
we do cross validation as well!!
using sklearn we do not need to add a column of ones to the data! we can specify in the function
C: inverse of regularization strength!
'''

# data:
iris = datasets.load_iris()
x = iris.data
print(x[1:10, ], x.shape)
y = iris.target
print(y[1:10])
# dividing the data into train and test:
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=0)  # equal size splits!
logisticModel = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty='l2', C=10)
rs = logisticModel.fit(xTrain, yTrain)
print('Coefficients:')
print(rs.coef_)
yPred = rs.predict(xTest)  # here the prediction returns the labels!!!! and not probabilities!!
print('-'*100)
print(yPred)
yProbs = rs.predict_proba(xTest)
print('Predicted Pprobabilities:\n')
print(yProbs)
accu = [1 for i in range(len(yTest)) if yTest[i] == yPred[i]]
print('Test Accuracy: ', round(sum(accu)*1.0/len(yTest), 2))
# lets plot the data on the first two features
plt.scatter(x[:, 1], x[:, 2], c=rs.predict(x), alpha=0.5, marker='+')
plt.show()
########
#Cross validation:

kf_total = cross_val_score(logisticModel, xTrain, yTrain, cv=10)  # k fold
print(kf_total)
print('10-Fold Average accuracy:', round(np.mean(kf_total), 2), '\n')

