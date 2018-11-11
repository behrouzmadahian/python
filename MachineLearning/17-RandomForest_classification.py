from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
iris = load_iris()
x = iris.data
# we turn y into factor as needed below!
y, _ = pd.factorize(iris.target)
print(x)
print(y)
# adding the response to data frame:
print(iris.target_names)
# dividing the data into train and test:
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=0) # equal size splits!
print('Xtest dimensions: ')
print(xTest.shape)
# n_jobs: for parallelizing accross CPU nodes!!
rf = RandomForestClassifier(n_estimators=10000, n_jobs=1)
# need to turn y into factor if classification!
# x must be samples*features in dimensions!
print(xTrain)

rf.fit(xTrain, yTrain)
predTrain = rf.predict(xTrain)
print(pd.crosstab(yTrain, predTrain, rownames=['Measured'], colnames=['Predicted']))

Train_accu = rf.score(xTrain, yTrain)
print(Train_accu)
predTest = rf.predict(xTest)
print(pd.crosstab(yTest, predTest, rownames=['Measured'], colnames=['Predicted']))
print(rf.score(xTest, yTest))