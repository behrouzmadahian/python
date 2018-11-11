from sklearn.datasets import load_iris
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
iris = load_iris()
x = iris.data[:, 0:3]
# we turn y into factor as needed below!
y = iris.data[:, 3]
print(x)
print(y)
# dividing the data into train and test:
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=0)  # equal size splits!
print ('Xtest dimensions: ')
print(xTest.shape)
# n_jobs: for parallelizing accross CPU nodes!!
rf = RandomForestRegressor(n_estimators=10000, n_jobs=2)
# need to turn y into factor if classification!
# x must be samples*features in dimensions!
print(xTrain)

rf.fit(xTrain, yTrain)
predTrain = rf.predict(xTrain)
print(predTrain)
# print(yTrain_f)
# coefficient of determination of the regression!
Train_R2 = rf.score(xTrain,yTrain)
print(Train_R2)
predTest = rf.predict(xTest)
print(predTest)
plt.subplot(121)
plt.plot(yTrain, predTrain, 'ro')
plt.plot(np.linspace(0, 2.5, 100), np.linspace(0, 2.5, 100), 'k-', color='blue')
plt.ylabel('Y value')
plt.xlabel('Predicted Value')
plt.title('Training Sample')
plt.text(1, 2.2, np.round(rf.score(xTrain, yTrain), 3))
plt.subplot(122)
plt.plot(yTest, predTest, 'ro')
plt.plot(np.linspace(0,2.5, 100), np.linspace(0, 2.5, 100), 'k-', color='blue')
plt.text(1, 2.2, np.round(rf.score(xTest, yTest), 3))

plt.ylabel('Y value')
plt.xlabel('Predicted Value')
plt.title('Test Sample')
plt.show()
print()

print('R2 based on test data:')
print(rf.score(xTest, yTest))
