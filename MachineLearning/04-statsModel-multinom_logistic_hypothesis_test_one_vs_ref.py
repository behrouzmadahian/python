import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import MNLogit

'''
It performs each class versus a reference class fit
generates k-1 set of betas
good for hypothesis testing and calculating effect sizes.
Not great at prediction!!

Note IRis data has perfect separation problem. add some noise to it!!!
'''

iris = datasets.load_iris()
x = iris.data  # lets use the first two features!!
x = np.c_[np.ones(x.shape[0]), x]
print(x.shape)
print(x[:4])
y = iris.target
y[1:5] = 2
y[-5:] = 1
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=50)  # equal size splits!
print(xTest.shape)
logitModel = MNLogit(yTrain, xTrain)
rs = logitModel.fit(maxiter=200)
print(rs.params)
print(rs.summary())
yPred = rs.predict()

print(yPred, yPred.shape, '\n\n')


def probtolabelPred(yPred):
    print(yPred[:10])
    labs = np.argmax(yPred, axis=1)
    return labs


predLabs = probtolabelPred(yPred)
print(predLabs)
print(yTrain)

# lets calculate average accuracy!!!!
cnt = [1 for i in range(len(yTrain)) if yTrain[i] == predLabs[i]]
print("Train Accuracy: ", round(sum(cnt)*1.0/len(yTrain), 2))

predTest = rs.predict(xTest)
predTestLabs = probtolabelPred(predTest)
cnt = [1 for i in range(len(yTest)) if yTest[i] == predTestLabs[i]]
print("Test Accuracy: ", round(sum(cnt)*1.0/len(yTrain),2))
#########################################################

