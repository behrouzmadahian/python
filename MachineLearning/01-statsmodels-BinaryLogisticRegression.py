import numpy as np
import statsmodels as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
mydata = sm.datasets.spector.load()  # load data from Spector and Mazzeo
x = np.append(np.ones((mydata.exog.shape[0], 1)), mydata.exog, axis=1)
print(x[1:10, ])
y = mydata.endog
print(x.shape)
logit_mod = smf.Logit(y, x)
rs = logit_mod.fit()
print(rs.summary())
xtest = [1, 6, 30, 0]
print('Testing:', rs.predict(xtest))
probs = rs.predict()
print(probs)


def binaryModelAnalysis(y, probs):
    '''
    In this function:
    :param y: binary values of responses
    :param probs: probabilities of each sample being 1. we threshold on 0.5
    :return: Accuracy, Sensitivity, and Specificity
    '''
    yPred = [1 if p >= 0.5 else 0 for p in probs]
    # calculating total accuracy:
    s = [1 for i in range(len(y)) if y[i] == yPred[i]]
    accu = round(np.sum(s) * 1.0 / len(y), 2)
    # Sensitivity:
    positives = [i for i, j in enumerate(y) if y[i] == 1]
    s = [1 for i in range(len(positives)) if yPred[positives[i]] == 1]
    sens = round(sum(s) * 1.0 / len(positives), 2)
    negatives = [i for i, j in enumerate(y) if y[i] == 0]
    s = [1 for i in range(len(negatives)) if yPred[negatives[i]] == 0]
    spec = round(sum(s) * 1.0 / len(negatives), 2)
    return "Accuracy: ", accu, 'Sensitivity: ', sens, 'Specificity: ', spec


print(binaryModelAnalysis(y, probs))
##########################
from sklearn.metrics import roc_curve, auc
# compute micro-average ROC curve and ROC area:
# y: binary [0,1] or [-1,1]
# ypred: probabilities of class=1 or confidence values
# generates array of tpr and fpr at different thresholds
fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)
print(roc_auc)

# fpr['micro'],tpr['micro'],_=roc_curve(y,yPred)
print(fpr)
print(tpr)
print(roc_auc)
# plotting:
plt.figure()
plt.plot(fpr, tpr, label='ROC curve(area= %0.2f)'%roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.05)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristics Curve')
plt.legend(loc='lower right')
plt.show()

