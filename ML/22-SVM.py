import numpy as np, pandas as pd
import os
import dataProcessing
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV,ShuffleSplit,RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
import scipy.stats
import pickle
# plt.plot(data['close_geo'])
# plt.axvline(4365)
# plt.text(2000,1800,'Train period')
# plt.text(5100,1800,'Test period')
# plt.title('SQ geometrically rolled')
#plt.show()

def data_gen(data,lookback,look_ahead,weired_feat=True,skip=0,every=0):
    if weired_feat:
        lookback=105
        trainX, trainY, train_weights,train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrixWeiredModoFeat(data,lookback=lookback, look_ahead=5,
                                       train_inds=(564,4070), test_inds=(4070, 6489),every=every)
    else:
        trainX, trainY, train_weights,train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrix(data,skip=skip,lookback=lookback, look_ahead=look_ahead,
                         train_inds=(564,4070), test_inds=(4070, 6489),every=every)

    return trainX, trainY,train_weights,train_prediction_start, testX, testY,test_prediction_start

def mySVM(trainX,trainY,train_weights, train_prediction_start,testX,testY,test_prediction_start,look_ahead):
    # randomized search for hyper parameters of SVM:
    cRange = scipy.stats.expon(scale=5)
    gammaRange = scipy.stats.expon(scale=0.1)
    parameter_dist = {'C': cRange, 'gamma': gammaRange}
    #parameter_dist = {'C': cRange}
    # clf = RandomizedSearchCV(estimator=svm.SVR(), param_distributions=parameter_dist,
    #                          n_iter=50, cv=10,n_jobs=-1, fit_params={'sample_weight': train_weights})

    clf = RandomizedSearchCV(estimator=svm.SVR(kernel='rbf'), param_distributions=parameter_dist,
                             n_iter=50, cv=10, n_jobs=-1)
    clf.fit(trainX, trainY)
    print('Best C:', clf.best_estimator_.C)
    print('Best Gamma:', clf.best_estimator_.gamma)
    svr = svm.SVR(kernel='rbf', gamma=clf.best_estimator_.gamma, C=clf.best_estimator_.C)
    svr = svr.fit(trainX, trainY)
    # pickling the SVM object to file:
    f = open('SVM_model_SQ', 'wb')
    pickle.dump(svr, f)
    train_preds = svr.predict(trainX)
    train_labs = [1 if trainY[i] > 0 else 0 for i in range(len(trainY))]
    train_predlabs = [1 if train_preds[i] > 0 else 0 for i in range(len(train_preds))]
    trainRs = pd.DataFrame(
        {'predicted_label': train_predlabs, 'True_label': train_labs, 'Y': trainY, 'Y_pred': train_preds,
         'dtStart': train_prediction_start})
    trainRs.to_csv('train_SQ_results_la%d.csv'%look_ahead, index=False)

    test_preds = svr.predict(testX)
    mse = mean_squared_error(testY, test_preds)
    test_labs = [1 if testY[i] > 0 else 0 for i in range(len(testY))]
    test_predlabs = [1 if test_preds[i] > 0 else 0 for i in range(len(test_preds))]
    testRs = pd.DataFrame({'predicted_label': test_predlabs, 'True_label': test_labs, 'Y': testY, 'Y_pred': test_preds,
                           'dtStart': test_prediction_start})
    testRs.to_csv('test_SQ_results_la%d.csv'%look_ahead, index=False)
    return mse

lookback=30
lookaheadrange=3
directory_tosave='C:/behrouz/Projects/dailyModelsSVMNewData_SQ/tmp'
os.chdir(directory_tosave)
datadir='C:/behrouz/Projects/dailyModelsSVMNewData_SQ/DailyDataFeaturesAndResponsesSQ2August2017_1.csv'
data = pd.read_csv(datadir)
print (data.shape)
if __name__=='__main__':
    for lookahead in range(1, lookaheadrange + 1):
        trainX, trainY,train_weights, train_prediction_start, testX, testY, test_prediction_start = \
            data_gen(data, lookback, lookahead, weired_feat=False, skip=0,every=0)
        print(trainX.shape)

        mySVM(trainX, trainY, train_weights,train_prediction_start,
              testX, testY, test_prediction_start, lookahead)










