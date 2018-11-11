import numpy as np
import pandas as pd
import os
import dataProcessing
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


def data_gen(data, lookback, look_ahead, weired_feat=True, skip=0, every=0):
    if weired_feat:
        lookback = 105
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrixWeiredModoFeat(data,lookback=lookback, look_ahead=5,
                                       train_inds=(564,4070), test_inds=(4070, 6489),every=every)
    else:
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrix(data,skip=skip,lookback=lookback, look_ahead=look_ahead,
                         train_inds=(564,4070), test_inds=(4070, 6489),every=every)

    return trainX, trainY,train_prediction_start, testX, testY,test_prediction_start


def myRidge(trainX,trainY, train_prediction_start,testX,testY,test_prediction_start,look_ahead,
            nBest=5):
    linearReg = LinearRegression()
    # step: number of features to remove at each step
    rfecv = RFECV(estimator=linearReg, step=1, cv=10, scoring='neg_mean_squared_error')
    rfecv.fit(trainX, trainY)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # Selected (i.e., estimated best) best feature is assigned rank 1.
    print('Feature Ranking: I want to take top 5 regressors ')
    print(rfecv.ranking_)
    if rfecv.n_features_> nBest:
        nBest = rfecv.n_features_
    inds = [i for i in range(30) if rfecv.ranking_[i] <= nBest]
    print(inds)
    trainX = trainX[:, inds]
    testX = testX[:, inds]
    print(trainX.shape, testX.shape)
    # computing the best alpha for Ridge:
    alphas = np.linspace(0, 2, 100)
    model = RidgeCV(cv=10, alphas=alphas).fit(trainX, trainY)
    print('Best Alpha Chosen: ', model.alpha_)
    clf = Ridge(alpha=model.alpha_)
    clf = clf.fit(trainX, trainY)
    print('%'*200)

    train_preds = clf.predict(trainX)
    mse = mean_squared_error(trainY, train_preds)
    train_labs = [1 if trainY[i] > 0 else 0 for i in range(len(trainY))]
    train_predlabs = [1 if train_preds[i] > 0 else 0 for i in range(len(train_preds))]
    trainRs = pd.DataFrame(
        {'predicted_label': train_predlabs, 'True_label': train_labs, 'Y': trainY, 'Y_pred': train_preds,
         'dtStart': train_prediction_start  })
    trainRs.to_csv('train_SQ_results_la%d.csv'%look_ahead, index=False)

    test_preds = clf.predict(testX)
    mse = mean_squared_error(testY, test_preds)
    test_labs = [1 if testY[i] > 0 else 0 for i in range(len(testY))]
    test_predlabs = [1 if test_preds[i] > 0 else 0 for i in range(len(test_preds))]
    testRs = pd.DataFrame({'predicted_label': test_predlabs, 'True_label': test_labs, 'Y': testY, 'Y_pred': test_preds,
                           'dtStart': test_prediction_start})
    testRs.to_csv('test_SQ_results_la%d.csv'%look_ahead, index=False)
    return mse

os.chdir('C:/behrouz/Projects/DailyModels_new/Ridge/dailyModelsRidgeNewData_SQ/tmp')
lookaheadrange=21
lookback=30
datadir='C:/behrouz/Projects/DailyModels_new/Ridge/dailyModelsRidgeNewData_SQ/DailyDataFeaturesAndResponsesSQ2August2017_1.csv'
data = pd.read_csv(datadir)
print (data.shape)

for look_ahead in range(1,lookaheadrange):
    print('look ahead: ',look_ahead)
    trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
        data_gen(data, lookback, look_ahead, weired_feat=False, skip=0, every=0)

    myRidge(trainX, trainY, train_prediction_start, testX, testY,
            test_prediction_start, look_ahead, nBest=5)







