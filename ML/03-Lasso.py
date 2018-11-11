import numpy as np, pandas as pd
import os
import dataProcessing
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV, lasso_path, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict


def data_gen(data, lookback, look_ahead, weired_feat=True, skip=0):
    if weired_feat:
        lookback=105
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrixWeiredModoFeat(data,lookback=lookback, look_ahead=5,
                                       train_inds=(564,4070), test_inds=(4070, 6489))
    else:
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrix(data,skip=skip,lookback=lookback, look_ahead=look_ahead,
                         train_inds=(564,4070), test_inds=(4070, 6489))

    return trainX, trainY,train_prediction_start, testX, testY,test_prediction_start


def myLasso(trainX,
            trainY,
            train_prediction_start,
            testX,
            testY,
            test_prediction_start,
            look_ahead):
    # computing the best alpha for lasso:
    model = LassoCV(cv=20).fit(trainX, trainY)
    print('Best Alpha Chosen: ', model.alpha_)
    clf = Lasso(alpha=model.alpha_)
    clf = clf.fit(trainX, trainY)
    print('%'*200)
    print(clf.coef_)
    print(clf.score(trainX,trainY))
##################################
    clf1 = LinearRegression()
    clf1 = clf1.fit(trainX, trainY)
    print('%' * 200)
    print(clf1.coef_)
    print(clf1.score(trainX, trainY))
    #######################

    train_preds = clf.predict(trainX)
    mse = mean_squared_error(trainY, train_preds)
    train_labs = [1 if trainY[i] > 0 else 0 for i in range(len(trainY))]
    train_predlabs = [1 if train_preds[i] > 0 else 0 for i in range(len(train_preds))]
    trainRs = pd.DataFrame(
        {'predicted_label': train_predlabs, 'True_label': train_labs, 'Y': trainY, 'Y_pred': train_preds,
         'dtStart': train_prediction_start,
         })
    trainRs.to_csv('train_SQ_results_la%d.csv'%look_ahead, index=False)

    test_preds = clf.predict(testX)
    mse = mean_squared_error(testY, test_preds)
    test_labs = [1 if testY[i] > 0 else 0 for i in range(len(testY))]
    test_predlabs = [1 if test_preds[i] > 0 else 0 for i in range(len(test_preds))]
    testRs = pd.DataFrame({'predicted_label': test_predlabs, 'True_label': test_labs, 'Y': testY, 'Y_pred': test_preds,
                           'dtStart': test_prediction_start,
                           })
    testRs.to_csv('test_SQ_results_la%d.csv'%look_ahead, index=False)
    return mse

lookahead=1
lookback=30
lookaheadrange=10
directory_tosave='C:/behrouz/Projects/dailyModelsLassoNewData/tmp'
os.chdir(directory_tosave)
datadir='C:/behrouz/Projects/dailyModelsLassoNewData/DailyDataFeaturesAndResponsesSQ2August2017_1.csv'
data = pd.read_csv(datadir)
print (data.shape)
for lookahead in range(1,lookaheadrange+1):
    trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
        data_gen(data, lookback, lookahead, weired_feat=False, skip=0)
    print (trainX.shape)

    myLasso(trainX, trainY, train_prediction_start, testX, testY, test_prediction_start, lookahead)








