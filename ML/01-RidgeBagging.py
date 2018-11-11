import numpy as np, pandas as pd
import os
import dataProcessing
from sklearn.linear_model import RidgeCV,Ridge
from sklearn.ensemble import BaggingRegressor

np.random.seed(123)

# plt.plot(data['close_geo'])
# plt.axvline(4365)
# plt.text(2000,1800,'Train period')
# plt.text(5100,1800,'Test period')
# plt.title('SQ geometrically rolled')
#plt.show()

def data_gen(data, lookback, look_ahead, skip=0,lookbackMethod=1, every=0,scale=False):
    if lookbackMethod==2:
        lookback = 105
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrixWeiredModoFeat(data, lookback=lookback,
                                                              look_ahead=look_ahead,
                                                              train_inds=(564, 4070),
                                                              test_inds=(4070, 6489),
                                                              every=every,scale=scale)
    else:
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrix(data, skip=skip, lookback=lookback,
                                                look_ahead=look_ahead,
                                                train_inds=(564, 4070),
                                                test_inds=(4070, 6489),
                                                every=every,scale=scale)

    return trainX, trainY, train_prediction_start, testX, testY, test_prediction_start

def baggingMyRidge(trainX,trainY, train_prediction_start,testX,testY,test_prediction_start,look_ahead,
                   bag_size=47,Nestimators=50,samp_size=0.95,
                   bagModels_times=50,lookbackMethod=1,every=0,market=1):

    #use bagsize of estimators to do prediction, repeat 50 times
    alphas = np.linspace(0, 20, 1000)
    model_CV = RidgeCV(cv=10, alphas=alphas,fit_intercept=True,normalize=False).fit(trainX, trainY)
    print('Best Alpha parameter found by 10 fold CV: ', model_CV.alpha_)

    model = BaggingRegressor(Ridge(alpha=model_CV.alpha_,normalize=False,fit_intercept=True),
                             n_estimators=Nestimators,
                             max_samples=samp_size, bootstrap=False, random_state=None,n_jobs=-1)
    model=model.fit(trainX,trainY)

    colnamesRaw = ['dtStart']
    cln = [i for i in range(1, Nestimators * 2 + 3, 1)]
    colnamesRaw.extend(cln)
    colnamesBagged = ['dtStart']
    cln1 = [i for i in range(1, bagModels_times * 2 + 3, 1)]
    colnamesBagged.extend(cln1)

    #(date, trainY, true_lab, pred_labs...)
    trainRs = np.zeros((trainX.shape[0], bagModels_times * 2 + 3))
    trainRs_raw = np.zeros((trainX.shape[0], Nestimators * 2 + 3))
    trainRs[:, 0] = train_prediction_start
    trainRs_raw[:, 0] = train_prediction_start
    trainRs[:, 1] = trainY
    trainRs_raw[:, 1] = trainY
    trainRs[:, 2] = [1 if trainY[i] > 0 else 0 for i in range(len(trainY))]
    trainRs_raw[:, 2] = [1 if trainY[i] > 0 else 0 for i in range(len(trainY))]
    #
    testRs = np.zeros((testX.shape[0], bagModels_times * 2 + 3))
    testRs_raw = np.zeros((testX.shape[0], Nestimators * 2 + 3))
    testRs[:, 0] = test_prediction_start
    testRs_raw[:, 0] = test_prediction_start
    testRs[:, 1] = testY
    testRs_raw[:, 1] = testY
    testRs[:, 2] = [1 if testY[i] > 0 else 0 for i in range(len(testY))]
    testRs_raw[:, 2] = [1 if testY[i] > 0 else 0 for i in range(len(testY))]
    for i in range(Nestimators):
        trainRs_raw[:, i + 3] = model.estimators_[i].predict(trainX)

        testRs_raw[:, i + 3] = model.estimators_[i].predict(testX)

        trainRs_raw[:, i + Nestimators + 3] = \
            [1 if trainRs_raw[j, i + 3] > 0 else 0 for j in range(len(trainRs_raw[:, i + 3]))]
        testRs_raw[:, i + Nestimators + 3] = \
            [1 if testRs_raw[j, i + 3] > 0 else 0 for j in range(len(testRs_raw[:, i + 3]))]

    # aggregating results!
    model_inds = [j for j in range(3, Nestimators + 3)]
    # print(model_inds)
    for i in range(bagModels_times):
        index_modelstoUse = np.random.choice(model_inds, bag_size, replace=False)
        tmp_train = trainRs_raw[:, index_modelstoUse]
        tmp_test = testRs_raw[:, index_modelstoUse]
        trainRs[:, i + 3] = np.sum(tmp_train, axis=1)
        testRs[:, i + 3] = np.sum(tmp_test, axis=1)

        trainRs[:, i + bagModels_times + 3] = \
            [1 if trainRs[j, i + 3] > 0 else 0 for j in range(len(trainRs[:, i + 3]))]
        testRs[:, i + bagModels_times + 3] = \
            [1 if testRs[j, i + 3] > 0 else 0 for j in range(len(testRs[:, i + 3]))]

    trainRs = pd.DataFrame(trainRs, columns=colnamesBagged)
    trainRs_raw = pd.DataFrame(trainRs_raw, columns=colnamesRaw)

    testRs = pd.DataFrame(testRs, columns=colnamesBagged)
    testRs_raw = pd.DataFrame(testRs_raw, columns=colnamesRaw)

    if every==0:
        trainRs.to_csv('%d_%d_%d_Ridge_train.csv' % (market, lookbackMethod, look_ahead), index=False)
        trainRs_raw.to_csv('%d_%d_%d_Ridge_train_Raw.csv' % (market, lookbackMethod, look_ahead), index=False)
        testRs.to_csv('%d_%d_%d_Ridge_test.csv' % (market, lookbackMethod, look_ahead), index=False)
        testRs_raw.to_csv('%d_%d_%d_Ridge_test_Raw.csv' % (market, lookbackMethod, look_ahead), index=False)
    else:

        trainRs.to_csv('%d_%d_%d_every_%d_Ridge_train.csv' %
                       (market,lookbackMethod,look_ahead, every), index=False)
        trainRs_raw.to_csv('%d_%d_%d_every_%d_Ridge_train_Raw.csv' %
                           (market, lookbackMethod, look_ahead, every), index=False)
        testRs.to_csv('%d_%d_%d_every_%d_Ridge_test.csv' %
                      (market, lookbackMethod, look_ahead, every), index=False)
        testRs_raw.to_csv('%d_%d_%d_every_%d_Ridge_test_Raw.csv' %
                          (market, lookbackMethod, look_ahead, every), index=False)
####
def baggingMyRidgeForModo(trainX,trainY, train_prediction_start,testX,testY,
                          test_prediction_start,look_ahead,
                   Nestimators=50,samp_size=0.95,feat_method=1,every=5):
    alphas = np.linspace(0, 2, 1000)
    model_CV = RidgeCV(cv=10, alphas=alphas, fit_intercept=True, normalize=False).fit(trainX, trainY)
    print('Best Alpha parameter found by 20 fold CV: ', model_CV.alpha_)

    model = BaggingRegressor(Ridge(alpha=model_CV.alpha_, normalize=False, fit_intercept=True),
                             n_estimators=Nestimators,
                             max_samples=samp_size, bootstrap=False, random_state=None,n_jobs=-1)
    model = model.fit(trainX, trainY)
    colnames=['dtStart','TrueY']
    cln=['pred_%d'%i for i in range(1,Nestimators+1,1)]
    colnames.extend(cln)
    #(date, trainY, true_lab, pred_labs...)
    trainRs_raw=np.zeros((trainX.shape[0],Nestimators+2))
    trainRs_raw[:,0]=train_prediction_start
    trainRs_raw[:, 1] =trainY

    testRs_raw = np.zeros((testX.shape[0], Nestimators+2))
    testRs_raw[:, 0] =test_prediction_start
    testRs_raw[:, 1] =testY
    for i  in range(Nestimators):
        trainRs_raw[:, i + 2] = model.estimators_[i].predict(trainX)
        testRs_raw[:, i + 2] = model.estimators_[i].predict(testX)

    trainRs_raw = pd.DataFrame(trainRs_raw,columns=colnames)
    testRs_raw = pd.DataFrame(testRs_raw,columns=colnames)

    if every>0:
        trainRs_raw.to_csv('1_%d_%d_every_%d_Lasso_train.csv'%(feat_method,look_ahead,every) , index=False)
        testRs_raw.to_csv('1_%d_%d_every_%d_Lasso_test.csv'%(feat_method,look_ahead,every),index=False)
    else:
        trainRs_raw.to_csv('1_%d_%d_Lasso_train.csv' % (feat_method, look_ahead), index=False)
        testRs_raw.to_csv('1_%d_%d_Lasso_test.csv' % (feat_method, look_ahead), index=False)


###
datadir='C:/behrouz/Projects/DailyModels_new/Ridge/' \
        'dailyModelsRidgeNewData_SQ/DailyDataFeaturesAndResponsesSQ2August2017_1.csv'
data = pd.read_csv(datadir)

#feat_methods:
#1: last 30 consecutive, 2: the staggered method,3: last30skip1, 4: last30skip2;
lookback=30
lookaheadrange=10
directory_tosave='C:/behrouz/Projects/DailyModels_new/Ridge/dailyModelsRidgeNewData_SQ/tmp_scaled1'
os.chdir(directory_tosave)
if __name__=='__main__':
    every=[0,5,6]
    skip_method={0:1,1:3,2:4}
    for skip in range(3):
        for eve in every:
            print('lookback method %d, every %d'% (skip_method[skip],eve))

            for lookahead in range(1, lookaheadrange + 1, 1):
                print('lookahead: ', lookahead)
                trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
                    data_gen(data, lookback, lookahead,
                             lookbackMethod=skip_method[skip], skip=skip, every=eve,scale=False)

                baggingMyRidge(trainX, trainY, train_prediction_start,
                               testX, testY, test_prediction_start, lookahead, bag_size=47,
                               Nestimators=50, samp_size=0.95,
                               bagModels_times=50, lookbackMethod=skip_method[skip], every=eve)
    for eve in every:
        print('lookback method %d, every %d' % (2, eve))

        for lookahead in range(1, lookaheadrange + 1, 1):
                print('lookahead: ', lookahead)
                trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
                    data_gen(data, lookback, lookahead, lookbackMethod=2, skip=0, every=eve,scale=False)
                # print(trainX.shape)
                baggingMyRidge(trainX, trainY, train_prediction_start,
                                       testX, testY, test_prediction_start, lookahead, bag_size=47,
                                       Nestimators=50, samp_size=0.95,
                                       bagModels_times=50, lookbackMethod=2, every=eve)



    # baggingMyLassoForModo(trainX ,trainY, train_prediction_start,
    #         testX, testY, test_prediction_start,lookahead,Nestimators=50,samp_size=.95,feat_method=4,every=6)





