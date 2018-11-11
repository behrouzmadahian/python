import numpy as np, pandas as pd
import os
import dataProcessing
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV, lasso_path, Lasso,LinearRegression,Ridge,RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor

np.random.seed(123)

# plt.plot(data['close_geo'])
# plt.axvline(4365)
# plt.text(2000,1800,'Train period')
# plt.text(5100,1800,'Test period')
# plt.title('SQ geometrically rolled')
# plt.show()

def data_gen(data, lookback, look_ahead, skip=0,lookbackMethod=1,
             every=0,scale=False,feat_inds=[],scaler='MinMaxScaler'):
    if lookbackMethod==2:
        lookback = 105
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrixWeiredModoFeat(data, lookback=lookback,
                                                              look_ahead=look_ahead,
                                                              train_inds=(564, 4070),
                                                              test_inds=(4070, 6489),
                                                              every=every,scale=scale,
                                                              feat_inds=feat_inds,scaler=scaler)
    else:
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrix(data, skip=skip, lookback=lookback,
                                                look_ahead=look_ahead,
                                                train_inds=(564, 4070),
                                                test_inds=(4070, 6489),
                                                every=every,scale=scale,
                                                feat_inds=feat_inds,scaler=scaler)

    return trainX, trainY, train_prediction_start, testX, testY, test_prediction_start


def baggingMyLasso(trainX, trainY, train_prediction_start, testX, testY, test_prediction_start,
                   look_ahead, bag_size=47, Nestimators=50, samp_size=0.95,
                   market=1,every=0,lookbackMethod=1, bagModels_times=50,
                   RandomLasso=False,modelCombination='lassoLasso',otherFeats=False,
                   top_ret_Feats=6, topFeats=3):
    '''

    :param modelCombination:
    Only true if Randomized Lasso
    'lassoLasso', 'lassoRidge', 'LassoSimpleLinear'
    :return:
    '''
    #if Random Lass0=True: we choose variables that appear at least in 40% of the models

    # use 47 of estimators to do prediction, repeat 50 times
    # computing the best alpha for lasso:
    #alphas = np.linspace(0.005, 1, 1000)

    model = LassoCV(cv=10, eps=0.0001, fit_intercept=True, normalize=False,
                    random_state=None).fit(trainX, trainY)
    print('Best Lasso alpha for lookahead %d: ' % lookahead, model.alpha_)

    if RandomLasso:
        svr = BaggingRegressor(Lasso(alpha=model.alpha_,
                                     fit_intercept=True,
                                     normalize=False,
                                     random_state=None),
                               n_estimators=200,
                               max_samples=0.95,
                               bootstrap=False,
                               random_state=None,
                               n_jobs=-1)
        svr = svr.fit(trainX, trainY)
        ###################################
        coef_matrix = np.zeros((Nestimators, trainX.shape[1]))
        for i in range(Nestimators):
            coef_matrix[i, :] = \
                [1 if svr.estimators_[i].coef_[j] != 0 else 0 for j in range(trainX.shape[1])]

        coef_freq = np.sum(coef_matrix, axis=0) / Nestimators
        #num_coefs = trainX.shape[1] + 1
        # plt.bar(np.arange(1, num_coefs), coef_freq, color='skyblue')
        # plt.ylabel('Freq selected')
        # plt.xlabel('Feature')
        # plt.ylim(0, 1.3)
        # plt.text(2, 1.1, 'lookahead %d' % lookahead)
        # plt.xlim(0, num_coefs)
        # plt.show()
        if not otherFeats:

            ret_feat_inds=[i for i in range(len(coef_freq)) if coef_freq[i]>=0.60]
            print('Number of Features with more than 60 percent frequency: ', len(ret_feat_inds))
            if len(ret_feat_inds) >= top_ret_Feats:
                trainX = trainX[:, ret_feat_inds]
                testX = testX[:, ret_feat_inds]
            else:
                print('No variable meets the frequency requirement! choosing the %d most frequent!'%topFeats)
                order = coef_freq.argsort()
                trainX = trainX[:, order[-top_ret_Feats:]]
                testX = testX[:, order[-top_ret_Feats:]]

        else:
            ret_feat_freqs = coef_freq[:lookback]
            otherFeat_freqs=coef_freq[lookback:]
            o_F_inds=np.arange(lookback,trainX.shape[1],1)

            ret_feat_inds = np.array([i for i in range(len(ret_feat_freqs)) if ret_feat_freqs[i] >= 0.60])

            other_feat_inds = np.array(
                [i for i in range(lookback,len(coef_freq),1) if coef_freq[i] >= 0.60])

            print('Number of return Features with more than 60 percent frequency: ', len(ret_feat_inds))
            print('Number of Other Features with more than 60 percent frequency: ', len(other_feat_inds))
            if len(ret_feat_inds) < top_ret_Feats:
                ret_order = ret_feat_freqs.argsort()
                ret_order=ret_order[-top_ret_Feats:]
            else:
                ret_order=ret_feat_inds
            print(ret_order, 'return feat indices')

            if len(other_feat_inds)<topFeats:
                otherFeat_order = otherFeat_freqs.argsort()
                o_F_inds=o_F_inds[otherFeat_order]
                otherFeat_order = o_F_inds[-topFeats:]
            else:
                otherFeat_order=other_feat_inds
            print(otherFeat_order, 'other feat indices')

            ret_order=list(ret_order)
            ret_order.extend(list(otherFeat_order))
            trainX = trainX[:, ret_order]
            testX = testX[:, ret_order]

        print(trainX.shape,testX.shape)
        if modelCombination=='lassoLasso':
            #alphas = np.linspace(0, 1, 200)
            model = LassoCV(cv=10, eps=0.0001,fit_intercept=True, normalize=False,
                            random_state=None).fit(trainX, trainY)
            print('Best alpha for lookahead %d: after feature Selection!' % lookahead, model.alpha_)
            svr = BaggingRegressor(Lasso(alpha=model.alpha_, fit_intercept=True, normalize=False,
                                 random_state=None), n_estimators=50,
                           max_samples=0.95, bootstrap=False, random_state=None, n_jobs=-1)
        elif modelCombination=='LassoSimpleLinear':
            svr = BaggingRegressor(LinearRegression(fit_intercept=True, normalize=False), n_estimators=50,
                               max_samples=0.95, bootstrap=False, random_state=None, n_jobs=-1)

        elif modelCombination=='LassoRidge':

            alphas = np.linspace(0.0001, 10, 200)
            #cv=None: a form of leave one out CV!
            model_CV = RidgeCV(cv=None, alphas=alphas, fit_intercept=True,
                               normalize=False).fit(trainX, trainY)
            print('Best Ridge alpha for lookahead %d: after feature Selection- Lasso Ridge!' %
                  lookahead, model_CV.alpha_)
            svr = BaggingRegressor(Ridge(alpha=model_CV.alpha_, normalize=False, fit_intercept=True),
                                     n_estimators=Nestimators,
                                     max_samples=samp_size, bootstrap=False, random_state=None, n_jobs=-1)

    else:
        svr = BaggingRegressor(Lasso(alpha=model.alpha_, fit_intercept=True, normalize=False,
                                     random_state=None), n_estimators=50,
                               max_samples=0.95, bootstrap=False, random_state=None, n_jobs=-1)

    svr = svr.fit(trainX, trainY)
    ##
    coef_matrix = np.zeros((Nestimators, trainX.shape[1]))
    for i in range(Nestimators):
        coef_matrix[i, :] = [1 if svr.estimators_[i].coef_[j] != 0 else 0 for j in range(trainX.shape[1])]

    coef_freq = np.sum(coef_matrix, axis=0) / Nestimators
    num_coefs = trainX.shape[1] + 1
    # plt.bar(np.arange(1, num_coefs), coef_freq, color='skyblue')
    # plt.ylabel('Freq selected')
    # plt.xlabel('Feature')
    # plt.ylim(0, 1.3)
    # plt.text(2, 1.1, 'lookahead %d' % lookahead)
    # plt.xlim(0, num_coefs)
   # plt.show()
    ###
    colnamesRaw = ['dtStart']
    cln = [i for i in range(1, Nestimators * 2 + 3, 1)]
    colnamesRaw.extend(cln)
    colnamesBagged = ['dtStart']
    cln1 = [i for i in range(1, bagModels_times * 2 + 3, 1)]
    colnamesBagged.extend(cln1)

    # (date, trainY, true_lab, pred_labs...)
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
        trainRs_raw[:, i + 3] = svr.estimators_[i].predict(trainX)

        testRs_raw[:, i + 3] = svr.estimators_[i].predict(testX)

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

    if every == 0:
        trainRs.to_csv('%d_%d_%d_Lasso_train.csv' % (market, lookbackMethod, look_ahead), index=False)
        trainRs_raw.to_csv('%d_%d_%d_Lasso_train_Raw.csv' % (market, lookbackMethod, look_ahead), index=False)
        testRs.to_csv('%d_%d_%d_Lasso_test.csv' % (market, lookbackMethod, look_ahead), index=False)
        testRs_raw.to_csv('%d_%d_%d_Lasso_test_Raw.csv' % (market, lookbackMethod, look_ahead), index=False)
    else:

        trainRs.to_csv('%d_%d_%d_every_%d_Lasso_train.csv' %
                       (market, lookbackMethod, look_ahead, every), index=False)
        trainRs_raw.to_csv('%d_%d_%d_every_%d_Lasso_train_Raw.csv' %
                           (market, lookbackMethod, look_ahead, every), index=False)
        testRs.to_csv('%d_%d_%d_every_%d_Lasso_test.csv' %
                      (market, lookbackMethod, look_ahead, every), index=False)
        testRs_raw.to_csv('%d_%d_%d_every_%d_Lasso_test_Raw.csv' %
                          (market, lookbackMethod, look_ahead, every), index=False)


    return coef_matrix, coef_freq

####
datadir = 'C:/behrouz/Projects/DailyModels_new/Lasso/SQ/SQ_RandomizedLasso/' \
          'DailyDataFeaturesAndResponsesSQ2August2017_1.csv'

directory_tosave = 'C:/behrouz/Projects/DailyModels_new/Lasso/SQ/' \
                   'SQ_RandomizedLasso/tmpOtherFeats'

os.chdir(directory_tosave)

data = pd.read_csv(datadir)
#print(data.shape)

# feat_methods:
# 1: last 30 consecutive, 2: the staggered method,3: last30skip1, 4: last30skip2;
lookback = 30
lookaheadrange = 10
#feat_inds=(120, 217)
feat_inds=[122,123,124,125,126,128,129,130,131,132,187,188,189,190,191,192]

#feat_inds=0
#scaler: MinMaxScaler, RobustScaler,meanNormScaler
if __name__=='__main__':
    #every=[0,5,6]
    every=[0]
    #skip_method={0:1,1:3,2:4}
    skip_method={0:1,1:3,2:4}
    for skip in range(3):
        for eve in every:
            print('lookback method %d, every %d'% (skip_method[skip],eve))
            for lookahead in range(1, lookaheadrange + 1, 1):
                print('lookahead: ', lookahead)
                trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
                    data_gen(data, lookback, lookahead,
                             lookbackMethod=skip_method[skip], skip=skip,
                             every=eve,scale=True,feat_inds=feat_inds,scaler='meanNormScaler')
                print(trainX.shape)

                coef_matrix, coef_freq=baggingMyLasso(trainX, trainY, train_prediction_start,
                               testX, testY, test_prediction_start, lookahead, bag_size=47,
                               Nestimators=50, samp_size=0.95,
                               bagModels_times=50, lookbackMethod=skip_method[skip],
                                                      every=eve,RandomLasso=True,
                                                      modelCombination='LassoLinearRegression',
                                                      otherFeats=True,top_ret_Feats=3,topFeats=2)
    for eve in every:
        print('lookback method %d, every %d' % (2, eve))
        for lookahead in range(1, lookaheadrange + 1, 1):
                print('lookahead: ', lookahead)
                trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
                    data_gen(data, lookback, lookahead,
                             lookbackMethod=2, skip=0, every=eve,
                             scale=True, feat_inds=feat_inds,scaler='meanNormScaler')
                print(trainX.shape)
                coef_matrix, coef_freq=  baggingMyLasso(trainX, trainY, train_prediction_start,
                                       testX, testY, test_prediction_start, lookahead, bag_size=47,
                                       Nestimators=50, samp_size=0.95,
                                       bagModels_times=50, lookbackMethod=2,
                                                        every=eve,RandomLasso=True,
                                                        modelCombination='LassoSimpleLinear',
                                                        otherFeats=True, top_ret_Feats=3,topFeats=2)












