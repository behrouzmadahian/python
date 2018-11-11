import numpy as np, pandas as pd
import os
import dataProcessing
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV, lasso_path, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV,ShuffleSplit,RandomizedSearchCV
import scipy.stats

np.random.seed(123)


# plt.plot(data['close_geo'])
# plt.axvline(4365)
# plt.text(2000,1800,'Train period')
# plt.text(5100,1800,'Test period')
# plt.title('SQ geometrically rolled')
# plt.show()

def data_gen(data, lookback, look_ahead, weired_feat=True, skip=0, every=0):
    if weired_feat:
        lookback = 105
        trainX, trainY, train_weights,train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrixWeiredModoFeat(data, lookback=lookback, look_ahead=5,
                                                              train_inds=(564, 4070), test_inds=(4070, 6489),
                                                              every=every)
    else:
        trainX, trainY, train_weights,train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrix(data, skip=skip, lookback=lookback, look_ahead=look_ahead,
                                                train_inds=(564, 4070), test_inds=(4070, 6489), every=every)

    return trainX, trainY,train_weights ,train_prediction_start, testX, testY, test_prediction_start


def baggingMySVM(trainX, trainY, train_prediction_start, testX, testY, test_prediction_start, look_ahead,
                   bag_size=47, Nestimators=50, samp_size=0.95, sampleModels=50,kernel='sigmoid'):
    cRange = scipy.stats.expon(scale=5)
    #gammaRange = scipy.stats.expon(scale=0.1)
    #parameter_dist = {'C': cRange, 'gamma': gammaRange}
    parameter_dist = {'C': cRange}
    clf = RandomizedSearchCV(estimator=svm.SVR(kernel=kernel), param_distributions=parameter_dist,
                             n_iter=50, cv=10, n_jobs=-1)
    clf.fit(trainX, trainY)
    print('Best C:', clf.best_estimator_.C)
    #print('Best Gamma:', clf.best_estimator_.gamma)



    svr = BaggingRegressor(svm.SVR(kernel=kernel, C=clf.best_estimator_.C),
                           n_estimators=Nestimators,
                           max_samples=samp_size, bootstrap=False, random_state=123)
    svr = svr.fit(trainX, trainY)

    colnames = ['dtStart']
    cln = [i for i in range(1, Nestimators * 2 + 3, 1)]
    colnames.extend(cln)
    # (date, trainY, true_lab, pred_labs...)
    trainRs = np.zeros((trainX.shape[0], sampleModels * 2 + 3))
    trainRs_raw = np.zeros((trainX.shape[0], Nestimators * 2 + 3))
    trainRs[:, 0] = train_prediction_start
    trainRs_raw[:, 0] = train_prediction_start
    trainRs[:, 1] = trainY
    trainRs_raw[:, 1] = trainY
    trainRs[:, 2] = [1 if trainY[i] > 0 else 0 for i in range(len(trainY))]
    trainRs_raw[:, 2] = [1 if trainY[i] > 0 else 0 for i in range(len(trainY))]
    #
    testRs = np.zeros((testX.shape[0], sampleModels * 2 + 3))
    testRs_raw = np.zeros((testX.shape[0], Nestimators * 2 + 3))
    testRs[:, 0] = test_prediction_start
    testRs_raw[:, 0] = test_prediction_start
    testRs[:, 1] = testY
    testRs_raw[:, 1] = testY
    testRs[:, 2] = [1 if testY[i] > 0 else 0 for i in range(len(testY))]
    testRs_raw[:, 2] = [1 if testY[i] > 0 else 0 for i in range(len(testY))]
    for i in range(sampleModels):
        trainRs_raw[:, i + 3] = svr.estimators_[i].predict(trainX)

        testRs_raw[:, i + 3] = svr.estimators_[i].predict(testX)

        trainRs_raw[:, i + Nestimators + 3] = \
            [1 if trainRs_raw[j, i + 3] > 0 else 0 for j in range(len(trainRs_raw[:, i + 3]))]
        testRs_raw[:, i + Nestimators + 3] = \
            [1 if testRs_raw[j, i + 3] > 0 else 0 for j in range(len(testRs_raw[:, i + 3]))]

    # aggregating results!
    model_inds = [j for j in range(3, Nestimators + 3)]
    # print(model_inds)
    for i in range(len(model_inds)):
        index_modelstoUse = np.random.choice(model_inds, bag_size, replace=False)
        tmp_train = trainRs_raw[:, index_modelstoUse]
        tmp_test = testRs_raw[:, index_modelstoUse]
        trainRs[:, i + 3] = np.sum(tmp_train, axis=1)
        testRs[:, i + 3] = np.sum(tmp_test, axis=1)

        trainRs[:, i + Nestimators + 3] = \
            [1 if trainRs[j, i + 3] > 0 else 0 for j in range(len(trainRs[:, i + 3]))]
        testRs[:, i + Nestimators + 3] = \
            [1 if testRs[j, i + 3] > 0 else 0 for j in range(len(testRs[:, i + 3]))]

    trainRs = pd.DataFrame(trainRs, columns=colnames)
    trainRs.to_csv('train_SQ_results_la%d.csv' % look_ahead, index=False)

    trainRs_raw = pd.DataFrame(trainRs_raw, columns=colnames)
    trainRs_raw.to_csv('train_SQ_Raw_results_la%d.csv' % look_ahead, index=False)

    testRs = pd.DataFrame(testRs, columns=colnames)
    testRs.to_csv('test_SQ_results_la%d.csv' % look_ahead, index=False)

    testRs_raw = pd.DataFrame(testRs_raw, columns=colnames)
    testRs_raw.to_csv('test_SQ_Raw_results_la%d.csv' % look_ahead, index=False)



####
def baggingMyLassoForModo(trainX, trainY, train_prediction_start, testX, testY,
                          test_prediction_start, look_ahead,
                          Nestimators=50, samp_size=0.95, feat_method=1, every=5):
    model = LassoCV(cv=10, random_state=123).fit(trainX, trainY)

    svr = BaggingRegressor(Lasso(alpha=model.alpha_), n_estimators=Nestimators,
                           max_samples=samp_size, bootstrap=False, random_state=123)
    svr = svr.fit(trainX, trainY)

    colnames = ['dtStart', 'TrueY']
    cln = ['pred_%d' % i for i in range(1, Nestimators + 1, 1)]
    colnames.extend(cln)
    # (date, trainY, true_lab, pred_labs...)
    trainRs_raw = np.zeros((trainX.shape[0], Nestimators + 2))
    trainRs_raw[:, 0] = train_prediction_start
    trainRs_raw[:, 1] = trainY
    #
    testRs_raw = np.zeros((testX.shape[0], Nestimators + 2))
    testRs_raw[:, 0] = test_prediction_start
    testRs_raw[:, 1] = testY
    for i in range(Nestimators):
        trainRs_raw[:, i + 2] = svr.estimators_[i].predict(trainX)

        testRs_raw[:, i + 2] = svr.estimators_[i].predict(testX)

    trainRs_raw = pd.DataFrame(trainRs_raw, columns=colnames)
    testRs_raw = pd.DataFrame(testRs_raw, columns=colnames)
    if every > 0:
        trainRs_raw.to_csv('1_%d_%d_every_%d_Lasso_train.csv' % (feat_method, look_ahead, every), index=False)
        testRs_raw.to_csv('1_%d_%d_every_%d_Lasso_test.csv' % (feat_method, look_ahead, every), index=False)
    else:
        trainRs_raw.to_csv('1_%d_%d_Lasso_train.csv' % (feat_method, look_ahead), index=False)
        testRs_raw.to_csv('1_%d_%d_Lasso_test.csv' % (feat_method, look_ahead), index=False)

###
datadir = 'C:/behrouz/Projects/dailyModelsSVMNewData_SQ/DailyDataFeaturesAndResponsesSQ2August2017_1.csv'
data = pd.read_csv(datadir)
#print(data.shape)

# feat_methods:
# 1: last 30 consecutive, 2: the staggered method,3: last30skip1, 4: last30skip2;
lookback = 30
lookaheadrange = 10
directory_tosave = 'C:/behrouz/Projects/dailyModelsSVMNewData_SQ/tmp'
os.chdir(directory_tosave)
if __name__=='__main__':

    for lookahead in range(1, lookaheadrange + 1, 1):
        #lookahead=3
        print('lookahead: ', lookahead)
        trainX, trainY, train_weights, train_prediction_start, testX, testY, test_prediction_start = \
            data_gen(data, lookback, lookahead, weired_feat=False, skip=0, every=0)
        print(trainX.shape)

        baggingMySVM(trainX, trainY, train_prediction_start,
                                           testX, testY, test_prediction_start, lookahead, bag_size=47, Nestimators=50,
                                           samp_size=.95, sampleModels=50,kernel='linear')

        # coef_mat, coef_freq = baggingMyLassoForModo(trainX, trainY, train_prediction_start,
        #                                             testX, testY, test_prediction_start, lookahead, Nestimators=50,
        #                                             samp_size=.95, feat_method=4, every=6)

    #     plt.subplot(4, 5, lookahead)
    #     num_coefs = trainX.shape[1] + 1
    #     plt.bar(np.arange(1, num_coefs), coef_freq, color='skyblue')
    #     plt.ylabel('Freq selected')
    #     plt.xlabel('Feature')
    #     plt.ylim(0, 1.3)
    #     plt.text(2, 1.1, 'lookahead %d' % lookahead)
    #     plt.xlim(0, num_coefs)
    # plt.show()










