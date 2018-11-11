import numpy as np, pandas as pd
import os
import dataProcessing
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV,Ridge,LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV
#np.random.seed(123)

# plt.plot(data['close_geo'])
# plt.axvline(4365)
# plt.text(2000,1800,'Train period')
# plt.text(5100,1800,'Test period')
# plt.title('SQ geometrically rolled')
#plt.show()
def AIC(estimator,X,y,k):
    yPred=estimator.predict(X)
    sse=np.sum(np.power(y-yPred,2))
    aic=y.shape[0]*np.log(sse)+2*k
    aicc=aic+2*k*(k+1)/(y.shape[0]-k-1)
    return aicc
    
    

def data_gen(data, lookback, look_ahead, weired_feat=True, skip=0, every=0):
    if weired_feat:
        lookback = 105
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrixWeiredModoFeat(data, lookback=lookback,
                                                              look_ahead=look_ahead,
                                                              train_inds=(564, 4070),
                                                              test_inds=(4070, 6489),
                                                              every=every)
    else:
        trainX, trainY, train_prediction_start, testX, testY, test_prediction_start = \
            dataProcessing.time_series_toMatrix(data, skip=skip, lookback=lookback, look_ahead=look_ahead,
                                                train_inds=(564, 4070), test_inds=(4070, 6489), every=every)

    return trainX, trainY, train_prediction_start, testX, testY, test_prediction_start

def baggingMyRidge(trainX,trainY, train_prediction_start,testX,testY,test_prediction_start,look_ahead,
                   bag_size=47,Nestimators=50,samp_size=0.95,
                   bagModels_times=50,nBest=5,feat_selection_method='recursive',simpleReg=False):
    #use 47 of estimators to do prediction, repeat 50 times
    if feat_selection_method=='recursive':
        alphas = np.linspace(0, 10, 100)
        param_dict={}
        param_dict['alpha']=alphas
        model = Ridge()
        grid_search = GridSearchCV(model, param_grid=param_dict, cv=10,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(trainX, trainY)
        # print(grid_search.best_estimator_)
        print(grid_search.best_params_, '===========')

        # step: number of features to remove at each step
        rfecv = RFECV(estimator=Ridge(alpha=grid_search.best_params_['alpha']),
                      step=1, cv=10, scoring='neg_mean_squared_error')
        rfecv.fit(trainX, trainY)
        print("Optimal number of features : %d" % rfecv.n_features_)
        # Selected (i.e., estimated best) best feature is assigned rank 1.
        print('Feature Ranking: I want to take top 5 regressors if best num is less than 5 else '
              'whatever it is! ')
        print(rfecv.ranking_)
        if rfecv.n_features_ > nBest:
            nBest = rfecv.n_features_
        inds = [i for i in range(30) if rfecv.ranking_[i] == 1]
        k = 2
        while len(inds) < nBest:
            inds1 = [i for i in range(30) if rfecv.ranking_[i] == k]
            inds.extend(inds1)
            k += 1
        print(inds)
        trainX = trainX[:, inds];
        testX = testX[:, inds]
        print(trainX.shape, testX.shape)

    if feat_selection_method == 'f1_regression':
        f_test, pval = f_regression(trainX, trainY,center=False)
        order=pval.argsort()
        print(order)
        print(pval[order[:nBest]])
        trainX=trainX[:,order[:nBest]]
        testX=testX[:,order[:nBest]]
    if feat_selection_method == 'mutual_Information':
        mi=mutual_info_regression(trainX,trainY)
        mi /= np.max(mi)
        print(mi)
        order = mi.argsort()
        print(order)
        print(mi[order[-nBest:]])
        trainX = trainX[:, order[-nBest:]]
        testX = testX[:, order[-nBest:]]

    if feat_selection_method == 'randomForest':
        param_dict={}
        param_dict['max_depth']=[2,3,4,5,6,7,8]
        param_dict['max_features']=[2,3,4,5,6]

        forest=RandomForestRegressor(n_estimators=200,n_jobs=-1)
        grid_search = GridSearchCV(forest, param_grid=param_dict,cv=3)
        grid_search.fit(trainX,trainY)
        #print(grid_search.best_estimator_)
        print(grid_search.best_params_)

        forest = RandomForestRegressor(n_estimators=200,max_depth=grid_search.best_params_['max_depth'],
                                       max_features=grid_search.best_params_['max_features'],n_jobs=-1)
        forest.fit(trainX,trainY)

        importances=forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        order = importances.argsort()
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(trainX.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(trainX.shape[1]), indices)
        plt.xlim([-1, trainX.shape[1]])
        plt.show()
        trainX=trainX[:,order[-nBest:]]
        testX=testX[:,order[-nBest:]]
    print(trainX.shape,testX.shape)
    if not simpleReg:
        alphas = np.linspace(0, 2, 100)
        model_CV = RidgeCV(cv=20, alphas=alphas).fit(trainX, trainY)
        print('Best Alpha parameter found by 10 fold CV: ', model_CV.alpha_)

        model = BaggingRegressor(Ridge(alpha=0), n_estimators=Nestimators,
                                 max_samples=samp_size, bootstrap=False, random_state=123)
    else:
        model = BaggingRegressor(LinearRegression(), n_estimators=Nestimators,
                                 max_samples=samp_size, bootstrap=False, random_state=123)

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
    trainRs.to_csv('train_SQ_results_la%d.csv' % look_ahead, index=False)

    trainRs_raw = pd.DataFrame(trainRs_raw, columns=colnamesRaw)
    trainRs_raw.to_csv('train_SQ_Raw_results_la%d.csv' % look_ahead, index=False)

    testRs = pd.DataFrame(testRs, columns=colnamesBagged)
    testRs.to_csv('test_SQ_results_la%d.csv' % look_ahead, index=False)

    testRs_raw = pd.DataFrame(testRs_raw, columns=colnamesRaw)
    testRs_raw.to_csv('test_SQ_Raw_results_la%d.csv' % look_ahead, index=False)

    # coef_freq = np.sum(coef_matrix, axis=0) / Nestimators
    # print(coef_freq)
    # return coef_matrix, coef_freq

####
def baggingMyLassoForModo(trainX,trainY, train_prediction_start,testX,testY,
                          test_prediction_start,look_ahead,
                   Nestimators=50,samp_size=0.95,feat_method=1,every=5,nBest=5):
    alphas = np.linspace(0, 200., 1000)

    model = RidgeCV(cv=10, alphas=alphas).fit(trainX, trainY)
    model = BaggingRegressor(RidgeCV(alpha=model.alpha_), n_estimators=Nestimators,
                             max_samples=samp_size, bootstrap=False, random_state=123)
    model = model.fit(trainX,trainY)
    colnames=['dtStart','TrueY']
    cln=['pred_%d'%i for i in range(1,Nestimators+1,1)]
    colnames.extend(cln)
    #(date, trainY, true_lab, pred_labs...)
    trainRs_raw=np.zeros((trainX.shape[0], Nestimators+2))
    trainRs_raw[:,0]=train_prediction_start
    trainRs_raw[:, 1] =trainY
    #
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
datadir='C:/behrouz/Projects/DailyModels_new/Ridge/dailyModelsRidgeNewData_SQ/DailyDataFeaturesAndResponsesSQ2August2017_1.csv'
data = pd.read_csv(datadir)
print (data.shape)

#feat_methods:
#1: last 30 consecutive, 2: the staggered method,3: last30skip1, 4: last30skip2;
lookback=30
lookaheadrange=10
nBest=5
directory_tosave='C:/behrouz/Projects/DailyModels_new/Ridge/dailyModelsRidgeNewData_SQ/tmp'
os.chdir(directory_tosave)
for lookahead in range(1,lookaheadrange+1,1):
    print('lookahead: ',lookahead)
    trainX, trainY, train_prediction_start, testX, testY, test_prediction_start=\
        data_gen(data,lookback, lookahead,weired_feat=False, skip=0,every=0)
    print(trainX.shape,trainY.shape)

    baggingMyRidge(trainX ,trainY, train_prediction_start,
           testX, testY, test_prediction_start,lookahead,bag_size=47,
                                       Nestimators=50,samp_size=0.95,
                   bagModels_times=50,nBest=nBest,feat_selection_method='randomForest',simpleReg=True)

    # baggingMyLassoForModo(trainX ,trainY, train_prediction_start,
    #         testX, testY, test_prediction_start,lookahead,Nestimators=50,samp_size=.95,feat_method=4,every=6)
    #
#     plt.subplot(4, 5, lookahead)
#     num_coefs = nBest
#     plt.bar(np.arange(1, num_coefs), coef_freq, color='skyblue')
#     plt.ylabel('Freq selected')
#     plt.xlabel('Feature')
#     plt.ylim(0, 1.3)
#     plt.text(2, 1.1, 'lookahead %d' % lookahead)
#     plt.xlim(0, num_coefs)
# plt.show()







