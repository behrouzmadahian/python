import numpy as np
import pandas as pd
import dataProcessing
import math
import copy

import os
from scipy import optimize
'''
A linear model optimized using Conjugate Gradient technique.
'''
markets = ('SQ', 'NQ', 'MQ', 'DQ', 'RN')
# objective function
# data parameters
Y_toUse = 1  # 1: scaled return, 2:1-day return
lookback = 30
lookahead = 1
rolling_sd_window = 100
date_train_end = 20070418
# I want to perform random search
l2_grid = np.random.uniform(0, 3, 30)
print(l2_grid)
np.save('randomSearch-reg_grid',l2_grid)
bias = 1
trainFrac = 0.25
slide_size = 40

def sum_sharpeLoss(outP_dict, train_dict, markets):
    neg_sharpe = 0
    for i in range(len(markets)):
        outP = outP_dict[markets[i]]
        tmp = outP * train_dict[markets[i]][2]
        neg_sharpe += -1 * np.mean(tmp) * math.sqrt(250.) / np.std(tmp)
    return neg_sharpe

def model_sumSharpes(weights, train_dict, bias, l2Reg, sum_sharpeLoss):
    '''
    X: n*p matrix of  data, y: response
    returns: the predicted response
    '''
    outP_dict = {}
    for i in range(len(markets)):
        outP = np.matmul(train_dict[markets[i]][0], weights) + bias
        outP_dict[markets[i]] = outP
    l2 = l2Reg * np.sum(weights**2)
    objective = sum_sharpeLoss(outP_dict, train_dict, markets) + l2
    return objective

def predict(x,weights,  bias):
    outP = np.matmul(x, weights) + bias
    return outP


# loading data
datadir = 'C:/behrouz/Projects/DailyModels_new/US-equities-GradFreeOptimization/data/%s_Commision-and-Slippage-limits-0.25.csv'
os.chdir('C:/behrouz/Projects/DailyModels_new/US-equities-GradFreeOptimization/CG')
#just matching by these since few days does not exactly match!
#get the common dates and then merge each data making sure they have common dates:
####################################################################################################
trans_data = pd.read_csv('C:/behrouz/Projects/DailyModels_new/NeuralNet/TransactionCosts.csv')
trans_data = trans_data.values
transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))
####################################################################################################
def  data_for_sumsharpeLoss(markets, date_train_end,lookback, lookahead, sd_window):
    # it is not necessary to  match by date
    test_dict = {}
    train_dict = {}
    # getting common dates:
    data = pd.read_csv(datadir % markets[0])
    for i in range(1, len(markets), 1):
        data1 = pd.read_csv(datadir % markets[i])
        data = pd.merge(data, data1, on='dtStart', how='inner')
    dates = data[['dtStart']]

    x = pd.read_csv(datadir % markets[0])
    x = pd.merge(x, dates, on='dtStart', how='inner')
    curr_market_data = dataProcessing.time_series_toMatrix(x, date_train_end, lookback=lookback,
                                            look_ahead=lookahead, sd_window=sd_window)
    train_dict[markets[0]] = copy.deepcopy(curr_market_data[:4])
    test_dict[markets[0]] = copy.deepcopy(curr_market_data[4:])
    for i in range(1, len(markets),1):
        x = pd.read_csv(datadir % markets[i])
        x = pd.merge(x, dates, on='dtStart', how='inner')
        curr_market_data = dataProcessing.time_series_toMatrix(x, date_train_end, lookback=lookback,
                                                               look_ahead=lookahead, sd_window=sd_window)
        train_dict[markets[i]] = copy.deepcopy(curr_market_data[:4])
        test_dict[markets[i]] = copy.deepcopy(curr_market_data[4:])
    return train_dict, test_dict

train_dict, test_dict  = data_for_sumsharpeLoss(markets, date_train_end,
                                                             lookback, lookahead, rolling_sd_window)
startInds = np.arange(0, 1700, slide_size)

finalWeights = np.zeros((lookback, len(startInds)))
for l2 in l2_grid:
    if not os.path.exists('rollTraining/sumSharpeLoss/predictions/l2-%.2f'%l2):
        os.makedirs('rollTraining/sumSharpeLoss/predictions/l2-%.2f'%l2)

    for i in range(len(startInds)):
        start_ind = startInds[i]
        curr_train_dict = dataProcessing.trainRoll_dict(train_dict, trainFrac, start_ind, slide_size, markets)
        # Optimizer for current training data:
        weights = np.random.randn(lookback, 1)
        finalWeights[:, i] = optimize.fmin_cg(model_sumSharpes, weights, fprime=None,
                                              args=(curr_train_dict,  bias, l2, sum_sharpeLoss))

    np.save('rollTraining/sumSharpeLoss/FinalWeightsRolls-%.2f'%l2, finalWeights)

    # predictions on the whole train and test:
    for i in range(len(markets)):
        colnames = ['dtStart', 'ret_1day']
        predCols = ['%s-pred%d' % (markets[i], kk + 1) for kk in range(len(startInds))]
        colnames.extend(predCols)

        train_data = train_dict[markets[i]]
        trainLen = train_data[0].shape[0]
        trainRes = np.zeros((trainLen, 2 + len(startInds)))
        trainRes[:, 0] = train_data[3]  # dates
        trainRes[:, 1] = train_data[2]  # returns
        for j in range(len(startInds)):
            trainRes[:, j + 2] = predict(train_data[0], finalWeights[:, j], bias)
        trainRes = pd.DataFrame(trainRes, columns=colnames)
        trainRes.to_csv('rollTraining/sumSharpeLoss/predictions/l2-%.2f/trainPreds-%s.csv' % (l2, markets[i]), index=False)
        #
        test_data = test_dict[markets[i]]
        testLen = test_data[0].shape[0]
        testRes = np.zeros((testLen, 2 + len(startInds)))
        testRes[:, 0] = test_data[3]
        testRes[:, 1] = test_data[2]
        for j in range(len(startInds)):
            testRes[:, j + 2] = predict(test_data[0], finalWeights[:, j], bias)
        testRes = pd.DataFrame(testRes, columns=colnames)
        testRes.to_csv('rollTraining/sumSharpeLoss/predictions/l2-%.2f/testPreds-%s.csv' % (l2, markets[i]), index=False)


