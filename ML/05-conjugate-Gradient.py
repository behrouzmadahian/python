import numpy as np
import pandas as pd
import dataProcessing
import math
import copy
from matplotlib import pyplot as plt
import multiprocessing
import time
import scipy
import os
from scipy import optimize
'''
A linear model optimized using Conjugate Gradient technique.
'''
Mycolors = ['blue', 'green', 'red', 'cyan', 'magenta']


def sharpe_posSized(outP, return_1day):
    ''' Treats all data as one market'''
    tmp = outP * return_1day
    neg_sharpe = -1 * np.mean(tmp) * math.sqrt(250.) / np.std(tmp)
    return neg_sharpe


def sharpe(outP, return_1day):
    ''' Treats all data as one market'''
    outP = outP.flatten()
    tmp = np.sign(outP) * return_1day
    neg_sharpe = -1 * np.mean(tmp) * math.sqrt(250.) / np.std(tmp)
    return neg_sharpe


markets = ('SQ', 'NQ', 'MQ', 'DQ', 'RN')
# objective function
# data parameters
Y_toUse = 1  # 1: scaled return, 2:1-day return
lookback = 30
lookahead = 1
rolling_sd_window = 100
date_train_end = 20070418
l2Reg = 1
bias = 1


def model(weights, x, y, bias, l2Reg, objective_func):
    '''
    X: n*p matrix of  data, y: response
    returns: the predicted response
    '''
    outP = np.matmul(x, weights) + bias
    l2 = l2Reg * np.sum(weights**2) / len(weights)
    objective = objective_func(outP, y) + l2
    return objective


def predict(weights, x, bias):
    outP = np.matmul(x, weights) + bias
    return outP



# loading data
datadir = '.csv'
os.chdir('/CG')
# just matching by these since few days does not exactly match!
# get the common dates and then merge each data making sure they have common dates:
####################################################################################################
trans_data = pd.read_csv('/TransactionCosts.csv')
trans_data = trans_data.values
transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))
####################################################################################################


def  data_for_sharpeLoss(markets, date_train_end,lookback, lookahead, sd_window):
    # it is not necessary to  match by date
    test_dict = {}
    train_dict = {}
    data = pd.read_csv(datadir % markets[0])
    for i in range(1, len(markets), 1):
        data1 = pd.read_csv(datadir % markets[i])
        data = pd.merge(data, data1, on='dtStart', how='inner')
    dates = data[['dtStart']]

    x = pd.read_csv(datadir % markets[0])
    x = pd.merge(x, dates, on='dtStart', how='inner')
    curr_market_data = dataProcessing.time_series_toMatrix(x, date_train_end, lookback=lookback,
                                            look_ahead=lookahead, sd_window=sd_window)
    trainX = curr_market_data[0]
    trainY = curr_market_data[1]
    train_dict[markets[0]] = copy.deepcopy(curr_market_data[:4])
    test_dict[markets[0]] = copy.deepcopy(curr_market_data[4:])
    for i in range(1, len(markets),1):
        x = pd.read_csv(datadir % markets[i])
        x = pd.merge(x, dates, on='dtStart', how='inner')
        curr_market_data = dataProcessing.time_series_toMatrix(x, date_train_end, lookback=lookback,
                                                               look_ahead=lookahead, sd_window=sd_window)
        trainX = np.append(trainX, curr_market_data[0], axis = 0)
        trainY = np.append(trainY, curr_market_data[1], axis = 0)
        train_dict[markets[i]] = copy.deepcopy(curr_market_data[:4])
        test_dict[markets[i]] = copy.deepcopy(curr_market_data[4:])
    return trainX, trainY, train_dict, test_dict


trainX, trainY, train_dict, test_dict  = data_for_sharpeLoss(markets,
                                                             date_train_end,
                                                             lookback,
                                                             lookahead,
                                                             rolling_sd_window)
print(trainX.shape, trainY.shape)
weights = np.random.randn(lookback, 1)

final_weights = optimize.fmin_cg(model, weights, fprime=None, args=(trainX, trainY, bias,  l2Reg, sharpe_posSized))

print(final_weights)
print(final_weights.shape)
np.save('FinalWeights', final_weights)

colnames = ['dtStart', 'ret_1day', 'pred']
for i in range(len(markets)):
    train_data = train_dict[markets[i]]
    trainLen = train_data[0].shape[0]
    trainRes =np.zeros((trainLen, 3))
    trainRes[:, 0] = train_data[3] # dates
    trainRes[:, 1] = train_data[2]# returns
    trainRes[:,-1] = predict(final_weights, train_data[0], 1)
    print(trainRes[:,-1].shape)
    trainRes = pd.DataFrame(trainRes, columns=colnames)
    trainRes.to_csv('train-%s.csv'%markets[i], index =False)
    #
    test_data = test_dict[markets[i]]
    testLen = test_data[0].shape[0]
    testRes = np.zeros((testLen, 3))
    testRes[:, 0] = test_data[3]
    testRes[:, 1] = test_data[2]
    testRes[:, -1] = predict(final_weights, test_data[0], 1)
    testRes = pd.DataFrame(testRes, columns=colnames)
    testRes.to_csv('test-%s.csv'%markets[i], index =False)

