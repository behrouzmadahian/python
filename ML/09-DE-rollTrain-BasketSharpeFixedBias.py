import numpy as np
import pandas as pd
import dataProcessing
import math
import copy
import os
from scipy import optimize
import multiprocessing
import time
'''
A linear model optimized using Conjugate Gradient technique.
'''
markets = ('SQ', 'NQ', 'MQ', 'DQ', 'RN')
# objective function
# data parameters
lookback = 30
lookahead = 1
rolling_sd_window = 100
date_train_end = 20070418
trainFrac = 0.25
slide_size = 40
bias = 1
res_folder = 'basketSharpe-fixedBias'
datadir = 'C:/behrouz/Projects/DailyModels_new/US-equities-Diff-evolution/data/%s_Commision-and-Slippage-limits-0.25.csv'
os.chdir('C:/behrouz/Projects/DailyModels_new/US-equities-Diff-evolution')

def basket_sharpe_likeModo(outP_dict, train_dict, markets, transCost_dict):
    basket_ret_stream = np.zeros(len(outP_dict[markets[0]]))
    for i in range(len(markets)):
        transCost = transCost_dict[markets[i]]
        outP = outP_dict[markets[i]]
        signPred = np.sign(outP)
        return_stream = signPred *  train_dict[markets[i]][2]
        t1 = [2 * transCost if signPred[i - 1] != signPred[i] else 0 for i in range(1, len(signPred), 1)]
        totalTransCost = [transCost]
        totalTransCost.extend(t1)
        return_stream -= totalTransCost
        return_stream /= np.std(return_stream)
        basket_ret_stream += return_stream
    basket_sharpe = -np.mean(basket_ret_stream) * math.sqrt(250.) / np.std(basket_ret_stream)
    return basket_sharpe


def model_sumSharpes(weights, bias, train_dict, l2Reg, sum_sharpe_likeModo,transCost_dict):
    outP_dict = {}
    for i in range(len(markets)):
        outP = np.matmul(train_dict[markets[i]][0], weights) + bias
        outP_dict[markets[i]] = outP
    l2 = l2Reg * np.sum(weights**2)
    objective = sum_sharpe_likeModo(outP_dict, train_dict, markets, transCost_dict) + l2
    return objective

def predict(x,weights, bias):
    outP = np.matmul(x, weights) + bias
    return outP

def  data_for_sumsharpeLoss(markets, date_train_end,lookback, lookahead, sd_window):
    # it is not necessary to  match by date
    # just matching by these since few days does not exactly match!
    # get the common dates and then merge each data making sure they have common dates:
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


def myparralelFunc(l2Reg,
                   train_dict,
                   test_dict,
                   trainFrac,
                   startInds,
                   slide_size,
                   markets,
                   transCost_dict):
    # bounds on parameters:
    bounds = [(-2, 2) for i in range(lookback)]
    finalWeights = np.zeros((lookback, len(startInds)))
    for i in range(len(startInds)):
        start_ind = startInds[i]
        curr_train_dict = dataProcessing.trainRoll_dict(train_dict,
                                                        trainFrac,
                                                        start_ind,
                                                        slide_size,
                                                        markets)
        rs = optimize.differential_evolution(model_sumSharpes,
                                             bounds,
                                             args=(bias,
                                                   curr_train_dict,
                                                   l2Reg,
                                                   basket_sharpe_likeModo,
                                                   transCost_dict))
        finalWeights[:, i] = rs.x
    np.save('rollTraining/%s/FinalWeightsRolls-l2-%.4f'%(res_folder,l2Reg), finalWeights)
    # predictions
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
            trainRes[:, j + 2] = predict(train_data[0], finalWeights[:, j],bias)
        trainRes = pd.DataFrame(trainRes, columns=colnames)
        trainRes.to_csv('rollTraining/%s/predictions/l2-%.4f/trainPreds-%s.csv' %
                                                                        (res_folder,l2Reg,markets[i]), index=False)
        test_data = test_dict[markets[i]]
        testLen = test_data[0].shape[0]
        testRes = np.zeros((testLen, 2 + len(startInds)))
        testRes[:, 0] = test_data[3]
        testRes[:, 1] = test_data[2]
        for j in range(len(startInds)):
            testRes[:, j + 2] = predict(test_data[0], finalWeights[:, j], bias)
        testRes = pd.DataFrame(testRes, columns=colnames)
        testRes.to_csv('rollTraining/%s/predictions/l2-%.4f/testPreds-%s.csv' % (res_folder,l2Reg,markets[i]), index=False)

if __name__ =='__main__':

    trans_data = pd.read_csv('/slippagesETA.csv')
    trans_data = trans_data.values
    transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))
    t1 = time.time()
    train_dict, test_dict = data_for_sumsharpeLoss(markets,
                                                   date_train_end,
                                                   lookback,
                                                   lookahead,
                                                   rolling_sd_window)
    startInds = np.arange(0, 1700, slide_size)
    l2_grid = np.linspace(0, 5, 21)
    for l2Reg in l2_grid:
        if not os.path.exists('rollTraining/%s/predictions/l2-%.4f' % (res_folder, l2Reg)):
            os.makedirs('rollTraining/%s/predictions/l2-%.4f' % (res_folder, l2Reg))
        processes = []
        p = multiprocessing.Process(target=myparralelFunc,
                                    args=(l2Reg,
                                          train_dict,
                                          test_dict,
                                          trainFrac,
                                          startInds,
                                          slide_size,
                                          markets,
                                          transCost_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    t2 = time.time()
    print('Total time(min)=', (t2-t1)/ 60.)


