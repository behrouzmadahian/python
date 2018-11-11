import numpy as np
import pandas as pd
import dataProcessing
import math
import copy
import os
from scipy import optimize
import multiprocessing
'''
A linear model optimized using Conjugate Gradient technique.
1. Impose norm of 1 on parameters
2. use floating Bias

'''
markets = ("AX","DA","DQ","EN","HI","IX","MI","MQ","MW",
           "NO","NQ","OX","RN","SA","SQ","TT","VB","ZX")

datadir = 'C:/behrouz/Projects/DailyModels_new/eta-ConjugateGradient/' \
          'data/%s_Commision-and-Slippage-limits-0.25.csv'
os.chdir('C:/behrouz/Projects/DailyModels_new/eta-ConjugateGradient/')
# objective function
# data parameters
Y_toUse = 1  # 1: scaled return, 2:1-day return
lookback = 30
lookahead = 1
rolling_sd_window = 100
date_train_end = 20070418
nfeatures = lookback + 1
trainFrac = 0.1
slide_size = 40
res_folder = 'basketSharpeLoss-floatBias-l2Norm-open-open'


def basket_sharpeLoss(outP_dict, train_dict, markets):
    ret_stream= np.zeros(outP_dict[markets[0]].shape[0])
    for i in range(len(markets)):
        outP = outP_dict[markets[i]]
        tmp = outP * train_dict[markets[i]][2]
        tmp /= np.std(tmp)
        ret_stream += tmp
    neg_sharpe = -1 * np.mean(ret_stream) * math.sqrt(250.) / np.std(ret_stream)
    return neg_sharpe


def model_basketSharpe_Objective_floatBias(weights, train_dict, l2Reg, basket_sharpeLoss):

    '''
    X: n*p matrix of  data, y: response
    returns: the predicted response
    '''
    l2Norm = np.sqrt(np.sum(weights**2))
    l2Norm_forL2 = np.sum(weights[1:]**2)
    outP_dict = {}
    for i in range(len(markets)):
        outP = np.matmul(train_dict[markets[i]][0], weights[1:]) + weights[0]
        outP_dict[markets[i]] = outP
    l2 = l2Reg * l2Norm_forL2
    objective = basket_sharpeLoss(outP_dict, train_dict, markets) + l2 + 1000 * (l2Norm-1)**2
    return objective

def predict(x,weights,  bias):
    outP = np.matmul(x, weights) + bias
    return outP

def predict_floatBias(x,weights):
    outP = np.matmul(x, weights[1:]) + weights[0]
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
    curr_market_data = dataProcessing.time_series_toMatrix_open_open(x, date_train_end, lookback=lookback,
                                            look_ahead=lookahead, sd_window=sd_window)
    train_dict[markets[0]] = copy.deepcopy(curr_market_data[:4])
    test_dict[markets[0]] = copy.deepcopy(curr_market_data[4:])
    for i in range(1, len(markets),1):
        x = pd.read_csv(datadir % markets[i])
        x = pd.merge(x, dates, on='dtStart', how='inner')
        curr_market_data = dataProcessing.time_series_toMatrix_open_open(x, date_train_end, lookback=lookback,
                                                               look_ahead=lookahead, sd_window=sd_window)
        train_dict[markets[i]] = copy.deepcopy(curr_market_data[:4])
        test_dict[markets[i]] = copy.deepcopy(curr_market_data[4:])
    return train_dict, test_dict

train_dict, test_dict  = data_for_sumsharpeLoss(markets, date_train_end,
                                                             lookback, lookahead, rolling_sd_window)
#
for i in range(len(markets)):
    print(train_dict[markets[i]][0].shape, train_dict[markets[i]][3][:3])

end_window = train_dict[markets[0]][0].shape[0]
window_size = int(trainFrac * end_window)
startInds = np.arange(0, end_window-window_size, slide_size)
print(len(startInds))

def GC_parallel(l2):
    if not os.path.exists('rollTraining/%s/predictions/l2-%.4f' % (res_folder, l2)):
        os.makedirs('rollTraining/%s/predictions/l2-%.4f' % (res_folder, l2))

    finalWeights = np.zeros((nfeatures, len(startInds)))
    for i in range(len(startInds)):

        start_ind = startInds[i]
        curr_train_dict = dataProcessing.trainRoll_dict(train_dict, trainFrac, start_ind, slide_size, markets)
        #print(curr_train_dict[markets[0]][0].shape)
        #print(curr_train_dict[markets[0]][3][-2:])
        # Optimizer for current training data:
        # run 3 times and get the best one!
        solution_dict = {}
        loss_iter = np.zeros(5)
        print('*' * 50)
        for iter in range(5):
            weights = np.random.randn(nfeatures, 1)
            weights /= np.linalg.norm(weights)
            solution_found = optimize.fmin_cg(model_basketSharpe_Objective_floatBias, weights, fprime=None,
                                              full_output=True,
                                              args=(curr_train_dict, l2, basket_sharpeLoss))
            solution_dict[iter] = solution_found[0]
            loss_iter[iter] = solution_found[1]
        best_iter = np.argmin(loss_iter)
        solution_found = solution_dict[best_iter]
        finalWeights[:, i] = solution_found

    np.save('rollTraining/%s/FinalWeightsRolls-%.4f' % (res_folder, l2), finalWeights)
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
            # trainRes[:, j + 2] = predict(train_data[0], finalWeights[:, j], bias)
            trainRes[:, j + 2] = predict_floatBias(train_data[0], finalWeights[:, j])
        trainRes = pd.DataFrame(trainRes, columns=colnames)
        trainRes.to_csv('rollTraining/%s/predictions/l2-%.4f/trainPreds-%s.csv' % (res_folder, l2, markets[i]),
                        index=False)
        test_data = test_dict[markets[i]]
        testLen = test_data[0].shape[0]
        testRes = np.zeros((testLen, 2 + len(startInds)))
        testRes[:, 0] = test_data[3]
        testRes[:, 1] = test_data[2]
        for j in range(len(startInds)):
            # testRes[:, j + 2] = predict(test_data[0], finalWeights[:, j], bias)
            testRes[:, j + 2] = predict_floatBias(test_data[0], finalWeights[:, j])
        testRes = pd.DataFrame(testRes, columns=colnames)
        testRes.to_csv('rollTraining/%s/predictions/l2-%.4f/testPreds-%s.csv' % (res_folder, l2, markets[i]),
                       index=False)

if __name__ == '__main__':
    l2_grid = np.linspace(0, 10, 101)
    pool = multiprocessing.Pool(processes = 4)

    for l2 in l2_grid:
        p = pool.apply_async(GC_parallel, args = (l2,))
    pool.close()
    pool.join()
#
#
