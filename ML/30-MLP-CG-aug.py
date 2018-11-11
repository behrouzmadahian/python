from scipy import optimize
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import initializers, os, dataProcessing, copy, multiprocessing, time, losses_and_metrics, shutil

'''An MLP using tensorflow. 
Used data: 5 market data: SQ, MQ, NQ, DQ, RN
'''
markets = ('AX', 'DA', 'IX', 'MI', 'MW','OX', 'SA', 'ZX')
MYcolors = ['blue', 'red', 'green', 'cyan', 'gold', 'magenta', 'yellow', 'lightCoral']

# data parameters
Y_toUse = 1  # 1: Binary Y, 2:1-day return
lookback = 10
lookahead = 1
#valFrac = 0.1
# training parameters:
# network parameters:
hidden1_size = 5
hidden2_size = 5
total_params = 2*lookback * hidden1_size +hidden1_size*hidden2_size+hidden2_size + hidden1_size+ hidden2_size+ 1

# loading data
datadir = 'C:/behrouz/Projects/MLP-EUEquities-CLASSIFICATION/data/%s_Commision-and-Slippage-limits-0.25.csv'

trans_data = pd.read_csv('C:/behrouz/Projects/MLP-EUEquities-CLASSIFICATION/TransactionCosts.csv')
trans_data = trans_data.values
transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))

def sharpeLoss(outP, return_1day):
    outP.flatten()
    tmp = outP * return_1day
    mean = np.mean(tmp)
    sd = np.std(tmp)
    neg_sharpe = -1 * mean*math.sqrt(250) / sd
    #print('Sharpe=', -1*neg_sharpe)
    return neg_sharpe

def glorot_normal_weight_initializer(shape):
    ''' Use for tanh activation  Glorot et al. 2012'''
    initial = np.random.randn(shape[0], shape[1])* np.sqrt(3. / (shape[0] + shape[1]))
    return initial
def weight_flatten(weights, biases):
    ''' parameters must be fed in using 1D vector'''
    weight_keys = ['h1','h2', 'out']
    b_keys = ['b1','b2', 'out']
    w = weights[weight_keys[0]].flatten()
    b = biases[b_keys[0]]
    for item in weight_keys[1:]:
        w = np.append(w, weights[item].flatten(), axis = 0)
    for item in b_keys[1:]:
        b = np.append(b, biases[item].flatten(), axis = 0)
    flattened_params = np.append(w,b, axis = 0)
    return flattened_params

def weight_shape(flat_params, lookback, size1, size2):
    '''returns weights and biases in a dict!'''
    weights = {}; biases = {}
    weights['h1'] = flat_params[:2*lookback*size1].reshape((2*lookback, size1))
    weights['h2'] = flat_params[2*lookback*size1: 2*lookback*size1+ size1*size2].reshape((size1, size2))
    weights['out'] = flat_params[2*lookback*size1+ size1*size2 : 2*lookback*size1+ size1*size2+ size2]
    W_end = 2*lookback*size1+ size1*size2+ size2
    biases['b1']  = flat_params[W_end: W_end+size1]
    biases['b2'] = flat_params[W_end+size1: W_end+size1 + size2]
    biases['out'] = flat_params[W_end+size1 + size2]
    return weights, biases


def MLP(flatParmas, x, rets, l2Reg, objectiveFunc):
    weights, biases = weight_shape(flatParmas, lookback, hidden1_size, hidden2_size)

    layer_1 = np.matmul(x, weights['h1'])+ biases['b1']
    layer_1 = np.tanh(layer_1)
    layer_2 = np.matmul(layer_1, weights['h2']) + biases['b2']
    layer_2 = np.tanh(layer_2)
    output = np.tanh(np.matmul(layer_2, weights['out']) + biases['out'])
    loss = sharpeLoss(output, rets)
    totWeights = len(flatParmas)
    l2Loss = sum([p**2 for p in flatParmas])
    l2Loss = l2Reg * l2Loss /totWeights
    totalLoss = loss + l2Loss
    return totalLoss


def MLP_predict(x, flatParmas):
    weights, biases = weight_shape(flatParmas, lookback, hidden1_size, hidden2_size)

    layer_1 = np.matmul(x, weights['h1'])+ biases['b1']
    layer_1 = np.tanh(layer_1)
    layer_2 = np.matmul(layer_1, weights['h2']) + biases['b2']
    layer_2 = np.tanh(layer_2)
    output = np.tanh(np.matmul(layer_2, weights['out']) + biases['out'])
    return output

def myparralelFunc(random_start_indicies, l2Reg, results_path):

    train_dict ={}
    test_dict ={}
    aug_multipliers = [2]
    aug_multipliers_val = [2]
    for i in range(len(markets)):
        data = pd.read_csv(datadir % markets[i])
        # Make sure we get data from all  markets on exact common dates
        curr_market_data = \
            dataProcessing.time_series_toMatrix(data, 20070418, lookback, lookahead)
        # print(markets[i], curr_market_data_aug[0].shape)
        if i ==0:
            trainX, trainY, trainRetY = curr_market_data[:3]
            testX, testY, testRet = curr_market_data[4:7]
        else:
            trainX = np.append(trainX, curr_market_data[0], axis =0)
            trainY = np.append(trainY, curr_market_data[1], axis =0)
            trainRetY = np.append(trainRetY, curr_market_data[2], axis =0)
            testX = np.append(testX, curr_market_data[4], axis=0)
            testY = np.append(testY, curr_market_data[5], axis=0)
            testRet = np.append(testRet, curr_market_data[6], axis=0)

    for i in range(len(markets)):
        data = pd.read_csv(datadir % markets[i])
        for aug in aug_multipliers:
            curr_market_data_aug = dataProcessing.time_series_toMatrix_AUG(data, 20070418, lookback, lookahead, aug)
            curr_market_data_aug_toTrain = dataProcessing.time_series_toMatrix_AUG(data, 20070418, lookback, lookahead, aug)

           # print(markets[i], curr_market_data_aug[0].shape)
            if i==0 and aug == aug_multipliers[0]:
                trainX1, trainY1, retY1 = curr_market_data_aug[:3]
                trainX11, trainY11, trainRetY11 = curr_market_data_aug_toTrain[:3]

            else:
                trainX1 = np.append(trainX1, curr_market_data_aug[0], axis = 0)
                trainY1 = np.append(trainY1, curr_market_data_aug[1], axis = 0)
                retY1 = np.append(retY1, curr_market_data_aug[2], axis = 0)
                trainX11 = np.append(trainX11, curr_market_data_aug_toTrain[0], axis=0)
                trainY11 = np.append(trainY11, curr_market_data_aug_toTrain[1], axis=0)
                trainRetY11 = np.append(trainRetY11, curr_market_data_aug_toTrain[2], axis=0)

        for aug in aug_multipliers_val:
            val_curr_market_data_aug = dataProcessing.time_series_toMatrix_AUG(data, 20070418, lookback, lookahead, aug)
            if i==0 and aug == aug_multipliers_val[0]:
                validX1, validY1, validretY1 = val_curr_market_data_aug[:3]
            else:
                validX1 = np.append(validX1, val_curr_market_data_aug[0], axis=0)
                validY1 = np.append(validY1, val_curr_market_data_aug[1], axis=0)
                validretY1 = np.append(validretY1, val_curr_market_data_aug[2], axis=0)

    print(trainX.shape, trainY.shape)
    train_dict['TrainPurturb'] = copy.deepcopy([trainX1, trainY1, retY1])
    train_dict['Train'] = copy.deepcopy((trainX, trainY, trainRetY))
    test_dict['Test'] = copy.deepcopy((testX, testY, testRet))
    print('Shape of training data=', trainX.shape, trainY.shape, trainRetY.shape)
    print('Shape of training perturbed data=', trainX1.shape, trainY1.shape)
    print('Shape of validation data=', validX1.shape, validY1.shape, validretY1.shape)
    print('Shape of test data:', testX.shape, testRet.shape)
    finalWeights = np.zeros((total_params, len(random_start_indicies)))
    for R in random_start_indicies:
        print('RUN %d optimization begins..' % R)
        ##################################################
        # a = np.arange(trainX.shape[0])
        # np.random.shuffle(a)
        # trainX = trainX[a,:]
        # trainY = trainY[a]
        # trainRet = retY[a]
        # validInd = int(trainX.shape[0] * valFrac)
        # validX, validY,validRet = trainX[:validInd,:], trainY[:validInd], trainRet[:validInd]
        # trainX, trainY = trainX[validInd:,:], trainY[validInd:]
        ###################################################
        #trainX = np.append(trainX,trainX11, axis = 0)
        #trainY = np.append(trainY,trainY11, axis = 0)
        #trainRetY = np.append(trainRetY,trainRetY11, axis = 0)
        ############################################
        #trainX = trainX11
        #trainY = trainY11
        #trainRetY = trainRetY11
        ##################################################
        weights = {
            'h1': glorot_normal_weight_initializer([lookback*2, hidden1_size]),
            'h2': glorot_normal_weight_initializer([hidden1_size, hidden2_size]),
            'out': glorot_normal_weight_initializer([hidden2_size, 1])
        }
        biases = {
            'b1': np.zeros(hidden1_size),
            'b2': np.zeros(hidden2_size),
            'out': np.zeros(1)
        }
        flat_params = weight_flatten(weights, biases)
        print('Length of Flattened parameters=', len(flat_params))
        # retall: returns the solution after each iteration
        finalWeights[:, R-1] = optimize.fmin_cg(MLP, flat_params, fprime=None,retall=False,
                                              args=(trainX, trainY,  l2Reg, sharpeLoss))
    np.save(results_path +'/'+ str(l2Reg)+'/FinalWeights-l2-%.2f' % l2Reg, finalWeights)
    # prediction and saving to file
    if not os.path.exists(results_path +'/'+ str(l2Reg)+'/predictions'):
        os.makedirs(results_path +'/'+ str(l2Reg)+'/predictions')
    #finalWeights = np.load(results_path +'/'+ str(l2Reg)+'/FinalWeights-l2-%.2f'%l2Reg+'.npy')
    for i in range(len(markets)):
        data = pd.read_csv(datadir % markets[i])
        # Make sure we get data from all  markets on exact common dates
        curr_market_data = dataProcessing.time_series_toMatrix(data, 20070418, lookback, lookahead)
        trainX = curr_market_data[0]
        testX =  curr_market_data[4]
        trainMat = np.zeros((trainX.shape[0], len(random_start_indicies)+2))
        testMat = np.zeros((testX.shape[0], len(random_start_indicies)+2))
        trainMat[:,0] =curr_market_data[3]
        trainMat[:,1] =curr_market_data[2]
        testMat[:,0]=curr_market_data[-1]
        testMat[:,1] = curr_market_data[-2]
        for R in range(len(random_start_indicies)):
            trainMat[:,R+2] = MLP_predict(trainX, finalWeights[:,R])
            testMat[:,R+2] = MLP_predict(testX, finalWeights[:,R])
        predsCols = ['dtStart', '%s-y-true' % markets[i]]
        predsCols.extend(['%s-pred%d' % (markets[i], j) for j in range(1, len(random_start_indicies) + 1, 1)])
        market_trainPred = pd.DataFrame(trainMat, columns=predsCols)
        market_trainPred.to_csv(results_path + str(l2Reg) + '/' + 'predictions/' + '%s-trainPreds.csv'%markets[i],
                                index=False)

        market_testPred = pd.DataFrame(testMat, columns=predsCols)
        market_testPred.to_csv(results_path +str(l2Reg)+ '/predictions/' + '%s-testPreds.csv'%markets[i], index=False)


if __name__ == '__main__':
    results_path = 'C:/behrouz/Projects/MLP-EUEquities_conjugateGrad/run1/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    random_start_indicies = np.arange(1, 101, 1)[:3]
    #l2_grid, lr_grid = pd.read_csv(results_path+'lrTuning/'+'best-Learning-rate-l2Grid.csv').values
    l2_grid = np.linspace(10,100,10)[:1]

    # # #results_loss,results_sharpe,seeds
    t1 = time.time()
    processes = []
    for l2 in l2_grid:
        if not os.path.exists(results_path +'/'+ str(l2)):
            os.makedirs(results_path +'/'+ str(l2))
            # train_parallel(0.5, 0.001, results_path, objective, train_data, test_data, transCost_dict)
        p = multiprocessing.Process(target=myparralelFunc, args=(random_start_indicies,l2, results_path))
        p.start()
        processes.append(p)
    results = []
    for p in processes:
        print('Running process = ', p)
        p.join()

    t2 = time.time()
    print('TOTAL ELAPSED TIME FOR %d runs='%len(random_start_indicies), np.round((t2 -t1)/ 60., 2))
