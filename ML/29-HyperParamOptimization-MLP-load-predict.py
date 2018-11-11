import tensorflow as tf
import numpy as np
import initializers
import os
import pandas as pd
import dataProcessing
import math
import copy
import losses_and_metrics
from matplotlib import pyplot as plt
from Models import MLP
import multiprocessing
import time
import shutil
from matplotlib import colors
'''An MLP using tensorflow. 
Used data: 5 market data: SQ, MQ, NQ, DQ, RN
'''
markets = ('SQ', 'MQ', 'NQ', 'DQ', 'RN')
MYcolors = [colors.cnames[name]  for name in colors.cnames ]

# objective function
#objective = losses_and_metrics.sum_sharpeLoss
objective = losses_and_metrics.sharpeLoss
#objective = losses_and_metrics.basket_trading_pos_size_sharpeLoss

curr_optimizer = tf.train.AdamOptimizer
#curr_optimizer = tf.train.RMSPropOptimizer

optim = 'Adam'
#optim = 'RMSprop'

#results_folder = 'dropout-sharpe-sumMarket-loss'
results_folder = 'sharpeloss1'
#results_folder = 'dropout-sharpe-basketloss'

# data parameters
Y_toUse = 1  # 1: scaled return, 2:1-day return
lookback = 30
lookahead = 1
rolling_sd_window = 100

# training parameters:


batch_size = 100


# network parameters:
network_activation = tf.nn.tanh
input_feats = lookback
test_start_date = 20070418
random_start_indicies = np.arange(1, 21, 1)
nRandom_start = 20
# hidden_size_grid = [5, 10, 15, 20]
# learning_rate_grid = [0.001, 0.0005, 0.0001, 0.00005]
# epoch_grid = [50, 100, 150, 200, 300]
# l2_grid = [0, 0.01, 0.1, 1, 5]
dropout = 1.


hidden_size_grid = [5, 10, 15, 20]
learning_rate_grid = [0.001, 0.0005, 0.0001, 0.00005]
epoch_grid = [50, 100, 150, 200, 300]
l2_grid = [0, 0.01, 0.1, 1, 5]
valid_frac = 0.2

# loading data
datadir = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/hyper-param-optimization/' \
          'tf-hyperParam-opt/data/%s_Commision-and-Slippage-limits.csv'

results_path = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/hyper-param-optimization/tf_hypeOpt-files-series/%s'


os.chdir(results_path% results_folder)


#get the common dates and then merge each data making sure they have common dates:
data = pd.read_csv(datadir % markets[0])
for i in range(1, len(markets), 1):
    data1 = pd.read_csv(datadir % markets[i])
    data = pd.merge(data, data1, on = 'dtStart', how = 'inner')
dates = data[['dtStart']]


trans_data = pd.read_csv('C:/behrouz/Projects/DailyModels_new/NeuralNet/TransactionCosts.csv')
trans_data = trans_data.values
transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))

test_start_ind = int( np.where(dates.values == test_start_date)[0] ) - rolling_sd_window - lookback - lookahead
inds = np.arange(test_start_ind)
valid_inds = pd.read_csv('Validation_indicies.csv').values
valid_inds = valid_inds.flatten()

print(valid_inds.shape)

train_inds = [ i for i in inds if i not in valid_inds]

# common dates
data = pd.read_csv(datadir % markets[0])
for i in range(1, len(markets), 1):
    data1 = pd.read_csv(datadir % markets[i])
    data = pd.merge(data, data1, on = 'dtStart', how = 'inner')

dates = data[['dtStart']]


for market in range(len(markets)):

    # get the data and then restore the model
    data = pd.read_csv(datadir % markets[market])
    # Make sure we get data from all  markets on exact common dates
    data = pd.merge(data, dates, on='dtStart', how='inner')
    curr_market_data = \
        dataProcessing.time_series_toMatrix(data, train_inds, valid_inds, 20070418,
                                            lookback=lookback,
                                            look_ahead=lookahead, sd_window=rolling_sd_window)

    train = curr_market_data[:4]
    validation = curr_market_data[4:8]
    test = curr_market_data[8:]

    print('Testing model on Market=', markets[market])
    trans_cost = transCost_dict[markets[market]]
    print(trans_cost)



    print(validation[0].shape)
    print(train[0].shape)



    for hidden_size in hidden_size_grid:

        for LR in learning_rate_grid:

            for training_epochs in epoch_grid:

                for l2Reg in l2_grid:

                    market_trainPred = np.zeros((train[0].shape[0], nRandom_start + 2))
                    market_validPred = np.zeros((validation[0].shape[0], nRandom_start + 2))
                    market_testPred = np.zeros((test[0].shape[0], nRandom_start + 2))

                    market_trainPred[:, 0] = train[3]  # date
                    market_trainPred[:, 1] = train[2]  # 1 day return
                    market_validPred[:, 0] = validation[3]
                    market_validPred[:, 1] = validation[2]
                    market_testPred[:, 0] = test[3]
                    market_testPred[:, 1] = test[2]


                    for R in range(len(random_start_indicies)):

                        print('Hidden Size =', hidden_size, 'Learning rate=', LR,
                              'TrainingEpochs=', training_epochs, 'L2 Reg=', l2Reg, 'Random Start=', R)

                        weights = {

                            'h1': initializers.xavier_from_tf_initializer([lookback, hidden_size], name='W_1'),
                            'h2': initializers.xavier_from_tf_initializer([hidden_size, hidden_size], name='W_2'),
                            'out': initializers.xavier_from_tf_initializer([hidden_size, 1], name='W_out')
                        }

                        biases = {

                            'b1': initializers.bias_initializer([hidden_size], name='B_1'),
                            'b2': initializers.bias_initializer([hidden_size], name='B_2'),
                            'out': initializers.bias_initializer([1], name='B_out')
                        }

                        x = tf.placeholder(tf.float32, [None, input_feats])
                        y = tf.placeholder(tf.float32, [None])
                        learning_rate = tf.placeholder(tf.float32)
                        keep_prob = tf.placeholder(tf.float32)

                        optimizer, output, sharpe_plus_l2_loss = \
                            MLP(x, y, weights, biases, keep_prob, curr_optimizer, learning_rate, objective,
                                batch_size=batch_size, markets=markets,
                                activation=network_activation, l2Reg=l2Reg, l2RegOutput=l2Reg * 1.,
                                l2Reg_biases=l2Reg * 1.)

                        saver = tf.train.Saver()

                        with tf.Session() as sess:

                            # init = tf.global_variables_initializer()
                            saver.restore(sess, './MLP-checkpointFiles/run%d-s-%d-LR-%.6f-epoch-%d-l2-%.5f.ckpt'
                                          % (R + 1, hidden_size, LR, training_epochs, l2Reg))

                            trainPred = sess.run(output, feed_dict={x: train[0], keep_prob: 1.})[:, 0]
                            validPred = sess.run(output, feed_dict={x: validation[0], keep_prob: 1.})[:, 0]

                            testPred = sess.run(output, feed_dict={x: test[0], keep_prob: 1.})[:, 0]

                            print(trainPred.shape)
                            market_trainPred[:, R + 2] = trainPred
                            market_validPred[:, R + 2] = validPred
                            market_testPred[:, R + 2] = testPred
                        tf.reset_default_graph()

                    predsCols = ['dtStart', '%s-y-true' % markets[market]]
                    predsCols.extend(['%s-pred%d' % (markets[market], j) for j in range(1, nRandom_start + 1, 1)])

                    market_trainPred = pd.DataFrame(market_trainPred, columns=predsCols)
                    market_trainPred.to_csv('./Results/%s-trainPreds-run%d-s-%d-LR-%.6f-epoch-%d-l2-%.5f.csv'
                                                % (markets[market],R + 1, hidden_size, LR, training_epochs, l2Reg),
                                                index=False)

                    market_validPred = pd.DataFrame(market_validPred, columns=predsCols)
                    market_validPred.to_csv('./Results/%s-validPreds-run%d-s-%d-LR-%.6f-epoch-%d-l2-%.5f.csv'
                                                 % (markets[market],R + 1, hidden_size, LR, training_epochs, l2Reg),
                                                index=False)

                    market_testPred = pd.DataFrame(market_testPred, columns=predsCols)
                    market_testPred.to_csv('./Results/%s-testPreds-run%d-s-%d-LR-%.6f-epoch-%d-l2-%.5f.csv'
                                               % (markets[market],R + 1, hidden_size, LR, training_epochs, l2Reg),
                                               index = False)
