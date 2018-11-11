import tensorflow as tf
import numpy as np
import initializers
import os
import pandas as pd
import dataProcessing
import multiprocessing
import time
import shutil
import math

'''An MLP using tensorflow. 
Used data:  SQ
'''
def sharpeLoss(outP, return_1day):
    ''' Treats all data as one market'''

    outP = tf.reshape(outP, [-1])
    tmp = tf.multiply(outP, return_1day)
    mean, var = tf.nn.moments(tmp, [0])
    sd = tf.sqrt(var)
    neg_sharpe = -1 * mean *math.sqrt(250)/ sd

    return neg_sharpe


def MLP_1layer(x, y,  weights, biases,  curr_optimizer,
               objective, activation, l2Reg , l2RegOutput,
               init_lr, decay_steps, decay_rate):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)
    output = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
    sharpe_loss = objective(output, y)

    # l2 regularization loss:
    l2Loss = l2Reg * tf.nn.l2_loss(weights['h1']) + l2RegOutput * tf.nn.l2_loss(weights['out'])
    sharpe_plus_l2_loss = sharpe_loss + l2Loss

    # cost function and optimization
    global_step = tf.Variable(0, trainable = False)
    # cost function and optimization
    learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate)
    train_op = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss, global_step)

    ema = tf.train.ExponentialMovingAverage(decay = 0.999)
    with tf.control_dependencies([train_op]):
        train_op_new = ema.apply(tf.trainable_variables())

    return train_op_new, output, sharpe_loss, sharpe_plus_l2_loss, l2Loss, ema

def MLP_1layer_fixedBiasOut_sigmoid(x, y,  weights, biases,  curr_optimizer,
                            objective, activation, l2Reg , l2RegOutput,
                           init_lr, decay_steps, decay_rate):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)

    output = tf.add(tf.matmul(layer_1, weights['out']), 0.1)
    output = tf.nn.sigmoid(output)
    sharpe_loss = objective(output - 0.5, y)
    classification_loss = ((tf.sign(y) + 1) / 2. ) * tf.log(output) + (1. - (tf.sign(y) + 1) / 2. ) * tf.log(1 - output)
    classification_loss *= -1
    classification_loss =  tf.reduce_mean(classification_loss *1.)

    # l2 regularization loss:

    l2Loss = l2Reg * tf.nn.l2_loss(weights['h1']) + l2RegOutput * tf.nn.l2_loss(weights['out'])

    sharpe_plus_l2_loss = sharpe_loss + l2Loss
    combined_loss = sharpe_plus_l2_loss + classification_loss

    # cost function and optimization
    global_step = tf.Variable(0, trainable = False)
    # cost function and optimization
    learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate)
    train_op = curr_optimizer(learning_rate=learning_rate).minimize(combined_loss, global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    with tf.control_dependencies([train_op]):
        train_op_new = ema.apply(tf.trainable_variables())

    return train_op_new, output, sharpe_loss, classification_loss

def MLP_1layerFixedOutBias(x, y,  weights, biases,  curr_optimizer,
                           objective, activation, l2Reg , l2RegOutput,
                           init_lr, decay_steps, decay_rate):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)

    output = tf.add(tf.matmul(layer_1, weights['out']) , 0.1)

    model_loss = objective(output, y)

    # l2 regularization loss:
    l2Loss = l2Reg * tf.nn.l2_loss(weights['h1']) + l2RegOutput * tf.nn.l2_loss(weights['out'])

    sharpe_plus_l2_loss = model_loss + l2Loss

    global_step =tf.Variable(0, trainable = False )
    # cost function and optimization
    learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate )
    train_op = curr_optimizer(learning_rate = learning_rate).minimize(sharpe_plus_l2_loss, global_step)

    ema =tf.train.ExponentialMovingAverage(decay = 0.999)
    with tf.control_dependencies([train_op]):
        train_op_new = ema.apply(tf.trainable_variables())

    return train_op_new, output, model_loss, sharpe_plus_l2_loss, l2Loss, ema

def MLP_1layerFixedOutBias_clipNorm(x, y,  weights, biases,  curr_optimizer,
                                    objective, l2Reg, l2RegOutput, init_lr,
                                    decay_steps, decay_rate, activation=tf.nn.tanh):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)

    output_weights = tf.clip_by_norm(weights['out'], 1, axes = None)
    output = tf.add(tf.matmul(layer_1, output_weights) , 0.1)

    model_loss = objective(output, y)

    l2Loss = l2Reg * tf.nn.l2_loss(weights['h1']) + l2RegOutput * tf.nn.l2_loss(weights['out'])

    sharpe_plus_l2_loss = model_loss + l2Loss

    # cost function and optimization
    global_step = tf.Variable(0, trainable=False)

    # cost function and optimization
    learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate)
    train_op = curr_optimizer(learning_rate = learning_rate).minimize(sharpe_plus_l2_loss, global_step)

    ema = tf.train.ExponentialMovingAverage(decay = 0.999)
    with tf.control_dependencies([train_op]):
        train_op_new = ema.apply(tf.trainable_variables())

    return train_op_new, output, model_loss, sharpe_plus_l2_loss, l2Loss, ema

def myMainFunc(market, hidden_size, l2Reg, Files_Folder, learning_rate_grid, epoch_grid):

    # objective function and optimizer
    objective = sharpeLoss
    curr_optimizer = tf.train.AdamOptimizer

    # data parameters
    lookback = 30
    lookahead = 1
    rolling_sd_window = 100
    network_activation = tf.nn.tanh
    test_start_date = 20070418
    random_start_indicies = np.arange(1, 11, 1)
    nRandom_start = len(random_start_indicies)
    batch_size = 100

    # loading data
    datadir = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-SQ-only/data/%s_Commision-and-Slippage-limits.csv'

    data = pd.read_csv(datadir % market)

    curr_market_data = \
        dataProcessing.time_series_toMatrix(data,  test_start_date,
                                            lookback=lookback,
                                            look_ahead=lookahead, sd_window=rolling_sd_window)

    train = curr_market_data[:4]
    test = curr_market_data[4:]

    total_batches = train[0].shape[0] // batch_size
    decay_steps = total_batches
    decay_rate = 0.99

    for LR in learning_rate_grid:

        for training_epochs in epoch_grid:

            market_trainPred = np.zeros((train[0].shape[0], nRandom_start + 2))
            market_testPred = np.zeros((test[0].shape[0], nRandom_start + 2))
            total_loss_matrix = np.zeros((nRandom_start, 6))


            market_trainPred[:, 0] = train[3]  # date
            market_trainPred[:, 1] = train[2]  # 1 day return
            market_testPred[:, 0] = test[3]
            market_testPred[:, 1] = test[2]

            for R in range(len(random_start_indicies)):
                print('Hidden Size =', hidden_size, 'Learning rate=', LR,
                      'TrainingEpochs=', training_epochs, 'L2 Reg=', l2Reg, 'Random Start=', R)

                weights = {

                     'h1': initializers.xavier_from_tf_initializer([lookback, hidden_size], name='W_1'),
                     'out': initializers.xavier_from_tf_initializer([hidden_size, 1], name='W_out')
                         }
                biases = {

                    'b1': initializers.bias_initializer([hidden_size], name='B_1')
                    #, 'out': initializers.bias_initializer([1], name='B_out')
                        }
                # placeholders
                x = tf.placeholder(tf.float32, [None, lookback])
                y = tf.placeholder(tf.float32, [None])

                optimizer, output, sharpe_loss, sharpe_plus_l2_loss, l2Loss, ema = \
                    MLP_1layerFixedOutBias(x, y, weights, biases, curr_optimizer,
                                           objective, network_activation, l2Reg, l2Reg,
                                           LR, decay_steps, decay_rate)

                # Getting EMA var names:
                ema_dict ={}
                for var in tf.trainable_variables():
                    ema_var_name = ema.average_name(var)
                     ema_dict[ema_var_name] = var
                saver = tf.train.Saver(ema_dict)

                saver = tf.train.Saver()

                with tf.Session() as sess:
                    try:
                        source_model_loc = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/' \
                                           '/tf-SQ-only/%s/' % Files_Folder

                        saver.restore(sess,
                                      source_model_loc + 'MLP-checkpointFiles/' + str(R + 1) +
                                      '/run%d-s-%d-LR-%.6f-epoch-%d-l2-%.5f.ckpt'
                                      % (R + 1, hidden_size, LR, training_epochs, l2Reg))
                        #print(weights['h1'].eval())

                    except IOError:
                        print('Could not find the checkpoint file, filling with previous model..')

                    trainPred, train_loss, train_total_loss, trainL2_loss = \
                        sess.run([output, sharpe_loss, sharpe_plus_l2_loss, l2Loss],
                                 feed_dict={x: train[0], y: train[1]})
                    trainPred = trainPred[:, 0]

                    testPred, test_loss, test_total_loss, test_l2_loss = \
                        sess.run([output, sharpe_loss, sharpe_plus_l2_loss, l2Loss],
                                 feed_dict={x: test[0], y: test[1]})

                    testPred = testPred[:, 0]

                    market_trainPred[:, R + 2] = trainPred
                    market_testPred[:, R + 2] = testPred
                    total_loss_matrix[R, 0:3] = train_loss, trainL2_loss, train_total_loss
                    total_loss_matrix[R, 3:] = test_loss, test_l2_loss, test_total_loss

                tf.reset_default_graph()

            total_loss_matrix_colnames = ['train_loss', 'train_l2_loss', 'train_total_loss',
                                          'test_loss', 'test_l2_loss', 'test_total_loss']
            total_loss_matrix = pd.DataFrame(total_loss_matrix, columns=total_loss_matrix_colnames)

            total_loss_matrix.to_csv('./Results/%s-loss-s-%d-LR-%.6f-epoch-%d-l2-%.5f.csv'
                                     % (market, hidden_size, LR, training_epochs, l2Reg),
                                     index=False)

            predsCols = ['dtStart', '%s-y-true' % market]
            predsCols.extend(['%s-pred%d' % (market, j) for j in range(1, nRandom_start + 1, 1)])

            market_trainPred = pd.DataFrame(market_trainPred, columns=predsCols)
            market_trainPred.to_csv('./Results/%s-trainPreds-s1-%d-LR-%.6f-epoch-%d-l2-%.5f.csv'
                                    % (market, hidden_size, LR, training_epochs, l2Reg),
                                    index=False)

            market_testPred = pd.DataFrame(market_testPred, columns=predsCols)
            market_testPred.to_csv('./Results/%s-testPreds-s1-%d-LR-%.6f-epoch-%d-l2-%.5f.csv'
                                   % (market, hidden_size, LR, training_epochs, l2Reg),
                                   index=False)

if __name__ == '__main__':

    results_path = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-SQ-only/opt_paradigm_comparison/'
    results_folder = 'l2Only-bConst-EMA'
    os.chdir(results_path + results_folder)

    t1 = time.time()
    hidden_size_grid = [3]
    l2_grid = np.linspace(0, 1, 11)
    learning_rate_grid = [0.001]
    epoch_grid = [300]

    market = 'SQ'
    processes = []

    for l2Reg in l2_grid:
        for hidden_size in hidden_size_grid:

            print('Hidden Sizes=', hidden_size, 'L2Reg =', l2Reg)
            p = multiprocessing.Process(target = myMainFunc, args = (market, hidden_size, l2Reg,
                                                                     results_folder, learning_rate_grid, epoch_grid))
            p.start()
            processes.append(p)

        for p in processes:
            print('Running process = ', p)
            p.join()

        processes = []

    t2 = time.time()
    random_start_indicies = np.arange(1, 11, 1)
    print('TOTAL ELAPSED TIME FOR %d runs=' % len(random_start_indicies), np.round((t2 - t1) / 60., 2))
