import tensorflow as tf
import numpy as np
import initializers
import os
import pandas as pd
import dataProcessing
import math
import copy
import random
import multiprocessing
'''An MLP using tensorflow. 
Used data: 5 market data: SQ, MQ, NQ, DQ, RN
'''
#########################################################################################################
markets = ('SQ', 'MQ', 'NQ', 'DQ', 'RN')
# objective function
curr_optimizer = tf.train.AdamOptimizer
# training parameters:
training_epochs = 1000
# network parameters:
network_activation = tf.nn.tanh
hidden1_size = 5
#hidden2_size = 30
dropout = 1.0
stationary_epochs = 500
decya_fact = 0.998
date_train_end = 20070418
features = 2
################################################################
def sharpeLoss(outPutMulY):
    outPutMulY = tf.reshape(outPutMulY, [-1])
    print('Shape of Flattened output=', outPutMulY.get_shape())
    mean, var = tf.nn.moments(outPutMulY, [0])
    sd = tf.sqrt(var)
    sharpe = mean * math.sqrt(250.) / sd
    sharpe *= -1
    return sharpe

def basket_sharpeLoss(outPutMulY):
    ret_seq = tf.reduce_sum(outPutMulY, axis =0)
    print('shape of summed sequence=', ret_seq.get_shape())
    mean, var = tf.nn.moments(ret_seq, [0])
    sd = tf.sqrt(var)
    sharpe = mean *math.sqrt(250.) / sd
    sharpe *= -1
    return sharpe
##################################################################
def vanilla_LSTM(x, y, weights, biases, keep_prob ,curr_optimizer, learning_rate, objective, markets,
                activation, l2Reg, hidden_size, n_layers):
    if n_layers > 1:
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, activation=activation)
            # if applying dropout after each layer is NOT desired, remove this and
            # maybe apply drop out to the output of the stack!
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
            return cell

        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
        lstm_out, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, activation=activation)
        # output of dynamic_rnn is  tensor of shape: [batch_size, timesteps, hidden_size]
        lstm_out, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    lstm_out = tf.reshape(lstm_out, [-1, hidden1_size])
    lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_prob)

    print('Shape of reshaped LSTM output and output weights', lstm_out.get_shape(), weights['out'].shape)
    output = tf.matmul(lstm_out, weights['out']) + biases['out']
    output = tf.reshape(output, [len(markets), -1])
    output = tf.nn.tanh(output)

    print('Shape of output in correct format for Sharpe:', output.get_shape(), y.shape)
    outPutMulY = tf.multiply(output, y)
    print('Out* y shape=', outPutMulY.shape)
    sharpe_loss = objective(outPutMulY)
    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg
    l2Loss = sum(tf.nn.l2_loss(curr_var) for curr_var in tf.trainable_variables() if not 'B_' in curr_var.name)

    l2Loss = l2Reg * l2Loss
    sharpe_plus_l2_loss = sharpe_loss + l2Loss
    # cost function and optimization
    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)
    return optimizer, output, sharpe_plus_l2_loss

def vanilla_RNN(x, y, weights, biases, keep_prob ,curr_optimizer, learning_rate, objective, markets,
                activation, l2Reg, hidden_size, n_layers):
    if n_layers > 1:
        def lstm_cell():
            cell = tf.contrib.rnn.BasicRNNCell(hidden_size, activation=activation)
            # if applying dropout after each layer is NOT desired, remove this and
            # maybe apply drop out to the output of the stack!
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
            return cell

        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
        lstm_out, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(hidden_size, activation=activation)
        # output of dynamic_rnn is  tensor of shape: [batch_size, timesteps, hidden_size]
        lstm_out, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    lstm_out = tf.reshape(lstm_out, [-1, hidden1_size])
    lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_prob)

    print('Shape of reshaped LSTM output and output weights', lstm_out.get_shape(), weights['out'].shape)
    output = tf.matmul(lstm_out, weights['out']) + biases['out']
    output = tf.reshape(output, [len(markets), -1])
    output = tf.nn.tanh(output)
    print('Shape of output in correct format for Sharpe:', output.get_shape(), y.shape)

    outPutMulY = tf.multiply(output, y)
    print('Out* y shape=', outPutMulY.shape)
    sharpe_loss = objective(outPutMulY)
    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg
    l2Loss = sum(tf.nn.l2_loss(curr_var) for curr_var in tf.trainable_variables() if not 'B_' in curr_var.name)

    l2Loss = l2Reg * l2Loss
    sharpe_plus_l2_loss = sharpe_loss + l2Loss
    # cost function and optimization
    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)
    return optimizer, output, sharpe_plus_l2_loss
###########################################################################
def vanilla_GRU(x, y, weights, biases, keep_prob ,curr_optimizer, learning_rate, objective, markets,
                activation, l2Reg, hidden_size, n_layers):
    if n_layers > 1:
        def lstm_cell():
            cell = tf.contrib.rnn.GRUCell(hidden_size, activation=activation)
            # if applying dropout after each layer is NOT desired, remove this and
            # maybe apply drop out to the output of the stack!
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
            return cell

        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
        lstm_out, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
    else:
        cell = tf.contrib.rnn.GRUCell(hidden_size, activation=activation)
        # output of dynamic_rnn is  tensor of shape: [batch_size, timesteps, hidden_size]
        lstm_out, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    lstm_out = tf.reshape(lstm_out, [-1, hidden1_size])
    lstm_out = tf.nn.dropout(lstm_out, keep_prob=keep_prob)

    print('Shape of reshaped LSTM output and output weights', lstm_out.get_shape(), weights['out'].shape)
    output = tf.matmul(lstm_out, weights['out']) + biases['out']
    output = tf.reshape(output, [len(markets), -1])
    output = tf.nn.tanh(output)
    #output = tf.
    print('Shape of output in correct format for Sharpe:', output.get_shape(), y.shape)
    #outNorm  = tf.norm(output, axis =1) /  5 # i want norm of each vector of signals! to be 5!!
    # print('Shape of Norm vector=', outNorm.get_shape(), tf.reshape(outNorm, [-1, 1]).get_shape())
    # output  /= tf.reshape(outNorm, [-1, 1])
    outPutMulY = tf.multiply(output, y)
    print('Out* y shape=', outPutMulY.shape)
    sharpe_loss = objective(outPutMulY)
    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg
    l2Loss = sum(tf.nn.l2_loss(curr_var) for curr_var in tf.trainable_variables() if not 'B_' in curr_var.name)

    l2Loss = l2Reg * l2Loss
    sharpe_plus_l2_loss = sharpe_loss + l2Loss
    # cost function and optimization
    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)
    return optimizer, output, sharpe_plus_l2_loss
###########################################################################
def myparralelFunc(LR_gird, l2Reg, dates, objective, results_path):
    for i in range(0, len(markets), 1):
        data = pd.read_csv(datadir % markets[i])
        # Make sure we get data from all  markets on exact common dates
        data = pd.merge(data, dates, on='dtStart', how='inner')
        curr_market_data = dataProcessing.time_series_toMatrix(data, date_train_end)
        if i == 0:
            trainX = curr_market_data[0]
            trainY = curr_market_data[1]
            train_Y_op_opD = curr_market_data[2]
            train_dates = curr_market_data[3]
            testX = curr_market_data[4]
            testY = curr_market_data[5]
            test_Y_op_opD = curr_market_data[6]
            test_dates = curr_market_data[7]
        else:
            trainX = np.append(trainX, copy.deepcopy(curr_market_data[0]), axis=0)
            trainY = np.dstack((trainY, copy.deepcopy(curr_market_data[1])))
            train_Y_op_opD = np.dstack((train_Y_op_opD, copy.deepcopy(curr_market_data[2])))
            testX = np.append(testX, copy.deepcopy(curr_market_data[4]), axis=0)
            testY = np.dstack((testY, copy.deepcopy(curr_market_data[5])))
            test_Y_op_opD = np.dstack((test_Y_op_opD, copy.deepcopy(curr_market_data[6])))

    trainY = np.transpose(trainY, [2, 1, 0])
    trainY = np.reshape(trainY, trainY.shape[:2])
    train_Y_op_opD = np.transpose(train_Y_op_opD, [2, 1, 0])
    train_Y_op_opD = np.reshape(train_Y_op_opD, train_Y_op_opD.shape[:2])
    testY = np.transpose(testY, [2, 1, 0])
    testY = np.reshape(testY, testY.shape[:2])
    test_Y_op_opD = np.transpose(test_Y_op_opD, [2, 1, 0])
    test_Y_op_opD = np.reshape(test_Y_op_opD, test_Y_op_opD.shape[:2])

    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)
    print('====')
    train_loss_mat = np.zeros((len(LR_gird), training_epochs))
    for i in range (len(LR_gird)):
        init_lr = LR_gird[i]
        random.seed(12345)
        np.random.seed(12345)
        tf.set_random_seed(12345)

        weights = {
            'out': initializers.xavier_from_tf_initializer([hidden1_size, 1], name='W_out')
        }
        biases = {
            'out': initializers.bias_initializer([1], name='B_out')
        }
        # placeholders
        x = tf.placeholder(tf.float32, [len(markets), None, features])
        y = tf.placeholder(tf.float32, [len(markets), None])
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)

        optimizer, output, sharpe_plus_l2_loss = vanilla_RNN(x, y, weights, biases, keep_prob, curr_optimizer,
                        learning_rate, objective, markets=markets, activation=tf.nn.tanh,
                        l2Reg=l2Reg, hidden_size=hidden1_size, n_layers=1)
        # initialize all tensors- to be run in Session!
        init = tf.global_variables_initializer()
        # saver for restoring the whole model graph of
        #  tensors from the  checkpoint file
        saver = tf.train.Saver()

        # launch default graph:
        with tf.Session() as sess:
            sess.run(init)
            # training cycle:
            decay_exponent = 1
            for epoch in range(training_epochs):
                if epoch >= stationary_epochs:
                    LR = init_lr * (decya_fact **decay_exponent)
                    decay_exponent += 1
                    feed_dict = {x: trainX, y: train_Y_op_opD, learning_rate: LR, keep_prob: dropout}
                else:
                    feed_dict = {x: trainX, y: train_Y_op_opD, learning_rate: init_lr, keep_prob: dropout}

                _ = sess.run(optimizer,feed_dict=feed_dict )

                train_loss = sess.run(sharpe_plus_l2_loss, feed_dict={x:trainX, y:train_Y_op_opD, keep_prob : 1.})
                print('L2 reg=',l2Reg,'Epoch=', epoch,'TrainLoss= ', train_loss)
                train_loss_mat[i, epoch] = train_loss

            saver.save(sess,results_path+ '/checkpointFiles/' + str(l2Reg)+
                                   '/checkPoint-LR-%.6f-l2-%.4f.ckpt' % (init_lr, l2Reg))

       # resetting the graph to be built again in the next iteration of for loop
        tf.reset_default_graph()
    return train_loss_mat

def learningRate_tuning_wrapper(init_lr_grid, l2Reg , return_dict, delta_thresh, dates, objective, results_path):
    if  not os.path.exists(results_path+'checkpointFiles/'+str(l2Reg)):
        os.makedirs(results_path+'checkpointFiles/'+str(l2Reg))
    iteration = 1
    train_losses =  myparralelFunc(init_lr_grid, l2Reg, dates,objective, results_path)
    train_lossesDF = np.transpose(train_losses)
    train_lossesDF = pd.DataFrame(train_lossesDF, columns=init_lr_grid)
    train_lossesDF.to_csv(results_path+'LR-tuning-losses-stage-%d-l2-%.4f.csv' % (iteration,l2Reg), index=False)
    #min_losses = np.min(train_losses, axis = 1)
    min_losses = train_losses[:, -1]
    best_loss_ind = np.argmin(min_losses)
    best_lossTraj = train_losses[best_loss_ind, :]
    best_LR = init_lr_grid[best_loss_ind]
    best_loss = min_losses[best_loss_ind]

    delta_best = np.inf
    p = 1
    print('Best LR for initial Grid=', best_LR)

    while delta_best > delta_thresh:
        losses_to_file = np.zeros((training_epochs, 3))
        step = best_LR * 0.5**p
        curr_grid = [best_LR - step,  best_LR + step]

        print('STEP=', step)
        iteration += 1
        train_losses = myparralelFunc(curr_grid, l2Reg,dates, objective, results_path)
        losses_to_file[:, 0] = train_losses [0, :]
        losses_to_file[:, 1] = best_lossTraj
        losses_to_file[:, 2] = train_losses [1, :]
        train_lossesDF = pd.DataFrame(losses_to_file, columns=[ curr_grid[0], best_LR, curr_grid[1] ])
        train_lossesDF.to_csv(results_path+'LR-tuning-losses-stage-%d-l2-%.4f.csv' % (iteration,l2Reg), index=False)

       # curr_min_losses = np.min(train_losses, axis=1)
        curr_min_losses = train_losses[:, -1]
        curr_min_ind  = np.argmin(curr_min_losses)
        curr_min = np.min(curr_min_losses)

        delta_best = np.abs( (best_loss - curr_min) / best_loss )
        if curr_min < best_loss:
            best_loss = curr_min
            best_LR = curr_grid[curr_min_ind]
            best_lossTraj = train_losses[curr_min_ind, :]
            p = 0
        p += 1
        print('Current_grid = ', curr_grid)
        print('Best Delta So far= %.5f'% delta_best)
        print('Best Learning rate So far=%.5f'%best_LR)
        if iteration > 10:
            break
    return_dict[l2Reg]=  ('Regularization=',l2Reg,'Best_lr=',best_LR,'best_loss=' ,best_loss )

datadir = 'C:/behrouz/Projects/daily_RNN_manyToMany/USequities/data/%s_Commision-and-Slippage-limits-0.25.csv'
# just matching by these since few days does not exactly match!
    # get the common dates and then merge each data making sure they have common dates:
data = pd.read_csv(datadir % markets[0])
for i in range(1, len(markets), 1):
    data1 = pd.read_csv(datadir % markets[i])
    data = pd.merge(data, data1, on='dtStart', how='inner')
dates = data[['dtStart']]
if __name__ == '__main__':
    results_path = 'C:/behrouz/Projects/daily_RNN_manyToMany/USequities/vanillaRNN/lrTuning/'
    objective = basket_sharpeLoss
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    os.chdir(results_path)
    pows = np.linspace(-5, -2, 4, endpoint=True)
    lR_grid = pow(10, pows)
    print(lR_grid)
    l2_grid = np.linspace(0, 10, 11)
    delta_thresh = 0.05
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for l2 in l2_grid:
        p = multiprocessing.Process(target=learningRate_tuning_wrapper, args=(lR_grid, l2, return_dict,
                                                                              delta_thresh, dates, objective, results_path))
        p.start()
        processes.append(p)
    results = []
    for p in processes:
        print('Running process = ', p)
        p.join()
    processes = []
    results = np.zeros((2, len(l2_grid)))
    for i in range(len(l2_grid)):
        results[0, i] = l2_grid[i]
        results[1, i] = return_dict[l2_grid[i]][3]

    print(results)
    print(lR_grid)
    results = pd.DataFrame(results)
    results.to_csv(results_path + 'best-Learning-rate-l2Grid.csv', index=False)
