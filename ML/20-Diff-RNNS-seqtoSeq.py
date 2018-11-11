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
import multiprocessing

'''
An MLP using tensorflow. 
Used data: 5 market data: SQ, MQ, NQ, DQ, RN
'''


def plot_sharpes(train_basket_sharpes_array,
                 test_basket_sharpes_array,
                 train_indiv_sharpes,
                 test_indiv_sharpes):
    xlim = len(train_basket_sharpes_array)
    step = int(xlim//10)
    fig = plt.figure(1, figsize=(20, 20))
    ax = fig.add_subplot(4, 2, 1)
    plt.plot(train_basket_sharpes_array, '-', label=' Train', c='red')
    plt.plot(test_basket_sharpes_array, '-', label=' Test', c='blue')

    plt.ylabel('Basket Sharpe')
    plt.legend(loc='lower right', ncol=2)
    plt.axhline(0.5, c='black', linestyle='--')
    plt.axhline(0.6, c='black', linestyle='--')
    plt.axhline(0.7, c='black', linestyle='--')
    plt.axhline(0.8, c='black', linestyle='--')
    for epoch in range(0, xlim, step):
        plt.axvline(x=epoch, linestyle='--', color='skyblue')
    plt.title('Basket Sharpe Learning Dynamics')
   # plt.text(50, 1.6, 'initial LR= %.1e. Exponential Decay = %.4f' % (init_LR, np.round(exponential_decay_rate, 2)))
    plt.ylim(-1, 2)
    #sharpes = ['ALL', 'Long', 'Short']
    colors = ['red', 'blue', 'green', 'skyblue', 'gold']
    plot_num = 3
    for i in range(3):
        ax = fig.add_subplot(4, 2, plot_num)
        for j in range(len(markets)):
            ax.plot(train_indiv_sharpes[j, :, i], '-', label=markets[j], c=colors[j])
            #ax.set_title('Train %s sharpe by market' % sharpes[i])
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=0.5, linestyle='--')
            plt.axhline(y=1, linestyle='--')
            plt.ylabel('Sharpe')

            plt.ylim(-1, 2)
            for epoch in range(0, xlim, step):
                plt.axvline(x=epoch, linestyle='--', color='skyblue')

        if plot_num == 3:
            plt.legend(loc='lower right', ncol=2)

        plot_num += 1

        ax = fig.add_subplot(4, 2, plot_num)
        for j in range(len(markets)):
            ax.plot(test_indiv_sharpes[j, :, i], '-', label=markets[j], c=colors[j])
            #ax.set_title('Test %s sharpe by market ' % sharpes[i])
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=0.5, linestyle='--')
            plt.axhline(y=1, linestyle='--')
            for epoch in range(0, xlim, step):
                plt.axvline(x=epoch, linestyle='--', color='skyblue')
            plt.ylim(-1, 2)
            plt.ylabel('Sharpe')

        plot_num += 1

    plt.xlabel('Epoch')
    plt.ylabel('Sharpe')
    # plt.show()
    return fig


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
    sharpe = mean * math.sqrt(250.) / sd
    sharpe *= -1
    return sharpe


def basket_sharpeLoss_kertosis(outPutMulY):
    ret_seq = tf.reduce_sum(outPutMulY, axis =0)
    print('shape of summed sequence=', ret_seq.get_shape())
    mean, var = tf.nn.moments(ret_seq, [0])
    sd = tf.sqrt(var)
    sharpe = mean *math.sqrt(250.) / sd
    sharpe *= -1
    outPutMulY -= mean
    kurtosis = tf.pow(outPutMulY, 4)
    sd4 = tf.pow(sd, 4)
    kurtosis = kurtosis / sd4
    kurtosis = tf.reduce_mean(kurtosis) - 3
    # we want to penalize for negative skew!!
    return sharpe + 0.2 * kurtosis
    # ################################################


def vanilla_LSTM(x,
                 y,
                 weights,
                 biases,
                 keep_prob,
                 curr_optimizer,
                 learning_rate,
                 objective,
                 markets,
                 activation,
                 l2Reg,
                 hidden_size,
                 n_layers):
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
    # output = tf.
    print('Shape of output in correct format for Sharpe:', output.get_shape(), y.shape)
    # outNorm  = tf.norm(output, axis =1) /  5 # i want norm of each vector of signals! to be 5!!
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


def peephole_LSTM(x,
                  y,
                  weights,
                  biases,
                  keep_prob,
                  curr_optimizer,
                  learning_rate,
                  objective,
                  markets,
                  activation,
                  l2Reg,
                  hidden_size,
                  n_layers):
    if n_layers > 1:
        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(hidden_size, activation=activation, use_peepholes=True)
            # if applying dropout after each layer is NOT desired, remove this and
            # maybe apply drop out to the output of the stack!
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
            return cell
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
        lstm_out, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
    else:
        cell = tf.contrib.rnn.LSTMCell(hidden_size, activation=activation, use_peepholes=True)
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


def attention_LSTM(x,
                   y,
                   weights,
                   biases,
                   keep_prob,
                   curr_optimizer,
                   learning_rate,
                   objective,
                   markets,
                   activation,
                   l2Reg,
                   hidden_size,
                   n_layers):
    if n_layers > 1:
        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, activation=activation)
            cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=2)
            # if applying dropout after each layer is NOT desired, remove this and
            # maybe apply drop out to the output of the stack!
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
            return cell
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
        lstm_out, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, activation=activation)
        cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=2)
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


def attention_BlockLSTM(x,
                        y,
                        weights,
                        biases,
                        keep_prob,
                        curr_optimizer,
                        learning_rate,
                        objective,
                        markets,
                        activation,
                        l2Reg,
                        hidden_size,
                        n_layers):
    if n_layers > 1:
        def lstm_cell():
            cell = tf.contrib.rnn.LSTMBlockCell(hidden_size)
            cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=5)
            # if applying dropout after each layer is NOT desired, remove this and
            # maybe apply drop out to the output of the stack!
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
            return cell
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
        lstm_out, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
    else:
        cell = tf.contrib.rnn.LSTMBlockCell(hidden_size)
        cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=5)

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


def vanilla_RNN(x,
                y,
                weights,
                biases,
                keep_prob,
                curr_optimizer,
                learning_rate,
                objective,
                markets,
                activation,
                l2Reg,
                hidden_size,
                n_layers):
    if n_layers > 1:
        def rnn_cell():
            cell = tf.contrib.rnn.BasicLRNNCell(hidden_size, activation=activation)
            # if applying dropout after each layer is NOT desired, remove this and
            # maybe apply drop out to the output of the stack!
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
            return cell
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(n_layers)])
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


def vanilla_GRU(x,
                y,
                weights,
                biases,
                keep_prob,
                curr_optimizer,
                learning_rate,
                objective,
                markets,
                activation,
                l2Reg,
                hidden_size,
                n_layers):
    if n_layers > 1:
        def gru_cell():
            cell = tf.contrib.rnn.GRUCell(hidden_size, activation=activation)
            # if applying dropout after each layer is NOT desired, remove this and
            # maybe apply drop out to the output of the stack!
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
            return cell
        stacked_gru = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(n_layers)])
        gru_out, states = tf.nn.dynamic_rnn(stacked_gru, x, dtype=tf.float32)
    else:
        cell = tf.contrib.rnn.GRUCell(hidden_size, activation=activation)
        # output of dynamic_rnn is  tensor of shape: [batch_size, timesteps, hidden_size]
        gru_out, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    gru_out = tf.reshape(gru_out, [-1, hidden1_size])
    gru_out = tf.nn.dropout(gru_out, keep_prob=keep_prob)

    print('Shape of reshaped LSTM output and output weights', gru_out.get_shape(), weights['out'].shape)
    output = tf.matmul(gru_out, weights['out']) + biases['out']
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


def residual_GRU(x,
                 y,
                 weights,
                 biases,
                 keep_prob,
                 curr_optimizer,
                 learning_rate,
                 objective,
                 markets,
                 activation,
                 l2Reg,
                 hidden_size,
                 n_layers):
    # Assumes at least two layers since the last dimension of input and output of LSTM layer must match
    def gru_cell():
        cell = tf.contrib.rnn.GRUCell(hidden_size, activation=activation)
        # if applying dropout after each layer is NOT desired, remove this and
        # maybe apply drop out to the output of the stack!
       # cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        return cell
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([gru_cell() if i == 0 else tf.contrib.rnn.ResidualWrapper(gru_cell())
                                                for i in range(n_layers)])
    gru_out, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)

    # output of dynamic_rnn is  tensor of shape: [batch_size, timesteps, hidden_size]
    print('Shape of reshaped LSTM output and output weights before reshaping', gru_out.get_shape(), weights['out'].shape)

    gru_out = tf.reshape(gru_out, [-1, hidden1_size])
    gru_out = tf.nn.dropout(gru_out, keep_prob=keep_prob)

    print('Shape of reshaped LSTM output and output weights', gru_out.get_shape(), weights['out'].shape)
    output = tf.matmul(gru_out, weights['out']) + biases['out']
    output = tf.reshape(output, [len(markets), -1])
    output = tf.nn.tanh(output)
    print('Shape of output in correct format for Sharpe:', output.get_shape(), y.shape)
    # outNorm  = tf.norm(output, axis =1) /  5 # i want norm of each vector of signals! to be 5!!
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

#################################################################################
markets = ('SQ', 'NQ', 'MQ', 'DQ', 'RN')
# objective function
objective = sharpeLoss
curr_optimizer = tf.train.AdamOptimizer
# data parameters
date_train_end = 20070418
# training parameters:
training_epochs = 1000
stationary_epochs = 500
decya_fact = 0.998
# network parameters:
network_activation = tf.nn.tanh
hidden1_size = 5
features = 2
dropout = 1.0 #1.0: do not use dropout!
nRandom_start = 5
n_layers = 1
##############################################################################
def train_parallel(l2Reg, init_lr, results_path, objective, train_data, test_data, transCost_dict):
    display_steps = 10
    total_evals = int(training_epochs // display_steps)
    for R in range(nRandom_start):
        ind = 0
        train_basket_sharpes_array = np.zeros(total_evals)
        test_basket_sharpes_array = np.zeros(total_evals)

        train_indiv_sharpes = np.zeros((len(markets), total_evals, 3))
        test_indiv_sharpes = np.zeros((len(markets), total_evals, 3))
        print('RUN %d optimization begins..' % (R + 1))
        weights = {
            'out': initializers.xavier_from_tf_initializer([hidden1_size, 1], name='W_out')
        }
        biases = {
            'out': initializers.bias_initializer([1], name='B_out')
        }
        x = tf.placeholder(tf.float32, [len(markets), None, features])
        y = tf.placeholder(tf.float32, [len(markets), None])
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)

        optimizer, output, sharpe_plus_l2_loss = vanilla_RNN(x, y, weights, biases, keep_prob, curr_optimizer,
                                                             learning_rate, objective, markets=markets,
                                                             activation=tf.nn.tanh, l2Reg=l2Reg,
                                                             hidden_size=hidden1_size, n_layers=n_layers)
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
                    LR = init_lr * (decya_fact ** decay_exponent)
                    decay_exponent += 1
                    feed_dict = {x: train_data[0], y: train_data[2], learning_rate: LR, keep_prob: dropout}
                else:
                    LR = init_lr
                    feed_dict = {x: train_data[0], y: train_data[2], learning_rate: LR, keep_prob: dropout}
                _ = sess.run(optimizer, feed_dict=feed_dict)
                ######################################################################################################
                if epoch % display_steps ==0:
                    train_preds_all_markets = sess.run(output, feed_dict={x: train_data[0], keep_prob: 1.0})
                    test_preds_all_markets = sess.run(output, feed_dict={x: test_data[0], keep_prob: 1.0})

                    train_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(train_preds_all_markets,
                                                                                          train_data[1], markets,
                                                                                          transCost_dict)
                    train_basket_sharpes_array[ind] = train_basket_sharpe
                    test_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(test_preds_all_markets,
                                                                                         test_data[1], markets, transCost_dict)
                    test_basket_sharpes_array[ind] = test_basket_sharpe
                    print('EPOCH %d- learning rate='%epoch,LR, 'Train basket sharpe=', round(train_basket_sharpe,3),
                          'Test Basket Sharpe=', round(test_basket_sharpe,3))
                    # Individual market sharpes :training data:
                    for m in range(len(markets)):
                        train_indiv_sharpes[m, ind, :] = losses_and_metrics.sharpe_likeModo(train_preds_all_markets[m, :],
                                                                    train_data[1][m, :], transCost_dict[markets[m]])
                        test_indiv_sharpes[m, ind, :] = losses_and_metrics.sharpe_likeModo(test_preds_all_markets[m, :],
                                                                    test_data[1][m, :], transCost_dict[markets[m]])
                    ind += 1
            print(' saving model graph of all tensors to file')
            saver.save(sess, results_path +str(l2Reg) + '/checkpointFiles/checkPoint-run%d.ckpt' % (R + 1))
            print('Optimization finished!')
            # resetting the graph to be built again in the next iteration of for loop
        tf.reset_default_graph()
        fig = plot_sharpes(train_basket_sharpes_array, test_basket_sharpes_array, train_indiv_sharpes,
                           test_indiv_sharpes)
        fig.savefig(results_path +str(l2Reg)+ '/Learning-dynamics-plot/%d-N-l2Reg-%.3f-run%d.png'
                    % (hidden1_size, l2Reg , R + 1),  bbox_inches='tight')
        plt.close()

def dataPrep(datadir, markets, dates):
    # dates are common dates between all markets
    for i in range(0, len(markets), 1):
        data = pd.read_csv(datadir % markets[i])
        # Make sure we get data from all  markets on exact common dates
        data = pd.merge(data, dates, on='dtStart', how='inner')
        curr_market_data = \
            dataProcessing.time_series_toMatrix(data, date_train_end)
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
    train_data = (trainX, trainY, train_Y_op_opD, train_dates)
    test_data = (testX, testY, test_Y_op_opD, test_dates)
    return train_data, test_data

if __name__ == '__main__':
    #loading  data
    datadir = 'C:/behrouz/Projects/daily_RNN_manyToMany/USequities/data/%s_Commision-and-Slippage-limits-0.25.csv'
    # just matching by these since few days does not exactly match!
    trans_data = pd.read_csv('C:/behrouz/Projects/daily_RNN_manyToMany/USequities/TransactionCosts.csv')
    trans_data = trans_data.values
    transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))
    data = pd.read_csv(datadir % markets[0])
    for i in range(1, len(markets), 1):
        data1 = pd.read_csv(datadir % markets[i])
        data = pd.merge(data, data1, on='dtStart', how='inner')
    dates = data[['dtStart']]

    train_data, test_data = dataPrep(datadir, markets,dates)
    print(train_data[0].shape, train_data[1].shape)
    print(test_data[0].shape, test_data[1].shape)
    print('====')

    results_path = 'C:/behrouz/Projects/daily_RNN_manyToMany/USequities/vanillaRNN/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    l2_grid, lr_grid = pd.read_csv(results_path+'lrTuning/'+'best-Learning-rate-l2Grid.csv').values
    print(l2_grid)
    print(lr_grid)
    processes = []
    for l2, lr in zip(l2_grid, lr_grid):
        if not os.path.exists(results_path + str(l2)):
            os.makedirs(results_path + str(l2))
        if not os.path.exists(results_path + str(l2) + '/Learning-dynamics-plot'):
            os.makedirs(results_path + str(l2) + '/Learning-dynamics-plot')
        # train_parallel(0.5, 0.001, results_path, objective, train_data, test_data, transCost_dict)
        p = multiprocessing.Process(target=train_parallel, args=(l2, lr, results_path, objective,
                                                                 train_data, test_data, transCost_dict))
        p.start()
        processes.append(p)
    results = []
    for p in processes:
        print('Running process = ', p)
        p.join()

