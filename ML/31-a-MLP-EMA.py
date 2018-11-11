import tensorflow as tf
import numpy as np
import initializers
import os
import pandas as pd
import dataProcessing
import multiprocessing
import time
import math
import shutil
import random

'''
An MLP using tensorflow. 
Used data: SQ
'''
def sharpeLoss(outP, return_1day):
    ''' Treats all data as one market'''

    outP = tf.reshape(outP, [-1])
    tmp = tf.multiply(outP, return_1day)
    mean, var = tf.nn.moments(tmp, [0])
    sd = tf.sqrt(var)
    neg_sharpe = -1 * mean * math.sqrt(250) / sd

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

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    with tf.control_dependencies([train_op]):
        train_op_new = ema.apply(tf.trainable_variables())

    return train_op_new, output, sharpe_loss

def MLP_1layer_fixedBiasOut_sigmoid(x, y,  weights, biases,  curr_optimizer,
                            objective, activation, l2Reg , l2RegOutput,
                           init_lr, decay_steps, decay_rate):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)

    output = tf.add(tf.matmul(layer_1, weights['out']), 0.1)
    output = tf.nn.sigmoid(output)
    sharpe_loss = objective(output - 0.5, y)
    classification_loss = ((tf.sign(y) + 1) / 2. ) * tf.log(output) + (1. - (tf.sign(y) + 1) / 2. ) * tf.log(1 - output)
    classification_loss =  -tf.reduce_mean(classification_loss)

    # l2 regularization loss:

    l2Loss = l2Reg * tf.nn.l2_loss(weights['h1']) + l2RegOutput * tf.nn.l2_loss(weights['out'])

    sharpe_plus_l2_loss = sharpe_loss + l2Loss
    combined_loss = sharpe_plus_l2_loss + 0.1 * classification_loss

    # cost function and optimization
    global_step = tf.Variable(0, trainable = False)
    # cost function and optimization
    learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate)
    train_op = curr_optimizer(learning_rate=learning_rate).minimize(combined_loss, global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.999)
    with tf.control_dependencies([train_op]):
        train_op_new = ema.apply(tf.trainable_variables())

    return train_op_new, output, sharpe_loss, classification_loss


seeds = np.loadtxt('C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-SQ-only/pythonCodes/seeds.txt').astype(int)

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

    global_step = tf.Variable(0, trainable = False )
    # cost function and optimization
    learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate )
    train_op = curr_optimizer(learning_rate = learning_rate).minimize(sharpe_plus_l2_loss, global_step)

    ema = tf.train.ExponentialMovingAverage(decay = 0.999)
    with tf.control_dependencies([train_op]):
        train_op_new = ema.apply(tf.trainable_variables())

    return train_op_new, output, model_loss

def MLP_1layerFixedOutBias_clipNorm(x, y,  weights, biases,  curr_optimizer,
                            objective, l2Reg, l2RegOutput, init_lr, decay_steps, decay_rate, activation=tf.nn.tanh):
    '''
    :param x: placeholder tensor for input
    :param y_for_train:  is 1-day return for optimizing for sharpe
    :param weights: dictionary of all the weight tensors in the model
    :param biases: dictionary of all the bias tensors in the model
    :param learning_rate:  placeholder for learning rate, we will feed this at training!
    :return: returns the output of the model.
    '''

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)

    output_weights = tf.clip_by_norm(weights['out'], 1, axes = None)
    output = tf.add(tf.matmul(layer_1, output_weights) , 0.1)

    model_loss = objective(output, y)

    l2Loss = l2Reg * tf.nn.l2_loss(weights['h1']) + l2RegOutput * tf.nn.l2_loss(weights['out'])

    sharpe_plus_l2_loss = model_loss + l2Loss

    # cost function and optimization
    global_step = tf.Variable(0, trainable = False)

    # cost function and optimization
    learning_rate = tf.train.exponential_decay(init_lr, global_step, decay_steps, decay_rate)
    train_op = curr_optimizer(learning_rate = learning_rate).minimize(sharpe_plus_l2_loss, global_step)

    ema = tf.train.ExponentialMovingAverage(decay = 0.999)
    with tf.control_dependencies([train_op]):
        train_op_new = ema.apply(tf.trainable_variables())

    return train_op_new, output, model_loss

def myMainFunc(rand_start, hidden_size, l2Reg, learning_rate_grid, epoch_grid ):

    #random.seed(seeds[rand_start - 1])
    #np.random.seed(seeds[rand_start - 1])
    #tf.set_random_seed(seeds[rand_start - 1])
    market = 'SQ'

    if not os.path.exists('./MLP-checkpointFiles/'+ str(rand_start)):
        os.makedirs('./MLP-checkpointFiles/'+ str(rand_start))

    # objective function
    objective = sharpeLoss
    curr_optimizer = tf.train.AdamOptimizer
    network_activation = tf.nn.tanh

    # data parameters
    lookback = 30
    lookahead = 1
    rolling_sd_window = 100

    # training parameters:
    batch_size = 100
    test_start_date = 20070418

    patience = 20   # stop training if  train loss does not improve after 20 epochs
    counter = 0
    best_train_loss = np.inf

    # loading data
    datadir = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-SQ-only/data/%s_Commision-and-Slippage-limits.csv'

    # get the common dates and then merge each data making sure they have common dates:
    data = pd.read_csv(datadir % market)

    curr_market_data = \
            dataProcessing.time_series_toMatrix(data,  test_start_date, lookback = lookback,
                                                look_ahead = lookahead, sd_window = rolling_sd_window)
    trainX, trainY, train_ret_1day, train_dates = curr_market_data[:4]

    total_batches = trainX.shape[0] // batch_size
    rem = trainX.shape[0] % batch_size

    decay_steps = total_batches
    decay_rate = 1.0

    for LR in learning_rate_grid:

        for training_epochs in epoch_grid:
            print('Hidden Size =', hidden_size, 'Learning rate=', LR,
                  'TrainingEpochs=', training_epochs, 'L2 Reg=', l2Reg)

            weights = {

                 'h1': initializers.xavier_from_tf_initializer([lookback, hidden_size], name='W_1'),
                 'out': initializers.xavier_from_tf_initializer([hidden_size, 1], name='W_out')
            }

            biases = {

                'b1': initializers.bias_initializer([hidden_size], name = 'B_1')
                #, 'out': initializers.bias_initializer([1], name='B_out')
            }
            # placeholders
            x = tf.placeholder(tf.float32, [None, lookback])
            y = tf.placeholder(tf.float32, [None])

            train_op, output, sharpe_plus_l2_loss, classification_loss = MLP_1layer_fixedBiasOut_sigmoid(x, y,  weights, biases,  curr_optimizer,
                                                                            objective, network_activation, l2Reg , l2Reg,
                                                                           LR, decay_steps, decay_rate)

            # initialize all tensors- to be run in Session!
            init = tf.global_variables_initializer()

            # saver for restoring the whole model graph of tensors from the  checkpoint file
            saver = tf.train.Saver()

            # launch default graph:
            with tf.Session() as sess:

                sess.run(init)

                # training cycle:
                for epoch in range(training_epochs):

                    # shuffle the training data at the beginning of each epoch!
                    # for now I turn this off to get very consistent results
                    a = np.arange(trainX.shape[0])
                    np.random.shuffle(a)
                    trainX = trainX[a,:]
                    trainY = trainY[a]
                    # loop over all batches:
                    for batch_number in range(total_batches):

                        if (batch_number + 1) == total_batches and rem != 0:
                            xBatch = trainX[(total_batches - 1) * batch_size + rem:, :]
                            trainY_batch = trainY[(total_batches - 1) * batch_size + rem:]

                        else:
                            xBatch = trainX[batch_number * batch_size: (batch_number + 1) * batch_size, :]
                            trainY_batch = trainY[batch_number * batch_size: (batch_number + 1) * batch_size]

                        # run optimization
                        _ = sess.run(train_op, feed_dict={x: xBatch, y: trainY_batch})

                    #curr_loss = sess.run(sharpe_plus_l2_loss, feed_dict={x: trainX, y: trainY})
                    curr_loss, curr_classification_loss = sess.run([sharpe_plus_l2_loss, classification_loss],
                                                              feed_dict={x: trainX, y: trainY})
                    print('='*20)
                    print('Epoch=', epoch, 'Current Train Loss=', curr_loss, 'Best Train Loss=', best_train_loss)
                    print('Epoch=', epoch, 'Classification Loss=', curr_classification_loss)


                    if curr_loss < best_train_loss:
                        counter = 0
                        best_train_loss = curr_loss
                        saver.save(sess, './MLP-checkpointFiles/'+ str(rand_start)+
                                       '/run%d-s-%d-LR-%.6f-epoch-%d-l2-%.5f.ckpt'
                                       % (rand_start, hidden_size, LR, training_epochs, l2Reg))
                    
                    else:
                        counter += 1
                    if counter >= patience:
                        break

            # resetting the graph to be built again in the next iteration of for loop
            tf.reset_default_graph()

if __name__ == '__main__':
     random_start_indicies = np.arange(1, 11, 1)

     results_path = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-SQ-only/%s'
     results_folder = 'hybrid_loss'
     os.chdir(results_path % results_folder)

     t1 = time.time()
     hidden_size_grid = [3]
     l2_grid = np.linspace(0, 1, 11)
     learning_rate_grid = [0.00018]
     epoch_grid = [300]

     for hidden_size in hidden_size_grid:
         for l2Reg in l2_grid:
                print('Hidden Sizes=', hidden_size, 'L2Reg =', l2Reg)

                processes = []

                for rand_start in random_start_indicies:
                    p = multiprocessing.Process(target = myMainFunc,
                                                args = (rand_start, hidden_size, l2Reg, learning_rate_grid,  epoch_grid))
                    p.start()
                    processes.append(p)

                for p in processes:
                    print('Running process = ', p)
                    p.join()

                processes = []

     t2 = time.time()
     random_start_indicies = np.arange(1, 11, 1)
     print('TOTAL ELAPSED TIME FOR %d runs='%len(random_start_indicies), np.round((t2 - t1)/ 60., 2))
