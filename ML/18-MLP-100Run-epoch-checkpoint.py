import tensorflow as tf
import numpy as np
import initializers
import os
import pandas as pd
import dataProcessing
import math
import copy
'''
An MLP using tensorflow. 
Used data: 5 market data: SQ, MQ, NQ, DQ, RN
'''

# data parameters

lookback = 30
lookahead = 1
rolling_sd_window = 100

# training parameters:

LR = 0.0005
batch_size = 100
display_steps = 1

# network parameters:

hidden1_size = 30
hidden2_size = 30
input_feats = lookback
l2Reg = 0.3

# loss function

def sharpeLoss(outP, return_1day):
    outP = tf.reshape(outP, [-1])

    print(outP.get_shape())
    print(return_1day.get_shape())

    tmp = tf.multiply(outP, return_1day)
    mean, var = tf.nn.moments(tmp, [0])
    sd = tf.sqrt(var)

    # note: sqrt(250 Does not Play any role in optimization so I remove it!
    neg_sharpe = -1 * mean / sd
    return neg_sharpe



def sharpe(outP, return_1day):

    outP = tf.reshape(outP, [-1])
    outP = tf.sign(outP)

    tmp = tf.multiply(outP, return_1day)

    mean, var = tf.nn.moments(tmp, [0])
    sd = tf.sqrt(var)

    sharpe = mean * tf.sqrt(250.) / sd

    return sharpe

def sharpe_likeModo(return_1day, y_pred, transCost = 0.00007):

    signPred = np.sign(y_pred)
    return_stream = signPred * return_1day

    # transaction costs:

    t1 = [transCost if signPred[i-1] != signPred[i] else 0
                        for i in range(1,len(signPred),1) ]

    # at t=0 we incure transcost on first trade!
    totalTransCost = [transCost]
    totalTransCost.extend(t1)

    #subtract transaction cost
    return_stream -= totalTransCost
    sharpe = np.round(np.mean(return_stream) * math.sqrt(250.) / np.std(return_stream), 3)

    tmpLong = [return_stream[i] for i in range(len(y_pred)) if y_pred[i] > 0]
    tmpShort = [return_stream[i] for i in range(len(y_pred)) if y_pred[i] < 0]

    sharpeLong = np.round(np.mean(tmpLong) * math.sqrt(250) / np.std(tmpLong), 3)
    sharpeShort = np.round(np.mean(tmpShort) * math.sqrt(250) / np.std(tmpShort), 3)

    return sharpe, sharpeLong, sharpeShort


def MLP(x,
        y,
        weights,
        biases,
        curr_optimizer,
        learning_rate,
        activation = tf.nn.tanh,
        l2Reg = 0.01):
    '''
    :param x: placeholder tensor for input
    :param y_for_train:  is 1-day return for optimizing for sharpe
    :param weights: dictionary of all the weight tensors in the model
    :param biases: dictionary of all the bias tensors in the model
    :param learning_rate:  placeholder for learning rate, we will feed this at training!
    :return: returns the output of the model (just the logits before the softmax (linear output)!
    '''

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'] )
    layer_1 = activation(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'] )
    layer_2 = activation(layer_2)
    output = tf.add(tf.matmul(layer_2, weights['out']) , biases['out'] )

    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg

    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_out' in curr_var.name))

    l2Loss_output_layer = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if ('W_out' in curr_var.name or 'B_out' in curr_var.name))

    l2Loss *= l2Reg
    l2Loss_output_layer *= 0.1 * l2Reg
    l2Loss += l2Loss_output_layer
    sharpe_loss =  sharpeLoss(output, y)
    sharpe_plus_l2_loss = sharpe_loss + l2Loss
    # cost function and optimization
    optimizer = curr_optimizer(learning_rate = learning_rate).minimize(sharpe_plus_l2_loss)
    return optimizer, output,  sharpe_plus_l2_loss

#
# loading data
markets = ['SQ', 'NQ', 'MQ', 'DQ', 'RN']

#dictionaries keeping the data by market
test_dict = {}
train_dict= {}

datadir = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tensorflow-5Market-model/%s_Adj.csv'

results_folder = '0.001-LR-25-epochsL2-Reg-0.05-ScaledY'

results_path = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tensorflow-5Market-model-epochCheckpoint/tanh/30Neurons/%s'

os.chdir(results_path%results_folder)


trans_data = pd.read_csv('C:/behrouz/Projects/DailyModels_new/NeuralNet/TransactionCosts.csv')
trans_data = trans_data.values
transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))


data = pd.read_csv(datadir % markets[0])

data_market_0 = dataProcessing.time_series_toMatrix(data, 20070418, lookback = lookback,
                                            look_ahead = lookahead, sd_window = rolling_sd_window)

trainX, trainY, train_ret_1day, train_pred_date, testX, testY, test_ret_1day, test_pred_dates = data_market_0



train_dict[markets[0]] = copy.deepcopy(data_market_0[:4])
test_dict[markets[0]] = copy.deepcopy(data_market_0[4:])

for i in range(1, len(markets), 1):
    print(markets[i])

    data = pd.read_csv(datadir % markets[i])

    curr_market_data = \
        dataProcessing.time_series_toMatrix(data, 20070418, lookback = 30, look_ahead = lookahead, sd_window = 100)

    trainX = np.append(trainX, curr_market_data[0], axis=0)
    trainY = np.append(trainY, curr_market_data[1], axis=0)
    train_ret_1day = np.append(train_ret_1day, curr_market_data[2], axis=0)


    train_dict[ markets[i] ] = copy.deepcopy(curr_market_data[:4])
    test_dict[ markets[i] ] = copy.deepcopy(curr_market_data[4:])

print(trainX.shape)

nRandom_start = 100
training_epochs = 25

total_batches = trainX.shape[0] // batch_size
rem = trainX.shape[0] % batch_size

for j in range(nRandom_start):

    print('RUN %d optimization begins..'%(j+1) )
    weights = {
    'h1':  initializers.tanh_uniform_weight_initializer([input_feats, hidden1_size]),
    'h2':  initializers.tanh_uniform_weight_initializer([hidden1_size, hidden2_size]),
    'out': initializers.tanh_uniform_weight_initializer([hidden2_size, 1], name='W_out')
    }

    biases = {
    'b1': initializers.bias_initializer([hidden1_size]),
    'b2': initializers.bias_initializer([hidden2_size]),
    'out': initializers.bias_initializer([1], name = 'B_out')
    }

    x = tf.placeholder(tf.float32, [None,input_feats])
    y = tf.placeholder(tf.float32, [None])
    learning_rate = tf.placeholder(tf.float32)

    optimizer, output, sharpe_plus_l2_loss = MLP(x,
                                                 y,
                                                 weights,
                                                 biases,
                                                 tf.train.AdamOptimizer,
                                                 learning_rate,
                                                 activation=tf.nn.tanh,
                                                 l2Reg=l2Reg)
    # initialize all tensors- to be run in Session!
    init = tf.global_variables_initializer()
    # saver for restoring the whole model graph of
    #  tensors from the  checkpoint file
    saver = tf.train.Saver()
    # launch default graph:
    with tf.Session() as sess:
        sess.run(init)
        # as  sanity check- performance at random start
        trainPred = sess.run(output,
                             feed_dict={x: trainX,
                                        y: trainY,
                                        learning_rate: LR})[:, 0]
        train_sharpe = sharpe_likeModo(train_ret_1day, trainPred, transCost=transCost_dict['SQ'])
        print('SQ- Pre training Train sharpe = ', train_sharpe)
        # training cycle:
        for epoch in range(training_epochs):
             # shuffle the training data at the begining of each batch!
            a = np.arange(trainX.shape[0])
            np.random.shuffle(a)
            trainX = trainX[a, :]
            train_ret_1day = train_ret_1day[a]
            trainY = trainY[a]
            # loop over all batches:
            for batch_number in range(total_batches):
                # if last batch is smaller than batch size:
                if (batch_number + 1) == total_batches and rem != 0:
                    xBatch = trainX[(total_batches - 1) * batch_size + rem:]
                    trainY_batch = trainY[ (total_batches - 1) * batch_size + rem:]
                else:
                    xBatch = trainX[batch_number * batch_size : (batch_number + 1) * batch_size]
                    trainY_batch = trainY[ batch_number * batch_size : (batch_number + 1) * batch_size]
                # run optimization
                _ =  sess.run(optimizer, feed_dict={x: xBatch,
                                                    y: trainY_batch,
                                                    learning_rate: LR})
            # sharpe train after each epoch:
            trainPred = sess.run(output, feed_dict={x: trainX,
                                                    y: trainY,
                                                    learning_rate: LR})[:, 0]
            print(trainPred.shape)
            train_sharpe = np.round(sharpe_likeModo(train_ret_1day,
                                                    trainPred,
                                                    transCost=transCost_dict['SQ']),
                                    3)
            print('EPOCH %dTrain_sharpe = ' % epoch, train_sharpe)
            #check point model at epoch 10, 15, 20 25
            if (epoch + 1) % 5 ==0 and (epoch + 1) > 5:
                print('Check pointing...')
                saver.save(sess, './MLP-checkpointFiles/MLP-tensors-checkPoint-run%d-epoch-%d.ckpt' % (j + 1, epoch + 1))

        print('Optimization finished!')
        # print(' saving model graph of all tensors to file')
        # save_path = saver.save(sess, './MLP-checkpointFiles/MLP-tensors-checkPoint-run%d.ckpt'%(j+1))
   # resetting the graph to be built again in the next iteration of for loop
    tf.reset_default_graph()
