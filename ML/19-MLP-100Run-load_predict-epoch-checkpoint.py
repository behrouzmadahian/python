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
l2Reg = 0.1

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


def MLP(x,  y,  weights, biases, curr_optimizer,learning_rate,
        activation = tf.nn.tanh,l2Reg = 0.01):
    '''
    :param x: placeholder tensor for input
    :param y_for_train:  is 1-day return for optimizing for sharpe
    :param weights: dictionary of all the weight tensors in the model
    :param biases: dictionary of all the bias tensors in the model
    :param learning_rate:  placeholder for learning rate, we will feed this at training!
    :return: returns the output of the model (just the logits before the softmax (linear output)!
    '''

    layer_1 = tf.add( tf.matmul(x, weights['h1']), biases['b1'] )
    layer_1 = activation(layer_1)

    layer_2 = tf.add( tf.matmul(layer_1, weights['h2']), biases['b2'] )
    layer_2 = activation(layer_2)

    output = tf.add( tf.matmul(layer_2, weights['out']) , biases['out'] )

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
neurons = 30
#dictionaries keeping the data by market

datadir = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tensorflow-5Market-model/%s_Adj.csv'

results_folder = '0.001-LR-25-epochsL2-Reg-0.05-ScaledY'

results_path = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tensorflow-5Market-model-epochCheckpoint/tanh/%dNeurons/%s'

os.chdir(results_path%(neurons,results_folder))


trans_data = pd.read_csv('C:/behrouz/Projects/DailyModels_new/NeuralNet/TransactionCosts.csv')
trans_data = trans_data.values
transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))



weights = {
    'h1':  initializers.tanh_normal_weight_initializer([input_feats, hidden1_size]),
    'h2':  initializers.tanh_normal_weight_initializer([hidden1_size, hidden2_size]),
    'out': initializers.tanh_normal_weight_initializer([hidden2_size, 1], name = 'W_out')
    }

biases = {
    'b1': initializers.bias_initializer([hidden1_size]),
    'b2': initializers.bias_initializer([hidden2_size]),
    'out': initializers.bias_initializer([1], name = 'B_out')
    }

x = tf.placeholder(tf.float32, [None,input_feats])
y = tf.placeholder(tf.float32, [None])
learning_rate = tf.placeholder(tf.float32)

optimizer, output, sharpe_plus_l2_loss =\
         MLP(x, y, weights, biases, tf.train.AdamOptimizer, learning_rate,
             activation = tf.nn.tanh, l2Reg = l2Reg )


nRandom_start = 100
epoch_checkpoint = [10, 15, 20, 25]
with tf.Session() as sess:
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    print('Testing current model on test data!')
    markets_sharpes_train = np.zeros((nRandom_start * len(epoch_checkpoint), 15))
    markets_sharpes_test = np.zeros((nRandom_start * len(epoch_checkpoint), 15))

    for i in range(len(markets)):
        # get the data and then restore the model
        data = pd.read_csv(datadir % markets[i])

        trainX, trainY, train_ret_1day, train_pred_date, testX, testY, test_ret_1day, test_pred_dates =\
            dataProcessing.time_series_toMatrix(data, 20070418, lookback=lookback,
                                                            look_ahead=lookahead, sd_window=rolling_sd_window)

        print('Testing model on Market=', markets[i])
        trans_cost = transCost_dict[markets[i]]
        print(trans_cost)

        market_trainPred = np.zeros((trainX.shape[0], nRandom_start * len(epoch_checkpoint) + 2))
        market_testPred = np.zeros((testX.shape[0], nRandom_start * len(epoch_checkpoint) + 2))

        market_trainPred[:, 0] = train_pred_date  # date
        market_trainPred[:, 1] = train_ret_1day  # 1 day return

        market_testPred[:, 0] = test_pred_dates
        market_testPred[:, 1] = test_ret_1day

        for j in range(nRandom_start):
            kk = 0
            for epoch in epoch_checkpoint:
                pred_ind = j * len(epoch_checkpoint) + kk
                print('Prediction Index=', pred_ind)

                sess.run(init)

                saver.restore(sess, './MLP-checkpointFiles/MLP-tensors-checkPoint-run%d-epoch-%d.ckpt' % (j + 1, epoch))

                trainPred = sess.run(output,
                                     feed_dict={x: trainX,
                                                y: trainY,
                                                learning_rate: LR})[:, 0]

                testPred = sess.run(output,
                                    feed_dict={x: testX,
                                               y: testY,
                                               learning_rate: LR})[:, 0]
                market_trainPred[:, pred_ind + 2] = trainPred
                market_testPred[:, pred_ind + 2] = testPred
                trainSharpes = sharpe_likeModo(train_ret_1day, trainPred, transCost=trans_cost)
                markets_sharpes_train[pred_ind, i] = trainSharpes[0]
                markets_sharpes_train[pred_ind, i + 5] = trainSharpes[1]
                markets_sharpes_train[pred_ind, i + 10] = trainSharpes[2]
                testSharpes = sharpe_likeModo(test_ret_1day, testPred, transCost=trans_cost)
                markets_sharpes_test[pred_ind, i] = testSharpes[0]
                markets_sharpes_test[pred_ind, i + 5] = testSharpes[1]
                markets_sharpes_test[pred_ind, i + 10] = testSharpes[2]

                kk += 1

        predsCols = ['dtStart', '%s-y-true' % markets[i]]
        predsCols.extend(['%s-pred%d' % (markets[i], j) for j in range(1, nRandom_start * len(epoch_checkpoint) + 1, 1)])

        market_trainPred = pd.DataFrame(market_trainPred, columns=predsCols)
        market_trainPred.to_csv('%s-trainPreds.csv' % markets[i], index=False)

        market_testPred = pd.DataFrame(market_testPred, columns=predsCols)
        market_testPred.to_csv('%s-testPreds.csv' % markets[i], index=False)

    sharpeCol_long = ['%s-long' % m for m in markets]
    sharpeCol_short = ['%s-short' % m for m in markets]

    cols = markets
    cols.extend(sharpeCol_long)
    cols.extend(sharpeCol_short)

    print(len(cols))
    print(markets_sharpes_test.shape)
    print(markets_sharpes_train.shape)

    markets_sharpes_train = pd.DataFrame(markets_sharpes_train, columns=cols)
    markets_sharpes_test = pd.DataFrame(markets_sharpes_test, columns=cols)

    markets_sharpes_train.to_csv('train-sharpe-5markets.csv', index=False)
    markets_sharpes_test.to_csv('test-sharpe-5markets.csv', index=False)




