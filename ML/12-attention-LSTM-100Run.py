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
from Models import attention_LSTM

'''
An MLP using tensorflow. 
Used data: 5 market data: SQ, MQ, NQ, DQ, RN
'''
markets = ('SQ', 'NQ', 'MQ', 'DQ', 'RN')

# objective function
objective = losses_and_metrics.sharpeLoss

#curr_optimizer = tf.train.AdamOptimizer
curr_optimizer = tf.train.RMSPropOptimizer

#optim = 'Adam'
optim = 'RMSprop'

results_folder = 'attention-LSTM'
#results_folder = 'L2-Reg-0.5-ScaledY-Adam'

# data parameters
Y_toUse = 1   # 1: scaled return, 2:1-day return
lookback = 50
lookahead = 1
rolling_sd_window = 100

# training parameters:
dropout = 1.0 #1.0: do not use dropout!
training_epochs = 3000
init_LR = 0.001
LR = copy.deepcopy(init_LR)
exponential_decay_rate = 0.999
batch_size = 100
display_steps = 1

# network parameters:
attention_length = 30
network_activation = tf.nn.tanh
hidden1_size = 10
hidden2_size = 10
input_feats = lookback
l2Reg = 0.06

nRandom_start = 1


# loading data
datadir = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-5Market-model-RNN/%s-all-Adj.csv'

results_path = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-5Market-model-RNN/%s'

os.chdir(results_path % results_folder)

#just matching by these since few days does not exactly match!

dates1 = pd.read_csv('C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-5Market-model-RNN/CommonDates_DQ.csv')

dates2 = pd.read_csv('C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-5Market-model-RNN/CommonDates_SQ.csv')

trans_data = pd.read_csv('C:/behrouz/Projects/DailyModels_new/NeuralNet/TransactionCosts.csv')
trans_data = trans_data.values
transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))


test_dict = {}
train_dict = {}

for i in range(0, len(markets), 1):

    data = pd.read_csv(datadir % markets[i])

    # Make sure we get data from all  markets on exact common dates
    data = pd.merge(data, dates1, on='dtStart', how='inner')
    data = pd.merge(data, dates2, on='dtStart', how='inner')

    curr_market_data = \
        dataProcessing.time_series_toMatrix(data, 20070418, lookback= lookback, look_ahead=lookahead,
                                            sd_window= rolling_sd_window)

    train_dict[markets[i]] = copy.deepcopy(curr_market_data[:4])
    test_dict[markets[i]] = copy.deepcopy(curr_market_data[4:])


total_batches = train_dict[markets[0]][0].shape[0] // batch_size
rem = train_dict[markets[0]][0].shape[0] % batch_size
print('TOTAL BATCHES+++++++++++++++++++++++', total_batches)

train_basket_sharpes_array = np.zeros(training_epochs)
test_basket_sharpes_array = np.zeros(training_epochs)

train_indiv_sharpes = np.zeros((len(markets), training_epochs, 3))
test_indiv_sharpes = np.zeros((len(markets), training_epochs, 3))


for R in range(nRandom_start):

    print('RUN %d optimization begins..' % (R + 1))

    weights = {
        'out': initializers.xavier_from_tf_initializer([hidden2_size, 1], name='W_out')
    }
    biases = {
        'out': initializers.bias_initializer([1], name='B_out')
    }

    x = tf.placeholder(tf.float32, [None, input_feats])
    y = tf.placeholder(tf.float32, [None])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    optimizer, output, sharpe_plus_l2_loss = attention_LSTM(x, y, weights, biases, keep_prob, curr_optimizer,
                                                                 learning_rate, objective, batch_size,
                                                                 markets=markets, activation=tf.nn.tanh,
                                                                 l2Reg=l2Reg, l2RegOutput=l2Reg * 1.,
                                                                 l2Reg_biases=l2Reg * 1.,
                                                                 lookback= lookback, hidden_size= hidden1_size,
                                                                 n_layers = 1, attention_length = attention_length)

    # initialize all tensors- to be run in Session!

    init = tf.global_variables_initializer()

    # saver for restoring the whole model graph of
    #  tensors from the  checkpoint file

    saver = tf.train.Saver()

    # launch default graph:
    with tf.Session() as sess:

        sess.run(init)

        # sharpe train after each epoch:
        train_preds_all_markets = []
        train_1day_returns_all_markets = []
        test_preds_all_markets = []
        test_1day_returns_all_markets = []

        #gathering predictions and 1-day returns for all markets for train and test
        for i in range(len(markets)):
            print(train_dict[markets[i]][0].shape)

            train_preds_all_markets.append(sess.run(output, feed_dict={x: train_dict[markets[i]][0],
                                                                       keep_prob : 1.0})[:, 0])

            test_preds_all_markets.append(sess.run(output, feed_dict={x: test_dict[markets[i]][0],
                                                                       keep_prob : 1.0})[:, 0])

            train_1day_returns_all_markets.append(train_dict[markets[i]][2])
            test_1day_returns_all_markets.append(test_dict[markets[i]][2])


        train_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(
            train_preds_all_markets, train_1day_returns_all_markets, markets, transCost_dict)

        test_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(
            test_preds_all_markets, test_1day_returns_all_markets, markets, transCost_dict)

        print(' Pre- training Train basket sharpe = ', np.round(train_basket_sharpe, 3))
        print(' Pre- training Test basket sharpe = ', np.round(test_basket_sharpe, 3))

        # training cycle:
        for epoch in range(training_epochs):

            if epoch > 100  :

                exponent = epoch - 50

            else:
                exponent = epoch + 1

            LR = init_LR * pow(exponential_decay_rate, exponent)


            print('Reducing learning rate-epoch = %d, current Learning rate= '%(epoch + 1), LR)
            print('-'*100)

            # shuffle the training data at the begining of each epoch!

            curr_train_dict = dataProcessing.shuffle_train_dict(train_dict, markets)

            # loop over all batches:
            for batch_number in range(total_batches):

                xBatch, trainY_batch = dataProcessing.next_batch_dict(curr_train_dict, batch_number,
                                                  batch_size, rem, Y_toUse, total_batches, markets)

                # run optimization

                _ = sess.run(optimizer, feed_dict={x: xBatch, y: trainY_batch, learning_rate: LR, keep_prob: dropout})

            # sharpe after each epoch on train and test

            train_preds_all_markets = []
            train_1day_returns_all_markets = []
            test_preds_all_markets = []
            test_1day_returns_all_markets = []

            for i in range(len(markets)):

                curr_epoch_train_preds = sess.run(output, feed_dict={x: train_dict[markets[i]][0],
                                                                       keep_prob : 1.0})[:, 0]
                curr_epoch_test_preds = sess.run(output, feed_dict={x: test_dict[markets[i]][0],
                                                                       keep_prob : 1.0})[:, 0]

                train_preds_all_markets.append(curr_epoch_train_preds)
                test_preds_all_markets.append( curr_epoch_test_preds)

                train_1day_returns_all_markets.append(train_dict[markets[i]][2])
                test_1day_returns_all_markets.append(test_dict[markets[i]][2])


            train_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(
                train_preds_all_markets, train_1day_returns_all_markets, markets, transCost_dict)

            train_basket_sharpes_array[epoch] = train_basket_sharpe

            print(' EPOCH %d- Train basket sharpe = '%epoch ,np.round( train_basket_sharpe, 3))

            # Individual market sharpes :training data:
            for m in range (len(markets)):

                train_indiv_sharpes[m, epoch, :] = losses_and_metrics.sharpe_likeModo(
                    train_preds_all_markets[m], train_1day_returns_all_markets[m],transCost_dict[markets[m]] )

            if (epoch + 1) % display_steps == 0:

                test_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(
                    test_preds_all_markets, test_1day_returns_all_markets, markets, transCost_dict)

                test_basket_sharpes_array[epoch] = test_basket_sharpe

                print(' EPOCH %d- Test basket sharpe = '%epoch, np.round(test_basket_sharpe, 3))

                print('*' * 10)

                # Individual market sharpes :training data:
                for m in range(len(markets)):

                    test_indiv_sharpes[m, epoch, :] = losses_and_metrics.sharpe_likeModo(
                        test_preds_all_markets[m], test_1day_returns_all_markets[m], transCost_dict[markets[m]])


        print(' saving model graph of all tensors to file')
        save_path = saver.save(sess, './MLP-checkpointFiles/MLP-checkPoint-run%d-%s.ckpt' %   (R + 1, optim))

        print('Optimization finished!')
        print(train_preds_all_markets[0][:10])
        # resetting the graph to be built again in the next iteration of for loop

    tf.reset_default_graph()

    fig = plt.figure(1, figsize=(20, 20))
    ax = fig.add_subplot(4, 2, 1)
    plt.plot(train_basket_sharpes_array, '-', label=' Train', c='red')
    plt.plot(test_basket_sharpes_array, '-', label=' Test', c='blue')

    plt.ylabel('Basket Sharpe')
    plt.legend(loc='lower right',ncol = 2)
    plt.axhline(0.5, c='black', linestyle='--')
    plt.axhline(0.6, c='black', linestyle='--')
    plt.axhline(0.7, c='black', linestyle='--')
    plt.axhline(0.8, c='black', linestyle='--')
    for epoch in range(100, training_epochs, 100):
        plt.axvline(x=epoch, linestyle='--', color='skyblue')
    plt.title('Basket Sharpe Learning Dynamics')
    plt.text(50, 1.6, 'initial LR= %.1e. Exponential Decay = %.4f' % (init_LR, np.round(exponential_decay_rate, 2)))
    plt.ylim(-1, 2)

    sharpes = ['ALL', 'Long', 'Short']
    colors = ['red', 'blue', 'green', 'skyblue', 'gold']

    plot_num = 3
    for i in range(3):
        ax = fig.add_subplot(4, 2, plot_num)
        for j in range(len(markets)):
            ax.plot(train_indiv_sharpes[j, :, i], '-', label=markets[j], c=colors[j])
            ax.set_title('Train %s sharpe by market' % sharpes[i])
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=0.5, linestyle='--')
            plt.axhline(y=1, linestyle='--')
            plt.ylabel('Sharpe')

            plt.ylim(-1, 2)
            for epoch in range(100, training_epochs, 100):
                plt.axvline(x=epoch, linestyle='--', color='skyblue')

        if plot_num == 3:
            plt.legend(loc='lower right', ncol=2)

        plot_num += 1

        ax = fig.add_subplot(4, 2, plot_num)
        for j in range(len(markets)):
            ax.plot(test_indiv_sharpes[j, :, i], '-', label=markets[j], c=colors[j])
            ax.set_title('Test %s sharpe by market ' % sharpes[i])
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=0.5, linestyle='--')
            plt.axhline(y=1, linestyle='--')
            for epoch in range(100, training_epochs, 100):
                plt.axvline(x=epoch, linestyle='--', color='skyblue')
            plt.ylim(-1, 2)
            plt.ylabel('Sharpe')

        plot_num += 1

    plt.xlabel('Epoch')
    plt.ylabel('Sharpe')
    #plt.show()


    fig.savefig('Learning-dynamics-plot/%d-N-l2Reg-%.3f-run%d.png' % (hidden1_size, l2Reg / 2., R + 1),
                bbox_inches='tight')
    plt.clf()
    plt.close()


