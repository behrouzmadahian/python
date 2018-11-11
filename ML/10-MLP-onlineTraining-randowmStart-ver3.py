import tensorflow as tf
import numpy as np
import initializers
import os
import pandas as pd
import math
import copy
import dataCLass
import losses_and_metrics
from Models import MLP
from matplotlib import pyplot as plt
import dataProcessing
import pickle
'''
An MLP using tensorflow. 
Used data: 5 market data: SQ, MQ, NQ, DQ, RN
I do k random start and check point twice during each run
for example if 50 epochs, checkpoint at epoch 25 and epoch 50!

'''
markets = ('SQ', 'NQ', 'MQ', 'DQ', 'RN')

# objective function
objective = losses_and_metrics.sum_sharpeLoss

# curr_optimizer = tf.train.AdamOptimizer
curr_optimizer = tf.train.RMSPropOptimizer

#optim = 'Adam'
optim = 'RMSprop'

results_folder = 'scaledY'
#results_folder = 'Adam'

# data parameters
lookback = 30
lookahead = 1
rolling_sd_window = 100
train_window = 500
slide_window = 40  # window of testing!

# training parameters:
init_LR = 0.00009
LR = copy.deepcopy(init_LR)
exponential_decay_rate1 = 0.9999
exponential_decay_rate2 = 0.998
exponential_decay_rate3 = 0.99

batch_size = 100
display_steps = 1

# network parameters:
network_activation = tf.nn.tanh
hidden1_size = 10
hidden2_size = 10
input_feats = lookback
l2Reg = 1.4

nRandom_start = 100
training_epochs = 300

# 1: scaled Y, 2: return 1 day
Y_toUse = 1
# loading data
datadir = '/%s-all-Adj.csv'
results_path = '/10Neurons/%s'
os.chdir(results_path % results_folder)

trans_data = pd.read_csv('TransactionCosts.csv')
trans_data = trans_data.values
transCost_dict = dict(zip(trans_data[:, 0], trans_data[:, 1]))

# initializing containers for predictions and getting data object
dataObj_init = dataCLass.Window_train_data(lookback=lookback,
                                           look_ahead=lookahead,
                                           sd_window=rolling_sd_window,
                                           train_window=train_window,
                                           slide_window=slide_window,
                                           markets=('SQ', 'NQ', 'MQ', 'DQ', 'RN'),
                                           nRandom_start=nRandom_start
                                           )
test_pred_dict = dataObj_init.prediction_containers()

print('Shape of test predcition array= N* RandomStart + 2=',test_pred_dict[markets[0]].shape)

total_training_steps = (dataObj_init.data_end_ind - dataObj_init.train_window) // dataObj_init.slide_window + 1

#total_training_steps = 5
# I will investigate performance for one window:
train_basket_sharpes_array = np.zeros((nRandom_start, total_training_steps ,training_epochs + 1)) #sharpe pretrainign the first row!
test_basket_sharpes_array = np.zeros((nRandom_start, total_training_steps, training_epochs + 1))


train_indiv_sharpes = np.zeros((nRandom_start, len(markets),total_training_steps ,training_epochs + 1, 3))
test_indiv_sharpes = np.zeros((nRandom_start, len(markets), total_training_steps,training_epochs + 1, 3))
print('Shape of sharpe array= ',test_indiv_sharpes.shape)

for R in range(nRandom_start):
    print('Random Start= ', R )
    print('Restarting data container indicies..')

    dataObj = dataCLass.Window_train_data(lookback = lookback, look_ahead = lookahead,
                                          sd_window = rolling_sd_window,
                                          train_window= train_window, slide_window= slide_window,
                                          markets=('SQ', 'NQ', 'MQ', 'DQ', 'RN'),
                                          nRandom_start=nRandom_start
                                          )

    trainX, trainY, train_ret_1day, train_dates, train_data_dict, test_data_dict =\
        dataObj.next_train_window()


    weights = {
    'h1':  initializers.glorot_uniform_weight_initializer([input_feats, hidden1_size]),
    'h2':  initializers.glorot_uniform_weight_initializer([hidden1_size, hidden2_size]),
    'out': initializers.glorot_uniform_weight_initializer([hidden2_size, 1], name='W_out')
    }
    biases = {
        'b1': initializers.bias_initializer([hidden1_size], name='B_h1'),
        'b2': initializers.bias_initializer([hidden2_size], name='B_h2'),
        'out': initializers.bias_initializer([1], name='B_out')
    }
    x = tf.placeholder(tf.float32, [None, input_feats])
    y = tf.placeholder(tf.float32, [None])
    learning_rate = tf.placeholder(tf.float32)
    optimizer, output, sharpe_plus_l2_loss = MLP(x,
                                                 y,
                                                 weights,
                                                 biases,
                                                 curr_optimizer,
                                                 learning_rate,
                                                 objective,
                                                 batch_size=batch_size,
                                                 markets=markets,
                                                 activation=network_activation,
                                                 l2Reg=l2Reg,
                                                 l2RegOutput=l2Reg * 0.5,
                                                 l2Reg_biases=l2Reg * 0.5)
    # initialize all tensors- to be run in Session!
    init = tf.global_variables_initializer()
    # saver for restoring the whole model graph of
    #  tensors from the  checkpoint file
    saver = tf.train.Saver()
    # launch default graph:
    with tf.Session() as sess:
        for train_step in range(total_training_steps):
            total_batches = train_data_dict[markets[0]][0].shape[0] // batch_size
            print("TRAIN STEP=", train_step, 'Prediction Start=', dataObj.pred_start[-1])
            print('='*20)
            # saver object for checkpointing
            saver = tf.train.Saver()
            # if initial basket sharpe on training is too high -> bad initialization- repeat initialization
            sess.run(init)
            train_preds_all_markets = {}
            train_1day_returns_all_markets = {}
            test_preds_all_markets = {}
            test_1day_returns_all_markets = {}
            for m in range(len(markets)):
                # DO NOT add duplicated training data in calculating metrics.
                train_preds_all_markets[markets[m]] = sess.run(output,
                                                               feed_dict={x: train_data_dict[markets[m]][0]})[:, 0][: dataObj.train_window]

                test_preds_all_markets[markets[m]] = sess.run(output,
                                                              feed_dict={x: test_data_dict[markets[m]][0]})[:, 0]

                train_1day_returns_all_markets[markets[m]] = train_data_dict[markets[m]][2][:dataObj.train_window]
                test_1day_returns_all_markets[markets[m]] = test_data_dict[markets[m]][2]

            train_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(train_preds_all_markets,
                                                                                  train_1day_returns_all_markets,
                                                                                  markets, transCost_dict)

            test_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(test_preds_all_markets,
                                                                                 test_1day_returns_all_markets,
                                                                                 markets, transCost_dict)

            train_basket_sharpes_array[R, train_step, 0] = train_basket_sharpe
            test_basket_sharpes_array[R, train_step, 0] = test_basket_sharpe

            # Per market Sharpes:
            for m in range(len(markets)):
                # Individual market sharpes :training data:
                train_indiv_sharpes[R, m, train_step, 0, :] = losses_and_metrics.sharpe_likeModo(
                    train_preds_all_markets[markets[m]],
                    train_1day_returns_all_markets[markets[m]],
                    transCost_dict[markets[m]])

                # Individual market sharpes :Test data:
                test_indiv_sharpes[R, m, train_step, 0, :] = losses_and_metrics.sharpe_likeModo(
                    test_preds_all_markets[markets[m]],
                    test_1day_returns_all_markets[markets[m]],
                    transCost_dict[markets[m]])

            print(' Pre- training Train basket sharpe = ', np.round(train_basket_sharpe, 3))
            print(' Pre- training Test basket sharpe = ', np.round(test_basket_sharpe, 3))
            print('=' * 100)

            for epoch in range(training_epochs):
                # reduce learning rate every 2 epochs
                if epoch > 300 and epoch <= 400:
                    exponential_decay_rate = exponential_decay_rate2
                elif epoch > 400:
                    exponential_decay_rate = exponential_decay_rate3
                else:
                    exponential_decay_rate = exponential_decay_rate1
                LR = init_LR * pow(exponential_decay_rate, epoch)
                # shuffle the training data at the begining of each batch!
                curr_train_dict = dataProcessing.shuffle_train_dict(train_data_dict, markets)
                rem = curr_train_dict[markets[0]][0].shape[0] % batch_size
                # loop over all batches:
                for batch_number in range(total_batches):
                    # I can replace this with next batch function that works on dictionaries, if I want to
                    # make sure batches have exact same date across markets
                    xBatch, trainY_batch = dataProcessing.next_batch_dict(
                                                        curr_train_dict, batch_number,
                                                        batch_size, rem,
                                                        Y_toUse, total_batches, markets)
                    # run optimization
                    _ = sess.run(optimizer, feed_dict={x: xBatch, y: trainY_batch, learning_rate: LR})

                # sharpe train and test after each epoch:
                # inidvidual markets Sharpes:

                train_preds_all_markets = {}
                train_1day_returns_all_markets = {}
                test_preds_all_markets = {}
                test_1day_returns_all_markets = {}

                for m in range(len(markets)):

                    curr_epoch_train_preds = \
                        sess.run(output, feed_dict={x: train_data_dict[markets[m]][0]})[:, 0][:dataObj.train_window]

                    curr_epoch_test_preds = sess.run(output, feed_dict={x: test_data_dict[markets[m]][0]})[:, 0]

                    train_preds_all_markets[markets[m]] = curr_epoch_train_preds
                    test_preds_all_markets[markets[m]] = curr_epoch_test_preds

                    train_1day_returns_all_markets[markets[m]] = train_data_dict[markets[m]][2][:dataObj.train_window]
                    test_1day_returns_all_markets[markets[m]] = test_data_dict[markets[m]][2]

                train_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(train_preds_all_markets,
                                                                                      train_1day_returns_all_markets,
                                                                                      markets, transCost_dict)


                test_basket_sharpe = losses_and_metrics.basket_trading_sharpeMetrics(test_preds_all_markets,
                                                                                     test_1day_returns_all_markets,
                                                                                     markets, transCost_dict)

                train_basket_sharpes_array[R, train_step, epoch + 1] = train_basket_sharpe
                test_basket_sharpes_array[R, train_step, epoch + 1] = test_basket_sharpe

                if (epoch + 1)  % 10 == 0:
                    print(' EPOCH %d- Train basket sharpe = ' % (epoch + 1), np.round(train_basket_sharpe, 3))
                    print(' EPOCH %d- Test basket sharpe = ' % (epoch + 1), np.round(test_basket_sharpe, 3))
                    print('Reducing learning rate-epoch = %d, current Learning rate= ' % (epoch + 1), LR)
                    print('=' * 50)


                # Per market Sharpes:
                for m in range(len(markets)):

                    # Individual market sharpes :training data:
                    train_indiv_sharpes[R, m, train_step, epoch + 1, :] = losses_and_metrics.sharpe_likeModo(
                                                                            train_preds_all_markets[markets[m]],
                                                                            train_1day_returns_all_markets[markets[m]],
                                                                            transCost_dict[markets[m]])

                    # Individual market sharpes :Test data:
                    test_indiv_sharpes[R ,m, train_step, epoch + 1, :] = losses_and_metrics.sharpe_likeModo(
                                                                            test_preds_all_markets[markets[m]],
                                                                            test_1day_returns_all_markets[markets[m]],
                                                                            transCost_dict[markets[m]])

            # Predictions
            for m in range(len(markets)):
                test_pred = sess.run(output, feed_dict={x: test_data_dict[markets[m]][0]})[:, 0]

                test_pred_dict[markets[m]][dataObj.test_start_ind - dataObj.train_window:
                dataObj.test_end_ind - dataObj.train_window, R + 2] = test_pred


            # check pointing after window
            print('End of training step. Saving weights to file...')
            saver.save(sess, './online-training-CPT/MLP-CP-run%d-pred-start-%d.ckpt' %
                               (R + 1, dataObj.pred_start[train_step]))
            print('=' * 100)

            try:
                # note the inidivual train x, trainy, .. have a dupicate
                #  copy of last 100 days for better learning current time!

                trainX, trainY, train_ret_1day, train_dates, train_data_dict, test_data_dict = dataObj.next_train_window()
            except:
                print("End of Training for current Start  reached!")
                print(test_data_dict[markets[0]][0].shape)
                print('*'*100)
                break


    tf.reset_default_graph()

    # resetting the index to the begining of data for new run:
    dataObj.start_over()


markets = ('SQ', 'NQ', 'MQ', 'DQ', 'RN')
#
for i in range(len(markets)):
    predsCols = ['dtStart', '%s-y-true' % markets[i]]
    predsCols.extend(['%s-pred%d' % (markets[i], j) for j in range(1, nRandom_start  + 1, 1)])

    test_df = pd.DataFrame(test_pred_dict[markets[i]], columns = predsCols)

    #train and test are both out of sample, for simplicity I add so other functions work
    # need to add train predictions~!

    test_df.to_csv('%s-testPreds.csv' % markets[i], index = False)
    test_df.to_csv('%s-trainPreds.csv' % markets[i], index = False)
#
#
# # plotting:
steps_to_Explore = np.arange(total_training_steps)
ranodmStart = 0
for step in steps_to_Explore:

    ymax = np.max(train_indiv_sharpes[ranodmStart, :, step, :, :])
    ymin = np.min(test_indiv_sharpes[ranodmStart, :, step, :, :])

    for R in range(nRandom_start):

        fig = plt.figure(1, figsize=(25, 20))
        ax = fig.add_subplot(4, 2, 1)
        plt.plot(train_basket_sharpes_array[R, step, :], '-', label=' Train', c='red')
        plt.plot(test_basket_sharpes_array[R, step, :], '-', label=' Test', c='blue')

        plt.ylabel('Basket Sharpe')
        plt.legend(loc='lower right', ncol=2)
        plt.axhline(0.5, c='black', linestyle='--')
        plt.axhline(0.6, c='black', linestyle='--')
        plt.axhline(0.7, c='black', linestyle='--')
        plt.axhline(0.8, c='black', linestyle='--')

        for epoch in range(100, training_epochs, 100):
            plt.axvline(x=epoch, linestyle='--', color='skyblue')

        plt.title('Basket Sharpe Learning Dynamics')
        plt.text(50, 1.6, 'initial LR= %.1e. Exponential Decay = %.4f' %
                 (init_LR, np.round(exponential_decay_rate, 2)))
        #plt.ylim(ymin, ymax)

        sharpes = ['ALL', 'Long', 'Short']
        colors = ['red', 'blue', 'green', 'skyblue', 'gold']
        plot_num = 3

        for sharpeType in range(3):
            ax = fig.add_subplot(4, 2, plot_num)

            for m in range(len(markets)):
                ax.plot(train_indiv_sharpes[R, m, step, :, sharpeType], '-', label=markets[m], c=colors[m])
                ax.set_title('Train %s sharpe by market' % sharpes[sharpeType])
                plt.axhline(y=0, linestyle='--')
                plt.axhline(y=0.5, linestyle='--')
                plt.axhline(y=1, linestyle='--')
                plt.ylabel('Sharpe')

               # plt.ylim(ymin, ymax)
                for epoch in range(100, training_epochs, 300):
                    plt.axvline(x=epoch, linestyle='--', color='skyblue')

            if plot_num == 3:
                plt.legend(loc='lower right', ncol=2)

            plot_num += 1

            ax = fig.add_subplot(4, 2, plot_num)
            for m in range(len(markets)):
                ax.plot(test_indiv_sharpes[R, m, step, :, sharpeType], '-', label=markets[m], c=colors[m])
                ax.set_title('Test %s sharpe by market ' % sharpes[sharpeType])
                plt.axhline(y=0, linestyle='--')
                plt.axhline(y=0.5, linestyle='--')
                plt.axhline(y=1, linestyle='--')
                for epoch in range(100, training_epochs, 300):
                    plt.axvline(x=epoch, linestyle='--', color='skyblue')
               # plt.ylim(ymin, ymax)
                plt.ylabel('Sharpe')

            plot_num += 1

        plt.xlabel('Epoch')
        plt.ylabel('Sharpe')
        #plt.show()

        fig.savefig('Learning-dynamics-plot/%d-N-l2Reg-%.3f-run%d-trainStep%d.png' %
                    (hidden1_size, l2Reg / 2., R + 1, step),
                    bbox_inches='tight')
        plt.clf()
        plt.close()


file_obj = open ('Train-individual-market-Sharpes', 'wb')
pickle.dump(train_indiv_sharpes, file_obj)
file_obj.close()
file_obj = open('Test-individual-market-Sharpes', 'wb')
pickle.dump(test_indiv_sharpes, file_obj)
file_obj.close()

# Rolling Sharpe Plot:

for R in range(nRandom_start):
    ymax = np.max(train_indiv_sharpes[R, :, :, :, -1])
    ymin = np.min(test_indiv_sharpes[R, :, :, :, -1])

    fig = plt.figure(1, figsize=(25, 20))
    ax = fig.add_subplot(4, 2, 1)
    plt.plot(train_basket_sharpes_array[R, :, -1], '--',marker = 'o', label=' Train', c='red')
    plt.plot(test_basket_sharpes_array[R, :, -1], '--', marker = 'o', label=' Test', c='blue')

    plt.ylabel('Basket Sharpe')
    plt.legend(loc='lower center', ncol=2)
    plt.axhline(0.5, c='black', linestyle='--')
    plt.axhline(0.6, c='black', linestyle='--')
    plt.axhline(0.7, c='black', linestyle='--')
    plt.axhline(0.8, c='black', linestyle='--')


    plt.title('Basket Sharpe Learning Dynamics')
    #plt.text(50, 1.6, 'initial LR= %.1e. Exponential Decay = %.4f' %
     #        (initial_LR, np.round(exponential_decay_rate, 2)))
    #plt.ylim(ymin, ymax)

    sharpes = ['ALL', 'Long', 'Short']
    colors = ['red', 'blue', 'green', 'skyblue', 'gold']
    plot_num = 3

    for sharpeType in range(3):
        ax = fig.add_subplot(4, 2, plot_num)

        for m in range(len(markets)):
            ax.plot(train_indiv_sharpes[R, m, :, -1, sharpeType], '--',marker = 'o', label=markets[m], c=colors[m])
            ax.set_title('Train %s sharpe by market' % sharpes[sharpeType])
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=0.5, linestyle='--')
            plt.axhline(y=1, linestyle='--')
            plt.ylabel('Sharpe')

            #plt.ylim(ymin, ymax)


        if plot_num == 3:
            plt.legend(loc='lower right', ncol=2)

        plot_num += 1

        ax = fig.add_subplot(4, 2, plot_num)
        for m in range(len(markets)):
            ax.plot(test_indiv_sharpes[R, m, :, -1, sharpeType], '--',marker = 'o', label=markets[m], c=colors[m])
            ax.set_title('Test %s sharpe by market ' % sharpes[sharpeType])
            plt.axhline(y=0, linestyle='--')
            plt.axhline(y=0.5, linestyle='--')
            plt.axhline(y=1, linestyle='--')

            #plt.ylim(ymin, ymax)
            plt.ylabel('Sharpe')

        plot_num += 1

    plt.xlabel('Training Step')
    plt.ylabel('Sharpe')
    # plt.show()
    #
    fig.savefig('Learning-dynamics-plot/RollingSharpe-%d-N-l2Reg-%.3f-run%d.png' % (hidden1_size, l2Reg / 2., R + 1),
                 bbox_inches='tight')
    plt.clf()
    plt.close()
