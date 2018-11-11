import tensorflow as tf
import numpy as np
import initializers
import os
import pandas as pd
import dataProcessing
import copy
import losses_and_metrics
from Models import MLP_1layer
import multiprocessing
import time
import shutil
from matplotlib import colors

'''
An MLP using tensorflow. 
Used data: 5 market data: SQ, MQ, NQ, DQ, RN
'''
# hidden_size_grid = [5, 10, 15, 20]
    # learning_rate_grid = [0.001, 0.0005, 0.0001, 0.00005]
    # epoch_grid = [5, 10, 15, 20, 30]
    # l2_grid = [0, 0.01, 0.1, 1, 5]


def myMainFunc(random_start_indicies):
    markets = ('SQ', 'MQ', 'NQ', 'DQ', 'RN')

    # objective function
    # objective = losses_and_metrics.sum_sharpeLoss
    objective = losses_and_metrics.sharpeLoss
    # objective = losses_and_metrics.basket_trading_pos_size_sharpeLoss

    curr_optimizer = tf.train.AdamOptimizer
    # curr_optimizer = tf.train.RMSPropOptimizer


    # data parameters
    Y_toUse = 1  # 1: scaled return, 2:1-day return
    lookback = 30
    lookahead = 1
    rolling_sd_window = 100

    # training parameters:

    batch_size = 100
    # network parameters:
    network_activation = tf.nn.tanh
    dropout = 1.

    input_feats = lookback
    test_start_date = 20070418

    hidden_size_grid = [5, 10, 15, 20]
    learning_rate_grid = [0.001, 0.0005, 0.0001, 0.00005]
    epoch_grid = [50, 100, 150, 200, 300]
    l2_grid = [0, 0.01, 0.1, 1, 5]
    valid_frac = 0.2

    # loading data
    datadir = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/hyper-param-optimization/tf-hyperParam-opt/data/%s_Commision-and-Slippage-limits.csv'

    # get the common dates and then merge each data making sure they have common dates:

    data = pd.read_csv(datadir % markets[0])
    for i in range(1, len(markets), 1):
        data1 = pd.read_csv(datadir % markets[i])
        data = pd.merge(data, data1, on='dtStart', how='inner')

    dates = data[['dtStart']]

    # getting random piece but common indicies from train  as validation

    test_start_ind = int(np.where(dates.values == test_start_date)[0]) - rolling_sd_window - lookback - lookahead
    inds = np.arange(test_start_ind)

    valid_inds = pd.read_csv('Validation_indicies.csv').values
    valid_inds = valid_inds.flatten()

    #valid_inds = np.random.choice(inds, size=int(valid_frac * test_start_ind), replace=False)
    #valid_inds = np.sort(valid_inds)
    # writing validation indicies to file
   # valid_inds_df = pd.DataFrame(valid_inds)
    #valid_inds_df.to_csv('Validation_indicies.csv', index=False)
    train_inds = [i for i in inds if i not in valid_inds]
    test_dict = {}
    train_dict = {}
    validation_dict ={}

    for i in range(0, len(markets), 1):
        data = pd.read_csv(datadir % markets[i])

        # Make sure we get data from all  markets on exact common dates
        data = pd.merge(data, dates, on='dtStart', how='inner')

        curr_market_data = \
            dataProcessing.time_series_toMatrix(data, train_inds, valid_inds , 20070418, lookback = lookback,
                                                look_ahead = lookahead, sd_window = rolling_sd_window)


        train_dict[markets[i]] = copy.deepcopy(curr_market_data[:4])
        validation_dict[markets[i]] = copy.deepcopy(curr_market_data[4:8])
        test_dict[markets[i]] = copy.deepcopy(curr_market_data[8:])

    total_batches = train_dict[markets[0]][0].shape[0] // batch_size
    rem = train_dict[markets[0]][0].shape[0] % batch_size

    print('TOTAL BATCHES+++++++++++++++++++++++', total_batches)

    for R in random_start_indicies:
        print('RUN %d optimization begins..' % R )

        for hidden_size in  hidden_size_grid:

            for LR in learning_rate_grid:

                for training_epochs in epoch_grid:

                    for l2Reg in l2_grid:

                        print('Hidden Size =', hidden_size, 'Learning rate=', LR,
                              'TrainingEpochs=', training_epochs, 'L2 Reg=', l2Reg)

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
                        # placeholders
                        x = tf.placeholder(tf.float32, [None, input_feats])
                        y = tf.placeholder(tf.float32, [None])
                        learning_rate = tf.placeholder(tf.float32)
                        keep_prob = tf.placeholder(tf.float32)

                        optimizer, output, sharpe_plus_l2_loss = \
                            MLP_1layer(x, y, weights, biases, keep_prob, curr_optimizer, learning_rate, objective,
                                batch_size=batch_size,
                                markets=markets,
                                activation=network_activation, l2Reg=l2Reg, l2RegOutput=l2Reg * 1.,
                                l2Reg_biases=l2Reg * 1.)

                        # initialize all tensors- to be run in Session!

                        init = tf.global_variables_initializer()

                        # saver for restoring the whole model graph of
                        #  tensors from the  checkpoint file

                        saver = tf.train.Saver()

                        # launch default graph:
                        with tf.Session() as sess:

                            sess.run(init)

                            # training cycle:
                            for epoch in range(training_epochs):

                                # shuffle the training data at the begining of each epoch!

                                curr_train_dict = dataProcessing.shuffle_train_dict(train_dict, markets)

                                # loop over all batches:
                                for batch_number in range(total_batches):
                                    xBatch, trainY_batch = dataProcessing.next_batch_dict(curr_train_dict, batch_number,
                                                                                          batch_size, rem, Y_toUse,
                                                                                          total_batches,
                                                                                          markets)
                                    # run optimization

                                    _ = sess.run(optimizer,
                                                 feed_dict={x: xBatch, y: trainY_batch, learning_rate: LR,
                                                            keep_prob: dropout})

                            #print(' Optimization finished! saving model graph of all tensors to file')

                            save_path = saver.save(sess, './MLP-checkpointFiles/run%d-s-%d-LR-%.6f-epoch-%d-l2-%.5f.ckpt'
                                                   % (R , hidden_size, LR,  training_epochs, l2Reg))

                        # resetting the graph to be built again in the next iteration of for loop

                        tf.reset_default_graph()
    return random_start_indicies

def myParrallelFunc():
    random_start_indicies = np.arange(1, 21, 1)

    cpu_cnt = 20  # only use K cores!
    #cpu_cnt = multiprocessing.cpu_count()
    print('Number of procesors: %d' % cpu_cnt)

    pool = multiprocessing.Pool(processes= cpu_cnt)  # we can define less number of processors

    print('Build a list of tasks for each processor..')

    tasks = []
    cpu_cnt = np.minimum(len(random_start_indicies), cpu_cnt)
    slice = int(len(random_start_indicies) / cpu_cnt)  # gives number of l2Reg to assign to one processor

    for i in range(cpu_cnt):
        a = random_start_indicies[i * slice:(i + 1) * slice]
        tasks.append([a])

    print(tasks)
    results = [pool.apply_async(myMainFunc, t) for t in tasks]
    return results


if __name__ == '__main__':
    # if results folder Exists remove and recreate it!
    # if os.path.exists(results_path%results_folder):
    #        shutil.rmtree(results_path%results_folder)
    # os.mkdir(results_path%results_folder)
    # os.mkdir('MLP-checkpointFiles')
    # os.mkdir('Learning-dynamics-plot')
    # os.mkdir('Results')

    cpu_cnt = 20

    results_path = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/hyper-param-optimization/tf_hypeOpt-files/%s'

    # results_folder = 'dropout-sharpe-sumMarket-loss'
    results_folder = '1Layer-sharpeloss1'
    # results_folder = 'dropout-sharpe-basketloss'

    os.chdir(results_path % results_folder)


    t1 = time.time()
    results = myParrallelFunc()

    print('Getting the results of each worker node..')

    for i in range(cpu_cnt):
        child_process_result = results[i]

        try:
            child_process_result = child_process_result.get(timeout = None)
            print('Chid Process %d finished completely'% i )

        except TimeoutError:
            print('Chid Process %d time out!'% i )

    t2 = time.time()
    random_start_indicies = np.arange(1, 21, 1)
    print('TOTAL ELAPSED TIME FOR %d runs='%len(random_start_indicies), np.round((t2 -t1)/ 60., 2))

