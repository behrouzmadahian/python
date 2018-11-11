import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

import pandas as pd
import tensorflow as tf
import time
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

'''
The idea is to approximate the function using a Gaussian process. In other words the function values are assumed
to follow a multivariate gaussian. The covariance of the function values are given by a GP 
kernel between the parameters.
 Then a smart choice to choose the next parameter to evaluate can be made by the acquisition function 
 over the Gaussian prior which is much quicker to evaluate.
We explore Gaussian Process optimization in the scikit-optimize package
dimensions :
we define the parameters we want to optimize. bounds for real ones and lists of categories for categorical ones.
boundsBayesOpt = [Real(-5, -3,'uniform', 'init_lr'),
                  Real(-8, -5, 'uniform', 'l2Reg'),
                  Real(0, 0.5, 'uniform')
                  ]
https://scikit-optimize.github.io/
Acquisition functions available:
-"LCB" for lower confidence bound.
-"EI" for negative expected improvement.
-"PI" for negative probability of improvement.
-"gp_hedge:"
Probabilistically choose one of the above three acquisition functions at every iteration.
The weightage given to these gains can be set by \eta through acq_func_kwargs.
1. The gains g_i are initialized to zero.
At every iteration,
2. Each acquisition function is optimised independently to propose an candidate point X_i.
3. Out of all these candidate points, the next point X_best is chosen by softmax(\eta g_i)
After fitting the surrogate model with (X_best, y_best), the gains are updated such that g_i -= \mu(X_i)

"EIps" for negated expected improvement per second to take into account the function compute time. 
Then, the objective function is assumed to return two values,
 the first being the objective value and the second being the time taken.
"PIps" for negated probability of improvement per second.

n_jobs:
 [int, default=1] Number of cores to run in parallel while running the lbfgs
optimizations over the acquisition function. Valid only when acq_optimizer is set to "lbfgs."
n_jobs= -1:
 all CPU cores!! 
'''


def performance_statistics(y_true, y_pred, y_probs):
    pos_recall = recall_score(y_true, y_pred, pos_label=1)
    pos_precision = precision_score(y_true, y_pred, pos_label=1)
    pos_f1 = f1_score(y_true, y_pred, pos_label=1)
    neg_recall = recall_score(y_true, y_pred, pos_label=0)
    neg_precision = precision_score(y_true, y_pred, pos_label=0)
    neg_f1 = f1_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(y_true, y_probs, average=None)
    return {'prec_recal_f1_pos':  (round(pos_precision, 3), round(pos_recall, 3), round(pos_f1, 3)),
            'prec_recall_f1_neg': (round(neg_precision, 3), round(neg_recall, 3), round(neg_f1, 3)),
            'AUC': round(auc, 3)}


def class_pred (probs, thresh):
    '''returns the associated class'''
    preds = [1 if p >= thresh else 0 for p in probs]
    return np.array(preds)


def data_class_div(x, y):
    '''
    given the dataframe, different data frame is returned for each class value.
    RETURNs numpy arrays!
    '''
    pos_x, pos_y = x[y == 1], y[y == 1]
    neg_x, neg_y = x[y == 0], y[y == 0]
    return pos_x, pos_y, neg_x, neg_y


def next_batch(x, y, n_samples, batch_size, iteration):
    '''
    x, and y are numpy arrays of data from a specific class!
    returns next batch for the class.
    ALL DATA ARE NUMPY ARRAYS.
    :return: a random batch of data from the associated class!
    '''
    if iteration * batch_size >= n_samples:
        inds = np.arange(n_samples)
        np.random.shuffle(inds)
        x = x[inds]
        y = y[inds]
        start_ind = 0
        end_ind = batch_size
        iteration = -1
    else:
        start_ind = iteration * batch_size
        end_ind = (iteration + 1) * batch_size
    return x[start_ind: end_ind], y[start_ind:end_ind], iteration


def validation_split(trainx, trainy, frac=0.2):
    '''
    train is data frame
    returns train and validation data
    '''
    pos_x, pos_y, neg_x, neg_y = data_class_div(trainx, trainy)
    n_pos = pos_x.shape[0]
    n_neg = neg_x.shape[0]
    pos_inds = np.arange(n_pos)
    neg_inds = np.arange(n_neg)
    np.random.shuffle(pos_inds)
    np.random.shuffle(neg_inds)
    trainx_pos = pos_x[pos_inds[int(frac * n_pos):]]
    trainy_pos = pos_y[pos_inds[int(frac * n_pos):]]
    trainx_neg = neg_x[neg_inds[int(frac * n_neg):]]
    trainy_neg = neg_y[neg_inds[int(frac * n_neg):]]
    validx_pos = pos_x[pos_inds[:int(frac * n_pos)]]
    validy_pos = pos_y[pos_inds[:int(frac * n_pos)]]
    validx_neg = neg_x[neg_inds[:int(frac * n_neg)]]
    validy_neg = neg_y[neg_inds[:int(frac * n_neg)]]
    trainx = np.append(trainx_pos, trainx_neg, axis=0)
    trainy = np.append(trainy_pos, trainy_neg)
    validx = np.append(validx_pos, validx_neg, axis=0)
    validy = np.append(validy_pos, validy_neg)
    return trainx, trainy, validx, validy


def model(x1,
          y,
          activation,
          istrain,
          h1_size,
          h2_size,
          l2Reg,
          dropoutRate,
          init_lr,
          optimizer):
    with tf.variable_scope('wide_model'):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
        else:
            bias_init = tf.zeros_initializer()
        h1 = tf.layers.dense(x1,
                             h1_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h1 = tf.layers.dropout(h1, rate=dropoutRate, training=istrain)

        h2 = tf.layers.dense(h1,
                             h2_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h2 = tf.layers.dropout(h2, rate=dropoutRate, training=istrain)
        logits = tf.squeeze(
            tf.layers.dense(h2,
                            1,
                            activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer()))
        out_prob = tf.nn.sigmoid(logits)
        out_pred = tf.cast(tf.greater(out_prob, 0.88), tf.float32)  # theshold set at statistics of train

        print('Shape of output= ', logits.get_shape())
        # loss = tf.reduce_mean(tf.reduce_sum(y * tf.log(output) + (1-y)*(1-tf.log(output)),
        #  name='output_layer', axis=1))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
        l2Loss = sum(tf.nn.l2_loss(p) for p in tf.trainable_variables()) * l2Reg
        loss += l2Loss
        global_step = tf.Variable(0, trainable=False)

        '''
        decayed_lr = init_lr / (1+ decay_rate* floor(global_step/ decay_step))
        '''
        learning_rate = tf.train.inverse_time_decay(init_lr,
                                                    global_step=global_step,
                                                    decay_steps=20000,
                                                    decay_rate=0.1,
                                                    staircase=True)
        #######################
        # every time minimize is called, global step will increment by 1!
        train_op = optimizer(learning_rate).minimize(loss)
        correct_pred = tf.equal(y, out_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return train_op, loss, out_prob, accuracy


data_dir = 'C:/behrouz/projects/data/O365_Business_Premium_solo_2017-06-28/'
train = pd.read_csv(data_dir + 'train_Processed_1branch_byCountry.csv')
# drop weight column!!!
train = train.drop(['Weight'], axis=1)
feature_columns = train.columns.values[2:]
class_col = 'outHasActiveO365'
trainx_np, trainy_np, validationx_np, validationy_np = validation_split(train[feature_columns].values,
                                                                        train[class_col].values,
                                                                        frac=0.2)
n_samples = trainx_np.shape[0]
activation = tf.nn.relu
optimizer = tf.train.AdamOptimizer
batch_size = 256
optimization_iters = 40000


def build_run_training(init_lr, dropoutRate):
    ############################
    # we search in log scale for learning rate and regularization
    l2Reg = 0
    init_lr = 10**init_lr
    dropoutRate = dropoutRate
    h1, h2 = 64, 64
    print('Size of layers= ', h1, h2)
    ############################
    print('Shape of Train and validation splits= ', trainx_np.shape, trainy_np.shape)
    x = tf.placeholder(tf.float32, shape=[None, trainx_np.shape[1]])
    y = tf.placeholder(tf.float32, [None])
    istrain = tf.placeholder(tf.bool)

    train_op, loss, out_prob, accuracy = model(x,
                                               y,
                                               activation,
                                               istrain,
                                               h1,
                                               h2,
                                               l2Reg,
                                               dropoutRate,
                                               init_lr,
                                               optimizer)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('-' * 200)
        print('h1=', h1, 'h2=', h2, 'init_lr=', init_lr, 'dropout=', dropoutRate, 'L2Reg=', l2Reg, '\n')
        k = 0  # used to track end of train cycle < epoch >
        for i in range(optimization_iters):
            x_batch, y_batch, k = next_batch(trainx_np, trainy_np, n_samples, batch_size, k)
            k += 1
            _ = sess.run(train_op, feed_dict={x: x_batch, y: y_batch, istrain: True})
            if (i+1) % 20000 == 0:
                print('Current Run-Iteration %d' % (i+1))

        print()
        print('Optimization finished. ')
        valid_loss, valid_prob, valid_accu = sess.run([loss, out_prob, accuracy],
                                                      feed_dict={x: validationx_np,
                                                                 y: validationy_np,
                                                                 istrain: False})
        valid_pred = class_pred(valid_prob, 0.88)

        valid__pref = performance_statistics(validationy_np, valid_pred, valid_prob)
        print('Performance on Validation=', ' Loss=', valid_loss, '   ', valid__pref, '\n')
    tf.reset_default_graph()
    return valid_loss


def run_model(dimensions):
    # init_lr, l2Reg, dropoutRate, h1, h2 = dimensions
    init_lr, dropoutRate = dimensions
    t1 = time.time()
    validation_loss = build_run_training(init_lr=init_lr,
                                         # l2Reg=l2Reg,
                                         dropoutRate=dropoutRate,
                                         )
    t2 = time.time()
    print('Run time of one iteration of Hyper optimization(Min)= ', round((t2-t1)/60, 2))
    print('-' * 200)
    return validation_loss


if __name__=='__main__':
    boundsBayesOpt = [Real(-7, -3, 'uniform', name='init_lr'),
                      # Real(-8, -5, 'uniform', name='l2Reg'),
                      Real(0, 0.5, 'uniform', name='dropoutRate'),
                      # Categorical([128], name='h1'),
                      # Categorical([16, 32, 64, 128, 256], name='h2')
                      ]
    t1 = time.time()
    results = gp_minimize(run_model,
                          boundsBayesOpt,
                          n_calls=100,
                          n_jobs=-1,
                          acq_func='gp_hedge',
                          acq_optimizer='lbfgs',
                          random_state=None,
                          verbose=True)  # set it for reproducible results
    t2 = time.time()
    print('Best parameters Obtained:')
    # print('Learning Rate=', 10 ** results.x[0], 'L2 Reg=',
    #       10 ** results.x[1], 'Dropout=', results.x[2], 'h1=', results.x[3], 'h2=', results.x[4])
    print('Learning Rate=', 10 ** results.x[0], 'Dropout= ', results.x[1])
    print('Validation Loss at the optimum obtaine=', results.fun)
    print('Function Value at each iteration plot:')
    print('Total Time(min) for optimization= ', round((t2-t1)/60, 2))
    from matplotlib import pyplot as plt

    plt.plot(results.func_vals)
    plt.ylabel('Validation Loss')
    plt.xlabel('Iteration')
    plt.show()



