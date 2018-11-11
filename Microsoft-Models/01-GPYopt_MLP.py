import numpy as np
import pandas as pd
import tensorflow as tf
import GPyOpt, time
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

'''
Second column is the binary outcome.
Last column is the integer code value for country and we use it to learn the embedding for it. 
Aquisition type describing how to select the next sample of parameters to try:

1. Maximum probability of improvement (MPI):
2. Expected improvement (EI):
3. Upper confidence bound (UCB):
'''


def performance_statistics(y_true, y_pred, y_probs):
    pos_recall = recall_score(y_true, y_pred, pos_label=1)
    pos_precision = precision_score(y_true, y_pred, pos_label=1)
    pos_f1 = f1_score(y_true, y_pred, pos_label=1)
    neg_recall = recall_score(y_true, y_pred, pos_label=0)
    neg_precision = precision_score(y_true, y_pred, pos_label=0)
    neg_f1 = f1_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(y_true, y_probs, average=None)
    return {'prec_recal_f1_pos':(round(pos_precision, 3), round(pos_recall, 3), round(pos_f1, 3)),
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


def model(x1, y, activation, istrain, h1_size,
          h2_size, l2Reg, dropoutRate, init_lr, optimizer):
    with tf.variable_scope('wide_model'):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
        else:
            bias_init = tf.zeros_initializer()
        h1 = tf.layers.dense(x1, h1_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h1 = tf.layers.dropout(h1, rate=dropoutRate, training=istrain)

        h2 = tf.layers.dense(h1, h2_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h2 = tf.layers.dropout(h2, rate=dropoutRate, training=istrain)
        logits = tf.squeeze(
            tf.layers.dense(h2, 1,
                            activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer()))
        out_prob = tf.nn.sigmoid(logits)
        out_pred = tf.cast(tf.greater(out_prob, 0.88), tf.float32)

        print('Shape of output= ', logits.get_shape())
        # loss = tf.reduce_mean(tf.reduce_sum(y * tf.log(output) + (1-y)*(1-tf.log(output)), name='output_layer', axis=1))
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
optimization_iters = 100000


def build_run_training(init_lr, dropoutRate, l2Reg):
    ############################
    # we search in log scale for learning rate and regularization
    l2Reg = 10**l2Reg
    init_lr = 10**init_lr
    dropoutRate = dropoutRate
    h1, h2 = 32, 32
    ############################
    print('Shape of Train and validation splits= ', trainx_np.shape, trainy_np.shape)
    x = tf.placeholder(tf.float32, shape=[None, trainx_np.shape[1]])
    y = tf.placeholder(tf.float32, [None])
    istrain = tf.placeholder(tf.bool)

    train_op, loss, out_prob, accuracy = model(x, y, activation, istrain, h1, h2, l2Reg,
                                               dropoutRate, init_lr, optimizer)
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


def run_model(bounds):
    t1 = time.time()
    validation_loss = build_run_training(
                                         init_lr=float(bounds[:, 0]),
                                         l2Reg=float(bounds[:, 1]),
                                         dropoutRate=float(bounds[:, 2]))
    t2 = time.time()
    print('Run time of one iteration of Hyper optimization(Min)= ', round((t2-t1)/60, 2))
    print('-' * 200)
    return validation_loss


def bayesOpt(max_iter):
    opt = GPyOpt.methods.BayesianOptimization(run_model,
                                              domain=boundsBayesOpt,
                                              aquisition_type='MPI',
                                              exact_feval=True)  # selects the MPI
    opt.run_optimization(max_iter=max_iter)
    print('Bayes Opt Results: ')
    print(opt.x_opt)
    print('Best Parameters found in optimization:')
    print('learning rate=', 10**opt.x_opt[0],
          'l2 regularization =', 10**opt.x_opt[1],
          'dropout= ', opt.x_opt[2])
    return opt.x_opt


if __name__ == '__main__':
    boundsBayesOpt = [#{'name': 'h1', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
                      #{'name': 'h2', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
                      {'name': 'init_lr', 'type': 'continuous', 'domain': (-5, -3)},
                      {'name': 'l2Reg', 'type': 'continuous', 'domain': (-8, -5)},
                      {'name': 'dropoutRate', 'type': 'continuous', 'domain': (0.0, 0.5)}
                      ]
    bayesOpt(max_iter=400)

