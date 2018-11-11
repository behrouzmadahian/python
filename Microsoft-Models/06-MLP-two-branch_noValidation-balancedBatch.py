import numpy as np
import pandas as pd
import tensorflow as tf
import os
import argparse, sys
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

'''
Second column is the binary outcome.
Last column is the integer code value for country and we use it to learn the embedding for it. 
To do:
Add another layer on top of concatenated hidden layer to get more interactions between countries and other feats!
GET AUC for test!!
equal sampling of positive and negative classes.
'''
FLAGS = None


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
    returns next batch for the class. The batch is taken randomly!
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


def validation_split(train, class_colname, frac=0.2):
    '''
    train is data frame
    returns train and validation data
    '''
    pos_x, pos_y, neg_x, neg_y = data_class_div(train, class_colname)
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


def wide_model(x1, activation, istrain, h1_size, h2_size):
    with tf.variable_scope('wide_model'):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
        else:
            bias_init = tf.zeros_initializer()
        h1 = tf.layers.dense(x1, h1_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h1 = tf.layers.dropout(h1, rate=FLAGS.dropoutRate, training=istrain)

        h2 = tf.layers.dense(h1, h2_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h2 = tf.layers.dropout(h2, rate=FLAGS.dropoutRate, training=istrain)
        return h2


def deep_model(ids,
               activation,
               istrain,
               embed_size,
               h1_size,
               h2_size,
               h3_size):
    with tf.variable_scope('deep_model'):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
        else:
            bias_init = tf.zeros_initializer()
        embedding_weights = tf.get_variable('embed_weight',
                                            shape=[FLAGS.n_countries, embed_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
        emb = tf.nn.embedding_lookup(embedding_weights, ids)
        print('Shape of embedding for the current batch= ', emb.get_shape())
        h1 = tf.layers.dense(emb,
                             h1_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h1 = tf.layers.dropout(h1, rate=FLAGS.dropoutRate, training=istrain)

        h2 = tf.layers.dense(h1,
                             h2_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h2 = tf.layers.dropout(h2, rate=FLAGS.dropoutRate, training=istrain)

        h3 = tf.layers.dense(h2,
                             h3_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        h3 = tf.layers.dropout(h3, rate=FLAGS.dropoutRate, training=istrain)
        return h3


def build_run_training():
    best_train_loss = np.inf
    data_dir = 'C:/behrouz/projects/MLP-2Branch/O365_Business_Premium_solo_2017-06-28/'
    if not os.path.exists(data_dir + '/balanced_batch_check_point/'):
        os.makedirs(data_dir + '/balanced_batch_check_point/')
    train = pd.read_csv(data_dir + 'train_Processed.csv')
    # drop weight column!!!
    train = train.drop(['Weight'], axis=1)
    feature_columns = train.columns.values[2:]
    class_col = 'outHasActiveO365'
    trainx_np, trainy_np = train[feature_columns].values, train[class_col].values
    print('Shape of Train = ', trainx_np.shape, trainy_np.shape)
    trainx_pos, trainy_pos, trainx_neg, trainy_neg = data_class_div(trainx_np, trainy_np)
    print(trainx_pos.shape, trainy_pos.shape, trainx_neg.shape, trainy_neg.shape)
    n_samples_pos = trainx_pos.shape[0]
    n_samples_neg = trainx_neg.shape[0]
    test = pd.read_csv(data_dir + 'test_Processed.csv')
    test = test.drop(['Weight'], axis=1)
    testx_np = test[test.columns.values[2:]].values
    testy_np = test['outHasActiveO365'].values

    x = tf.placeholder(tf.float32, shape=[None, FLAGS.wide_feats + 1])
    y = tf.placeholder(tf.float32, [None])
    istrain = tf.placeholder(tf.bool)
    # -1: include all data in  this dimension
    x1 = tf.slice(x, begin=[0, 0], size=[-1, FLAGS.wide_feats])
    x2 = tf.cast(tf.slice(x, begin=[0, FLAGS.wide_feats], size=[-1, 1]), tf.int32)
    print('Shape of input to wide and deep models= ', x1.get_shape(), x2.get_shape())

    wide_out = wide_model(x1, FLAGS.activation, istrain, FLAGS.wh1, FLAGS.wh2)
    deep_out = deep_model(x2, FLAGS.activation, istrain, FLAGS.embedding_size, FLAGS.dh1, FLAGS.dh2, FLAGS.dh3)
    # removing dimension with size=1
    # deep_out = tf.squeeze(deep_out)
    deep_out = tf.reshape(deep_out, (-1, deep_out.get_shape()[2]))
    print('Shape of output of wide and deep models= ', wide_out.get_shape(), deep_out.get_shape())
    hidden = tf.concat([wide_out, deep_out], axis=1)
    print('Shape of concatenated hidden layer = ', hidden.get_shape())

    logits = tf.squeeze(tf.layers.dense(hidden,
                                        1,
                                        activation=None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer()))
    out_prob = tf.nn.sigmoid(logits)
    out_pred = tf.cast(tf.greater(out_prob, 0.5), tf.float32)

    print('Shape of output= ', logits.get_shape())
    # loss = tf.reduce_mean(tf.reduce_sum(y * tf.log(output) + (1-y)*(1-tf.log(output)), name='output_layer', axis=1))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(FLAGS.init_lr,
                                                global_step=global_step,
                                                decay_steps=FLAGS.decay_steps,
                                                decay_rate=FLAGS.decay_rate,
                                                staircase=True)
    #######################
    # every time minimize is called, global step will increment by 1!
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    correct_pred = tf.equal(y, out_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # recall, rec_op = tf.metrics.recall(labels=tf.cast(y, tf.int32),
    #                                    predictions=tf.cast(out_pred, tf.int32))
    # precision, prec_op = tf.metrics.precision(labels=tf.cast(y, tf.int32),
    #                                           predictions=tf.cast(out_pred, tf.int32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        k_pos, k_neg = 0, 0
        for i in range(FLAGS.iterations):
            batchx_pos, batchy_pos, k_pos = next_batch(trainx_pos, trainy_pos, n_samples_pos,FLAGS.batch_size, k_pos)
            batchx_neg, batchy_neg, k_neg = next_batch(trainx_neg, trainy_neg, n_samples_neg,FLAGS.batch_size, k_neg)
            k_pos += 1
            k_neg += 1
            x_batch = np.append(batchx_pos, batchx_neg, axis=0)
            y_batch = np.append(batchy_pos, batchy_neg)
            _ = sess.run(train_op, feed_dict={x: x_batch, y: y_batch, istrain: True})

            train_loss, train_prob, train_accu = sess.run([loss, out_prob, accuracy],
                                                          feed_dict={x: trainx_np, y: trainy_np, istrain: False})
            if train_loss < best_train_loss:
                print('-' * 100)
                # print('validation AUC improved from {} to {} '.format(best_train_auc, train_pref['AUC']))
                print('Train loss improved from {} to {} '.format(best_train_loss, train_loss))
                print('Saving Model to file..')
                # best_train_auc = train_pref['AUC']
                best_train_loss = train_loss
                saver.save(sess,
                           'C:/behrouz/projects/MLP-2Branch' + '/balanced_batch_check_point/' + 'model.ckpt')
            if (i + 1) % 100 == 0:
                test_loss, test_prob, test_accu = sess.run([loss, out_prob, accuracy],
                                                               feed_dict={x: testx_np,
                                                                          y: testy_np,
                                                                          istrain: False})

                train_pred = class_pred(train_prob, 0.88)
                test_pred = class_pred(test_prob, 0.88)
                train_pref = performance_statistics(trainy_np, train_pred, train_prob)
                test_pref = performance_statistics(testy_np, test_pred, test_prob)
                print('iteration ', (i + 1))
                print('Train: ', 'Accuracy= ', train_accu, train_pref)
                print('Test: ', 'Accuracy= ', test_accu, test_pref)
                print('-' * 100)
                print('-' * 100)
        print('Optimization finished. ')
        print('Restoring Best model and evaluating...')
        saver.restore(sess, 'C:/behrouz/projects/MLP-2Branch' + '/balanced_batch_check_point/' + 'model.ckpt')
        train_loss, train_prob, train_accu = sess.run([loss, out_prob, accuracy],
                                                      feed_dict={x: trainx_np,
                                                                 y: trainy_np,
                                                                 istrain: False})

        test_loss, test_prob, test_accu = sess.run([loss, out_prob, accuracy],
                                                   feed_dict={x: testx_np,
                                                              y: testy_np,
                                                              istrain: False})
        train_pred = class_pred(train_prob, 0.88)
        test_pred = class_pred(test_prob, 0.88)
        train_pref = performance_statistics(trainy_np, train_pred, train_prob)
        test_pref = performance_statistics(testy_np, test_pred, test_prob)
        print('=' * 100)
        print('Train: ', 'Accuracy= ', train_accu, train_pref)
        print('Test: ', 'Accuracy= ', test_accu, test_pref)

        train_res = pd.DataFrame({'OMSTenantId': train['OMSTenantId'].values,
                                  'outHasActiveO365': train['outHasActiveO365'].values,
                                  'Probs': train_prob})

        train_res.to_csv('C:/behrouz/projects/MLP-2Branch/' + 'results/' + 'train_balanced_batch.csv', index=False)

        test_res = pd.DataFrame({'OMSTenantId': test['OMSTenantId'].values,
                                 'outHasActiveO365': test['outHasActiveO365'].values,
                                 'Probs': test_prob})
        test_res.to_csv('C:/behrouz/projects/MLP-2Branch/' + 'results/' + 'test_balanced_batch.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wh1',
        type=int,
        default=32,
        help='wide model hidden1_size',
    )
    parser.add_argument(
        '--wh2',
        type=int,
        default=32,
        help='wide model hidden2_size'
    )
    parser.add_argument(
        '--dh1',
        type=int,
        default=10,
        help='deep model hidden1_size'
    )
    parser.add_argument(
        '--dh2',
        type=int,
        default=10,
        help='deep model hidden2_size'
    )
    parser.add_argument(
        '--dh3',
        type=int,
        default=16,
        help='deep model hidden2_size'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100000,
        help='NUmber of training epochs'
    )
    parser.add_argument(
        '--init_lr',
        type=int,
        default=0.001,
        help='Initial learning Rate'
    )
    parser.add_argument(
        # decrease learning rate in half every 100 epochs!
        '--decay_rate',
        type=float,
        default=0.01,
        help='decrease learning rate every 100 epochs!'
    )
    parser.add_argument(
        # decrease learning rate every decay steps
        '--decay_steps',
        type=int,
        default=100000,
        help='decrease learning rate every 1000 steps-batches'
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=10,
        help='size of embedding vector'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='size of embedding vector'
    )
    parser.add_argument(
        '--wide_feats',
        type=int,
        default=42,
        help='size of embedding vector'
    )
    parser.add_argument(
        '--n_countries',
        type=int,
        default=21,
        help='size of embedding vector'
    )

    parser.add_argument(
        '--activation',
        type=int,
        default=tf.nn.tanh,
        help='Initial learning Rate'
    )
    parser.add_argument(
        '--dropoutRate',
        type=float,
        default=0.01,
        help='Dropout Rate- what percent of activation in a layer to drop!'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS, unparsed, sys.argv[0])
    build_run_training()
