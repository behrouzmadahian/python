import numpy as np
import pandas as pd
import tensorflow as tf
import os
import argparse, sys
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

'''
last column is the binary outcome.
To do:
GET AUC for test!!
The Hinge Loss here is for Binay classification Problem ONLY!
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
    return {'prec_recal_f1_pos':(round(pos_precision, 4), round(pos_recall, 4), round(pos_f1, 4)),
            'prec_recall_f1_neg': (round(neg_precision, 4), round(neg_recall, 4), round(neg_f1, 4)),
            'AUC': round(auc, 3)}


def class_pred (probs, thresh):
    '''returns the associated class'''
    preds = [1 if p >= thresh else 0 for p in probs]
    return np.array(preds)


def data_class_div(x, y):
    '''
    given the dataframe, different data frames are returned for each class value.
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


def validation_split(data, frac=0.2):
    '''
    data is data frame. first column tenant ID last column Y variable
    returns train and validation data
    '''
    m = data.shape[0]
    inds = np.arange(m)
    np.random.shuffle(inds)
    train = data.iloc[inds[int(frac * m):]]
    valid = data.iloc[inds[:int(frac * m)]]
    return train, valid


def rank_net_loss(y_true, logits):
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(logits, y_true, 2)
    score_neg = parts[0]
    score_pos = parts[1]
    score_neg = tf.expand_dims(score_neg, axis=-1)
    score_pos = tf.expand_dims(score_pos, axis=0)
    # score_pos = tf.slice(logits, [0], [FLAGS.batch_size])
    # score_neg = tf.slice(logits, [FLAGS.batch_size], [-1])
    rank_loss = score_pos - score_neg
    rank_loss = tf.nn.sigmoid(-2*rank_loss)
    # rank_loss = tf.log(tf.reduce_sum(rank_loss, axis=1))  # cross entropy!
    # rank_loss = tf.log(rank_loss)  # cross entropy!
    return tf.reduce_mean(rank_loss), score_pos, score_neg, rank_loss


def model(x1, activation, istrain, h1_size, h2_size, h3_size):
    with tf.variable_scope('wide_model'):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
        else:
            bias_init = tf.zeros_initializer()
        h1 = tf.layers.dense(x1, h1_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2))
        # h1 = tf.layers.dropout(h1, rate=FLAGS.dropoutRate, training=istrain)

        h2 = tf.layers.dense(h1, h2_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2))
        # h2 = tf.layers.dropout(h2, rate=FLAGS.dropoutRate, training=istrain)
        h3 = tf.layers.dense(h2, h3_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)

        # h3 = tf.layers.dropout(h3, rate=FLAGS.dropoutRate, training=istrain)
        logits = tf.squeeze(
            tf.layers.dense(h3, 1,
                            activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer()))
        l2_loss = tf.losses.get_regularization_loss()  # total regularization loss
        return logits, l2_loss


data_dir = 'C:/behrouz/projects/behrouz-Rui-Gaurav-project/Anna/data/'
model_dir = 'C:/behrouz/projects/behrouz-Rui-Gaurav-project/Anna/MLP/checkpoint/'
results_dir = 'C:/behrouz/projects/behrouz-Rui-Gaurav-project/Anna/MLP/results/'


def build_run_training():
    best_validation_auc = - np.inf
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    data = pd.read_csv(data_dir + 'Anna-train-Processed.csv')

    feature_columns = data.columns.values[1:-1]
    class_col = 'Label'
    train, validation = validation_split(data, frac=0.2)
    print('Shape of train and validationd data:', train.shape, validation.shape)
    trainx_np, trainy_np = train[feature_columns].values, train[class_col].values
    validationx_np, validationy_np = validation[feature_columns].values, validation[class_col].values

    print('Shape of Train data ', trainx_np.shape, trainy_np.shape)

    # test = pd.read_csv(data_dir + 'Anna-test-Processed-independently.csv')
    # testx_np = test[test.columns.values[1:-1]].values
    # testy_np = test['Label'].values

    x = tf.placeholder(tf.float32, shape=[None, trainx_np.shape[1]])
    y = tf.placeholder(tf.float32, [None])
    istrain = tf.placeholder(tf.bool)

    logits, l2loss = model(x, FLAGS.activation, istrain, FLAGS.h1, FLAGS.h2, FLAGS.h3)

    out_prob = tf.nn.sigmoid(logits)
    out_pred = tf.cast(tf.greater(out_prob, 0.5), tf.float32)

    # cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(y * tf.log(out_prob) + (1-y)*(1-tf.log(out_prob)), name='output_layer', axis=1))
    # sample_weights = y * 0.1 + 1.  # positive samples get 1.25 times more weight! -> increasing recall!!
    # we want to reduce false positives!! Give more eights to negative samples!!
    sample_weights = 1.0  # 100% more weight on negative class!
    cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) * sample_weights
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    rankLoss, score_pos, score_neg, loss_mat = rank_net_loss(y, logits)
    # hinge loss:
    # sample_weights = 1.0
    # loss = tf.losses.hinge_loss(labels=y, logits=logits, weights=sample_weights)
    total_loss = rankLoss + l2loss

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(FLAGS.init_lr,
                                                global_step=global_step,
                                                decay_steps=FLAGS.decay_steps,
                                                decay_rate=FLAGS.decay_rate,
                                                staircase=True)
    #######################
    # every time minimize is called, global step will be incremented by 1!
    train_op = FLAGS.optimizer(learning_rate).minimize(total_loss)
    correct_pred = tf.equal(y, out_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    n_samples = trainx_np.shape[0]
    train_validation_loss = np.zeros((2, FLAGS.iterations))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        kk = 0  # used to track end of train cycle < epoch >
        for i in range(FLAGS.iterations):
            x_batch, y_batch, kk = next_batch(trainx_np, trainy_np, n_samples, FLAGS.batch_size, kk)
            kk += 1
            # print('Batch shape=', x_batch.shape)
            _ = sess.run(train_op, feed_dict={x: x_batch, y: y_batch, istrain: True})
            # print('Loss Matrix=\n')
            # print(logit.shape, s_P.shape, s_N.shape, loss_m.shape)
            train_loss, train_prob, train_accu = sess.run([cross_entropy_loss, out_prob, accuracy],
                                                          feed_dict={x: trainx_np,
                                                                     y: trainy_np,
                                                                     istrain: False})

            validation_loss, validation_prob, validation_accu = sess.run([cross_entropy_loss, out_prob, accuracy],
                                                                         feed_dict={x: validationx_np,
                                                                                    y: validationy_np,
                                                                                    istrain: False})
            validation_pred = class_pred(validation_prob, 0.5)
            validation_pref = performance_statistics(validationy_np, validation_pred, validation_prob)

            train_validation_loss[:, i] = train_loss, validation_loss
            if validation_pref['AUC'] > best_validation_auc:
                print('-' * 100)
                # print('validation AUC improved from {} to {} '.format(best_train_auc, train_pref['AUC']))
                print('Validation AUC improved from {} to {} '.format(best_validation_auc, validation_pref['AUC']))
                print('Saving Model to file..')
                # best_train_auc = train_pref['AUC']
                best_validation_auc = validation_pref['AUC']
                saver.save(sess, model_dir + 'model.ckpt')
                if i >= FLAGS.iterations / 100:
                    # test_loss, test_prob, test_accu = sess.run([loss, out_prob, accuracy],
                    #                                            feed_dict={x: testx_np,
                    #                                                       y: testy_np,
                    #                                                       istrain: False})
                    train_pred = class_pred(train_prob, 0.5)
                    # test_pred = class_pred(test_prob, 0.5)
                    train_pref = performance_statistics(trainy_np, train_pred, train_prob)
                    # test_pref = performance_statistics(testy_np, test_pred, test_prob)
                    print('iteration ', (i + 1))
                    print('Train: ', 'Accuracy= ', train_accu, train_pref)
                    print('Validation: ', 'Accuracy= ', validation_accu, validation_pref)
                    # print('Test: ', 'Accuracy= ', test_accu, test_pref)
                    print('-' * 100)

        print('Optimization finished. ')
        print('Restoring Best model and evaluating...')
        saver.restore(sess,model_dir + 'model.ckpt')
        train_loss, train_prob, train_accu = sess.run([cross_entropy_loss, out_prob, accuracy],
                                                      feed_dict={x: train[feature_columns].values,
                                                                 y: train[class_col].values,
                                                                 istrain: False})
        validation_loss, validation_prob, validation_accu = sess.run([cross_entropy_loss, out_prob, accuracy],
                                                                     feed_dict={x: validation[feature_columns].values,
                                                                                y: validation[class_col].values,
                                                                                istrain: False})
        # test_loss, test_prob, test_accu = sess.run([cross_entropy_loss, out_prob, accuracy],
        #                                            feed_dict={x: testx_np,
        #                                                       y: testy_np,
        #                                                       istrain: False})
        train_pred = class_pred(train_prob, 0.5)
        validation_pred = class_pred(validation_prob, 0.5)
        # test_pred = class_pred(test_prob, 0.5)
        train_pref = performance_statistics(trainy_np, train_pred, train_prob)
        validation_pref = performance_statistics(validationy_np, validation_pred, validation_prob)
        # test_pref = performance_statistics(testy_np, test_pred, test_prob)
        print('=' * 100)
        print('Train: ', 'Accuracy= ', train_accu, train_pref)
        print('Validation: ', 'Accuracy= ', validation_accu, validation_pref)
        # print('Test: ', 'Accuracy= ', test_accu, test_pref)

        train_res = pd.DataFrame({'OMSTenantId': train['OMSTenantId'].values,
                                  'Label': train['Label'].values,
                                  'Probs': train_prob})
        train_res.to_csv(results_dir + 'train-results.csv', index=False)

        validation_res = pd.DataFrame({'OMSTenantId': validation['OMSTenantId'].values,
                                       'Label': validation['Label'].values,
                                       'Probs': validation_prob})
        validation_res.to_csv(results_dir + 'validation-results.csv', index=False)

        # test_res = pd.DataFrame({'OMSTenantId': test['OMSTenantId'].values,
        #                          'Label': test['Label'].values,
        #                          'Probs': test_prob})
        # test_res.to_csv(results_dir + 'test-results.csv', index=False)

        fig = plt.figure(figsize=(6, 6))
        plt.plot(train_validation_loss[0], color='blue', label='Train')
        plt.plot(train_validation_loss[1], color='red', label='Validation')
        plt.xlim(0, FLAGS.iterations)
        plt.ylim(0, 1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        fig.savefig(results_dir + 'Train-validation-loss-plot.png')


if __name__ == '__main__':
    '''
    lr = lr / (1 + decay_rate * global_step / decay_step)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--optimizer',
        default=tf.train.AdamOptimizer,
        help='optimizer'
    )
    parser.add_argument(
        '--h1',
        type=int,
        default=64,
        help='model hidden1_size',
    )
    parser.add_argument(
        '--h2',
        type=int,
        default=64,
        help='model hidden2_size'
    )
    parser.add_argument(
        '--h3',
        type=int,
        default=32,
        help='model hidden2_size'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=60000,
        help='NUmber of training iterations'
    )
    parser.add_argument(
        '--init_lr',
        type=int,
        default=0.005,
        help='Initial learning Rate'
    )
    parser.add_argument(
        '--l2',
        type=int,
        default=0.0,
        help='L@ Regularization'
    )
    parser.add_argument(
        # decrease learning rate in half every 100 epochs!
        '--decay_rate',
        type=float,
        default=0.5,
        help='decrease learning rate every 100 epochs!'
    )
    parser.add_argument(
        # decrease learning rate every decay steps
        '--decay_steps',
        type=int,
        default=5000,
        help='decrease learning rate every default steps-batches'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='batch size'
    )
    parser.add_argument(
        '--activation',
        type=int,
        default=tf.nn.relu,
        help='Activation Function'
    )
    parser.add_argument(
        '--dropoutRate',
        type=float,
        default=0.0,
        help='Dropout Rate- what percent of activation in a layer to drop!'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS, unparsed, sys.argv[0])
    build_run_training()
