import numpy as np
import tensorflow as tf
import time
import argparse
import sys
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, \
    roc_auc_score, roc_curve, auc, precision_recall_curve
import pickle
FLAGS = None
N_FEATURES = 1500
# data scaled already- if data is massive, would be a better approach to scale at the time of batching
train_min_max = pd.read_csv('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                            'excel-pbi-modeling/ExcelUsageData/Train_normalizing_param.csv', index_col=0).values
trainx_min = tf.constant(train_min_max[1], dtype=tf.float32, shape=[N_FEATURES])
trainx_max = tf.constant(train_min_max[0], dtype=tf.float32, shape=[N_FEATURES])
TRAIN_SIZE = 299952
VALIDATION_SIZE = 99984
TEST_SIZE = 99985


def _parse_func(example_proto):
    """
    :param example_proto:
    :return:
    """
    f_range = tf.subtract(trainx_max, trainx_min)
    features = {
        'x_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    x_raw = parsed_features['x_raw']
    x_raw = tf.decode_raw(x_raw, tf.int64)
    x_raw.set_shape([N_FEATURES])
    x_raw = tf.cast(x_raw, tf.float32)
    x_scaled = tf.subtract(x_raw, trainx_min)
    x_scaled = tf.div(x_scaled, f_range)
    # print('Shape of scaled x: ', x_scaled.get_shape())
    return x_scaled, parsed_features['label']


def inputs(filenames, data_size):
    """
    Only for test and validation. Ideally just read this files <especilly the test> into memory
    without creating tfrecords.
    :param filenames: list of full path to  .tfrecords files
    :param data_size: total size of the data in all .tfrecords
    :param batch_size:
    :return: a batch of data and labels
    """
    # batching yields a [1, batch_size] label! we will flatten it
    # filenames: The filenames argument to the TFRecordDataset initializer can
    # either be a string, a list of strings, or a tf.Tensor of strings.
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
    # applies the function to each sample in data set.
    dataset = dataset.map(_parse_func)
    # repeat indefinitely
    dataset = dataset.repeat(count=None)
    dataset = dataset.batch(data_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_labels = iterator.get_next()
    print('Shape of BATCH coming out of iterator= ', batch_x.get_shape())
    batch_labels = tf.reshape(batch_labels, [-1])
    return batch_x, tf.cast(batch_labels, tf.float32), iterator


def inputs_train(filename_pattern):
    # shuffle list of files
    files = tf.data.Dataset.list_files(FLAGS.data_path+filename_pattern, shuffle=True)
    # 1. read files in parallel and interleave the examples read.
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset,
        cycle_length=FLAGS.num_parallel_readers))
    # shuffle the items as they pass through when size becomes: data_size(each epoch)
    # it maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer.
    # count= None: repeat forever; train_buffer_size: min number of elements in the buffer before shuffling
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.train_buffer_size, count=None))
    # 2. Parallelize Data Transformation
    # 3. Parallelize batch creation
    # if your batch size is in the hundreds or thousands, your pipeline will likely
    #  additionally benefit from parallelizing the batch creation.
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_parse_func,
        batch_size=FLAGS.batch_size,
        num_parallel_batches=None,  # Only set on of  num_.. arguments
        num_parallel_calls=FLAGS.num_parallel_readers  # if None:= batch_size * num_parallel_batches
        ))
    # prefetch next batch while training on curr batch
    dataset = dataset.prefetch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_labels = iterator.get_next()
    batch_labels = tf.reshape(batch_labels, [-1])
    return batch_x, tf.cast(batch_labels, tf.float32), iterator


def performance_statistics(y_true, y_pred, y_probs):
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # fpr = fp / (fp + tn)
    # tpr = tp / (tp +fn)
    pos_recall = recall_score(y_true, y_pred, pos_label=1)
    pos_precision = precision_score(y_true, y_pred, pos_label=1)
    pos_f1 = f1_score(y_true, y_pred, pos_label=1)
    neg_recall = recall_score(y_true, y_pred, pos_label=0)
    neg_precision = precision_score(y_true, y_pred, pos_label=0)
    neg_f1 = f1_score(y_true, y_pred, pos_label=0)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs, pos_label=1)
    auc_ = auc(fpr, tpr)
    # area under precision recall score:
    precision, recalls, thresholds = precision_recall_curve(y_true, y_probs, pos_label=1)
    p_r_auc = auc(recalls, precision)
    return pos_precision, pos_recall, pos_f1,\
           neg_precision, neg_recall, neg_f1, \
           p_r_auc, auc_


def model(x, y, activation, h1_size, h2_size, h3_size, h4_size,
          h5_size, h6_size, h7_size, h8_size, h9_size, h10_size, istrain=True, reuse=False):
    with tf.variable_scope('Model', reuse=reuse):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
            # relu initializer He et al 2015
            kernel_init = tf.contrib.layers.variance_scaling_initializer(factor=2,
                                                                         mode='FAN_IN',
                                                                         uniform=False)
        else:
            bias_init = tf.zeros_initializer()
            kernel_init = tf.contrib.layers.xavier_initializer()
        h1 = tf.layers.dense(x,
                             h1_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        h1 = tf.layers.dropout(h1, rate=FLAGS.dropoutRate, training=istrain)
        h2 = tf.layers.dense(h1,
                             h2_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        h2 = tf.layers.dropout(h2, rate=FLAGS.dropoutRate, training=istrain)
        h3 = tf.layers.dense(h2,
                             h3_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        h3 = tf.layers.dropout(h3, rate=FLAGS.dropoutRate, training=istrain)
        h4 = tf.layers.dense(h3,
                             h4_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        h4 = tf.layers.dropout(h4, rate=FLAGS.dropoutRate, training=istrain)
        h5 = tf.layers.dense(h4,
                             h5_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        h5 = tf.layers.dropout(h5, rate=FLAGS.dropoutRate, training=istrain)

        # h6 = tf.layers.dense(h5,
        #                      h6_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init,
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        # h6 = tf.layers.dropout(h6, rate=FLAGS.dropoutRate, training=istrain)

        # h7 = tf.layers.dense(h6,
        #                      h7_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init,
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        # h7 = tf.layers.dropout(h7, rate=FLAGS.dropoutRate, training=istrain)

        # h8 = tf.layers.dense(h7,
        #                      h8_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init),
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        # h8 = tf.layers.dropout(h8, rate=FLAGS.dropoutRate, training=istrain)

        # h9 = tf.layers.dense(h8,
        #                      h9_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init,
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        # h9 = tf.layers.dropout(h9, rate=FLAGS.dropoutRate, training=istrain)

        # h10 = tf.layers.dense(h9,
        #                       h10_size,
        #                       activation=activation,
        #                       kernel_initializer=kernel_init,
        #                       bias_initializer=bias_init,
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_r))
        # h10 = tf.layers.dropout(h10, rate=FLAGS.dropoutRate, training=istrain)

        logits = tf.squeeze(tf.layers.dense(h5, 1,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer()))
        out_probs = tf.nn.sigmoid(logits)
        out_preds = tf.cast(tf.greater_equal(out_probs, 0.5), tf.float32)
        correct_pred = tf.equal(y, out_preds)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return logits, accuracy, out_probs, out_preds


def run_training():
    # Tell TensorFlow that the model will be built into the default Graph.
    validation_file_list = [FLAGS.data_path + 'validation_' + '%d.tfrecords' % i for i in range(1, 11)]
    train_file_list = [FLAGS.data_path + 'train_' + '%d.tfrecords' % i for i in range(1, 11)]
    test_file_list = [FLAGS.data_path + 'test_' + '%d.tfrecords' % i for i in range(1, 11)]
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step(graph=None)
        with tf.device('/cpu:0'):
            batch_x_pos, batch_labels_pos, batch_iterator_pos = inputs_train('train_*_pos.tfrecords')
            batch_x_neg, batch_labels_neg, batch_iterator_neg = inputs_train('train_*_neg.tfrecords')
            batch_x = tf.concat([batch_x_pos, batch_x_neg], axis=0)
            batch_labels = tf.concat([batch_labels_pos, batch_labels_neg], axis=0)

            print(batch_x.get_shape(), batch_labels.get_shape())
            print(batch_labels.dtype, '====================')
            train_x, train_labels, train_iterator = inputs(train_file_list,
                                                           FLAGS.train_size
                                                           )
            validation_x, validation_labels, val_iterator = inputs(validation_file_list,
                                                                   FLAGS.valid_size
                                                                   )
            test_x, test_labels, test_iterator = inputs(test_file_list,
                                                        FLAGS.test_size
                                                        )
        print(batch_x.get_shape(), ', ', batch_labels.get_shape())
        with tf.device("/cpu:0"):
            train_logits, train_accuracy, train_probs, train_preds = model(batch_x,
                                                                           batch_labels,
                                                                           FLAGS.activation,
                                                                           FLAGS.h1,
                                                                           FLAGS.h2,
                                                                           FLAGS.h3,
                                                                           FLAGS.h4,
                                                                           FLAGS.h5,
                                                                           FLAGS.h6,
                                                                           FLAGS.h7,
                                                                           FLAGS.h8,
                                                                           FLAGS.h9,
                                                                           FLAGS.h10,
                                                                           istrain=True,
                                                                           reuse=False
                                                                           )
            tot_train_logits, tot_train_accuracy, tot_train_probs, tot_train_preds = model(train_x,
                                                                                           train_labels,
                                                                                           FLAGS.activation,
                                                                                           FLAGS.h1,
                                                                                           FLAGS.h2,
                                                                                           FLAGS.h3,
                                                                                           FLAGS.h4,
                                                                                           FLAGS.h5,
                                                                                           FLAGS.h6,
                                                                                           FLAGS.h7,
                                                                                           FLAGS.h8,
                                                                                           FLAGS.h9,
                                                                                           FLAGS.h10,
                                                                                           istrain=False,
                                                                                           reuse=True
                                                                                           )
            train_logits_ev, train_accuracy_ev, train_probs_ev, train_preds_ev = model(batch_x,
                                                                                       batch_labels,
                                                                                       FLAGS.activation,
                                                                                       FLAGS.h1,
                                                                                       FLAGS.h2,
                                                                                       FLAGS.h3,
                                                                                       FLAGS.h4,
                                                                                       FLAGS.h5,
                                                                                       FLAGS.h6,
                                                                                       FLAGS.h7,
                                                                                       FLAGS.h8,
                                                                                       FLAGS.h9,
                                                                                       FLAGS.h10,
                                                                                       istrain=False,
                                                                                       reuse=True
                                                                                       )
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=train_logits))
            l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.reduce_sum(l2_loss)
            cost += l2_loss
            # cross entropy loss when drop out turned off!
            cost_ev_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels,
                                                                                   logits=train_logits_ev))
            tot_train_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_labels
                                                                                    , logits=tot_train_logits))
            # its like above cost except weights positive examples!
            # cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(batch_labels, train_logits, 1))
            learning_rate = tf.train.inverse_time_decay(FLAGS.learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=FLAGS.decay_steps,
                                                        decay_rate=FLAGS.decay_rate,
                                                        staircase=True
                                                        )
            train_op = tf.contrib.layers.optimize_loss(loss=cost,
                                                       global_step=global_step,
                                                       learning_rate=learning_rate,
                                                       optimizer=tf.train.AdamOptimizer(),
                                                       # clip_gradients=2.0,
                                                       name='d_optimize_loss',
                                                       variables=tf.trainable_variables())
            ema = tf.train.ExponentialMovingAverage(decay=0.999)
            # EMA weights:
            with tf.control_dependencies([train_op]):
                train_op_new = ema.apply(tf.trainable_variables())
            validation_logits, validation_accuracy, validation_probs, validation_preds = model(validation_x,
                                                                                               validation_labels,
                                                                                               FLAGS.activation,
                                                                                               FLAGS.h1,
                                                                                               FLAGS.h2,
                                                                                               FLAGS.h3,
                                                                                               FLAGS.h4,
                                                                                               FLAGS.h5,
                                                                                               FLAGS.h6,
                                                                                               FLAGS.h7,
                                                                                               FLAGS.h8,
                                                                                               FLAGS.h9,
                                                                                               FLAGS.h10,
                                                                                               istrain=False,
                                                                                               reuse=True)
            test_logits, test_accuracy, test_probs, test_preds = model(test_x,
                                                                       test_labels,
                                                                       FLAGS.activation,
                                                                       FLAGS.h1,
                                                                       FLAGS.h2,
                                                                       FLAGS.h3,
                                                                       FLAGS.h4,
                                                                       FLAGS.h5,
                                                                       FLAGS.h6,
                                                                       FLAGS.h7,
                                                                       FLAGS.h8,
                                                                       FLAGS.h9,
                                                                       FLAGS.h10,
                                                                       istrain=False,
                                                                       reuse=True)
            validation_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=validation_labels,
                                                                                     logits=validation_logits))

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        # allow to use smaller amount of GPU memory and grow if needed!
        config.gpu_options.allow_growth = False
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            valid_perfs = dict()
            metrics = ['pos_percision', 'pos_recall', 'pos_f1',
                       'neg_percision', 'neg_recall', 'neg_f1',
                       'p_r_auc', 'auc']
            valid_perfs['pos_percision'] = []
            valid_perfs['pos_recall'] = []
            valid_perfs['pos_f1'] = []
            valid_perfs['neg_percision'] = []
            valid_perfs['neg_recall'] = []
            valid_perfs['neg_f1'] = []
            valid_perfs['p_r_auc'] = []
            valid_perfs['auc'] = []
            sess.run(init_op)
            # b, l = sess.run([batch_x, batch_labels])
            # np.savetxt(FLAGS.data_path + 'batch1.txt', b[:1000], delimiter='\t')
            # print(len(l), np.sum(l))
            # print('Shape of Batch=', b.shape, l.shape, batch_x.get_shape(), batch_labels.get_shape())
            checkpoint_rate = int(0.5* (FLAGS.train_size // FLAGS.batch_size) // 100) * 100
            n_iterations = checkpoint_rate * FLAGS.num_epochs
            # checkpoint_rate = 10000
            print('Total iterations= ', n_iterations)
            best_validation_auc = 0
            train_per_epoch_loss = 0
            val_loss_list = []
            train_per_epoch_loss_list = []
            # pre training performance measure:
            tr_loss = sess.run(tot_train_cost)
            train_per_epoch_loss_list.append(tr_loss)
            val_probs, val_preds, val_labs, val_loss = sess.run([validation_probs,
                                                                 validation_preds,
                                                                 validation_labels,
                                                                 validation_cost])
            val_loss_list.append(val_loss)
            validation_pref = performance_statistics(val_labs, val_preds, val_probs)
            for m in range(len(metrics)):
                valid_perfs[metrics[m]].append(validation_pref[m])
            # training
            for i in range(n_iterations):
                start_time = time.time()
                batch_loss, batch_accu, _ = sess.run([cost_ev_train, train_accuracy_ev, train_op_new])
                train_per_epoch_loss += batch_loss
                duration = time.time() - start_time
                if i % 100 == 0:
                    print('Step %d: BATCH loss = %.3f, accuracy=%.3f (%.3f sec)' %
                          (i, batch_loss, batch_accu, duration))
                train_per_epoch_loss += batch_loss
                if i % 100 == 0:
                    print('Step %d: BATCH loss = %.2f, accuracy=%.2f (%.3f sec)' %
                          (i, batch_loss, batch_accu, duration))
                if (i+1) % checkpoint_rate == 0:
                    t1 = time.time()
                    val_probs, val_preds, val_labs, val_loss = sess.run([validation_probs,
                                                                         validation_preds,
                                                                         validation_labels,
                                                                         validation_cost])
                    validation_pref = performance_statistics(val_labs, val_preds, val_probs)
                    print('Epoch end validation performance:')
                    print(validation_pref)
                    val_loss_list.append(val_loss)

                    train_per_epoch_loss_list.append(train_per_epoch_loss / checkpoint_rate)
                    train_per_epoch_loss = 0
                    for m in range(len(metrics)):
                        valid_perfs[metrics[m]].append(validation_pref[m])
                    if best_validation_auc < validation_pref[-1]:
                        print('-' * 100)
                        print('validation AUC improved from {} to {} '.format(best_validation_auc,
                                                                              validation_pref[-1]),
                              '(%.3f sec)' % (time.time()-t1))
                        best_validation_auc = validation_pref[-1]
                        # print('Train loss improved from {} to {} '.format(best_train_loss, train_loss))
                        print('Saving Model to file..')
                        print('-' * 100)
                        saver.save(sess,
                                   'C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                                   'excel-pbi-modeling/balanced-batch/checkpoint/model.ckpt')
            test_accu = sess.run(test_accuracy)
            train_val_loss = np.zeros((2, len(val_loss_list)))
            train_val_loss[0, :] = train_per_epoch_loss_list
            train_val_loss[1, :] = val_loss_list
            np.save('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                    'excel-pbi-modeling/balanced-batch/train_val_loss.npy',
                    train_val_loss)
            print(train_val_loss.shape)
            with open('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                      'excel-pbi-modeling/balanced-batch/val_perfs.pickle', 'wb') as f:
                pickle.dump(valid_perfs, f)
            print('Out of Sample Accuracy= ', test_accu)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/ExcelUsageData/tfrecords-byclass/',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--activation',
        default=tf.nn.relu,
        help='activation function'
    )
    parser.add_argument(
        '--h1',
        type=int,
        default=1024,
        help='hidden1_size',
    )
    parser.add_argument(
        '--h2',
        type=int,
        default=512,
        help='hidden2_size'
    )
    parser.add_argument(
        '--h3',
        type=int,
        default=256,
        help='hidden3_size',
    )
    parser.add_argument(
        '--h4',
        type=int,
        default=128,
        help='hidden4_size'
    )
    parser.add_argument(
        '--h5',
        type=int,
        default=64,
        help='hidden5_size'
    )
    parser.add_argument(
        '--h6',
        type=int,
        default=256,
        help='hidden6_size'
    )
    parser.add_argument(
        '--h7',
        type=int,
        default=256,
        help='hidden7_size'
    )
    parser.add_argument(
        '--h8',
        type=int,
        default=128,
        help='hidden8_size'
    )
    parser.add_argument(
        '--h9',
        type=int,
        default=128,
        help='hidden9_size'
    )
    parser.add_argument(
        '--h10',
        type=int,
        default=64,
        help='hidden10_size'
    )
    parser.add_argument(
        '--dropoutRate',
        type=float,
        default=0.0,
        help='Dropout Rate- what percent of activation in a layer to drop!'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        # decrease learning rate in half every 100 epochs!
        '--decay_rate',
        type=float,
        default=0.1,
        help='decrease learning rate every 100 epochs!'
    )
    parser.add_argument(
        # decrease learning rate every decay steps
        '--decay_steps',
        type=int,
        default=20000,
        help='decrease learning rate every default steps-batches'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=299952,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--valid_size',
        type=int,
        default=99984,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=99985,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--num_parallel_readers',
        type=int,
        default=10,
        help='number of parallel readers to read from files.'
    )
    parser.add_argument(
        '--train_buffer_size',
        type=int,
        default=20000,
        help='min # of samples to have in buffer to perform shuffling'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

