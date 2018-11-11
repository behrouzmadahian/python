import numpy as np
import tensorflow as tf
import time
import argparse
import pandas as pd
from skopt import dummy_minimize, load
from skopt.space import Real, Categorical
from skopt.callbacks import CheckpointSaver
"""
features are scaled at batching time using train data statistics.
"""
FLAGS = None
N_FEATURES = 1500
train_min_max = pd.read_csv('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                            'excel-pbi-modeling/ExcelUsageData/Train_normalizing_param.csv', index_col=0).values
trainx_min = tf.constant(train_min_max[1], dtype=tf.float32, shape=[N_FEATURES])
trainx_max = tf.constant(train_min_max[0], dtype=tf.float32, shape=[N_FEATURES])
TRAIN_SIZE = 299952
VALIDATION_SIZE = 99984
TEST_SIZE = 99985


def _parse_func(example_proto):
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
    return x_scaled, parsed_features['label']


def inputs(filenames, data_size, cnt):
    """
    Only for test and validation. optionally,  just read
    this files into memory without creating tfrecords.
    :param filenames: list of full paths to  .tfrecords files
    :param data_size: total size of the data in all .tfrecords
    :param cnt: # of times to go over the data, None: indefinitely
    :return: data and labels
    """
    # batching yields a [1, batch_size] label! we will flatten it
    # filenames: The filenames argument to the TFRecordDataset initializer can
    # either be a string, a list of strings, or a tf.Tensor of strings.
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=1)
    # applies the function to each sample in data set.
    dataset = dataset.map(_parse_func)
    # repeat indefinitely
    dataset = dataset.repeat(count=cnt)
    dataset = dataset.batch(data_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_labels = iterator.get_next()
    batch_labels = tf.reshape(batch_labels, [-1])
    return batch_x, tf.cast(batch_labels, tf.float32), iterator


def inputs_train(filename_pattern, batch_size):
    # 1. shuffle list of files
    files = tf.data.Dataset.list_files(FLAGS.data_path+filename_pattern, shuffle=True)
    # 2. read files in parallel and interleave the examples read.
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset,
                                                              cycle_length=FLAGS.num_parallel_readers))
    # shuffle the items as they pass through when size becomes: buffer_size
    # it maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer.
    # count= None: repeat forever; train_buffer_size: min number of elements in the buffer before shuffling
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.train_buffer_size, count=None))
    # 3. Parallel Data Transformation
    # 4. Parallel batch creation
    # if your batch size is in the hundreds or thousands, your pipeline will likely
    # additionally benefit from parallel batch creation.
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_func,
                                                          batch_size=batch_size,
                                                          num_parallel_batches=None,
                                                          num_parallel_calls=FLAGS.num_parallel_readers
                                                          ))
    # prefetch next batch while training on curr batch
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_x, batch_labels = iterator.get_next()
    batch_labels = tf.reshape(batch_labels, [-1])
    return batch_x, tf.cast(batch_labels, tf.float32), iterator


def model(x, y, activation, h1_size, h2_size, h3_size, log_init_lr, dropout_rate, log_l2_r, istrain=True, reuse=False):
    init_lr = 10**log_init_lr
    l2_r = 10**log_l2_r
    with tf.variable_scope('Model', reuse=reuse):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
            # relu initializer He et al 2015.
            kernel_init = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
        else:
            bias_init = tf.zeros_initializer()
            kernel_init = tf.contrib.layers.xavier_initializer()
        h1 = tf.layers.dense(x,
                             h1_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        h1 = tf.layers.dropout(h1, rate=dropout_rate, training=istrain)
        h2 = tf.layers.dense(h1,
                             h2_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        h2 = tf.layers.dropout(h2, rate=dropout_rate, training=istrain)
        h3 = tf.layers.dense(h2,
                             h3_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        h3 = tf.layers.dropout(h3, rate=dropout_rate, training=istrain)
        logits = tf.squeeze(tf.layers.dense(h3, 1,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer()))
        out_probs = tf.nn.sigmoid(logits)
        out_preds = tf.cast(tf.greater_equal(out_probs, 0.5), tf.float32)
        correct_pred = tf.equal(y, out_preds)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
        # cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(batch_labels, train_logits, 4.))
        l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.reduce_sum(l2_loss)
        cost_plus_l2 = cost + l2_loss
        global_step = tf.train.get_or_create_global_step(graph=None)

        train_op = tf.contrib.layers.optimize_loss(loss=cost_plus_l2,
                                                   global_step=global_step,
                                                   learning_rate=init_lr,
                                                   optimizer=tf.train.AdamOptimizer(),
                                                   # clip_gradients=4.0,
                                                   name='d_optimize_loss',
                                                   variables=tf.trainable_variables())
        return logits, accuracy, out_probs, out_preds, cost, train_op


def run_training(init_lr, l2_r, dropout_rate, batch_size, activation):
    validation_file_list = [FLAGS.data_path + 'validation_' + '%d.tfrecords' % i for i in range(1, 11)]
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            batch_x, batch_labels, batch_iterator = inputs_train('train_*.tfrecords', batch_size)
            validation_x, validation_labels, val_iterator = inputs(validation_file_list, FLAGS.valid_size, cnt=None)
        print(batch_x.get_shape(), ', ', batch_labels.get_shape())
        with tf.device("/device:gpu:0"):
            train_logits, train_accuracy, train_probs, train_preds, cost, train_op = model(batch_x,
                                                                                           batch_labels,
                                                                                           activation,
                                                                                           FLAGS.h1,
                                                                                           FLAGS.h2,
                                                                                           FLAGS.h3,
                                                                                           init_lr,
                                                                                           dropout_rate,
                                                                                           l2_r,
                                                                                           istrain=True,
                                                                                           reuse=False
                                                                                           )

            val_logits, val_accuracy, val_probs, val_preds, val_cost, _ = model(validation_x,
                                                                                validation_labels,
                                                                                activation,
                                                                                FLAGS.h1,
                                                                                FLAGS.h2,
                                                                                FLAGS.h3,
                                                                                init_lr,
                                                                                dropout_rate,
                                                                                l2_r,
                                                                                istrain=False,
                                                                                reuse=True)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            n_iterations = int(np.floor(FLAGS.train_size // batch_size)) * FLAGS.num_epochs
            print('Total iterations= ', n_iterations)
            print('Optimization started..')
            t1 = time.time()
            for i in range(n_iterations):
                _ = sess.run(train_op)
            print('Optimization FINISHED..', 'Total time(Min):', round((time.time() - t1) / 60., 2))
            val_loss = sess.run(val_cost)
    print('End of training Validation cross entropy loss: ', val_loss)
    return val_loss


def run_model(dimensions):
    init_lr, l2_reg, batch_size, activation_name, dropout_rate = dimensions
    if activation_name == 'relu':
        activation = tf.nn.relu
    elif activation_name == 'tanh':
        activation = tf.nn.tanh
    elif activation_name == 'softsign':
        activation = tf.nn.softsign
    validation_loss = run_training(init_lr=init_lr, l2_r=l2_reg, dropout_rate=dropout_rate,
                                   batch_size=batch_size, activation=activation)
    return validation_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/ExcelUsageData/tfrecords/',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--h1',
        type=int,
        default=64,
        help='hidden1_size',
    )
    parser.add_argument(
        '--h2',
        type=int,
        default=32,
        help='hidden2_size'
    )
    parser.add_argument(
        '--h3',
        type=int,
        default=32,
        help='hidden3_size',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of epochs to run trainer.'
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
        '--num_parallel_readers',
        type=int,
        default=10,
        help='number of parallel readers to read from files.'
    )
    parser.add_argument(
        '--train_buffer_size',
        type=int,
        default=50000,
        help='min # of samples to have in buffer to perform shuffling'
    )
    FLAGS, unparsed = parser.parse_known_args()
    boundsOpt = [Real(-8, -1, name='log_init_lr'),
                 Real(-6, -1, name='log_l2_r'),
                 Categorical([512, 256], name='batch_size'),
                 Categorical(['relu', 'tanh'], name='activation'),
                 Real(0, 0.8, 'uniform', name='dropout_rate'),
                 ]
    t1 = time.time()
    checkpoint_saver = CheckpointSaver('C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/'
                                       'imbalanced_batch/randomSearch_checkpoint_continue.pkl')
    restore_search = load('C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/'
                          'imbalanced_batch/randomSearch_checkpoint.pkl')
    x0 = restore_search.x_iters    # already examined values
    y0 = restore_search.func_vals  # observed values for x0
    results = dummy_minimize(run_model,
                             boundsOpt,
                             x0=x0,
                             y0=y0,
                             n_calls=50,
                             random_state=None,  # set it for reproducible results
                             verbose=True,
                             callback=[checkpoint_saver])
    t2 = time.time()
    print('-' * 100)
    print('Best parameters Obtained:')
    print('Minimum Validation Loss obtained=', results.fun)
    print('Learning Rate:', 10 ** results.x[0],
          'l2:', 10 ** results.x[1],
          'Batch size:', results.x[2],
          'activation= ', results.x[3],
          'Dropout= ', results.x[4])
    print('-' * 100)
    print('Total Time(min) for optimization= ', round((t2 - t1) / 60, 2))
    from matplotlib import pyplot as plt

    plt.plot(results.func_vals)
    plt.ylabel('Validation Loss')
    plt.xlabel('Iteration')
    plt.title('Validation loss')
    plt.show()