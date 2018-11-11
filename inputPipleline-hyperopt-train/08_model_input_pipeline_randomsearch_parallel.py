"""
Interuptible hyper param search:
if for any reason the hyper params earch does not finish completely, the intermediate results will be lost.
We save the state of the hyper parameter after each run and in case it is interupted we load fro file using:
CheckpointSaver callback.
-------------------------------------------------------------------
Loading the last state of optimization and continue the search:
from skopt import load
res = load('./checkpoint.pkl')
x0 = res.x_iters
y0 = res.func_vals
results = dummy_minimize(run_model,
                             boundsOpt,
                             n_calls=100,
                             x0=x0, # already examined values
                             y0=y0  # observed loss for x0
                             random_state=None, # set it for reproducible results
                             verbose=True,
                             callback=[checkpoint_saver])
Use of subset of data for hyper param optimization:
I use half of training and validation data for hyper param optimization

"""
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, \
     roc_curve, auc, precision_recall_curve
from skopt import dummy_minimize
from skopt.space import Real, Categorical
from skopt.callbacks import CheckpointSaver
import multiprocessing
N_FEATURES = 1500
h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = 64, 32, 32, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
decay_rate = 0.1
decay_steps = 20000
num_epochs = 100
train_size, valid_size = 299952, 99984
num_parallel_readers = 2
train_buffer_size = 50000

data_path = 'C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/ExcelUsageData/tfrecords-hyperopt/'

# data scaled already- if data is massive, would be a better approach to scale at the time of batching
train_min_max = pd.read_csv('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                            'excel-pbi-modeling/ExcelUsageData/Train_normalizing_param.csv', index_col=0).values
trainx_min = tf.constant(train_min_max[1], dtype=tf.float32, shape=[N_FEATURES])
trainx_max = tf.constant(train_min_max[0], dtype=tf.float32, shape=[N_FEATURES])
TRAIN_SIZE = 299952
VALIDATION_SIZE = 99984


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


def inputs_train(filename_pattern, batch_size):
    # shuffle list of files
    files = tf.data.Dataset.list_files(data_path+filename_pattern, shuffle=True)
    # 1. read files in parallel and interleave the examples read.
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_parallel_readers))
    # shuffle the items as they pass through when size becomes: data_size(each epoch)
    # it maintains a fixed-size buffer and chooses the next element uniformly at random from that buffer.
    # count= None: repeat forever; train_buffer_size: min number of elements in the buffer before shuffling
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=train_buffer_size, count=None))
    # 2. Parallelize Data Transformation
    # 3. Parallelize batch creation
    # if your batch size is in the hundreds or thousands, your pipeline will likely
    #  additionally benefit from parallelizing the batch creation.
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_parse_func,
        batch_size=batch_size,
        num_parallel_batches=None,  # Only set on of  num_.. arguments
        num_parallel_calls=num_parallel_readers  # if None:= batch_size * num_parallel_batches
        ))
    # prefetch next batch while training on curr batch
    dataset = dataset.prefetch(batch_size)
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
          h5_size, h6_size, h7_size, h8_size, h9_size, h10_size,
          init_lr, dropout_rate, l2_r, istrain=True, reuse=False):
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
        # h4 = tf.layers.dense(h3,
        #                      h4_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init,
        #                      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        # h4 = tf.layers.dropout(h4, rate=dropout_rate, training=istrain)
        # h5 = tf.layers.dense(h4,
        #                      h5_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init,
        #                      kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        # h5 = tf.layers.dropout(h5, rate=dropout_rate, training=istrain)

        # h6 = tf.layers.dense(h5,
        #                      h6_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init,
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        # h6 = tf.layers.dropout(h6, rate=dropout_rate, training=istrain)

        # h7 = tf.layers.dense(h6,
        #                      h7_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init,
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        # h7 = tf.layers.dropout(h7, rate=dropout_rate, training=istrain)

        # h8 = tf.layers.dense(h7,
        #                      h8_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init),
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        # h8 = tf.layers.dropout(h8, rate=dropout_rate, training=istrain)

        # h9 = tf.layers.dense(h8,
        #                      h9_size,
        #                      activation=activation,
        #                      kernel_initializer=kernel_init,
        #                      bias_initializer=bias_init,
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        # h9 = tf.layers.dropout(h9, rate=dropout_rate, training=istrain)

        # h10 = tf.layers.dense(h9,
        #                       h10_size,
        #                       activation=activation,
        #                       kernel_initializer=kernel_init,
        #                       bias_initializer=bias_init,
        #                              kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_r))
        # h10 = tf.layers.dropout(h10, rate=dropout_rate, training=istrain)

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
        # its like above cost except weights positive examples!
        # cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(batch_labels, train_logits, 1))
        global_step = tf.train.get_or_create_global_step(graph=None)
        learning_rate = tf.train.inverse_time_decay(init_lr,
                                                    global_step=global_step,
                                                    decay_steps=decay_steps,
                                                    decay_rate=decay_rate,
                                                    staircase=True
                                                    )
        train_op = tf.contrib.layers.optimize_loss(loss=cost_plus_l2,
                                                   global_step=global_step,
                                                   learning_rate=learning_rate,
                                                   optimizer=tf.train.AdamOptimizer(),
                                                   clip_gradients=4.0,
                                                   name='d_optimize_loss',
                                                   variables=tf.trainable_variables())
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        # EMA weights:
        with tf.control_dependencies([train_op]):
            train_op_new = ema.apply(tf.trainable_variables())
        return logits, accuracy, out_probs, out_preds, cost, train_op_new


def run_training(init_lr, l2_r, dropout_rate, batch_size, activation):
    # change this if want to optimize l2 as well
    # Tell TensorFlow that the model will be built into the default Graph.
    validation_file_list = [data_path + 'validation_' + '%d.tfrecords' % i for i in range(1, 11)]
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            batch_x, batch_labels, batch_iterator = inputs_train('train_*.tfrecords', batch_size)
            validation_x, validation_labels, val_iterator = inputs(validation_file_list,
                                                                   valid_size
                                                                   )
        print(batch_x.get_shape(), ', ', batch_labels.get_shape())
        with tf.device("/cpu:0"):
            train_logits, train_accuracy, train_probs, train_preds, cost, train_op = model(batch_x,
                                                                                           batch_labels,
                                                                                           activation,
                                                                                           h1,
                                                                                           h2,
                                                                                           h3,
                                                                                           h4,
                                                                                           h5,
                                                                                           h6,
                                                                                           h7,
                                                                                           h8,
                                                                                           h9,
                                                                                           h10,
                                                                                           init_lr,
                                                                                           dropout_rate,
                                                                                           l2_r,
                                                                                           istrain=True,
                                                                                           reuse=False
                                                                                           )

            val_logits, val_accuracy, val_probs, val_preds, val_cost, _ = model(validation_x,
                                                                                validation_labels,
                                                                                activation,
                                                                                h1,
                                                                                h2,
                                                                                h3,
                                                                                h4,
                                                                                h5,
                                                                                h6,
                                                                                h7,
                                                                                h8,
                                                                                h9,
                                                                                h10,
                                                                                init_lr,
                                                                                dropout_rate,
                                                                                l2_r,
                                                                                istrain=False,
                                                                                reuse=True)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        # allow to use smaller amount of GPU memory and grow if needed!
        config.gpu_options.allow_growth = False
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            checkpoint_rate = int((train_size // batch_size) // 100) * 100
            n_iterations = checkpoint_rate * num_epochs
            print('Total iterations= ', n_iterations)
            # training
            print('Optimization started..')
            t1 = time.time()
            for i in range(n_iterations):
                _ = sess.run(train_op)
            print('Optimization FINISHED..', 'Total time(Min):', round((time.time() - t1) / 60., 2))
            val_probs, val_preds, val_labs, val_loss = sess.run([val_probs,
                                                                 val_preds,
                                                                 validation_labels,
                                                                 val_cost])
            # validation_pref = performance_statistics(val_labs, val_preds, val_probs)
    print('-'*20)
    print('current parameters: ')
    print(init_lr, l2_r, dropout_rate, batch_size, activation)
    print('End of training Validation cross entropy loss: ', val_loss)
    print('-'*20)
    return val_loss


def run_model(dimensions):
    # init_lr, l2_reg, dropoutRate, h1, h2 = dimensions
    init_lr, l2_reg, batch_size, activation_name, dropout_rate = dimensions
    if activation_name == 'relu':
        activation = tf.nn.relu
    elif activation_name == 'tanh':
        activation = tf.nn.tanh
    elif activation_name == 'softsign':
        activation = tf.nn.softsign
    validation_loss = run_training(init_lr=init_lr, l2_r=l2_reg,
                                   dropout_rate=dropout_rate, batch_size=batch_size,
                                   activation=activation)
    print('-' * 200)
    return validation_loss


def random_search_wrapper(res_dict, i, run_model, boundsOpt, n_calls=10,
                          random_state=None, verbose=True, callback=[]):
    rslt = dummy_minimize(run_model, boundsOpt, n_calls=n_calls,
                          random_state=random_state, verbose=verbose, callback=callback)
    res_dict[i] = rslt


if __name__ == '__main__':
    space = np.logspace(-8, -1, 8)
    space = list(np.sort(np.append(space, space / 2.)))
    boundsOpt = [Categorical(space, name='init_lr'),
                 Categorical(space, name='l2_r'),
                 Categorical([512, 256], name='batch_size'),
                 Categorical(['relu', 'tanh', 'softsign'], name='activation'),
                 Real(0, 0.8, 'uniform', name='dropout_rate'),
                ]
    t1 = time.time()

    manager = multiprocessing.Manager()
    res_dict = manager.dict()
    processes = []
    for i in range(1, 4):
        checkpoint_saver = CheckpointSaver('C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/'
                                           'imbalanced_batch/randomSearch_checkpoint%d.pkl' % i)
        p = multiprocessing.Process(target=random_search_wrapper,
                                    args=(res_dict, i, run_model, boundsOpt),
                                    kwargs={'n_calls': 60,
                                            'random_state': None,  # set it for reproducible results
                                            'verbose': True,
                                            'callback': [checkpoint_saver]})
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Hyper opt Total Time: %.2f' % ((time.time() - t1) / 60.))
    x, y, func_vals = [], [], []
    for item in res_dict.keys():
        x.append(res_dict[item]['x'])
        y.append(res_dict[item]['fun'])
        func_vals.extend(list(res_dict[item]['func_vals']))
    print('Minimum validation loss obtained:')
    print(y)
    print('Optimal values:')
    print('learning rate, l2_r, batch_size, activation, dropout')
    print(x[np.argmin(np.array(func_vals))])
    print(len(func_vals))
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(func_vals)
    ax.set_ylabel('loss')
    ax.set_xlabel('points searched')
    ax.set_ylim(0, 0.8)
    ax.axvline(x=np.argmin(np.array(func_vals)), c='red', linestyle='--')
    ax.axhline(y=np.amin(np.array(func_vals)), c='red', linestyle='--')
    ax.text(0.25 * len(func_vals), 0.75, 'Optimum point:')
    ax.text(0.25 * len(func_vals), 0.65, 'learning rate, l2_r, batch_size, activation, dropout')
    ax.text(0.25 * len(func_vals), 0.6, str(x[np.argmin(y)]))
    ax.set_xticks(np.arange(len(func_vals)))
    ax.set_xticklabels(np.arange(len(func_vals)))
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_title('Hyper parameter optimization result')
    fig.savefig('C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/'
                'imbalanced_batch/hyperopt_perf.png')
    plt.show()

