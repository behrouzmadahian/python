import numpy as np
import pandas as pd
import tensorflow as tf
import time
import argparse
import sys
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

FLAGS = None

N_FEATURES = 1500
# data scaled already- if data is massive, would be a better approach to scale at the time of batching
train_min_max = pd.read_csv('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                            'excel-pbi-modeling/ExcelUsageData/Train_normalizing_param.csv', index_col=0).values
trainx_min = tf.constant(train_min_max[1], dtype=tf.float32, shape=[N_FEATURES])
trainx_max = tf.constant(train_min_max[0], dtype=tf.float32, shape=[N_FEATURES])
TRAIN_SIZE = 298242
VALIDATION_SIZE = 99414
TEST_SIZE = 99415


def _parse_func(example_proto):
    """
    :param example_proto:
    :return:
    """
    # f_range = tf.subtract(trainx_max, trainx_min)
    features = {
        'x_raw': tf.FixedLenFeature([N_FEATURES], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    x_raw = parsed_features['x_raw']
    print(x_raw)
    # x_scaled = tf.subtract(x_raw, trainx_min)
    print('Shape of scaled x: ', x_scaled.get_shape())
    # x_scaled = tf.div(x_scaled, f_range)
    print('Shape of scaled x: ', x_scaled.get_shape())
    return x_scaled, parsed_features['label']


def inputs(filenames, data_size, batch_size, is_train):
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


# this function has all the optimization guidelines except caching!


def inputs_train(filename_pattern):
    # shuffle list of files
    files = tf.data.Dataset.list_files(FLAGS.data_path+filename_pattern, shuffle=True)
    # 1. read files in parallel and interleave the examples read.
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset,
        cycle_length=FLAGS.num_parallel_readers))
    # shuffle the items as they pass through when size become: data_size(each epoch)
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
    batch_images, batch_labels = iterator.get_next()
    batch_labels = tf.reshape(batch_labels, [-1])
    return batch_images, tf.cast(batch_labels, tf.float32), iterator


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


def model(x, y, activation, h1_size, h2_size, h3_size, h4_size,
          h5_size, h6_size, h7_size, h8_size, h9_size, h10_size, istrain=True, reuse=False):
    with tf.variable_scope('Model', reuse=reuse):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
        else:
            bias_init = tf.zeros_initializer()
        h1 = tf.layers.dense(x,
                             h1_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        # h1 = tf.layers.dropout(h1, rate=FLAGS.dropoutRate, training=istrain)

        h2 = tf.layers.dense(h1,
                             h2_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        # h2 = tf.layers.dropout(h2, rate=FLAGS.dropoutRate, training=istrain)
        h3 = tf.layers.dense(h2,
                             h3_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        # h3 = tf.layers.dropout(h3, rate=FLAGS.dropoutRate, training=istrain)
        h4 = tf.layers.dense(h3,
                             h4_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        # h4 = tf.layers.dropout(h4, rate=FLAGS.dropoutRate, training=istrain)
        h5 = tf.layers.dense(h4,
                             h5_size,
                             activation=activation,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             bias_initializer=bias_init)
        # h6 = tf.layers.dense(h5,
        #                      h6_size,
        #                      activation=activation,
        #                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                      bias_initializer=bias_init)
        # h7 = tf.layers.dense(h6,
        #                      h7_size,
        #                      activation=activation,
        #                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                      bias_initializer=bias_init)
        # h8 = tf.layers.dense(h7,
        #                      h8_size,
        #                      activation=activation,
        #                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                      bias_initializer=bias_init)
        # h9 = tf.layers.dense(h8,
        #                      h9_size,
        #                      activation=activation,
        #                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                      bias_initializer=bias_init)
        # h10 = tf.layers.dense(h9,
        #                       h10_size,
        #                       activation=activation,
        #                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                       bias_initializer=bias_init)
        logits = tf.squeeze(tf.layers.dense(h5, 1,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer()))
        out_probs = tf.nn.sigmoid(logits)
        out_preds = tf.cast(tf.greater(out_probs, 0.5), tf.float32)
        correct_pred = tf.equal(y, out_preds)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return logits, accuracy, out_probs, out_preds


def run_training():
    # Tell TensorFlow that the model will be built into the default Graph.
    validation_file_list = [FLAGS.data_path + 'validation_scaled_' + '%d.tfrecords' % i for i in range(1, 11)]
    test_file_list = [FLAGS.data_path + 'test_scaled_' + '%d.tfrecords' % i for i in range(1, 11)]
    print(test_file_list)
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step(graph=None)
        with tf.device('/cpu:0'):
            batch_x, batch_labels, batch_iterator = inputs_train('train_scaled_*.tfrecords')
            print(batch_x.get_shape())
            print(batch_labels.dtype, '====================')
            validation_x, validation_labels, val_iterator = inputs(validation_file_list,
                                                                   FLAGS.valid_size,
                                                                   FLAGS.valid_size,
                                                                   False)
            test_x, test_labels, test_iterator = inputs(test_file_list,
                                                        FLAGS.test_size,
                                                        FLAGS.test_size,
                                                        False)
        print(batch_x.get_shape(), ', ', batch_labels.get_shape())
        with tf.device("/device:GPU:0"):
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
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=train_logits))
            learning_rate = tf.train.inverse_time_decay(FLAGS.learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=FLAGS.decay_steps,
                                                        decay_rate=FLAGS.decay_rate,
                                                        staircase=True
                                                        )
            train_op = tf.contrib.layers.optimize_loss(loss=cost,
                                                       global_step=global_step,
                                                       learning_rate=learning_rate,
                                                       optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                       # clip_gradients=2.0,
                                                       name='d_optimize_loss',
                                                       variables=tf.trainable_variables())
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
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # next_elem = batch_iterator.get_next()[0]
            # next_el = next_elem.eval()
            # np.savetxt(FLAGS.data_path + 'batch.txt', next_el, delimiter='\t')
            # print(np.shape(next_el))
            # print(batch_iterator.get_next()[1].eval()[-10:])
            b, l = sess.run([batch_x, batch_labels])
            np.savetxt(FLAGS.data_path + 'batch1.txt', b[:1000], delimiter='\t')

            print('Shape of Batch=', b.shape, l.shape, batch_x.get_shape(), batch_labels.get_shape())
            n_iterations = int(FLAGS.num_epochs * FLAGS.train_size // FLAGS.batch_size)
            print('Total iterations= ', n_iterations)
            best_validation_auc = 0
            for i in range(n_iterations):
                start_time = time.time()
                batch_loss, batch_accu, _ = sess.run([cost, train_accuracy, train_op])

                duration = time.time() - start_time
                if i % FLAGS.checkpoint_rate == 0:
                    print('Step %d: BATCH loss = %.2f, accuracy=%.2f (%.3f sec)' %
                          (i, batch_loss, batch_accu, duration))
                    t1 = time.time()
                    val_accu, val_probs, val_preds, val_labs = sess.run([validation_accuracy,
                                                                         validation_probs,
                                                                         validation_preds,
                                                                         validation_labels])
                    # print(val_probs[:20])
                    # print(val_labs[:20])
                    validation_pref = performance_statistics(val_labs, val_preds, val_probs)
                    if best_validation_auc < validation_pref['AUC']:
                        print('-' * 100)
                        print('validation AUC improved from {} to {} '.format(best_validation_auc,
                                                                              validation_pref['AUC']),
                              '(%.3f sec)' % (time.time()-t1))

                        # print('Train loss improved from {} to {} '.format(best_train_loss, train_loss))
                        print('Saving Model to file..')
                        print('-' * 100)
                        saver.save(sess,
                                   'C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                                   'excel-pbi-modeling/checkpoint/model.ckpt')
                        best_validation_auc = validation_pref['AUC']
            test_accu = sess.run(test_accuracy)
            print('Out of Sample Accuracy= ', test_accu)

        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/ExcelUsageData/tfrecords/scaled/',
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
        default=1024,
        help='hidden2_size'
    )
    parser.add_argument(
        '--h3',
        type=int,
        default=512,
        help='hidden3_size',
    )
    parser.add_argument(
        '--h4',
        type=int,
        default=512,
        help='hidden4_size'
    )
    parser.add_argument(
        '--h5',
        type=int,
        default=512,
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
        default=0.002,
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
        default=100,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=298242,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--valid_size',
        type=int,
        default=99414,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=99415,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--train_buffer_size',
        type=int,
        default=10000,
        help='min # of samples to have in buffer to perform shuffling'
    )
    parser.add_argument(
        '--checkpoint_rate',
        type=int,
        default=100,
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--num_parallel_readers',
        type=int,
        default=10,
        help='number of parallel readers to read from files.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

