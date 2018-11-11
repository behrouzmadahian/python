import tensorflow as tf
import argparse
import pandas as pd
FLAGS = None
N_FEATURES = 1500
train_min_max = pd.read_csv('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'
                            'excel-pbi-modeling/ExcelUsageData/Train_normalizing_param.csv', index_col=0).values
trainx_min = tf.constant(train_min_max[1], dtype=tf.float32, shape=[N_FEATURES])
trainx_max = tf.constant(train_min_max[0], dtype=tf.float32, shape=[N_FEATURES])
TRAIN_SIZE = 299952
VALIDATION_SIZE = 99984
TEST_SIZE = 99985
N_FEATURES = 1500
data_path = 'C:/behrouz/projects/behrouz-Rui-Gaurav-project/excel-pbi-modeling/ExcelUsageData/'
checkpoint_path = 'C:/behrouz/projects/behrouz-Rui-Gaurav-project/' \
                  'excel-pbi-modeling/imbalanced_batch/rs-checkpoint/train-ckpt/model.ckpt'
TRAIN_FILE = 'train'
VALIDATION_FILE = 'validation'
TEST_FILE = 'test'


def _scale_func(x_raw):
    """
    :param x_raw: unscaled data
    :return: scaled data using train statistics
    """
    f_range = tf.subtract(trainx_max, trainx_min)
    x_scaled = tf.subtract(x_raw, trainx_min)
    x_scaled = tf.div(x_scaled, f_range)
    return x_scaled


def model(x, y, activation, h1_size, h2_size, h3_size, reuse=False):
    x = _scale_func(x)
    with tf.variable_scope('Model', reuse=reuse):
        if activation == tf.nn.relu:
            bias_init = tf.constant_initializer(0.1)
            # relu initializer He et al 2015
            kernel_init = tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=False)
        else:
            bias_init = tf.zeros_initializer()
            kernel_init = tf.contrib.layers.xavier_initializer()
        h1 = tf.layers.dense(x,
                             h1_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             )
        h2 = tf.layers.dense(h1,
                             h2_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                            )
        h3 = tf.layers.dense(h2,
                             h3_size,
                             activation=activation,
                             kernel_initializer=kernel_init,
                             bias_initializer=bias_init,
                             )
        logits = tf.squeeze(tf.layers.dense(h3, 1,
                                            activation=None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer()))
        out_probs = tf.nn.sigmoid(logits)
        out_preds = tf.cast(tf.greater_equal(out_probs, 0.5), tf.float32)
        correct_pred = tf.equal(y, out_preds)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return logits, out_probs, out_preds, accuracy


def load_predict(data_file, ema_weights):
    data_set = pd.read_csv(data_path + data_file + '.csv', index_col='UserObjectId')
    x = tf.placeholder(tf.float32, shape=[None, N_FEATURES])
    y = tf.placeholder(tf.float32, [None])
    labels = data_set['Label'].values
    print(data_set.columns[1:10])
    features = data_set[data_set.columns[1:]].values
    print(features.shape)
    logits, probs, preds, accu = model(x,
                                       y,
                                       FLAGS.activation,
                                       FLAGS.h1,
                                       FLAGS.h2,
                                       FLAGS.h3,
                                       reuse=False
                                       )

    if ema_weights:
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        ema_dict = {}
        for var in tf.trainable_variables():
            ema_var_name = ema.average_name(var)
            ema_dict[ema_var_name] = var
        saver = tf.train.Saver(ema_dict)
    else:
        saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        l, prob, pred, accuracy = sess.run([logits, probs, preds, accu], feed_dict={x: features, y: labels})
        print(prob[:10])
        print('Accuracy= ', accuracy)
        results = pd.DataFrame({'UserObjectId':data_set.index.tolist(), 'Label': labels, 'probs': prob})
        results.to_csv('C:/behrouz/projects/behrouz-Rui-Gaurav-project/'\
                       'excel-pbi-modeling/imbalanced_batch/gpopt-checkpoint/results/' + data_file + 'predictions.csv',
                       index=False)
    tf.reset_default_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--activation',
        default=tf.nn.relu,
        help='activation function'
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
    FLAGS, unparsed = parser.parse_known_args()
    load_predict(TEST_FILE, ema_weights=True)
