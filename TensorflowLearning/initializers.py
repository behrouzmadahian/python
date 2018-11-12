import numpy as np
import tensorflow as tf


def glorot_normal_weight_initializer(shape, name=None):
    ''' Use for tanh activation'''
    # Glorot et al. 2012
    initial = tf.random_normal(shape, stddev=np.sqrt(3. / (shape[0] + shape[1])))
    if name:
        return tf.Variable(initial, name=name)

    else:
        return tf.Variable(initial)


def glorot_uniform_weight_initializer(shape, name = None):
    ''' Use for tanh activation'''
    limit = np.sqrt(6. / (shape[0] + shape[1]) )
    initial = tf.random_uniform(shape, minval=- limit, maxval=limit)
    if name:
        return tf.Variable(initial, name = name)

    else:
        return tf.Variable(initial)


def xavier_from_tf_initializer(shape, name = None):
    ''' Use for tanh activation'''
    return tf.get_variable(name=name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer())


def bias_initializer(shape, name = None):
    initial = tf.constant(0., shape=shape)
    if name:
        return tf.Variable(initial, name = name)

    else:
        return tf.Variable(initial)


def relu_weight_initializer(shape, name=None):

    # He et al 2015
    # values whose magnitude is more than 2 standard deviations
    # from the mean are dropped and re-picked

    initial = tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[0]))
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)



def relu_bias_initializer(shape, name = None):
    initial = tf.constant(0.1, shape=shape)
    if name:
        return tf.Variable(initial, name = name)

    else:
        return tf.Variable(initial)

def variable_summaries(var, varname):
    with tf.name_scope(varname+'-summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.reduce_mean(tf.square(var - mean))
            tf.summary.scalar('stddev', stddev)

        tf.summary.histogram('Histogram', var)
