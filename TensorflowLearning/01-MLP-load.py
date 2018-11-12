import tensorflow as tf

'''
An MLP using tensorflow. 
Used data: MNIST dataset of handwitten digits
'''

# import MNIST data:
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

train, validation, test = mnist

# network parameters:

hidden1_size = 256
hidden2_size = 256
input_feats = 784  # 28 * 28 pictures with 1 channel!
n_classes = 10

# two functions to initialize and return weights.
# we can use different intializers as we see fit
# here we just use a simple method!


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def MLP(x,
        weights,
        biases,
        activation = tf.nn.relu,
        ):
    '''
    :param x: placeholder tensor for input
    :param weights: dictionary of all the weight tensors in the model
    :param biases: dictionary of all the bias tensors in the model
    :return: returns the output of the model (just the logits before the softmax (linear output)!
    '''

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = activation(layer_2)

    logits = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    # calculating accuracy measure:
    # binary true false vector of length of training data

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return logits, accuracy


# assume we want to load the model in a session, and load weights from 5 different times
# and do testing
# this is akin to training models in for loop!
nRandomStart = 3
for i in range(nRandomStart):
    print('RUN %d loading graph begins..' % (i+1))

    with tf.Session() as sess:
        # define the placeholders and weight matrices.
        x = tf.placeholder(tf.float32, [None, input_feats])
        y = tf.placeholder(tf.float32, [None, n_classes])

        weights = \
            {
                'h1': weight_variable([input_feats, hidden1_size]),
                'h2': weight_variable([hidden1_size, hidden2_size]),
                'out': weight_variable([hidden2_size, n_classes])
            }

        biases = \
            {
                'b1': weight_variable([hidden1_size]),
                'b2': weight_variable([hidden2_size]),
                'out': weight_variable([n_classes])
            }
        logits, accuracy = MLP(x,
                              weights,
                              biases,
                              activation=tf.nn.relu)

        # we need to initialize before being able to restore weights from file!
        # init = tf.global_variables_initializer()
        # we perform sess.run(init) ONLY if we want to randomly initialize!
        # sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, './MLP-checkpointFiles/'
                            'MLP-tensors-checkPoint-run%d.ckpt' % (i + 1))

        print('Graph loaded for checkpoint file of RUN %d ' % (i + 1))
        # test the model on all test data using the loaded model
        testX = test.images
        testY = test.labels
        test_accu = sess.run(accuracy,
                             feed_dict={x: testX, y: testY})

        print('Test Accuracy= ', test_accu)
    tf.reset_default_graph()

