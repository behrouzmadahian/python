'''
A convolutional neural network using Tensorflow.
Used data: MNIST database of handwrittern digits.
'''

import tensorflow as tf

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# network parameters
n_inputs = 784  # MNIST data input image shape is 28x28=784
n_classes = 10  # MNIST data output classes are 10 (0-9 digits)
dropout = 0.75  # dropout (probability of keeping units)

# tf graph input
x = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
# create a wrapper for 2d convolutional network


def conv2d(x, w, b, strides=1):
    # 2d convolutional neural network, with bias and ReLu activation
    # strides: the stride of the sliding window for each dimension
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# create a wrapper for 2d max pooling

def maxpool2d(x, k=2):
    # 2d max pool
    # kzise: size of the window for each dimension;
    # strides: the stride of the sliding window for each dimension
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# create convolutional model

def conv_net(x, weights, biases, dropout):
    # reshape input
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # first convolutional layer
    conv1 = conv2d(x, weights['w1'], biases['b1'])
    # first max pooling layer
    conv1 = maxpool2d(conv1, k=2)

    # second convolutional layer
    conv2 = conv2d(conv1, weights['w2'], biases['b2'])
    # second max pooling layer
    conv2 = maxpool2d(conv2, k=2)

    # fully connected layer
    # reshape conv2 output to fit fully connected layer input
    fc = tf.reshape(conv2, [-1, weights['w_fc'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc'])
    fc = tf.nn.relu(fc)
    # dropout
    fc = tf.nn.dropout(fc, dropout)
    # output class prediction
    out = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return out


# store layers' weights and biases
weights = \
    {
        # 5x5 convolution (filter), 1 input channel, 32 outputs
        'w1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 convolution (filter), 32 inputs, 64 outputs
        'w2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected 7*7*64 inputs (max pool k=2 with SAME padding makes a 7x7 filters after two layers),
        #  1024 outputs
        'w_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

'''
a better practice for defining W is as follows:
def weights_variable(shape):
    initial = tf.truncated_random(shape, stddev=0.1)
    return tf.Variable(initial)

# a better practice for defining bias is as follows:
def bias_variable(shape):
     initial = tf.constant(0.1, shape=shape)
     return tf.Variable(initial)
'''

biases =\
    {
        'b1': tf.Variable(tf.random_normal([32])),
        'b2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

# prediction model
pred = conv_net(x, weights, biases, keep_prob)

# cost and optimization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# evaluate model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize variables
init = tf.global_variables_initializer()

# launch default graph
with tf.Session() as session:
    session.run(init)
    step = 1
    # training cycle
    while step * batch_size < training_iters:
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        # run optimization
        session.run(optimizer, feed_dict={x: x_batch, y: y_batch, keep_prob: dropout})
        if step % display_step == 0:
            batch_cost, batch_accuracy = session.run([cost, accuracy],
                                                     feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
            print("Iteration:", (step * batch_size), "Batch cost=", batch_cost, "Training accuracy=", batch_accuracy)
        step += 1

    print("Optimization finished!")

    # test model with test data (for 256 images)
    print("Testing accuracy=", session.run(accuracy,
                                           feed_dict={x: mnist.test.images[:256],
                                                      y: mnist.test.labels[:256],
                                                      keep_prob: 1.0}))