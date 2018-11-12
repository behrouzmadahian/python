import tensorflow as tf
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
'''
Improving linear classifiers with kernel methods:
Map the feaures to higher dimensional space using Random Fourier feature mapper.
Its very sentitive to the choice of standard deviation of kernel
MUST do hyper param optimization to choose this hyper parameter.
'''



'''
An MLP using tensorflow. 
Used data: MNIST dataset of handwritten digits
'''
# import MNIST data:
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot=True)

train, validation, test = mnist

# parameters:
learning_rate = 0.001
# training_epochs = 10
batch_size = 256
display_steps = 1

# network parameters:
hidden1_size = 256
hidden2_size = 256
input_feats = 784  # 28 * 28 pictures with 1 channel!
n_classes = 10
kernel_output_dim = 2000
n_batches = 200

# two functions to initialize and return weights.
# we can use different initializers as we see fit
# here we just use a simple method!


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def MLP(x,
        weights,
        biases,
        activation=tf.nn.relu,
        l2Reg=0.01,
        curr_optimizer=tf.train.AdamOptimizer):
    '''
    :param x: placeholder tensor for input
    :param weights: dictionary of all the weight tensors in the model
    :param biases: dictionary of all the bias tensors in the model
    :return: returns the output of the model (just the logits before the softmax (linear output)!
    '''
    # mapping features to higher dimension:
    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
        input_dim=784, output_dim=kernel_output_dim, stddev=5.0, name='rffm')
    x = kernel_mapper.map(x)
    print('Shape of mapped features: ', x.get_shape())

    logits = tf.add(tf.matmul(x, weights['out']), biases['out'])
    # l2 regularization loss:
    l2Loss = 0
    for key in weights.keys():
        l2Loss += tf.nn.l2_loss(weights[key])
    for key in biases.keys():
        l2Loss += tf.nn.l2_loss(biases[key])
    l2Loss *= l2Reg
    # calculating accuracy measure:
    # binary true false vector of length of training data
    correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # cost function and optimization
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    cost_plus_l2Loss = cost + l2Loss
    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(cost_plus_l2Loss)
    return logits, l2Loss, optimizer, accuracy, cost, cost_plus_l2Loss


nRandom_start = 1
training_epochs = [2000, 20, 200]

for j in range(nRandom_start):
    print('RUN %d optimization begins..' % (j+1))
    weights =\
        {
            'h1':  weight_variable([784, hidden1_size]),
        }

    biases = \
        {
            'out': bias_variable([n_classes])
        }
    x = tf.placeholder(tf.float32, [None, input_feats])
    y = tf.placeholder(tf.float32, [None, n_classes])
    logits, l2Loss, optimizer, accuracy, cost, cost_plus_l2Loss = MLP(x,
                                                                      weights,
                                                                      biases,
                                                                      activation=tf.nn.relu,
                                                                      l2Reg=0.002,
                                                                      curr_optimizer=tf.train.AdamOptimizer)
    # initialize all tensors- to be run in Session!
    init = tf.global_variables_initializer()
    # saver for restoring the whole model graph of
    #  tensors from the  checkpoint file <
    saver = tf.train.Saver()
    # launch default graph:
    with tf.Session() as sess:
        sess.run(init)
        # training cycle:
        for epoch in range(training_epochs[j]):
            avg_cost = 0.0
            avg_accu = 0

            # total_batches = int(mnist.train.num_examples / batch_size)
            # i want to train for 10 batches per epoch

            # loop over all batches:
            for nBatchesh in range(n_batches):
                x_batch, y_batch = train.next_batch(batch_size)
                # run optimization
                _, batch_data_cost, batch_accu = sess.run([optimizer, cost, accuracy],
                                                          feed_dict={x: x_batch, y: y_batch})
                # compute total cost:
                avg_cost += batch_data_cost
                avg_accu += batch_accu
            # display logs every display step epochs:
            # display_steps = 1:so it correctly returns averages!
            if epoch % display_steps == 0:
                avg_cost /= n_batches
                avg_accu /= n_batches

                print('Epoch:', epoch, 'cost=', avg_cost, 'Accuracy=', avg_accu)
                print(sess.run([cost, accuracy],
                               feed_dict={x: train.images, y: train.labels}))

                print('Test performance: data cost, accuracy')
                print(sess.run([cost, accuracy], feed_dict={x: test.images, y: test.labels}))

        print('Optimization finished!')
        print(' saving model graph of all tensors to file')
        save_path = saver.save(sess, './MLP-checkpointFiles/MLP-tensors-checkPoint-run%d.ckpt'%
                               (j+1))

    # resetting the graph to be built again in the next iteration of for loop
    tf.reset_default_graph()














