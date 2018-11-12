import tensorflow as tf

'''
An MLP using tensorflow. 
Used data: MNIST dataset of handwritten digits
'''
# import MNIST data:
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist', one_hot = True)

train, validation, test = mnist

# parameters:
learning_rate = 0.001
# training_epochs = 10
batch_size = 100
display_steps = 1

# network parameters:
hidden1_size = 256
hidden2_size = 256
input_feats = 784  # 28 * 28 pictures with 1 channel!
n_classes = 10

# two functions to initialize and return weights.
# we can use different initializers as we see fit
# here we just use a simple method!


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def MLP(x, weights, biases, activation = tf.nn.relu,
        l2Reg = 0.01, curr_optimizer = tf.train.AdamOptimizer):
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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    cost_plus_l2Loss = cost + l2Loss
    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(cost_plus_l2Loss)
    return logits, l2Loss, optimizer, accuracy, cost, cost_plus_l2Loss

nRandom_start = 3
training_epochs = [2, 20, 200]

for j in range(nRandom_start):
    print('RUN %d optimization begins..' % (j+1))
    weights =\
        {
            'h1':  weight_variable([input_feats, hidden1_size]),
            'h2':  weight_variable([hidden1_size, hidden2_size]),
            'out': weight_variable([hidden2_size, n_classes])
        }

    biases = \
        {
            'b1':  bias_variable([hidden1_size]),
            'b2':  bias_variable([hidden2_size]),
            'out': bias_variable([n_classes])
        }
    x = tf.placeholder(tf.float32, [None, input_feats])
    y = tf.placeholder(tf.float32, [None, n_classes])
    logits, l2Loss, optimizer, accuracy, cost, cost_plus_l2Loss = MLP(x,
                                                                      weights,
                                                                      biases,
                                                                      activation=tf.nn.relu,
                                                                      l2Reg=0.01,
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
            for nBatchesh in range(10):
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
                avg_cost /= 10
                avg_accu /= 10

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














